"""Federated antenna surrogates with per-client conformal certificates.

Setting: K labs each hold their own full-wave simulations (designs + S11 sweeps) and will not
pool them. They train one shared S11 surrogate by federated learning (only model weights leave
a lab) and certify its predictions with conformal bounds computed without sharing data.

Datasets
  B  HF rectangular patch (real full-wave, 2051 designs x 1000 freq; 100 freq kept, 1-10 GHz)
  C  transmission-line-model patch (synthetic, 1500 designs x 64 freq, substrate eps_r varies)

Partitions of DESIGNS (never rows, so no design is split across train/test or across clients)
  iid       designs dealt at random to clients
  shift     B: clients = contiguous bands of the design's resonant frequency
            C: clients = contiguous ranges of substrate permittivity
Within each client: 60 / 20 / 20 % of its designs -> train / calibration / test.

Training methods (same 3x256 SiLU MLP, same number of steps per client)
  local      each client trains on its own data only (10k steps)
  fedavg     FedAvg (McMahan et al. 2017), 50 rounds x 200 local steps
  fedprox    FedProx (Li et al. 2020) with a DECOUPLED proximal step after Adam, mu = 0.01
  fedavg_ft  FedAvg then 1000 local fine-tuning steps at lr 2e-4 (personalised)
  central    pooled data, 20k steps (upper bound; violates the privacy constraint)

Conformal certificates (design-level, so exchangeability is over designs, not correlated rows)
  scores    sup    = max_f |S11 - S11_hat|        (dB, simultaneous band over the sweep)
            rmse_d = per-design sweep RMSE          (dB)
            dfr    = resonant-frequency error      (GHz)
  local-CP  each client uses only its own calibration designs
  pooled-CP all calibration scores pooled (needs scores to be shared; reference)
  QQ-CP     quantile of quantiles, one communication round (Humbert et al. 2023); each client
            sends one number. Orders (l, k) chosen by distribution-free rank simulation.

    python fl_antenna.py run --seed 0 [--threads 2] [--quick]   -> ../results/fl_*_s0.csv
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
B_PARQUET = os.path.join(HERE, "..", "data", "rect_patch.parquet")
B_URL = ("https://huggingface.co/datasets/becklabash/rectangular-patch-antenna-freq-response/"
         "resolve/main/data/train-00000-of-00001.parquet")
ALPHA = 0.10
Y_SHIFT, Y_SCALE = -10.0, 10.0          # fixed, public output scaling (dB)


# ======================================================================== data
def load_B(n_freq=100):
    if not os.path.exists(B_PARQUET):            # first run: fetch the public dataset once
        os.makedirs(os.path.dirname(B_PARQUET), exist_ok=True)
        print("downloading dataset B from Hugging Face ...", flush=True)
        pd.read_parquet(B_URL).to_parquet(B_PARQUET)
    df = pd.read_parquet(B_PARQUET)
    F = np.stack(df["frequencies"].values) / 1e9
    S = np.stack(df["s11"].values).astype(np.float32)
    keep = np.linspace(0, F.shape[1] - 1, n_freq).round().astype(int)
    F, S = F[:, keep], S[:, keep]
    G = np.stack([df["length"], df["width"], df["feed_y"]], 1).astype(np.float32)
    fr = F[np.arange(len(S)), S.argmin(1)]
    return dict(name="B", G=G, F=F.astype(np.float32), S=S, shift_key=fr,
                gbounds=[(7.5, 47.5), (6.0, 95.0), (-15.833, 0.0)], fbounds=(1.0, 10.0),
                elec=lambda G: G[:, :2], elec_max=np.array([47.5, 95.0], np.float32))


def load_C(n_designs=1500, n_freq=64, seed=0):
    """Cavity / transmission-line patch model (as dataset C of the journal paper)."""
    rng = np.random.default_rng(seed)
    c, Z0 = 3e8, 50.0
    G, F, S = [], [], []
    for _ in range(n_designs):
        L, W = rng.uniform(28, 45), rng.uniform(35, 50)
        h, er = rng.uniform(0.8, 1.6), rng.uniform(2.2, 6.15)
        y0 = rng.uniform(0, 0.35 * L)
        eeff = (er + 1) / 2 + (er - 1) / 2 * (1 + 12 * h / W) ** -0.5
        dL = 0.412 * h * (eeff + 0.3) * (W / h + 0.264) / ((eeff - 0.258) * (W / h + 0.8))
        fr = c / (2 * (L + 2 * dL) / 1000.0 * np.sqrt(eeff))
        Q = 28.0 * (1.6 / h)
        Rin = max(8.0, 90.0 * (er ** 2 / (er - 1)) * (L / W) * np.cos(np.pi * y0 / L) ** 2)
        f = np.linspace(fr * 0.85, fr * 1.15, n_freq)
        Zin = Rin / (1 + 1j * Q * (f / fr - fr / f))
        s = 20 * np.log10(np.abs((Zin - Z0) / (Zin + Z0)) + 1e-9) + rng.normal(0, 0.1, n_freq)
        G.append([L, W, h, er, y0]); F.append(f / 1e9); S.append(s)
    G = np.asarray(G, np.float32)
    return dict(name="C", G=G, F=np.asarray(F, np.float32), S=np.asarray(S, np.float32),
                shift_key=G[:, 3], gbounds=[(28, 45), (35, 50), (0.8, 1.6), (2.2, 6.15), (0, 15.75)],
                fbounds=(1.0, 4.5),
                elec=lambda G: G[:, :2] * np.sqrt(G[:, 3:4]),
                elec_max=np.array([45 * 6.15 ** 0.5, 50 * 6.15 ** 0.5], np.float32))


def features(ds, idx):
    """rows (design x freq) -> scaled inputs by the PUBLIC design-space bounds (no stats shared)"""
    G, F = ds["G"][idx], ds["F"][idx]
    lo = np.array([b[0] for b in ds["gbounds"]], np.float32)
    hi = np.array([b[1] for b in ds["gbounds"]], np.float32)
    g = (G - lo) / (hi - lo)
    f = (F - ds["fbounds"][0]) / (ds["fbounds"][1] - ds["fbounds"][0])
    nf = F.shape[1]
    # physics-informed inputs: electrical size f*L, f*W (times sqrt(eps_r) when it varies);
    # the patch resonances sit at fixed values of these, so dips line up across designs
    el = ds["elec"](G) / ds["elec_max"]                          # (n, 2)
    fe = (F / ds["fbounds"][1])[:, :, None] * el[:, None, :]     # (n, nf, 2)
    X = np.concatenate([np.repeat(g, nf, 0), f.reshape(-1, 1), fe.reshape(-1, 2)], 1)
    y = (ds["S"][idx].reshape(-1) - Y_SHIFT) / Y_SCALE
    return torch.tensor(X), torch.tensor(y.astype(np.float32))


def partition(ds, K, mode, seed):
    n = len(ds["G"])
    rng = np.random.default_rng(seed)
    if mode == "iid":
        order = rng.permutation(n)
    else:                                   # contiguous ranges of the shift key
        order = np.argsort(ds["shift_key"] + 1e-6 * rng.random(n), kind="stable")
    groups = np.array_split(order, K)
    clients = []
    for g in groups:
        g = rng.permutation(g)
        a, b = int(0.6 * len(g)), int(0.8 * len(g))
        clients.append(dict(train=g[:a], cal=g[a:b], test=g[b:]))
    return clients


# ======================================================================== model
def make_model(d, seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                         nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1))


LR_MAX, LR_MIN, BATCH = 2e-3, 2e-5, 1024


def cos_lr(frac):
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + np.cos(np.pi * min(frac, 1.0)))


def train_steps(model, X, y, steps, seed, lr=None, prox_ref=None, mu=0.0, batch=BATCH):
    """`steps` minibatch Adam steps. lr=None -> cosine schedule LR_MAX -> LR_MIN over the
    steps; a number -> constant (used inside one federated round)."""
    opt = torch.optim.Adam(model.parameters(), lr=LR_MAX if lr is None else lr)
    g = torch.Generator().manual_seed(seed)
    ref = [p.detach().clone() for p in prox_ref.parameters()] if prox_ref is not None else None
    model.train()
    for s in range(steps):
        if lr is None:
            for pg in opt.param_groups:
                pg["lr"] = cos_lr(s / steps)
        j = torch.randint(0, len(X), (batch,), generator=g)
        loss = ((model(X[j]).squeeze(1) - y[j]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ref is not None and mu > 0:
            # FedProx proximal term, DECOUPLED from Adam (as AdamW decouples weight decay):
            # w <- w - lr * mu * (w - w_global). Adding mu/2 ||w - w_g||^2 to the loss instead
            # lets Adam rescale the proximal gradient to full step size wherever the data
            # gradient is small, which froze training on the early plateau (see tune_mu.py).
            cur_lr = opt.param_groups[0]["lr"]
            with torch.no_grad():
                for p, r in zip(model.parameters(), ref):
                    p.sub_(cur_lr * mu * (p - r))
    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[i:i + 65536]).squeeze(1) for i in range(0, len(X), 65536)])


def average(models, weights):
    w = np.asarray(weights, float) / np.sum(weights)
    avg = [sum(wk * m.state_dict()[k] for wk, m in zip(w, models)) for k in models[0].state_dict()]
    return dict(zip(models[0].state_dict().keys(), avg))


def copy_model(src, d):
    m = make_model(d, 0)
    m.load_state_dict(src.state_dict())
    return m


def federated(data, d, rounds, local_steps, seed, mu=0.0, log=None, eval_fn=None):
    glob = make_model(d, seed)
    sizes = [len(c["y"]) for c in data]
    for r in range(rounds):
        locs = []
        lr = cos_lr(r / rounds)                  # server-side cosine schedule over rounds
        for k, c in enumerate(data):
            m = copy_model(glob, d)
            train_steps(m, c["X"], c["y"], local_steps, seed * 1000 + r * 31 + k, lr=lr,
                        prox_ref=glob if mu > 0 else None, mu=mu)
            locs.append(m)
        glob.load_state_dict(average(locs, sizes))
        if log is not None and eval_fn is not None and (r % 5 == 4 or r == rounds - 1):
            log.append(dict(round=r + 1, **eval_fn(glob)))
    return glob


# ======================================================================== evaluation
def sweep_scores(model, ds, idx):
    """per design: sup-norm sweep error (dB), resonance-frequency error (GHz), RMSE parts"""
    X, y = features(ds, idx)
    p = predict(model, X).numpy() * Y_SCALE + Y_SHIFT
    t = y.numpy() * Y_SCALE + Y_SHIFT
    nf = ds["F"].shape[1]
    P, T = p.reshape(-1, nf), t.reshape(-1, nf)
    Fr = ds["F"][idx]
    rows = np.arange(len(idx))
    dfr = np.abs(Fr[rows, P.argmin(1)] - Fr[rows, T.argmin(1)])
    return dict(sup=np.abs(P - T).max(1), rmse_d=np.sqrt(((P - T) ** 2).mean(1)), dfr=dfr, P=P,
                se=((P - T) ** 2).sum(), n=P.size,
                ss=((T - T.mean()) ** 2).sum(), tsum=T.sum(), t2=(T ** 2).sum())


def summarise(sc):
    rmse = np.sqrt(sc["se"] / sc["n"])
    r2 = 1 - sc["se"] / sc["ss"]
    return dict(rmse=rmse, r2=r2, sup_med=float(np.median(sc["sup"])), dfr_med=float(np.median(sc["dfr"])))


def conformal_level(n, alpha=ALPHA):
    return int(min(np.ceil((n + 1) * (1 - alpha)), n))     # order statistic (1-based)


_QQ_CACHE = {}


def qq_orders(m, n, alpha=ALPHA, sims=30_000, seed=0):
    """(l, k): each client sends its l-th smallest score, server takes the k-th smallest of
    those m values. Under exchangeability the coverage depends only on ranks, so it is
    computed exactly enough by simulating uniform scores. Pick the valid pair with the
    smallest expected rank (tightest bound)."""
    key = (m, n, alpha)
    if key in _QQ_CACHE:
        return _QQ_CACHE[key]
    rng = np.random.default_rng(seed)
    U = np.sort(rng.random((sims, m, n)), axis=2)
    test = rng.random(sims)
    best = None
    for l in range(1, n + 1):
        Ql = np.sort(U[:, :, l - 1], axis=1)              # (sims, m)
        for k in range(1, m + 1):
            q = Ql[:, k - 1]
            cov = (test <= q).mean()
            if cov >= 1 - alpha + 0.002:                  # small margin for MC error
                if best is None or q.mean() < best[2]:
                    best = (l, k, q.mean(), cov)
                break                                     # larger k only widens
    _QQ_CACHE[key] = best[:2] if best else (n, m)
    return _QQ_CACHE[key]


SCORES = {"sup": "max |error| over the sweep (dB)", "rmse_d": "per-design sweep RMSE (dB)",
          "dfr": "resonant-frequency error (GHz)"}
PRED_STORE = {}


def certificates(model_for, ds, clients, tag):
    """model_for(k) -> model used by client k. Returns per-client coverage / width rows for
    each score in SCORES; also keeps the predictions in PRED_STORE for later re-analysis."""
    cal_sc = [sweep_scores(model_for(k), ds, c["cal"]) for k, c in enumerate(clients)]
    tst_sc = [sweep_scores(model_for(k), ds, c["test"]) for k, c in enumerate(clients)]
    PRED_STORE[tag] = dict(cal=[s["P"].astype(np.float16) for s in cal_sc],
                           test=[s["P"].astype(np.float16) for s in tst_sc])
    m = len(clients)
    rows = []
    for score in SCORES:
        cal = [s[score] for s in cal_sc]
        tst = [s[score] for s in tst_sc]
        n = min(len(s) for s in cal)
        pooled = np.sort(np.concatenate(cal))
        q_pool = pooled[conformal_level(len(pooled)) - 1]
        l, kk = qq_orders(m, n)
        q_qq = np.sort([np.sort(s[:n])[l - 1] for s in cal])[kk - 1]
        for k in range(m):
            q_loc = np.sort(cal[k])[conformal_level(len(cal[k])) - 1]
            for meth, q in (("local-CP", q_loc), ("pooled-CP", q_pool), ("QQ-CP", q_qq)):
                rows.append(dict(model=tag, client=k, score=score, cp=meth, q=float(q),
                                 coverage=float((tst[k] <= q).mean()), n_test=len(tst[k])))
    return rows


# ======================================================================== experiment
def run_config(ds, K, mode, seed, rounds, local_steps, central_steps, ft_steps):
    """Budget: every client takes rounds*local_steps minibatch steps in local-only, FedAvg and
    FedProx alike; the pooled model takes central_steps."""
    clients = partition(ds, K, mode, seed)
    d = ds["G"].shape[1] + 3
    data = []
    for c in clients:
        X, y = features(ds, c["train"])
        data.append(dict(X=X, y=y))
    all_test = np.concatenate([c["test"] for c in clients])
    client_steps = rounds * local_steps
    out, curves, certs = [], [], []

    def evaluate(tag, model_for, t_fit, comm_mb=0.0):
        for k, c in enumerate(clients):
            own = summarise(sweep_scores(model_for(k), ds, c["test"]))
            other = np.concatenate([cc["test"] for j, cc in enumerate(clients) if j != k])
            oth = summarise(sweep_scores(model_for(k), ds, other))
            out.append(dict(dataset=ds["name"], K=K, partition=mode, seed=seed, method=tag, client=k,
                            n_train_designs=len(c["train"]), r2_own=own["r2"], rmse_own=own["rmse"],
                            sup_own=own["sup_med"], dfr_own=own["dfr_med"], r2_other=oth["r2"],
                            rmse_other=oth["rmse"], dfr_other=oth["dfr_med"], fit_s=t_fit, comm_mb=comm_mb))
        certs.extend(dict(dataset=ds["name"], K=K, partition=mode, seed=seed, **r)
                     for r in certificates(model_for, ds, clients, tag))

    n_par = sum(p.numel() for p in make_model(d, 0).parameters())
    mb_round = 2 * K * n_par * 4 / 1e6                   # up + down, float32

    t = time.time()
    Xc = torch.cat([c["X"] for c in data]); yc = torch.cat([c["y"] for c in data])
    cen = train_steps(make_model(d, seed), Xc, yc, central_steps, seed)
    evaluate("central", lambda k: cen, time.time() - t)

    t = time.time()
    locs = [train_steps(make_model(d, seed), c["X"], c["y"], client_steps, seed + 17 * k)
            for k, c in enumerate(data)]
    evaluate("local", lambda k: locs[k], time.time() - t)

    def curve_eval(tag):
        def f(model):
            s = summarise(sweep_scores(model, ds, all_test))
            return dict(dataset=ds["name"], K=K, partition=mode, seed=seed, method=tag,
                        r2=s["r2"], rmse=s["rmse"])
        return f

    for tag, mu in (("fedavg", 0.0), ("fedprox", 0.01)):
        t = time.time()
        log = []
        g = federated(data, d, rounds, local_steps, seed, mu=mu, log=log, eval_fn=curve_eval(tag))
        curves.extend(log)
        evaluate(tag, lambda k, g=g: g, time.time() - t, mb_round * rounds)
        if tag == "fedavg":
            ft = []
            for k, c in enumerate(data):
                m = copy_model(g, d)
                ft.append(train_steps(m, c["X"], c["y"], ft_steps, seed + 99 + k, lr=2e-4))
            evaluate("fedavg_ft", lambda k: ft[k], time.time() - t, mb_round * rounds)
    store = {f"{tag}__{part}__{k}": arr for tag, v in PRED_STORE.items()
             for part, lst in v.items() for k, arr in enumerate(lst)}
    for k, c in enumerate(clients):
        for part in ("cal", "test"):
            store[f"idx__{part}__{k}"] = c[part]
    os.makedirs(os.path.join(RES, "preds"), exist_ok=True)
    np.savez_compressed(os.path.join(RES, "preds", f"{ds['name']}_K{K}_{mode}_s{seed}.npz"), **store)
    PRED_STORE.clear()
    return out, curves, certs


def append_csv(name, rows):
    if not rows:
        return
    path = os.path.join(RES, name)
    pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def done(seed, dsname, K, mode):
    path = os.path.join(RES, f"fl_clients_s{seed}.csv")
    if not os.path.exists(path):
        return False
    df = pd.read_csv(path)
    return bool(((df.dataset == dsname) & (df.K == K) & (df.partition == mode)).any())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(RES, exist_ok=True)
    # rounds, local steps / round, central steps, fine-tune steps
    budget = (3, 20, 120, 20) if a.quick else (50, 200, 20000, 1000)
    plan = [("C", 5, "iid"), ("C", 5, "shift"), ("B", 5, "iid"), ("B", 5, "shift"),
            ("B", 3, "shift"), ("B", 10, "shift")]
    cache = {}
    sfx = ("_quick" if a.quick else "") + f"_s{a.seed}"
    for dsname, K, mode in plan:
        if not a.quick and done(a.seed, dsname, K, mode):
            continue
        if dsname not in cache:
            cache[dsname] = load_B() if dsname == "B" else load_C()
        t = time.time()
        out, curves, certs = run_config(cache[dsname], K, mode, a.seed, *budget)
        append_csv(f"fl_clients{sfx}.csv", out)
        append_csv(f"fl_curves{sfx}.csv", curves)
        append_csv(f"fl_certs{sfx}.csv", certs)
        print(f"{dsname} K={K} {mode} seed={a.seed}: {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
