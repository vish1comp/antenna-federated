"""Two ablations for the paper (same budget, seeds and partitions as fl_antenna.py run).

  python ablate.py phys --ds B --seed 0
      physics-informed inputs REMOVED (geometry + frequency only); centralized and FedAvg on the
      IID K=5 partition. The with-physics numbers come from the main run (same seeds/partitions).
      -> ../results/ablation_phys_{ds}_s{seed}.csv

  python ablate.py prox --seed 0
      FedProx with the COUPLED proximal loss (mu/2 ||w - w_g||^2 added to the loss and passed
      through Adam), C shift K=5, 50 rounds; global test R2 every 5 rounds, as in fl_curves.
      -> ../results/ablation_prox_s{seed}.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

import fl_antenna as F

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ROUNDS, LOCAL, CENTRAL = 50, 200, 20000

ap = argparse.ArgumentParser()
ap.add_argument("what", choices=["phys", "prox"])
ap.add_argument("--ds", default="C")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--mu", type=float, default=0.01)
a = ap.parse_args()
torch.set_num_threads(a.threads)


def per_client_r2(model, ds, clients):
    return [F.summarise(F.sweep_scores(model, ds, c["test"]))["r2"] for c in clients]


if a.what == "phys":
    _orig = F.features

    def features_nophys(ds, idx):                     # drop the two electrical-size columns
        X, y = _orig(ds, idx)
        return X[:, :-2].contiguous(), y

    F.features = features_nophys                      # sweep_scores looks it up at call time
    ds = F.load_B() if a.ds == "B" else F.load_C()
    clients = F.partition(ds, 5, "iid", a.seed)
    d = ds["G"].shape[1] + 1
    data = [dict(zip(("X", "y"), F.features(ds, c["train"]))) for c in clients]
    rows = []
    Xc = torch.cat([c["X"] for c in data]); yc = torch.cat([c["y"] for c in data])
    cen = F.train_steps(F.make_model(d, a.seed), Xc, yc, CENTRAL, a.seed)
    for k, r in enumerate(per_client_r2(cen, ds, clients)):
        rows.append(dict(dataset=a.ds, seed=a.seed, inputs="no_physics", method="central", client=k, r2=r))
    print("central done", flush=True)
    g = F.federated(data, d, ROUNDS, LOCAL, a.seed)
    for k, r in enumerate(per_client_r2(g, ds, clients)):
        rows.append(dict(dataset=a.ds, seed=a.seed, inputs="no_physics", method="fedavg", client=k, r2=r))
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"ablation_phys_{a.ds}_s{a.seed}.csv"), index=False)
    print(pd.DataFrame(rows).groupby("method").r2.mean(), flush=True)

else:
    def train_steps_coupled(model, X, y, steps, seed, lr=None, prox_ref=None, mu=0.0, batch=F.BATCH):
        opt = torch.optim.Adam(model.parameters(), lr=F.LR_MAX if lr is None else lr)
        g = torch.Generator().manual_seed(seed)
        ref = [p.detach().clone() for p in prox_ref.parameters()] if prox_ref is not None else None
        model.train()
        for s in range(steps):
            if lr is None:
                for pg in opt.param_groups:
                    pg["lr"] = F.cos_lr(s / steps)
            j = torch.randint(0, len(X), (batch,), generator=g)
            loss = ((model(X[j]).squeeze(1) - y[j]) ** 2).mean()
            if ref is not None and mu > 0:
                loss = loss + 0.5 * mu * sum(((p - r) ** 2).sum() for p, r in zip(model.parameters(), ref))
            opt.zero_grad()
            loss.backward()
            opt.step()
        return model

    F.train_steps = train_steps_coupled
    ds = F.load_C()
    clients = F.partition(ds, 5, "shift", a.seed)
    d = ds["G"].shape[1] + 3
    data = [dict(zip(("X", "y"), F.features(ds, c["train"]))) for c in clients]
    all_test = np.concatenate([c["test"] for c in clients])
    log = []
    ev = lambda m: dict(r2=F.summarise(F.sweep_scores(m, ds, all_test))["r2"])
    F.federated(data, d, ROUNDS, LOCAL, a.seed, mu=a.mu, log=log, eval_fn=ev)
    df = pd.DataFrame(log).assign(seed=a.seed, method="fedprox_coupled", mu=a.mu)
    df.to_csv(os.path.join(RES, f"ablation_prox_s{a.seed}.csv"), index=False)
    print(df, flush=True)
