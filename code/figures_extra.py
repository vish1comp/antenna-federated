"""Additional figures for the paper from saved results (no retraining).

fl_framework.png    schematic of the protocol
fl_data.png         example sweeps + how the shifted partition splits the design space
fl_examples.png     predicted vs full-wave/cavity sweeps of held-out designs, with certificate band
fl_perclient.png    own-range R2 per client and per-design error vs permittivity (C shift)
fl_K.png            own-band R2 vs number of clients (B shift)
fl_tradeoff.png     worst-client coverage vs threshold width, per calibration rule
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import fl_antenna as F

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "..", "results"), os.path.join(HERE, "..", "figures")
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
ORDER = ["local", "fedavg", "fedprox", "fedavg_ft", "central"]
NAMES = {"local": "Local only", "fedavg": "FedAvg", "fedprox": "FedProx",
         "fedavg_ft": "FedAvg + fine-tune", "central": "Centralized"}
COL = {"local": "#9a6b2f", "fedavg": "#1f6f8b", "fedprox": "#4a9c7a", "fedavg_ft": "#6b4fa0",
       "central": "#7a7a7a"}
CPCOL = {"local-CP": "#9a6b2f", "pooled-CP": "#7a7a7a", "QQ-CP": "#1f6f8b"}
CPMK = {"local-CP": "o", "pooled-CP": "s", "QQ-CP": "^"}


def load(stem):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RES, f"{stem}_s[0-9].csv")))],
                     ignore_index=True)


cl, ce = load("fl_clients"), load("fl_certs")
DS = {"B": F.load_B(), "C": F.load_C()}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ======================================================================== framework schematic
def box(ax, x, y, w, h, text, fc, ec="#333333", fs=7.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def arrow(ax, p, q, txt=None, col="#333333", rad=0.0, off=(0, 0)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=0.9, color=col,
                                 connectionstyle=f"arc3,rad={rad}"))
    if txt:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], txt, fontsize=6.5,
                ha="center", va="center", color=col)


fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.7); ax.axis("off")
box(ax, 3.6, 3.75, 2.8, 0.75, "Server\nweighted average of weights\n(never sees designs)", "#e3eef3", bold=False)
labs = [("Lab 1", "band / substrate 1"), ("Lab k", "band / substrate k"), ("Lab K", "band / substrate K")]
for i, (nm, rng) in enumerate(labs):
    x = 0.15 + i * 3.35
    ax.add_patch(FancyBboxPatch((x, 0.15), 3.0, 2.75, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#fbf7f0", ec="#9a6b2f", lw=1.0, ls="-"))
    ax.text(x + 1.5, 2.72, f"{nm}  ({rng})", ha="center", fontsize=7.5, fontweight="bold")
    box(ax, x + 0.15, 1.95, 2.7, 0.55, "private archive (g, |S11| sweeps)\nsplit by design: 60/20/20 %",
        "#ffffff", fs=6.3)
    box(ax, x + 0.15, 1.12, 1.3, 0.6, "local Adam\nsteps (+ prox.)\n& fine-tune", "#ede7f6", fs=6.3)
    box(ax, x + 1.55, 1.12, 1.3, 0.6, "local conformal\ncalibration\n→ threshold $q_k$", "#f3ece0", fs=6.3)
    box(ax, x + 0.15, 0.3, 2.7, 0.58, "new design:  $\\hat{s}(g)$  with  $e(g)\\leq q_k$\n(90 % certificate, no data shared)",
        "#eef6ee", fs=6.5)
    arrow(ax, (x + 0.8, 1.95), (x + 0.8, 1.74))
    arrow(ax, (x + 2.2, 1.95), (x + 2.2, 1.74))
    arrow(ax, (x + 0.8, 1.12), (x + 0.8, 0.9))
    arrow(ax, (x + 2.2, 1.12), (x + 2.2, 0.9))
    cx = x + 1.5
    arrow(ax, (cx - 0.25, 2.92), (4.4 + i * 0.6, 3.73), col="#1f6f8b", rad=0.0)
    arrow(ax, (4.6 + i * 0.6, 3.73), (cx + 0.25, 2.92), col="#6b4fa0", rad=0.0)
ax.text(1.4, 3.55, "upload local weights", fontsize=6.5, color="#1f6f8b")
ax.text(7.0, 3.55, "download global weights\n(R = 50 rounds)", fontsize=6.5, color="#6b4fa0")
ax.text(3.33, 1.5, "…", fontsize=14, ha="center"); ax.text(6.68, 1.5, "…", fontsize=14, ha="center")
save(fig, "fl_framework.png")

# ======================================================================== data + partition
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(width_ratios=[1.2, 1.2, 1]))
rng = np.random.default_rng(3)
for ax, dsn in zip(axes[:2], ("B", "C")):
    ds = DS[dsn]
    cls = F.partition(ds, 5, "shift", 0)
    cmap = plt.get_cmap("viridis")
    for k, c in enumerate(cls):
        j = rng.choice(c["test"])
        ax.plot(ds["F"][j], ds["S"][j], color=cmap(k / 4), lw=1.0)
    ax.set_xlabel("frequency (GHz)")
    ax.set_title(f"Dataset {dsn}: one design per client", fontsize=8.5)
axes[0].set_ylabel("$|S_{11}|$ (dB)")
ax = axes[2]
ax2 = ax.twiny()
for dsn, axx, yb in (("B", ax2, 1.0), ("C", ax, 0.0)):
    ds = DS[dsn]
    cls = F.partition(ds, 5, "shift", 0)
    for k, c in enumerate(cls):
        key = ds["shift_key"][np.concatenate([c["train"], c["cal"], c["test"]])]
        axx.barh(yb, key.max() - key.min(), left=key.min(), height=0.55,
                 color=plt.get_cmap("viridis")(k / 4), ec="white")
ax.set_ylim(-0.6, 1.6)
ax.set_yticks([0, 1], ["C", "B"])
ax.set_xlabel("C: permittivity $\\varepsilon_r$", fontsize=7.5)
ax2.set_xlabel("B: resonant frequency (GHz)", fontsize=7.5)
ax2.spines["top"].set_visible(True)
ax.set_title("")
ax.text(0.5, 0.5, "five contiguous clients per dataset", transform=ax.transAxes, ha="center",
        va="center", fontsize=7)
fig.tight_layout()
save(fig, "fl_data.png")

# ======================================================================== example predictions
def preds(dsn, K, part, seed):
    return np.load(os.path.join(RES, "preds", f"{dsn}_K{K}_{part}_s{seed}.npz"))


fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=False)
cases = [("C", 0, "lowest-$\\varepsilon_r$ client"), ("C", 4, "highest-$\\varepsilon_r$ client"),
         ("B", 2, "middle band")]
for ax, (dsn, k, ttl) in zip(axes, cases):
    ds, z = DS[dsn], preds(dsn, 5, "shift", 0)
    idx = z[f"idx__test__{k}"]
    T = ds["S"][idx]
    e = np.sqrt(((z[f"fedavg__test__{k}"].astype(np.float32) - T) ** 2).mean(1))
    i = int(np.argsort(e)[len(e) // 2])                # median-error design for FedAvg
    f = ds["F"][idx[i]]
    q = ce[(ce.dataset == dsn) & (ce.K == 5) & (ce.partition == "shift") & (ce.seed == 0) &
           (ce.model == "fedavg_ft") & (ce.cp == "local-CP") & (ce.score == "sup") & (ce.client == k)].q
    pf = z[f"fedavg_ft__test__{k}"][i].astype(np.float32)
    if dsn == "C" and len(q):
        ax.fill_between(f, pf - q.iloc[0], pf + q.iloc[0], color=COL["fedavg_ft"], alpha=0.15, lw=0,
                        label="FT ± local-CP $e_\\infty$ bound")
    ax.plot(f, T[i], "k", lw=1.6, label="simulated")
    for m, ls in (("local", ":"), ("fedavg", "--"), ("fedavg_ft", "-")):
        ax.plot(f, z[f"{m}__test__{k}"][i].astype(np.float32), ls, color=COL[m], lw=1.1, label=NAMES[m])
    ax.set_title(f"{dsn}, {ttl}", fontsize=8.5)
    ax.set_xlabel("frequency (GHz)")
axes[0].set_ylabel("$|S_{11}|$ (dB)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, "fl_examples.png")

# ======================================================================== per-client accuracy (C shift)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1.1, 1]))
ax = axes[0]
g = cl[(cl.dataset == "C") & (cl.K == 5) & (cl.partition == "shift")]
w = 0.16
for j, m in enumerate(ORDER):
    s = g[g.method == m].groupby("client").r2_own.agg(["mean", "std"])
    ax.bar(s.index + (j - 2) * w, s["mean"], w, yerr=s["std"], color=COL[m], capsize=1.5,
           label=NAMES[m], error_kw=dict(lw=0.6))
ax.set_xticks(range(5), [f"{k}" for k in range(5)])
ax.set_xlabel("client (ordered by $\\varepsilon_r$)")
ax.set_ylabel("own-range test $R^2$")
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=6.3, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
ax.set_title("(a) C, substrate shift: per-client accuracy", fontsize=8.5)
ax = axes[1]
ds = DS["C"]
for m in ("local", "fedavg", "fedavg_ft"):
    xs, ys = [], []
    for s in range(3):
        z = preds("C", 5, "shift", s)
        for k in range(5):
            idx = z[f"idx__test__{k}"]
            P = z[f"{m}__test__{k}"].astype(np.float32)
            xs.append(ds["G"][idx, 3]); ys.append(np.sqrt(((P - ds["S"][idx]) ** 2).mean(1)))
    x, y = np.concatenate(xs), np.concatenate(ys)
    ax.scatter(x, y, s=3, color=COL[m], alpha=0.35, lw=0)
    bins = np.linspace(2.2, 6.15, 9)
    mid = 0.5 * (bins[1:] + bins[:-1])
    med = [np.median(y[(x >= a) & (x < b)]) for a, b in zip(bins[:-1], bins[1:])]
    ax.plot(mid, med, color=COL[m], lw=1.5, label=NAMES[m] + " (median)")
ax.set_yscale("log")
ax.set_xlabel("substrate permittivity $\\varepsilon_r$")
ax.set_ylabel("per-design sweep RMSE (dB)")
ax.legend(frameon=False, fontsize=6.3)
ax.set_title("(b) error of each test design", fontsize=8.5)
fig.tight_layout()
save(fig, "fl_perclient.png")

# ======================================================================== number of clients
fig, ax = plt.subplots(figsize=(3.4, 2.4))
g = cl[(cl.dataset == "B") & (cl.partition == "shift")]
for m in ("local", "fedavg", "fedavg_ft", "central"):
    s = g[g.method == m].groupby(["K", "seed"]).r2_own.mean().groupby("K").agg(["mean", "std"])
    ax.errorbar(s.index, s["mean"], s["std"], marker="o", ms=4, capsize=2, color=COL[m], label=NAMES[m])
ax.set_xticks([3, 5, 10])
ax.set_xlabel("number of clients $K$")
ax.set_ylabel("own-band test $R^2$")
ax.set_title("Dataset B, band shift", fontsize=8.5)
ax.legend(frameon=False, fontsize=6.5, loc="center right")
fig.tight_layout()
save(fig, "fl_K.png")

# ======================================================================== coverage / width trade-off
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
for ax, part in zip(axes, ("iid", "shift")):
    for (dsn, score), mk_fill in ((("B", "rmse_d"), True), (("B", "dfr"), False), (("C", "rmse_d"), True), (("C", "sup"), False)):
        g = ce[(ce.dataset == dsn) & (ce.K == 5) & (ce.partition == part) & (ce.score == score)]
        for m in ORDER:
            h = g[g.model == m]
            ql = h[h.cp == "local-CP"].groupby("seed").q.mean().mean()
            for cp in ("local-CP", "pooled-CP", "QQ-CP"):
                hh = h[h.cp == cp]
                worst = hh.groupby(["seed", "client"]).coverage.mean().groupby("client").mean().min()
                ax.scatter(hh.groupby("seed").q.mean().mean() / ql, worst, marker=CPMK[cp], s=16,
                           color=CPCOL[cp] if mk_fill else "none", edgecolors=CPCOL[cp], lw=0.8)
    ax.axhline(0.9, color="k", ls=":", lw=0.8)
    ax.axhspan(0.835, 0.847, color="#cccccc", alpha=0.6, lw=0)
    ax.axvline(1.0, color="k", ls=":", lw=0.5)
    ax.set_xlabel("threshold width relative to local CP")
    ax.set_title(f"({'a' if part == 'iid' else 'b'}) {'IID' if part == 'iid' else 'shifted'} clients",
                 fontsize=8.5)
axes[0].set_ylabel("lowest per-client coverage\n(seed-averaged)")
for cp in CPMK:
    axes[1].scatter([], [], marker=CPMK[cp], color=CPCOL[cp], label=cp, s=16)
axes[1].scatter([], [], marker="o", color="none", edgecolors="k", label="hollow: $e_f$ (B), $e_\\infty$ (C)", s=16)
axes[1].scatter([], [], marker="s", color="#cccccc", label="expected lowest\n(valid CP)", s=30)
axes[1].legend(frameon=False, fontsize=6.3, loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()
save(fig, "fl_tradeoff.png")
