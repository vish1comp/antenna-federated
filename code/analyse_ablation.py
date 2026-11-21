"""Ablation outputs: physics-input table (markdown) and coupled-vs-decoupled FedProx figure."""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "..", "results"), os.path.join(HERE, "..", "figures")
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})


def cat(pat):
    fs = sorted(glob.glob(os.path.join(RES, pat)))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


cl = cat("fl_clients_s[0-9].csv")
ph = cat("ablation_phys_*_s[0-9].csv")
out = ["## Physics-informed inputs (IID, K = 5; mean over clients, mean ± sd over seeds)\n",
       "| Dataset | Method | with physics | without | n seeds (without) |", "|---|---|---|---|---|"]
for dsn in ("B", "C"):
    for m in ("central", "fedavg"):
        w = cl[(cl.dataset == dsn) & (cl.K == 5) & (cl.partition == "iid") & (cl.method == m)]
        w = w.groupby("seed").r2_own.mean()
        n = ph[(ph.dataset == dsn) & (ph.method == m)].groupby("seed").r2.mean() if len(ph) else pd.Series(dtype=float)
        out.append(f"| {dsn} | {m} | {w.mean():.3f} ± {w.std():.3f} | "
                   f"{(f'{n.mean():.3f} ± {n.std():.3f}' if len(n) else 'n/a')} | {len(n)} |")
open(os.path.join(RES, "ablation_summary.md"), "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))

cu = cat("fl_curves_s[0-9].csv")
pr = cat("ablation_prox_s[0-9].csv")
fig, ax = plt.subplots(figsize=(3.4, 2.4))
g = cu[(cu.dataset == "C") & (cu.K == 5) & (cu.partition == "shift")]
for m, col, lab in (("fedavg", "#1f6f8b", "FedAvg"), ("fedprox", "#4a9c7a", "FedProx, decoupled (ours)")):
    s = g[g.method == m].groupby("round").r2.agg(["mean", "std"])
    ax.plot(s.index, s["mean"], color=col, label=lab)
    ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], color=col, alpha=0.15, lw=0)
if len(pr):
    s = pr.groupby("round").r2.agg(["mean", "std"])
    ax.plot(s.index, s["mean"], "--", color="#c0392b", label="FedProx, coupled (standard)")
    ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], color="#c0392b", alpha=0.15, lw=0)
    print(s.round(3))
ax.set_xlabel("communication round")
ax.set_ylabel("global test $R^2$")
ax.set_title("Dataset C, substrate shift, $\\mu=0.01$", fontsize=8.5)
ax.legend(frameon=False, fontsize=6.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fl_prox.png"), dpi=300, bbox_inches="tight")
print("wrote fl_prox.png")
