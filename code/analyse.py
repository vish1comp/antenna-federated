"""Tables and figures from ../results/fl_*.csv  ->  ../results/summary.md, ../figures/*.png"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "..", "results"), os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)
ORDER = ["local", "fedavg", "fedprox", "fedavg_ft", "central"]
NAMES = {"local": "Local only", "fedavg": "FedAvg", "fedprox": "FedProx",
         "fedavg_ft": "FedAvg + fine-tune", "central": "Centralised (pooled)"}
COL = {"local": "#9a6b2f", "fedavg": "#1f6f8b", "fedprox": "#4a9c7a", "fedavg_ft": "#6b4fa0",
       "central": "#7a7a7a"}
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

import glob


def load(stem):
    files = sorted(glob.glob(os.path.join(RES, f"{stem}_s[0-9].csv")))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


cl, ce, cu = load("fl_clients"), load("fl_certs"), load("fl_curves")
md = []


def ms(x):
    return "%.3f ± %.3f" % (np.mean(x), np.std(x))


# ---------------------------------------------------------------- accuracy table (K = 5)
md.append("## Accuracy, K = 5 (mean over clients, then mean ± sd over seeds)\n")
for (dsn, part), g in cl[cl.K == 5].groupby(["dataset", "partition"]):
    md.append(f"### Dataset {dsn}, {part}\n")
    md.append("| Method | R² own | R² other clients | RMSE own (dB) | Δf_r own (GHz) | worst-client R² own |")
    md.append("|---|---|---|---|---|---|")
    for m in ORDER:
        h = g[g.method == m]
        if h.empty:
            continue
        per_seed = h.groupby("seed").agg(r2o=("r2_own", "mean"), r2x=("r2_other", "mean"),
                                         rm=("rmse_own", "mean"), df=("dfr_own", "mean"),
                                         worst=("r2_own", "min"))
        md.append(f"| {NAMES[m]} | {ms(per_seed.r2o)} | {ms(per_seed.r2x)} | {ms(per_seed.rm)} | "
                  f"{ms(per_seed.df)} | {ms(per_seed.worst)} |")
    md.append("")

# ---------------------------------------------------------------- number of clients
md.append("## Dataset B, shift: effect of the number of clients\n")
md.append("| K | Local R² own | FedAvg R² own | FedAvg+FT R² own | Central R² own |")
md.append("|---|---|---|---|---|")
for K, g in cl[(cl.dataset == "B") & (cl.partition == "shift")].groupby("K"):
    cells = []
    for m in ("local", "fedavg", "fedavg_ft", "central"):
        cells.append(ms(g[g.method == m].groupby("seed").r2_own.mean()))
    md.append(f"| {K} | " + " | ".join(cells) + " |")
md.append("")

# ---------------------------------------------------------------- certificates
if "score" not in ce.columns:
    ce["score"] = "sup"
UNITS = {"sup": "dB", "rmse_d": "dB", "dfr": "GHz"}


def chance_worst(n_cal, n_test, K, sims=20000, alpha=0.1, seed=0):
    """Expected worst-client coverage of LOCAL split conformal when every client's scores are
    exchangeable: the dip below 90 % that finite calibration/test sets produce by chance."""
    rng = np.random.default_rng(seed)
    l = int(min(np.ceil((n_cal + 1) * (1 - alpha)), n_cal))
    q = np.sort(rng.random((sims, K, n_cal)), axis=2)[:, :, l - 1]
    cov = rng.binomial(n_test, q) / n_test
    return float(cov.min(1).mean())


_ch = {}
for (dsn, K, part), g in ce.groupby(["dataset", "K", "partition"]):
    nt = int(g.n_test.median())
    _ch[(dsn, K, part)] = chance_worst(nt, nt, K)       # cal and test sets are both 20 %
md.append("## Chance level of worst-client coverage (exactly valid local CP)\n")
md.append("| Dataset | K | partition | designs per client test set | expected worst-client coverage |")
md.append("|---|---|---|---|---|")
for (dsn, K, part), v in sorted(_ch.items()):
    nt = int(ce[(ce.dataset == dsn) & (ce.K == K) & (ce.partition == part)].n_test.median())
    md.append(f"| {dsn} | {K} | {part} | {nt} | {v:.3f} |")
md.append("")
for score in ("rmse_d", "sup", "dfr"):
    md.append(f"## Conformal certificates on `{score}` (target 90 %), K = 5\n")
    md.append("Coverage per client then: mean, min over clients (mean over seeds); bound q.\n")
    for (dsn, part), g in ce[(ce.K == 5) & (ce.score == score)].groupby(["dataset", "partition"]):
        md.append(f"### Dataset {dsn}, {part}\n")
        md.append(f"| Model | CP | mean coverage | worst-client coverage | mean bound ({UNITS[score]}) |")
        md.append("|---|---|---|---|---|")
        for m in ORDER:
            for cp in ("local-CP", "pooled-CP", "QQ-CP"):
                h = g[(g.model == m) & (g.cp == cp)]
                if h.empty:
                    continue
                s_ = h.groupby("seed").agg(c=("coverage", "mean"), w=("coverage", "min"), q=("q", "mean"))
                md.append(f"| {NAMES[m]} | {cp} | {s_.c.mean():.3f} | {s_.w.mean():.3f} | {s_.q.mean():.3f} |")
        md.append("")

open(os.path.join(RES, "summary.md"), "w", encoding="utf-8").write("\n".join(md))
print("\n".join(md))

# ---------------------------------------------------------------- fig 1: own vs other R2
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
for ax, dsn in zip(axes, ("B", "C")):
    g = cl[(cl.dataset == dsn) & (cl.K == 5) & (cl.partition == "shift")]
    x = np.arange(len(ORDER))
    own = [g[g.method == m].groupby("seed").r2_own.mean() for m in ORDER]
    oth = [g[g.method == m].groupby("seed").r2_other.mean() for m in ORDER]
    ax.bar(x - 0.19, [o.mean() for o in own], 0.38, yerr=[o.std() for o in own],
           color=[COL[m] for m in ORDER], capsize=2, label="own band")
    ax.bar(x + 0.19, [o.mean() for o in oth], 0.38, yerr=[o.std() for o in oth],
           color=[COL[m] for m in ORDER], alpha=0.45, hatch="//", capsize=2, label="other clients' bands")
    ax.set_xticks(x, [NAMES[m].replace(" (pooled)", "").replace(" + ", "+\n") for m in ORDER], fontsize=7.5)
    ax.set_title(f"Dataset {dsn} ({'resonant-band' if dsn == 'B' else 'substrate'} shift, K = 5)")
    ax.axhline(0, color="k", lw=0.6)
axes[0].set_ylabel("test R²")
axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
axes[0].set_ylim(min(-0.2, axes[0].get_ylim()[0]), 1.0)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fl_own_other.png"), dpi=300)

# ---------------------------------------------------------------- fig 2: convergence
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=False)
for ax, dsn in zip(axes, ("B", "C")):
    for part, ls in (("iid", "--"), ("shift", "-")):
        for m in ("fedavg", "fedprox"):
            g = cu[(cu.dataset == dsn) & (cu.K == 5) & (cu.partition == part) & (cu.method == m)]
            if g.empty:
                continue
            s = g.groupby("round").r2.agg(["mean", "std"])
            ax.plot(s.index, s["mean"], ls, color=COL[m], label=f"{NAMES[m]}, {part}")
            ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], color=COL[m], alpha=0.15)
        c = cl[(cl.dataset == dsn) & (cl.K == 5) & (cl.partition == part) & (cl.method == "central")]
        ax.axhline(c.groupby("seed").r2_own.mean().mean(), color="#7a7a7a", ls=ls, lw=0.8)
    ax.set_xlabel("communication round")
    ax.set_title(f"Dataset {dsn}")
axes[0].set_ylabel("global test R²")
axes[1].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fl_convergence.png"), dpi=300)

# ---------------------------------------------------------------- fig 3: per-band coverage
def chance_band(n, seeds=3, sims=20000, alpha=0.1, seed=1):
    """2.5-97.5 % range of a client's seed-averaged coverage under exactly valid split CP"""
    rng = np.random.default_rng(seed)
    l = int(min(np.ceil((n + 1) * (1 - alpha)), n))
    q = np.sort(rng.random((sims, seeds, n)), axis=2)[:, :, l - 1]
    cov = (rng.binomial(n, q) / n).mean(1)
    return np.percentile(cov, [2.5, 97.5])


fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
PAN = (("B", "rmse_d", "B: sweep RMSE"), ("B", "dfr", "B: resonance error"), ("C", "rmse_d", "C: sweep RMSE"))
for ax, (dsn, score, ttl) in zip(axes, PAN):
    g = ce[(ce.dataset == dsn) & (ce.K == 5) & (ce.partition == "shift") & (ce.model == "fedavg_ft")
           & (ce.score == score)]
    if g.empty:
        ax.set_axis_off()
        continue
    lo, hi = chance_band(int(g.n_test.median()))
    ax.axhspan(lo, hi, color="#cccccc", alpha=0.5, lw=0, label="chance range (valid CP)")
    ax.axhline(0.9, color="k", ls=":", lw=0.8)
    for j, (cp, c, mk) in enumerate((("local-CP", "#9a6b2f", "o"), ("pooled-CP", "#7a7a7a", "s"),
                                     ("QQ-CP", "#1f6f8b", "^"))):
        s_ = g[g.cp == cp].groupby("client").coverage.mean()
        ax.plot(s_.index + (j - 1) * 0.15, s_.values, mk, ms=4.5, color=c, label=cp)
    ax.set_xticks(range(5))
    ax.set_xlabel("client (by %s)" % ("resonant freq." if dsn == "B" else "permittivity"))
    ax.set_title(ttl, fontsize=9)
axes[0].set_ylabel("coverage (mean of 3 seeds)")
axes[0].legend(frameon=False, fontsize=6.5, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fl_coverage.png"), dpi=300)
print("figures written")
