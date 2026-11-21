"""Pilot: FedProx proximal weight mu, chosen on the CALIBRATION designs (never the test designs)."""
import sys, numpy as np, torch
import fl_antenna as F
torch.set_num_threads(2)
dsn, mu = sys.argv[1], float(sys.argv[2])
ds = F.load_B() if dsn == "B" else F.load_C()
cl = F.partition(ds, 5, "shift", 0)
data = [dict(zip(("X", "y"), F.features(ds, c["train"]))) for c in cl]
cal = np.concatenate([c["cal"] for c in cl])
d = ds["G"].shape[1] + 3
log = []
ev = lambda m: dict(r2=F.summarise(F.sweep_scores(m, ds, cal))["r2"])
F.federated(data, d, 30, 200, 0, mu=mu, log=log, eval_fn=ev)
print(dsn, "mu", mu, " ".join("%d:%.3f" % (r["round"], r["r2"]) for r in log), flush=True)
