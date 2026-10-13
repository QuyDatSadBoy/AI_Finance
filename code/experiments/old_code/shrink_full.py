"""Rescale the network correction: pred = level + base + lam * residual (parts saved by cfg parts=True), lam chosen on DEV only."""
import sys, hashlib, numpy as np, pandas as pd, ht_base as B, ht_common as h, ht_arch as A, ht_new as nw
from full_data import make_full
from full_cfgs import FCFGS
name = sys.argv[1]; NS = int(sys.argv[2])
D = make_full(with_H=False, with_win=False); P = pd.read_parquet(h.CACHE / "panel_eb.parquet")
keys = pd.DataFrame(dict(family=D.F.family.values, signalid=D.F.signalid.values, oos_year=D.yr))
Pc = P[P.family == "acct_ew"].merge(keys, on=["family", "signalid", "oos_year"]); yr, sd = D.yr, D.sd
parts = []
for s in range(NS):
    cfg = {**FCFGS[name], "seed": s}; nm = "|".join(f"{k}={cfg[k]}" for k in sorted(cfg))
    parts.append(np.load(B.PRED_DIR / ("fullparts_" + hashlib.md5(nm.encode()).hexdigest() + ".npz")))
res = np.mean([p["res"] for p in parts], 0); base = parts[0]["base"]; lvl = parts[0]["lvl"]
def rets(p): return A.evaluate(Pc, D, p.astype(np.float32), "T0", "sr", 1.0, schemes=("eq",))[1]
def ic(p):
    d = pd.DataFrame(dict(yr=yr, p=p, y=D.y_or)); d = d[d.yr >= 1983]
    s = d.groupby("yr").apply(lambda g: g.p.corr(g.y, method="spearman"), include_groups=False); return s.loc[:2004].mean(), s.loc[2005:].mean()
print("lam  | dev top1/5/10 mean | hold top1/5/10 mean | IC dev hold")
for lam in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
    p = lvl + base + lam * res; r = rets(p)
    dv = [nw.perf(r[(k, "eq")], *nw.DEV)["sharpe"] for k in (1, 5, 10)]; ho = [nw.perf(r[(k, "eq")], *nw.HOLD)["sharpe"] for k in (1, 5, 10)]; i = ic(p)
    print(f"{lam:4.2f} | {dv[0]:.2f}/{dv[1]:.2f}/{dv[2]:.2f} {np.mean(dv):.3f} | {ho[0]:.2f}/{ho[1]:.2f}/{ho[2]:.2f} {np.mean(ho):.3f} | {i[0]:.3f} {i[1]:.3f}", flush=True)
