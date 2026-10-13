"""python quick_full.py cfg1 cfg2 ... (env NS = seeds): dev / holdout Sharpe of top 1/5/10% and mean within-year IC on the FULL acct_ew universe, no tests."""
import os, sys, hashlib, numpy as np, pandas as pd, ht_base as B, ht_common as h, ht_arch as A, ht_new as nw
from full_data import make_full
from full_cfgs import FCFGS
NS = int(os.environ.get("NS", 1)); D = make_full(with_H=False, with_win=False); P = pd.read_parquet(h.CACHE / "panel_eb.parquet")
keys = pd.DataFrame(dict(family=D.F.family.values, signalid=D.F.signalid.values, oos_year=D.yr))
Pc = P[P.family == "acct_ew"].merge(keys, on=["family", "signalid", "oos_year"]); yr, sd = D.yr, D.sd
gk = D.famcode * 10000 + yr
def ld(cfg):
    nm = "|".join(f"{k}={cfg[k]}" for k in sorted(cfg)); return np.load(B.PRED_DIR / ("full_" + hashlib.md5(nm.encode()).hexdigest() + ".npy"))
def rets(p): return A.evaluate(Pc, D, p.astype(np.float32), "T0", "sr", 1.0, schemes=("eq",))[1]
def ic(p):
    d = pd.DataFrame(dict(yr=yr, p=p, y=D.y_or)); d = d[d.yr >= 1983]
    s = d.groupby("yr").apply(lambda g: g.p.corr(g.y, method="spearman"), include_groups=False); return s.loc[:2004].mean(), s.loc[2005:].mean()
def row(lab, p):
    r = rets(p); dv = [nw.perf(r[(k, "eq")], *nw.DEV)["sharpe"] for k in (1, 5, 10)]; ho = [nw.perf(r[(k, "eq")], *nw.HOLD)["sharpe"] for k in (1, 5, 10)]
    i = ic(p); print(f"{lab:22s} dev {dv[0]:.2f}/{dv[1]:.2f}/{dv[2]:.2f} mean {np.mean(dv):.3f} | hold {ho[0]:.2f}/{ho[1]:.2f}/{ho[2]:.2f} mean {np.mean(ho):.3f} | IC dev {i[0]:.3f} hold {i[1]:.3f}", flush=True)
row("EB", D.eb_or)
t = np.abs(D.F.tstat.values); zt = (t - pd.Series(t).groupby(gk).transform("mean").values) / pd.Series(t).groupby(gk).transform("std").values + 10
row("|t| rule", zt * sd)
for nm in sys.argv[1:]:
    ps = []
    for s in range(NS):
        try: ps.append(ld({**FCFGS[nm], "seed": s}))
        except FileNotFoundError: pass
    if ps: row(f"{nm} (n={len(ps)})", np.mean(ps, 0))
