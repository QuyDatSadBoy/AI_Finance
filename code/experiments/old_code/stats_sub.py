"""Significance tests on a family subset (CPU only): median- (or mean-, AGG=mean) of-seeds ensembles vs EB on the SAME families, both universes
(cand: candidate rows only; all: every strategy of the families).  Sharpe: Jobson-Korkie-Memmel and year-block bootstrap; mean: Newey-West.
Usage: [NSEED=10] [AGG=mean] python stats_sub.py acct_ew 'xs+ydm+mha=xs:True,prep:ydm,pool:mha' 'goc=' ..."""
import os, sys, numpy as np, pandas as pd, ht_common as h, ht_base as B, ht_arch as A, ht_new as nw
from ht_sub import make_sub
fams = sys.argv[1]
D = make_sub(fams, with_X=False, with_H=False); P = pd.read_parquet(h.CACHE / "panel_eb.parquet")
def parse(spec):
    nm, kv = spec.split("=", 1); c = {**B.BASE, "fams": fams}
    for p in filter(None, kv.split(",")):
        k, v = p.split(":"); c[k] = {"True": True, "False": False}.get(v, v)
    return nm, c
def median_pred(c, n):
    full = [B.pred_of({**c, "seed": s}) for s in range(n)]                # predictions over all candidate rows; keep this subset
    return (np.mean if os.environ.get("AGG", "median") == "mean" else np.median)([f[D.idx_full] for f in full], 0)
rows = []
m = np.ones(len(D.yr), bool)
keys = pd.DataFrame(dict(family=D.F.family.values, signalid=D.F.signalid.values, oos_year=D.yr))
P_all = P[P.family.isin(fams.split("+"))]; P_cand = P_all.merge(keys, on=["family", "signalid", "oos_year"])
eb = {u: A.evaluate(PU, D, D.eb_or.astype(np.float32), "T0", "sr", 1.0, schemes=("eq",))[1] for u, PU in (("cand", P_cand), ("all", P_all))}
n = int(os.environ.get("NSEED", 3))                                    # seeds in the median ensemble
for spec in sys.argv[2:]:
    nm, c = parse(spec); p = median_pred(c, n).astype(np.float32); acc = B.accuracy(D, p)
    for u, PU in (("cand", P_cand), ("all", P_all)):
        _, rets = A.evaluate(PU, D, p, "T0", "sr", 1.0, schemes=("eq",))
        for r in (1, 5, 10):
            a, e = rets[(r, "eq")], eb[u][(r, "eq")]
            for per, (lo, hi) in (("dev", nw.DEV), ("hold", nw.HOLD), ("full", (1983, 2020))):
                pa, pe = nw.perf(a, lo, hi), nw.perf(e, lo, hi); _, z, pj = nw.sharpe_test_jkm(a, e, lo, hi); _, ci, pb = nw.sharpe_diff_bootstrap_p(a, e, lo, hi, n=int(os.environ.get("NBOOT", 10000))); ph = nw.sharpe_test_hac(a, e, lo, hi)
                dm_, tm, pm = nw.mean_diff_test(a, e, lo, hi)
                rows.append(dict(model=nm, uni=u, top=r, per=per, sr_ai=pa["sharpe"], sr_eb=pe["sharpe"], p_jkm=pj, p_hac=ph, p_boot=pb, ret_ai=pa["mean"], ret_eb=pe["mean"], p_ret=pm,
                                 mse_dev=acc["mse_vs_eb_dev"], dm_dev=acc["dm_t_dev"], mse_hold=acc["mse_vs_eb_hold"], dm_hold=acc["dm_t_hold"]))
out = pd.DataFrame(rows); out.to_csv(h.CACHE / f"stats_sub_{fams}_n{n}_{os.environ.get('AGG', 'median')}.csv", index=False)
pd.set_option("display.width", 220)
for nm in out.model.unique():
    x = out[out.model == nm]; print(f"\n== {nm} on {fams} | MSE vs EB dev {x.mse_dev.iloc[0]:+.1f}% (t {x.dm_dev.iloc[0]:.2f}) hold {x.mse_hold.iloc[0]:+.1f}% (t {x.dm_hold.iloc[0]:.2f})")
    print(x[["uni", "top", "per", "sr_ai", "sr_eb", "p_jkm", "p_hac", "p_boot", "ret_ai", "ret_eb", "p_ret"]].round(3).to_string(index=False))
