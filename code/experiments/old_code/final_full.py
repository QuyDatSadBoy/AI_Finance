"""Final evaluation of the accounting-EW model (CPU only, from saved predictions).
1. Selection on DEV only: among the given configs (20-seed mean ensembles), the highest mean dev candidate Sharpe of the top 1/5/10%.
2. The selected forecast with the ratio level (lvl=ratio20, chosen on dev; the level is added after the network and does not change the
   within-family-year ranking, so applying it to the saved forecast is exact).
3. Versus EB: MSE (DM), within-year IC, top-k Sharpe (HAC, year-block bootstrap 10,000, JKM), raw return (Newey-West), dev / holdout / full.
4. Versus the |t| rule, versus the reference notebook model (paired seeds), versus the non-pretrained counterparts.
Usage: python final_acct.py Dt0 Dt0_chr2_240"""
import sys, numpy as np, pandas as pd, ht_base as B, ht_common as h, ht_arch as A, ht_new as nw
import hashlib
from full_data import make_full
from full_cfgs import FCFGS as CFGS

import os
NS = int(os.environ.get("NS", 20)); names = sys.argv[1:]
D = make_full(with_H=False, with_win=False); P = pd.read_parquet(h.CACHE / "panel_eb.parquet")
keys = pd.DataFrame(dict(family=D.F.family.values, signalid=D.F.signalid.values, oos_year=D.yr))
Pc = P[P.family == "acct_ew"].merge(keys, on=["family", "signalid", "oos_year"])
gk = D.famcode * 10000 + D.yr; yr, sd = D.yr, D.sd
_, R_EB = A.evaluate(Pc, D, D.eb_or.astype(np.float32), "T0", "sr", 1.0, schemes=("eq",))
PER = (("dev", nw.DEV), ("hold", nw.HOLD), ("full", (1983, 2020)))


def pred_full(cfg):
    nm = "|".join(f"{k}={cfg[k]}" for k in sorted(cfg)); return np.load(B.PRED_DIR / ("full_" + hashlib.md5(nm.encode()).hexdigest() + ".npy"))


def mean_pred(nm, seeds):
    ps = []
    for s in seeds:
        try: ps.append(pred_full({**CFGS[nm], "seed": s}))
        except FileNotFoundError: pass
    return np.mean(ps, 0), len(ps)


def ratio_level(p, win=20):
    """Replace the family-year level (Sharpe units) by c * mean in-sample Sharpe, c = realised / in-sample over the last `win` years."""
    s = p / sd; sm = pd.Series(s).groupby(gk).transform("mean").values
    fy = pd.DataFrame(dict(yr=yr, y=D.y_or / sd, x=D.F.mean_is.values * D.sg / sd)).groupby("yr").mean()
    q = s.copy()
    for Y in np.unique(yr):
        past = fy[(fy.index <= Y - 1) & (fy.index >= Y - win)]
        if len(past): q[yr == Y] = s[yr == Y] - sm[yr == Y] + past.y.sum() / past.x.sum() * fy.x.loc[Y]
    return (q * sd).astype(np.float32)


def rets(p): return A.evaluate(Pc, D, p.astype(np.float32), "T0", "sr", 1.0, schemes=("eq",))[1]


def dev_sr(p):
    r = rets(p); return np.mean([nw.perf(r[(k, "eq")], *nw.DEV)["sharpe"] for k in (1, 5, 10)])


def ic(p, lo, hi):
    d = pd.DataFrame(dict(g=gk, yr=yr, p=p, y=D.y_or)); d = d[(d.yr >= lo) & (d.yr <= hi)]
    return d.groupby("g").apply(lambda q: q.p.corr(q.y, method="spearman"), include_groups=False).mean()


def compare(a_r, b_r, la, lb):
    rows = []
    for k in (1, 5, 10):
        for per, (lo, hi) in PER:
            a, b = a_r[(k, "eq")], b_r[(k, "eq")]
            rows.append(dict(top=f"{k}%", per=per, **{f"SR_{la}": nw.perf(a, lo, hi)["sharpe"], f"SR_{lb}": nw.perf(b, lo, hi)["sharpe"]},
                             p_hac=nw.sharpe_test_hac(a, b, lo, hi), p_boot=nw.sharpe_diff_bootstrap_p(a, b, lo, hi, n=10000)[2],
                             p_jkm=nw.sharpe_test_jkm(a, b, lo, hi)[2], **{f"ret_{la}": nw.perf(a, lo, hi)["mean"], f"ret_{lb}": nw.perf(b, lo, hi)["mean"]},
                             p_ret=nw.mean_diff_test(a, b, lo, hi)[2]))
    return pd.DataFrame(rows).round(3)


pd.set_option("display.width", 250)
print("== selection on dev (20-seed mean ensembles, mean dev Sharpe top 1/5/10, FULL universe)")
sel = {}
for nm in names:
    p, n = mean_pred(nm, range(NS)); sel[nm] = (dev_sr(p), n); print(f"  {nm:16s} seeds {n:2d}  mean dev SR {sel[nm][0]:.3f}")
best = max(sel, key=lambda k: sel[k][0]); print("  selected:", best)
p0, n = mean_pred(best, range(NS)); p = ratio_level(p0)
print("\n== accuracy vs EB (selected + ratio20 level; positive = AI better; DM > 0 = AI worse)")
print("  original level:", {k: round(v, 2) for k, v in B.accuracy(D, p0).items()})
print("  ratio20 level :", {k: round(v, 2) for k, v in B.accuracy(D, p).items()})
print(f"  within-year IC dev {ic(p, 1983, 2004):.3f} (EB {ic(D.eb_or, 1983, 2004):.3f}) | hold {ic(p, 2005, 2020):.3f} (EB {ic(D.eb_or, 2005, 2020):.3f})")
def ic_years(q):
    d = pd.DataFrame(dict(g=gk, yr=yr, p=q, y=D.y_or))
    return d.groupby("yr").apply(lambda x: x.p.corr(x.y, method="spearman"), include_groups=False)
ia, ie = ic_years(p), ic_years(D.eb_or)
import statsmodels.api as sm
print("\n== IC by year (Spearman within family-year), AI - EB, Newey-West t")
for nm, lo, hi in (("dev", 1983, 2004), ("hold", 2005, 2020), ("full", 1983, 2020)):
    dd = (ia - ie).loc[lo:hi]; m = sm.OLS(dd.values, np.ones(len(dd))).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    print(f"  {nm:4s} IC AI {ia.loc[lo:hi].mean():.3f} EB {ie.loc[lo:hi].mean():.3f} diff {dd.mean():+.3f} t {m.tvalues[0]:+.2f} p {m.pvalues[0]:.3f} | years AI>EB {int((dd > 0).sum())}/{len(dd)}")
print("\n== unbiasedness (paper Fig. 4-5 style): 20 groups by forecast within each year; realised vs forecast mean per group, slope of realised on forecast")
for nm, lo, hi in (("dev", 1983, 2004), ("hold", 2005, 2020)):
    for lab, q in (("AI", p), ("EB", D.eb_or)):
        d = pd.DataFrame(dict(yr=yr, f=q, y=D.y_or)); d = d[(d.yr >= lo) & (d.yr <= hi)]
        d["bin"] = d.groupby("yr").f.transform(lambda x: pd.qcut(x.rank(method="first"), 20, labels=False))
        g = d.groupby("bin")[["f", "y"]].mean(); b = np.polyfit(g.f, g.y, 1)
        print(f"  {nm:4s} {lab}: slope {b[0]:.2f} intercept {b[1]:+.3f} | mean |realised - forecast| over groups {np.abs(g.y - g.f).mean():.3f} | top group forecast {g.f.iloc[-1]:.3f} realised {g.y.iloc[-1]:.3f}")
        g.to_csv(h.CACHE / f"calib_{nm}_{lab}.csv")
R = rets(p)
print("\n== portfolios vs EB (FULL universe)"); print(compare(R, R_EB, "AI", "EB").to_string(index=False))
t = np.abs(D.F.tstat.values); zt = (t - pd.Series(t).groupby(gk).transform("mean").values) / pd.Series(t).groupby(gk).transform("std").values + 10
print("\n== portfolios vs the |t| rule (no model)"); print(compare(R, rets(zt * sd), "AI", "absT").to_string(index=False))
print("\n== portfolios of the |t| rule vs EB"); print(compare(rets(zt * sd), R_EB, "absT", "EB").to_string(index=False))
print("\n== versus the reference-notebook model on the same data (goc: FinCast -> Bi-LSTM -> attention -> concat -> FC, trained on all strategies, walk-forward; seed-mean)")
pg, ng = mean_pred("goc", range(NS)); Rg = rets(pg.astype(np.float32))
print(f"  goc seeds {ng}; IC dev {ic(pg, 1983, 2004):.3f} hold {ic(pg, 2005, 2020):.3f}"); print(compare(R, Rg, "AI", "goc").to_string(index=False))
