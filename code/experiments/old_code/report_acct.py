"""Helpers for the accounting-EW report notebook (code/notebook/02_ai_vs_eb_accounting_ew.ipynb).  CPU only, saved predictions only.
Everything is evaluated on the candidate universe of accounting EW (10% largest |t| per year), with EB re-scored on the same rows."""
import glob, numpy as np, pandas as pd, statsmodels.api as sm
import ht_base as B, ht_common as h, ht_arch as A, ht_new as nw
from ht_sub import make_sub
from run_q6_cfgs import CFGS

D = make_sub("acct_ew", with_X=False, with_H=False)
P = pd.read_parquet(h.CACHE / "panel_eb.parquet")
_keys = pd.DataFrame(dict(family=D.F.family.values, signalid=D.F.signalid.values, oos_year=D.yr))
PC = P[P.family == "acct_ew"].merge(_keys, on=["family", "signalid", "oos_year"])
GK, YR, SD = D.famcode * 10000 + D.yr, D.yr, D.sd
PER = (("dev 1983-2004", nw.DEV), ("holdout 2005-2020", nw.HOLD), ("full 1983-2020", (1983, 2020)))
CFGS = {**CFGS, "goc": {**B.BASE, "fams": "acct_ew"}}


def mean_pred(name, seeds=range(20)):
    ps = []
    for s in seeds:
        try: ps.append(B.to_local(D, B.pred_of({**CFGS[name], "seed": s})))
        except FileNotFoundError: pass
    return np.mean(ps, 0).astype(np.float32), len(ps)


def ratio_level(p, win=20):
    """Family-year level (Sharpe units) = c * mean in-sample Sharpe, c = realised / in-sample over the last `win` years (lvl=ratio20)."""
    s = p / SD; sm_ = pd.Series(s).groupby(GK).transform("mean").values
    fy = pd.DataFrame(dict(yr=YR, y=D.y_or / SD, x=D.F.mean_is.values * D.sg / SD)).groupby("yr").mean()
    q = s.copy()
    for Y in np.unique(YR):
        past = fy[(fy.index <= Y - 1) & (fy.index >= Y - win)]
        if len(past): q[YR == Y] = s[YR == Y] - sm_[YR == Y] + past.y.sum() / past.x.sum() * fy.x.loc[Y]
    return (q * SD).astype(np.float32)


def abs_t_forecast():
    """The 'naive mining' benchmark (paper Table 2 Panel B analogue): rank by in-sample |t|, no model."""
    t = np.abs(D.F.tstat.values)
    z = (t - pd.Series(t).groupby(GK).transform("mean").values) / pd.Series(t).groupby(GK).transform("std").values + 10
    return (z * SD).astype(np.float32)


def portfolios(pred):
    return A.evaluate(PC, D, np.asarray(pred, np.float32), "T0", "sr", 1.0, schemes=("eq",))[1]


def table2(rets_by_model, ref="EB"):
    """Paper Table 2 style: mean return (% ann), t-stat, Sharpe for each model, top 1/5/10%, each period; p-values vs `ref`."""
    rows = []
    for k in (1, 5, 10):
        for per, (lo, hi) in PER:
            r = dict(top=f"{k}%", period=per)
            for m, R in rets_by_model.items():
                p_ = nw.perf(R[(k, "eq")], lo, hi); n = len(R[(k, "eq")].query("year >= @lo and year <= @hi"))
                r[f"{m} ret"], r[f"{m} t"], r[f"{m} SR"] = p_["mean"], p_["sharpe"] * np.sqrt(n / 12), p_["sharpe"]   # t = monthly mean / sd * sqrt(months)
            a, e = rets_by_model["AI"][(k, "eq")], rets_by_model[ref][(k, "eq")]
            r[f"p HAC (SR AI vs {ref})"] = nw.sharpe_test_hac(a, e, lo, hi)
            r[f"p boot (SR AI vs {ref})"] = nw.sharpe_diff_bootstrap_p(a, e, lo, hi, n=10000)[2]
            r[f"p NW (ret AI vs {ref})"] = nw.mean_diff_test(a, e, lo, hi)[2]
            rows.append(r)
    return pd.DataFrame(rows)


def ic_years(pred):
    d = pd.DataFrame(dict(g=GK, yr=YR, p=pred, y=D.y_or))
    return d.groupby("yr").apply(lambda x: x.p.corr(x.y, method="spearman"), include_groups=False)


def ic_test(ia, ie, lo, hi):
    dd = (ia - ie).loc[lo:hi]; m = sm.OLS(dd.values, np.ones(len(dd))).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    return dict(IC_AI=ia.loc[lo:hi].mean(), IC_EB=ie.loc[lo:hi].mean(), diff=dd.mean(), t=m.tvalues[0], p=m.pvalues[0], years_AI_better=f"{int((dd > 0).sum())}/{len(dd)}")


def calib_bins(pred, lo, hi, nb=20):
    """Paper Fig. 4-5 style: nb groups by forecast within each year; mean forecast and mean realised return per group."""
    d = pd.DataFrame(dict(yr=YR, f=pred, y=D.y_or)); d = d[(d.yr >= lo) & (d.yr <= hi)]
    d["bin"] = d.groupby("yr").f.transform(lambda x: pd.qcut(x.rank(method="first"), nb, labels=False))
    return d.groupby("bin")[["f", "y"]].mean()


def ensembles(files=("q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "q16", "q17")):
    """Every 10/20-seed mean ensemble row logged in the queues (dev / holdout candidate Sharpe, MSE vs EB)."""
    L = []
    for f in files:
        for p_ in glob.glob(str(h.CACHE / f"{f}_results.tsv")):
            d = pd.read_csv(p_, sep="\t"); d = d[d.name.str.contains("ENSEMBLE1|ENSEMBLE2") & d.name.str.contains("mea")]; d["queue"] = f; L.append(d)
    e = pd.concat(L, ignore_index=True)
    e["config"] = e.name.str.extract(r"\|Q\d+ (.+) on acct_ew")[0]
    e["seeds"] = e.name.str.extract(r"ENSEMBLE(\d+)")[0].astype(int)
    for per in ("dev", "hold"):
        e[f"SR {per} mean"] = e[[f"cand_top{k}_{per}_sr" for k in (1, 5, 10)]].mean(1)
    return e
