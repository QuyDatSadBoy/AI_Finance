"""Helpers for the full-data notebook (code/notebook/03_full_data_ai_vs_eb_accounting_ew.ipynb).  CPU only, saved forecasts only.
Everything is evaluated on ALL Accounting-EW strategies (about 26,000 per year), EB re-scored on the same rows, top-k picked by predicted Sharpe."""
import hashlib, numpy as np, pandas as pd, statsmodels.api as sm
import ht_base as B, ht_common as h, ht_arch as A, ht_new as nw
from full_data import make_full
from full_cfgs import FCFGS

D = make_full(with_H=False, with_win=False)
P = pd.read_parquet(h.CACHE / "panel_eb.parquet")
_keys = pd.DataFrame(dict(family=D.F.family.values, signalid=D.F.signalid.values, oos_year=D.yr))
PC = P[P.family == "acct_ew"].merge(_keys, on=["family", "signalid", "oos_year"])
YR, SD = D.yr, D.sd
GK = D.famcode * 10000 + YR
PER = (("dev 1983-2004", nw.DEV), ("holdout 2005-2020", nw.HOLD), ("full 1983-2020", (1983, 2020)))


def _file(cfg):
    nm = "|".join(f"{k}={cfg[k]}" for k in sorted(cfg))
    return B.PRED_DIR / ("full_" + hashlib.md5(nm.encode()).hexdigest() + ".npy")


def mean_pred(name, seeds=range(20)):
    """Mean forecast over the available seeds of a named configuration (oriented %/month); returns (forecast, number of seeds)."""
    ps = []
    for s in seeds:
        f = _file({**FCFGS[name], "seed": s})
        if f.exists(): ps.append(np.load(f))
    return (np.mean(ps, 0).astype(np.float32) if ps else None), len(ps)


def portfolios(pred):
    return A.evaluate(PC, D, np.asarray(pred, np.float32), "T0", "sr", 1.0, schemes=("eq",))[1]


def abs_t_forecast():
    """Ranking by in-sample |t| alone (no model): the paper's 'naive mining' analogue."""
    t = np.abs(D.F.tstat.values)
    g = pd.Series(t).groupby(GK)
    return (((t - g.transform("mean").values) / g.transform("std").values + 10) * SD).astype(np.float32)


def sharpe_table(preds, rets=None):
    """One row per model: Sharpe of the top 1/5/10% in dev / holdout / full, plus the mean and the yearly IC. preds: {name: forecast}."""
    rows = []
    for nm, p in preds.items():
        R = rets[nm] if rets and nm in rets else portfolios(p)
        r = dict(model=nm)
        for per, (lo, hi) in PER:
            v = [nw.perf(R[(k, "eq")], lo, hi)["sharpe"] for k in (1, 5, 10)]
            r[f"{per.split()[0]} top1/5/10"] = " / ".join(f"{x:.2f}" for x in v); r[f"{per.split()[0]} mean"] = round(float(np.mean(v)), 3)
        i = ic_years(p); r["IC dev"], r["IC hold"] = round(i.loc[1983:2004].mean(), 3), round(i.loc[2005:2020].mean(), 3)
        rows.append(r)
    return pd.DataFrame(rows)


def compare(a_r, b_r, la, lb):
    """Sharpe / raw return of two portfolios per top-k and period, with HAC, year-block bootstrap (10,000) and JKM p-values for the Sharpe difference."""
    rows = []
    for k in (1, 5, 10):
        for per, (lo, hi) in PER:
            a, b = a_r[(k, "eq")], b_r[(k, "eq")]
            rows.append(dict(top=f"{k}%", period=per.split()[0], **{f"SR {la}": nw.perf(a, lo, hi)["sharpe"], f"SR {lb}": nw.perf(b, lo, hi)["sharpe"]},
                             p_HAC=nw.sharpe_test_hac(a, b, lo, hi), p_boot=nw.sharpe_diff_bootstrap_p(a, b, lo, hi, n=10000)[2],
                             p_JKM=nw.sharpe_test_jkm(a, b, lo, hi)[2], **{f"ret {la}": nw.perf(a, lo, hi)["mean"], f"ret {lb}": nw.perf(b, lo, hi)["mean"]}))
    return pd.DataFrame(rows).round(3)


def ic_years(pred):
    d = pd.DataFrame(dict(g=GK, yr=YR, p=pred, y=D.y_or)); d = d[d.yr >= 1983]
    return d.groupby("yr").apply(lambda x: x.p.corr(x.y, method="spearman"), include_groups=False)


def ic_test(ia, ie, lo, hi):
    dd = (ia - ie).loc[lo:hi]; m = sm.OLS(dd.values, np.ones(len(dd))).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    return dict(IC_AI=ia.loc[lo:hi].mean(), IC_EB=ie.loc[lo:hi].mean(), diff=dd.mean(), t=m.tvalues[0], p=m.pvalues[0], years_AI_better=f"{int((dd > 0).sum())}/{len(dd)}")


def calib_bins(pred, lo, hi, nb=20):
    """Paper Fig. 4-5 style: nb groups by forecast within each year; mean forecast and mean realised return per group."""
    d = pd.DataFrame(dict(yr=YR, f=pred, y=D.y_or)); d = d[(d.yr >= lo) & (d.yr <= hi)]
    d["bin"] = d.groupby("yr").f.transform(lambda x: pd.qcut(x.rank(method="first"), nb, labels=False))
    return d.groupby("bin")[["f", "y"]].mean()


def ensemble_table(names, seeds=range(20)):
    """Dev / holdout mean Sharpe (top 1/5/10) of the seed-mean forecast of each named configuration, with the number of seeds."""
    rows = []
    for nm in names:
        p, n = mean_pred(nm, seeds)
        if p is None: continue
        R = portfolios(p)
        dv = np.mean([nw.perf(R[(k, "eq")], *nw.DEV)["sharpe"] for k in (1, 5, 10)]); ho = np.mean([nw.perf(R[(k, "eq")], *nw.HOLD)["sharpe"] for k in (1, 5, 10)])
        i = ic_years(p); rows.append(dict(config=nm, seeds=n, SR_dev=round(dv, 3), SR_hold=round(ho, 3), IC_dev=round(i.loc[1983:2004].mean(), 3), IC_hold=round(i.loc[2005:2020].mean(), 3)))
    return pd.DataFrame(rows)
