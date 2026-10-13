"""Shared helpers for the two High-Throughput Asset Pricing notebooks.

Data layer + the pieces of the Chen & Dim pipeline that both notebooks need.
Returns are monthly long-short returns in percent. Every strategy is a row of a
wide matrix (strategies x months); months are counted from 1960-01 (month index 0).
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
CACHE = Path(__file__).resolve().parent / "_cache"
CKPT = Path(__file__).resolve().parent / "checkpoints"      # trained model weights (git-ignored)
CKPT.mkdir(exist_ok=True, parents=True)
CACHE.mkdir(exist_ok=True, parents=True)

BASE_YEAR = 1960
IS_N_YEARS = 20          # in-sample window length (years)
IS_MIN_OBS = 36          # min months in IS window
OOS_MIN_OBS = 12         # full OOS year required
OOS_START = 1983
YEAR_MAX = 2020          # last OOS year used in the paper's tables
FAMILIES = ["acct_ew", "acct_vw", "past_ret_ew", "past_ret_vw", "ticker_ew", "ticker_vw"]
LONG_NAME = {"acct_ew": "Accounting EW", "acct_vw": "Accounting VW",
             "past_ret_ew": "Past Return EW", "past_ret_vw": "Past Return VW",
             "ticker_ew": "Ticker EW", "ticker_vw": "Ticker VW"}


# ----------------------------------------------------------------------------- data layer
def _to_wide(ids, year, month, ret, n_ids_hint=None):
    """Long (id, year, month, ret) -> float32 matrix [n_strategies, n_months]."""
    m = (year.astype(np.int32) - BASE_YEAR) * 12 + (month.astype(np.int32) - 1)
    uid, inv = np.unique(ids, return_inverse=True)
    wide = np.full((len(uid), int(m.max()) + 1), np.nan, dtype=np.float32)
    wide[inv, m] = ret.astype(np.float32)
    return uid, wide


def _load_family_files():
    out = {}
    for tag in ("EW", "VW"):
        df = pd.read_csv(RAW / f"DataMinedLongShortReturns{tag}.csv",
                         usecols=["signalid", "year", "month", "ret"],
                         dtype={"signalid": "int32", "year": "int16", "month": "int8", "ret": "float32"})
        out[f"acct_{tag.lower()}"] = _to_wide(df.signalid.values, df.year.values, df.month.values, df.ret.values)
        del df
    for name, key in (("PastReturnSignalsLongShort", "past_ret"), ("TickerSignalsLongShort", "ticker")):
        df = pd.read_csv(RAW / f"{name}.csv", dtype={"signalid": "int32", "ret_ew": "float32", "ret_vw": "float32",
                                                      "nshort": "int32", "nlong": "int32"})
        df = df[(df.nshort >= 2) & (df.nlong >= 2)]          # same filter as the authors' code
        d = pd.to_datetime(df["date"])
        y, mo = d.dt.year.values, d.dt.month.values
        for tag in ("ew", "vw"):
            out[f"{key}_{tag}"] = _to_wide(df.signalid.values, y, mo, df[f"ret_{tag}"].values)
        del df
    return out


def load_wide(family):
    """Returns (signal ids, wide matrix). Cached as .npz next to the notebooks."""
    f = CACHE / f"wide_{family}.npz"
    if not f.exists():
        for fam, (uid, wide) in _load_family_files().items():
            np.savez(CACHE / f"wide_{fam}.npz", ids=uid, wide=wide)
    z = np.load(f)
    return z["ids"], z["wide"]


# ----------------------------------------------------------------------------- rolling stats
def rolling_stats(wide, oos_year):
    """IS (20y before oos_year) and OOS (oos_year) mean / t-stat / nobs for every strategy.

    t-stat = mean / (std_ddof0 / sqrt(n)), identical to statsmodels OLS 'ret ~ 1' with HAC maxlags=0.
    Returns dict of arrays + boolean mask of strategies that have enough data (same rule as the paper).
    """
    a = (oos_year - IS_N_YEARS - BASE_YEAR) * 12
    b = (oos_year - BASE_YEAR) * 12
    a = max(a, 0)
    if b + 12 > wide.shape[1]:
        return None
    IS, OOS = wide[:, a:b], wide[:, b:b + 12]
    n_is = np.sum(~np.isnan(IS), 1)
    n_oos = np.sum(~np.isnan(OOS), 1)
    with np.errstate(all="ignore"):
        m_is = np.nanmean(IS, 1)
        s_is = np.nanstd(IS, 1)                              # ddof=0
        t_is = m_is / (s_is / np.sqrt(n_is))
        m_oos = np.nanmean(OOS, 1)
    ok = (n_is >= IS_MIN_OBS) & (n_oos >= OOS_MIN_OBS) & np.isfinite(t_is)
    return dict(mean_is=m_is, tstat=t_is, nobs=n_is, sd_is=s_is, mean_oos=m_oos, ok=ok)


# ----------------------------------------------------------------------------- Empirical Bayes
# theta ~ p*N(mua, siga^2) + (1-p)*N(mub, sigb^2);   t = theta + N(0,1)
# => t is a 2-component normal mixture with variances sig^2+1.  Posterior E[theta|t] is analytic.
def _unpack(par):
    mua, lsa, logit_p, mub, lsb = par
    return mua, np.exp(lsa), np.clip(expit(logit_p), 1e-9, 1 - 1e-9), mub, np.exp(lsb)


def eb_negloglik(par, t):
    mua, sa, p, mub, sb = _unpack(par)
    lf = np.logaddexp(
        np.log(p) - 0.5 * np.log(2 * np.pi * (sa**2 + 1)) - 0.5 * (t - mua) ** 2 / (sa**2 + 1),
        np.log(1 - p) - 0.5 * np.log(2 * np.pi * (sb**2 + 1)) - 0.5 * (t - mub) ** 2 / (sb**2 + 1))
    return -np.sum(lf)


def fit_eb(t, starts=((0.5, 0.0, 0.0, -0.5, 0.0), (1.0, 0.0, 0.0, -1.0, -0.5), (0.0, 0.5, 1.0, 0.0, -1.0))):
    """Quasi-ML fit of the mixture. Multiple starts + L-BFGS (the authors use NLopt BOBYQA from one start)."""
    best = None
    for x0 in starts:
        r = minimize(eb_negloglik, np.array(x0, float), args=(t,), method="Nelder-Mead",
                     options=dict(maxiter=2000, xatol=1e-4, fatol=1e-6))
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def eb_posterior_mean(par, t):
    """E[theta | t] for the fitted mixture prior (closed form, no numerical integration)."""
    mua, sa, p, mub, sb = _unpack(par)
    out_num, out_logw = [], []
    for w, mu, s in ((p, mua, sa), (1 - p, mub, sb)):
        v = s**2 + 1
        logw = np.log(w) - 0.5 * np.log(2 * np.pi * v) - 0.5 * (t - mu) ** 2 / v     # marginal density of t
        post_mean = mu + (s**2 / v) * (t - mu)                                          # E[theta|t, component]
        out_logw.append(logw); out_num.append(post_mean)
    lw = np.stack(out_logw); pm = np.stack(out_num)
    w = np.exp(lw - logsumexp(lw, axis=0))
    return (w * pm).sum(0)


# ----------------------------------------------------------------------------- panel builder
def build_panel(family, years=None):
    """One row per (strategy, oos_year): IS stats, EB prediction, realised OOS mean return."""
    ids, wide = load_wide(family)
    rows = []
    for y in (years or range(OOS_START, YEAR_MAX + 1)):
        st = rolling_stats(wide, y)
        if st is None:
            continue
        ok = st["ok"]
        t = st["tstat"][ok]
        par = fit_eb(t)
        th = eb_posterior_mean(par, t)
        mean_is = st["mean_is"][ok]
        rows.append(pd.DataFrame(dict(
            family=family, signalid=ids[ok], oos_year=y, mean_is=mean_is, tstat=t, nobs=st["nobs"][ok],
            sd_is=st["sd_is"][ok], thetahat=th, pred_ret=th * mean_is / t, mean_oos=st["mean_oos"][ok])))
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------- selection + evaluation
_WIDE = {}


def wide_of(family):
    if family not in _WIDE:
        ids, w = load_wide(family)
        _WIDE[family] = (pd.Series(np.arange(len(ids)), index=ids), w)
    return _WIDE[family]


def signed_monthly_returns(panel, sign):
    """Realised monthly OOS returns of every panel row (rows x 12), multiplied by sign (+1 long / -1 flipped)."""
    R = np.full((len(panel), 12), np.nan, dtype=np.float32)
    for fam, idx in panel.groupby("family").indices.items():
        pos, wide = wide_of(fam)
        g = panel.iloc[idx]
        rows = pos.loc[g.signalid.values].values
        b = ((g.oos_year.values - BASE_YEAR) * 12)[:, None] + np.arange(12)[None, :]
        R[idx] = wide[rows[:, None], b]
    return R * sign[:, None]


def monthly_top_returns(panel, score, sign, ranks=(1, 5, 10)):
    """Paper's selection rule. Per oos_year, across ALL families pooled:
    orient each strategy by `sign` (+1 keep, -1 flip long/short), rank by the oriented `score` (percentile),
    keep the top r%, return the equal-weighted monthly OOS return of that portfolio.
    panel must have columns family, signalid, oos_year. score/sign are arrays aligned with panel rows.
    """
    sign = np.asarray(sign, float)
    R = signed_monthly_returns(panel, sign)
    pct = pd.Series(np.asarray(score, float)).groupby(panel.oos_year.values).rank(pct=True).values * 100
    years = panel.oos_year.values
    out = []
    for y in np.unique(years):
        m = years == y
        for r in ranks:
            sel = m & (pct >= 100 - r)
            out.append(pd.DataFrame(dict(year=y, month=np.arange(1, 13), rank=r,
                                         ret=np.nanmean(R[sel], 0), nstrat=int(sel.sum()))))
    return pd.concat(out, ignore_index=True)


def summarize(mr, lo=1983, hi=2020):
    """Annualised mean (%), t-stat and annualised Sharpe of the monthly portfolio returns."""
    d = mr[(mr.year >= lo) & (mr.year <= hi)]
    g = d.groupby("rank").ret
    s = pd.DataFrame(dict(mean_ann=g.mean() * 12, sd=g.std(), n=g.count(), nstrat=d.groupby("rank").nstrat.mean()))
    s["sharpe"] = g.mean() / s["sd"] * np.sqrt(12)
    s["tstat"] = g.mean() / s["sd"] * np.sqrt(s["n"])
    return s[["nstrat", "mean_ann", "tstat", "sharpe"]]


def build_all_panels(families=FAMILIES, years=None):
    return pd.concat([build_panel(f, years) for f in families], ignore_index=True)
