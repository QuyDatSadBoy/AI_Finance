"""Building blocks of the new idea (notebook 2): online calibration, GPU risk-aware weighting, evaluation tools.

All quantities used to rank / weight strategies in year Y come from data strictly before Y:
 * IS returns (<= Y-1) for statistics, covariance and EB;
 * the calibration slope and the learners only see OOS years <= Y-1 (walk-forward, no peeking).
"""
import numpy as np, pandas as pd, torch
import ht_common as h

DEV = (1983, 2004)        # design / hyper-parameter period
HOLD = (2005, 2020)       # untouched until the final report


# ------------------------------------------------------------------------------ 1. online calibration of EB
def calibration_slopes(P, K=10, lam=0.5, cand_q=0.85):
    """Slope b[family, Y] of  realised_oriented_return = b * EB_pred  learned ONLY from years Y-K..Y-1.

    EB assumes b = 1.  Past-return strategies reverse after 2004 (b < 0), accounting EW decays (b ~ 0.5): EB's stationarity
    assumption fails.  lam shrinks the estimate toward b = 1 (a fraction `lam` of the data weight is a prior at 1).
    P needs: family, oos_year, pred_ret, mean_oos, abs_t_rank.
    """
    sg = np.where(P.pred_ret >= 0, 1.0, -1.0)
    x, y = np.abs(P.pred_ret.values), P.mean_oos.values * sg
    top = P.abs_t_rank.values >= cand_q
    fy = pd.DataFrame(dict(family=P.family.values, yr=P.oos_year.values, sxy=np.where(top, x * y, 0.0),
                           sxx=np.where(top, x * x, 0.0))).groupby(["family", "yr"]).sum()
    b = {}
    for fam in fy.index.get_level_values(0).unique():
        d = fy.loc[fam]
        for Y in range(h.OOS_START, h.YEAR_MAX + 1):
            past = d.loc[max(Y - K, d.index.min()):Y - 1] if Y - 1 >= d.index.min() else d.iloc[:0]
            if len(past) == 0:
                b[(fam, Y)] = 1.0
                continue
            sxy, sxx = past.sxy.sum(), past.sxx.sum()
            b[(fam, Y)] = (sxy + lam * sxx) / ((1 + lam) * sxx)
    return np.array([b[(f, y_)] for f, y_ in zip(P.family.values, P.oos_year.values)])


# ------------------------------------------------------------------------------ 2. GPU risk-aware weights
def _is_window(rows, Y, months):
    out = np.empty((len(rows), months), np.float32)
    for fam, idx in rows.groupby("family").indices.items():
        pos, wide = h.wide_of(fam)
        b = (Y - h.BASE_YEAR) * 12
        out[idx] = wide[pos.loc[rows.signalid.values[idx]].values, b - months:b]
    return out


@torch.no_grad()
def shrunk_mv_weights(Wis, mu, delta=0.9, iters=2, device="cuda"):
    """w ∝ max(Sigma^-1 mu, 0) with Sigma = (1-delta)*S + delta*diag(S)  (constrained to w >= 0, sum w = 1).

    S is the IS sample covariance of the (oriented) strategy returns (NaNs -> 0 after demeaning).
    Strategies that are near-duplicates (same accounting variable) are highly correlated, so their joint weight is
    automatically capped: this is the "de-clustering" the equal-weight top-k rule lacks.
    """
    X = torch.tensor(np.asarray(Wis, np.float32), device=device)
    m = torch.isnan(X)
    cnt = (~m).sum(1, keepdim=True).clamp(min=1)
    X = torch.where(m, torch.zeros_like(X), X)
    Xc = torch.where(m, torch.zeros_like(X), X - X.sum(1, keepdim=True) / cnt) / cnt.sqrt()
    S = Xc @ Xc.T
    Sig = (1 - delta) * S + delta * torch.diag(S.diag())
    mu_t = torch.tensor(mu, device=device, dtype=torch.float32)
    act = torch.ones(len(mu), dtype=torch.bool, device=device)
    w = torch.zeros_like(mu_t)
    for _ in range(iters + 1):
        idx = act.nonzero().squeeze(1)
        w = torch.zeros_like(mu_t)
        w[idx] = torch.linalg.solve(Sig[idx][:, idx], mu_t[idx])
        neg = w < 0
        if not neg.any():
            break
        act &= ~neg
    w = w.clamp(min=0)
    return (w / w.sum()).cpu().numpy()


def portfolio_returns(P, sign, score, mu, r, scheme="eq", delta=0.9, months=120, return_weights=False):
    """Top r% (pooled across families, per year) by `score`; weights 'eq' (paper) or 'mv' (shrunk risk-aware).
    mu: oriented monthly expected return used by the MV step (ignored for 'eq'). Returns monthly DataFrame."""
    sign = np.asarray(sign, float)
    R = h.signed_monthly_returns(P, sign)
    pct = pd.Series(np.asarray(score, float)).groupby(P.oos_year.values).rank(pct=True).values * 100
    years = P.oos_year.values
    out, wts = [], []
    for Y in np.unique(years):
        sel = np.where((years == Y) & (pct >= 100 - r))[0]
        if len(sel) == 0:                                   # no scored strategy this year (e.g. no forecast for it): no portfolio
            continue
        ro = R[sel]
        if scheme == "eq":
            w = np.full(len(sel), 1.0 / len(sel))
        else:
            Wis = _is_window(P.iloc[sel], Y, months) * sign[sel][:, None]
            w = shrunk_mv_weights(Wis, np.nan_to_num(mu[sel]), delta)
        ok = ~np.isnan(ro)
        ret = np.nansum(ro * w[:, None], 0) / (w[:, None] * ok).sum(0)
        out.append(pd.DataFrame(dict(year=Y, month=np.arange(1, 13), ret=ret)))
        if return_weights:
            wts.append(pd.DataFrame(dict(year=Y, row=sel, w=w)))
    out = pd.concat(out, ignore_index=True)
    return (out, pd.concat(wts)) if return_weights else out


# ------------------------------------------------------------------------------ 3. evaluation helpers
def perf(ret, lo, hi):
    d = ret[(ret.year >= lo) & (ret.year <= hi)].ret.values
    return dict(mean=d.mean() * 12, sharpe=d.mean() / d.std(ddof=1) * np.sqrt(12), vol=d.std(ddof=1) * np.sqrt(12))


def sharpe_diff_bootstrap(a, b, lo, hi, n=2000, seed=0):
    """Year-block bootstrap of Sharpe(a) - Sharpe(b) on monthly return DataFrames (same months)."""
    A = a[(a.year >= lo) & (a.year <= hi)].pivot(index="year", columns="month", values="ret").values
    B = b[(b.year >= lo) & (b.year <= hi)].pivot(index="year", columns="month", values="ret").values
    rng = np.random.default_rng(seed)
    sh = lambda M: M.mean() / M.std(ddof=1)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(A), len(A))
        d.append((sh(A[i].ravel()) - sh(B[i].ravel())) * np.sqrt(12))
    d = np.array(d)
    return d.mean(), np.percentile(d, [2.5, 97.5]), (d <= 0).mean()


def sharpe_test_jkm(a, b, lo, hi):
    """Jobson-Korkie (1981) test with the Memmel (2003) correction for H0: SR_a = SR_b on paired monthly returns.
    Returns (annualised SR difference, z, two-sided p)."""
    from scipy.stats import norm
    A = a[(a.year >= lo) & (a.year <= hi)].sort_values(["year", "month"]).ret.values
    B = b[(b.year >= lo) & (b.year <= hi)].sort_values(["year", "month"]).ret.values
    T = len(A); s1, s2 = A.mean() / A.std(ddof=1), B.mean() / B.std(ddof=1); rho = np.corrcoef(A, B)[0, 1]
    V = (2 - 2 * rho + 0.5 * (s1 ** 2 + s2 ** 2 - 2 * s1 * s2 * rho ** 2)) / T
    z = (s1 - s2) / np.sqrt(V)
    return (s1 - s2) * np.sqrt(12), z, 2 * (1 - norm.cdf(abs(z)))


def sharpe_test_hac(a, b, lo, hi, L=12):
    """Ledoit-Wolf (2008)-style HAC test of SR_a = SR_b: delta method on the moments (means, second moments) of the paired monthly
    returns with a Bartlett-kernel long-run covariance (L lags), robust to autocorrelation and heteroskedasticity.  Returns two-sided p."""
    from scipy.stats import norm
    x = a[(a.year >= lo) & (a.year <= hi)].sort_values(["year", "month"]).ret.values
    y = b[(b.year >= lo) & (b.year <= hi)].sort_values(["year", "month"]).ret.values
    T = len(x); u = np.array([x.mean(), y.mean(), (x ** 2).mean(), (y ** 2).mean()])
    V = np.c_[x - u[0], y - u[1], x ** 2 - u[2], y ** 2 - u[3]]; S = V.T @ V / T
    for l in range(1, L + 1):
        G = V[l:].T @ V[:-l] / T; S += (1 - l / (L + 1)) * (G + G.T)
    f = lambda v: v[0] / np.sqrt(v[2] - v[0] ** 2) - v[1] / np.sqrt(v[3] - v[1] ** 2)
    grad = np.array([(f(u + 1e-6 * e) - f(u - 1e-6 * e)) / 2e-6 for e in np.eye(4)])
    return 2 * norm.sf(abs(f(u) / np.sqrt(grad @ S @ grad / T)))


def sharpe_diff_bootstrap_p(a, b, lo, hi, n=5000, seed=0):
    """Year-block bootstrap, centred under H0 (difference re-centred to 0): two-sided p-value for SR_a - SR_b."""
    A = a[(a.year >= lo) & (a.year <= hi)].pivot(index="year", columns="month", values="ret").values
    B = b[(b.year >= lo) & (b.year <= hi)].pivot(index="year", columns="month", values="ret").values
    sh = lambda M: M.mean() / M.std(ddof=1)
    d0 = (sh(A.ravel()) - sh(B.ravel())) * np.sqrt(12)
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(A), len(A)); d.append((sh(A[i].ravel()) - sh(B[i].ravel())) * np.sqrt(12))
    d = np.array(d)
    return d0, np.percentile(d, [2.5, 97.5]), np.mean(np.abs(d - d.mean()) >= abs(d0))


def mean_diff_test(a, b, lo, hi):
    """Paired t-test (Newey-West, 12 lags) on the monthly return difference: (annualised mean diff %, t, p)."""
    import statsmodels.api as sm
    A = a[(a.year >= lo) & (a.year <= hi)].sort_values(["year", "month"]).ret.values
    B = b[(b.year >= lo) & (b.year <= hi)].sort_values(["year", "month"]).ret.values
    m = sm.OLS(A - B, np.ones(len(A))).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    return m.params[0] * 12, m.tvalues[0], m.pvalues[0]
