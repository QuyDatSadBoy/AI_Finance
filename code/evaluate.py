"""Đánh giá theo cách của paper (Chen & Dim): danh mục top 1/5/10%, dự báo không chệch, IC theo năm, và các kiểm định.

Danh mục năm Y (chỉ dùng dự báo, không dùng lãi thật của năm Y để chọn):
    1) điểm = |dự báo Sharpe| = |dự báo / sd_is|   (xếp theo Sharpe dự báo, như paper)
    2) chiều long/short = dấu của dự báo           (dự báo âm -> đảo chiều chiến lược)
    3) chọn top r% theo điểm trong năm, chia đều, lãi tháng của 12 tháng năm Y.
"""
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import norm
from data import DEV, HOLD, BASE_YEAR

PERIODS = (("dev 1983-2004", DEV), ("holdout 2005-2020", HOLD), ("full 1983-2020", (1983, 2020)))


def signal_from_oriented(T, pred):
    """Dự báo của mô hình (%/tháng, đã nhân dấu sgn) -> (dự báo theo dấu gốc, chiều long/short, điểm xếp hạng)."""
    z = pred / T.sd_is.values                            # Sharpe tháng dự báo, theo dấu đã nhân
    zs = z * T.sgn.values                                # đổi về dấu gốc của chiến lược
    return zs, np.where(zs >= 0, 1.0, -1.0), np.abs(z)


def signal_from_eb(T):
    """EB của paper (dự báo theo dấu gốc) -> cùng ba đại lượng. Chỉ để so sánh."""
    z = T.eb.values / T.sd_is.values
    return z, np.where(z >= 0, 1.0, -1.0), np.abs(z)


def signal_abs_t(T):
    """Mốc không mô hình: xếp theo |t-stat| trong năm (cùng luật chọn). Dự báo = (z-score |t| + 10) x sd, cùng dấu với mean in-sample."""
    at = np.abs(T.tstat.values)
    g = pd.Series(at).groupby(T.oos_year.values)
    z = (at - g.transform("mean").values) / g.transform("std").values + 10
    return signal_from_oriented(T, (z * T.sd_is.values).astype(np.float32))


def top_portfolios(T, ids, wide, sig, ranks=(1, 5, 10), years=(1983, 2020)):
    """sig = (dự báo, sign, score) -> {r: DataFrame(year, month, ret)}: lãi tháng của danh mục top r%, chia đều."""
    _, sign, score = sig
    m = (T.oos_year.values >= years[0]) & (T.oos_year.values <= years[1]) & np.isfinite(score)
    pos = pd.Series(np.arange(len(ids)), index=ids)
    rows = pos.loc[T.signalid.values[m]].values
    yr = T.oos_year.values[m]
    R = wide[rows[:, None], ((yr - BASE_YEAR) * 12)[:, None] + np.arange(12)[None, :]] * sign[m][:, None]
    pct = pd.Series(score[m]).groupby(yr).rank(pct=True).values * 100
    out = {}
    for r in ranks:
        parts = []
        for Y in np.unique(yr):
            sel = (yr == Y) & (pct >= 100 - r)
            ro = R[sel]; ok = ~np.isnan(ro)
            parts.append(pd.DataFrame(dict(year=Y, month=np.arange(1, 13), ret=np.nansum(ro, 0) / ok.sum(0))))
        out[r] = pd.concat(parts, ignore_index=True)
    return out


def perf(ret, lo, hi):
    d = ret[(ret.year >= lo) & (ret.year <= hi)].ret.values
    return dict(mean=d.mean() * 12, sharpe=d.mean() / d.std(ddof=1) * np.sqrt(12), vol=d.std(ddof=1) * np.sqrt(12))


def _pair(a, b, lo, hi):
    f = lambda d: d[(d.year >= lo) & (d.year <= hi)].sort_values(["year", "month"]).ret.values
    return f(a), f(b)


def sharpe_test_hac(a, b, lo, hi, L=12):
    """Ledoit-Wolf (2008): kiểm định SR_a = SR_b, HAC (Bartlett, L trễ). Trả p hai phía."""
    x, y = _pair(a, b, lo, hi)
    T = len(x); u = np.array([x.mean(), y.mean(), (x ** 2).mean(), (y ** 2).mean()])
    V = np.c_[x - u[0], y - u[1], x ** 2 - u[2], y ** 2 - u[3]]; S = V.T @ V / T
    for l in range(1, L + 1):
        G = V[l:].T @ V[:-l] / T; S += (1 - l / (L + 1)) * (G + G.T)
    f = lambda v: v[0] / np.sqrt(v[2] - v[0] ** 2) - v[1] / np.sqrt(v[3] - v[1] ** 2)
    grad = np.array([(f(u + 1e-6 * e) - f(u - 1e-6 * e)) / 2e-6 for e in np.eye(4)])
    return 2 * norm.sf(abs(f(u) / np.sqrt(grad @ S @ grad / T)))


def sharpe_test_jkm(a, b, lo, hi):
    """Jobson-Korkie (1981) + Memmel (2003). Trả p hai phía."""
    A, B = _pair(a, b, lo, hi)
    T = len(A); s1, s2 = A.mean() / A.std(ddof=1), B.mean() / B.std(ddof=1); rho = np.corrcoef(A, B)[0, 1]
    V = (2 - 2 * rho + 0.5 * (s1 ** 2 + s2 ** 2 - 2 * s1 * s2 * rho ** 2)) / T
    return 2 * (1 - norm.cdf(abs((s1 - s2) / np.sqrt(V))))


def sharpe_test_boot(a, b, lo, hi, n=10000, seed=0):
    """Bootstrap theo khối năm (n lần), hiệu Sharpe được dịch về 0 dưới H0. Trả p hai phía."""
    piv = lambda d: d[(d.year >= lo) & (d.year <= hi)].pivot(index="year", columns="month", values="ret").values
    A, B = piv(a), piv(b)
    sh = lambda M: M.mean() / M.std(ddof=1)
    d0 = (sh(A.ravel()) - sh(B.ravel())) * np.sqrt(12)
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(A), len(A)); d.append((sh(A[i].ravel()) - sh(B[i].ravel())) * np.sqrt(12))
    d = np.array(d)
    return np.mean(np.abs(d - d.mean()) >= abs(d0))


def compare(a_r, b_r, la, lb):
    """Bảng so sánh Sharpe top 1/5/10% của hai mô hình ở dev / holdout / cả kỳ, kèm p HAC, bootstrap, JKM."""
    rows = []
    for k in (1, 5, 10):
        for per, (lo, hi) in PERIODS:
            a, b = a_r[k], b_r[k]
            rows.append({"top": f"{k}%", "kỳ": per.split()[0], f"SR {la}": perf(a, lo, hi)["sharpe"], f"SR {lb}": perf(b, lo, hi)["sharpe"],
                         "p_HAC": sharpe_test_hac(a, b, lo, hi), "p_boot": sharpe_test_boot(a, b, lo, hi), "p_JKM": sharpe_test_jkm(a, b, lo, hi)})
    return pd.DataFrame(rows).round(3)


def sharpe_table(port):
    """{tên: {r: DataFrame}} -> bảng Sharpe top 1/5/10% và trung bình ở dev / holdout / cả kỳ."""
    rows = []
    for nm, R in port.items():
        r = {"mô hình": nm}
        for per, (lo, hi) in PERIODS:
            v = [perf(R[k], lo, hi)["sharpe"] for k in (1, 5, 10)]
            r[per.split()[0] + " top1/5/10"] = " / ".join(f"{x:.2f}" for x in v); r[per.split()[0] + " TB"] = round(float(np.mean(v)), 3)
        rows.append(r)
    return pd.DataFrame(rows)


def ic_years(T, pred):
    """IC theo năm: tương quan hạng (Spearman) giữa dự báo và lãi thật của năm, trên mọi chiến lược của năm (1983-2020)."""
    d = pd.DataFrame(dict(yr=T.oos_year.values, p=pred, y=T.y.values)); d = d[d.yr >= 1983]
    return d.groupby("yr").apply(lambda x: x.p.corr(x.y, method="spearman"), include_groups=False)


def ic_test(ia, ie, lo, hi):
    dd = (ia - ie).loc[lo:hi]; m = sm.OLS(dd.values, np.ones(len(dd))).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    return dict(IC_AI=ia.loc[lo:hi].mean(), IC_EB=ie.loc[lo:hi].mean(), hiệu=dd.mean(), p=m.pvalues[0], năm_AI_hơn=f"{int((dd > 0).sum())}/{len(dd)}")


def calib_bins(T, pred, lo, hi, nb=20):
    """Hình 4-5 của paper: chia nb nhóm theo dự báo trong từng năm; trung bình dự báo và lãi thật mỗi nhóm (%/tháng, dấu đã nhân)."""
    d = pd.DataFrame(dict(yr=T.oos_year.values, f=pred, y=T.y.values)); d = d[(d.yr >= lo) & (d.yr <= hi)]
    d["bin"] = d.groupby("yr").f.transform(lambda x: pd.qcut(x.rank(method="first"), nb, labels=False))
    return d.groupby("bin")[["f", "y"]].mean()
