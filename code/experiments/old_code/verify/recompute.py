"""Independent recomputation of B3 (10-seed mean) vs EB on acct_ew candidate universe. CPU only."""
import numpy as np, pandas as pd, sys
from scipy.stats import norm, t as tdist
import ht_common as h, ht_base as B, ht_arch as A, ht_new as nw
from ht_sub import make_sub
from run_q6_cfgs import CFGS

fam = "acct_ew"
D = make_sub(fam, with_X=False, with_H=False)
cfg = CFGS["B3"]
preds = np.array([B.pred_of({**cfg, "seed": s})[D.idx_full] for s in range(10)])
p = preds.mean(0)
print("rows", len(p), "NaN preds by year:", pd.Series(np.isnan(p)).groupby(D.yr).sum().loc[lambda s: s > 0].to_dict())
p_full = p.copy()
class S: pass
K = D.yr >= 1983
E = S(); E.F = D.F[K].reset_index(drop=True); E.sd, E.sg, E.yr = D.sd[K], D.sg[K], D.yr[K]
preds, p = preds[:, K], p[K]
print("NaN preds in 1983+:", np.isnan(p).sum(), " rows per year range:", pd.Series(E.yr).value_counts().agg(["min", "max"]).to_dict())
print("seed corr of preds/sd (dev):", np.corrcoef((preds / E.sd)[:, E.yr <= 2004])[np.triu_indices(10, 1)].mean().round(3))
Dfull = D; D = E

# ---- own portfolio construction from wide matrix
ids, wide = h.load_wide(fam)
pos = pd.Series(np.arange(len(ids)), index=ids)
F = D.F
rowpos = pos.loc[F.signalid.values].values
b = (F.oos_year.values - h.BASE_YEAR) * 12
Rraw = wide[rowpos[:, None], b[:, None] + np.arange(12)[None, :]].astype(float)   # unsigned raw strategy returns
# check panel consistency: oriented y equals mean OOS * sgn
P = pd.read_parquet(h.CACHE / "panel_eb.parquet"); P = P[P.family == fam]
M = F[["signalid", "oos_year"]].merge(P[["signalid", "oos_year", "pred_ret", "sd_is", "mean_oos"]], on=["signalid", "oos_year"], how="left")
assert M.pred_ret.notna().all()
print("max |nanmean(Rraw) - mean_oos|:", np.nanmax(np.abs(np.nanmean(Rraw, 1) - M.mean_oos.values)))
print("max |sd_is diff|:", np.abs(M.sd_is.values - D.sd).max(), " sgn check:", np.mean(np.sign(F.mean_is.values) == D.sg))

def port(z_signed, k_rule="pct"):
    """z_signed: forecast in monthly-Sharpe units, in the strategy's own (raw) direction."""
    sign = np.where(z_signed >= 0, 1.0, -1.0); score = np.abs(z_signed)
    out = []
    for Y in np.unique(D.yr):
        idx = np.where(D.yr == Y)[0]; N = len(idx)
        o = idx[np.argsort(-score[idx], kind="stable")]
        if k_rule == "pct":   # replicate rank(pct) >= 99 incl. ties
            r = pd.Series(score[idx]).rank(pct=True).values * 100
            sel = idx[r >= 99]
        else:
            k = max(1, int(round(0.01 * N))); sel = o[:k]
        R = Rraw[sel] * sign[sel, None]
        out.append(pd.DataFrame(dict(year=Y, month=np.arange(1, 13), ret=np.nanmean(R, 0), n=len(sel))))
    return pd.concat(out, ignore_index=True)

z_ai = (p / D.sd) * D.sg                      # oriented %/month -> Sharpe -> raw direction
z_eb = M.pred_ret.values / D.sd               # EB thetahat/sqrt(n)
print("EB check thetahat/sqrt(n):", np.nanmax(np.abs(z_eb - P.set_index(["signalid", "oos_year"]).loc[list(zip(F.signalid, F.oos_year))].eval("thetahat/sqrt(nobs)").values)))
a, e = port(z_ai), port(z_eb)
a2, e2 = port(z_ai, "round"), port(z_eb, "round")
print("n selected per year (pct rule) ai:", a.groupby("year").n.first().describe()[["min", "max"]].to_dict(), " round rule:", a2.groupby("year").n.first().describe()[["min", "max"]].to_dict())

def sr(x): return x.mean() / x.std(ddof=1) * np.sqrt(12)
def sub(df, lo, hi): return df[(df.year >= lo) & (df.year <= hi)].sort_values(["year", "month"])

def jkm(x, y):
    T = len(x); s1, s2 = x.mean() / x.std(ddof=1), y.mean() / y.std(ddof=1); r = np.corrcoef(x, y)[0, 1]
    V = (2 - 2 * r + 0.5 * (s1**2 + s2**2 - 2 * s1 * s2 * r**2)) / T
    return 2 * norm.sf(abs((s1 - s2) / np.sqrt(V)))

def lw_hac(x, y, L=12):
    """Ledoit-Wolf (2008) style HAC delta-method test of SR diff (Bartlett kernel)."""
    T = len(x); m1, m2 = x.mean(), y.mean(); g1, g2 = (x**2).mean(), (y**2).mean()
    V = np.c_[x - m1, y - m2, x**2 - g1, y**2 - g2]
    S = V.T @ V / T
    for l in range(1, L + 1):
        G = V[l:].T @ V[:-l] / T; S += (1 - l / (L + 1)) * (G + G.T)
    def f(u): return u[0] / np.sqrt(u[2] - u[0]**2) - u[1] / np.sqrt(u[3] - u[1]**2)
    u = np.array([m1, m2, g1, g2]); eps = 1e-6
    grad = np.array([(f(u + eps * np.eye(4)[i]) - f(u - eps * np.eye(4)[i])) / (2 * eps) for i in range(4)])
    se = np.sqrt(grad @ S @ grad / T)
    return 2 * norm.sf(abs(f(u) / se))

def yblock(x, y, nyr, n=20000, seed=1):
    X, Y = x.reshape(nyr, 12), y.reshape(nyr, 12); rng = np.random.default_rng(seed)
    d0 = sr(x) - sr(y); d = np.empty(n)
    for i in range(n):
        j = rng.integers(0, nyr, nyr); d[i] = sr(X[j].ravel()) - sr(Y[j].ravel())
    return d0, np.mean(np.abs(d - d.mean()) >= abs(d0)), np.mean(np.abs(d - d0) >= abs(d0)), np.percentile(d, [2.5, 97.5])

def stationary(x, y, Lmean=12, n=20000, seed=2):
    T = len(x); rng = np.random.default_rng(seed); d0 = sr(x) - sr(y); d = np.empty(n); q = 1 / Lmean
    for i in range(n):
        idx = np.empty(T, int); idx[0] = rng.integers(T)
        newb = rng.random(T) < q; starts = rng.integers(0, T, T)
        for t in range(1, T):
            idx[t] = starts[t] if newb[t] else (idx[t - 1] + 1) % T
        d[i] = sr(x[idx]) - sr(y[idx])
    return np.mean(np.abs(d - d0) >= abs(d0)), np.percentile(d, [2.5, 97.5])

def stdz_diff(x, y):
    z = x / x.std(ddof=1) - y / y.std(ddof=1); T = len(z)
    tt = z.mean() / (z.std(ddof=1) / np.sqrt(T))
    import statsmodels.api as sm
    m = sm.OLS(z, np.ones(T)).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    return z.mean() * np.sqrt(12), tt, 2 * tdist.sf(abs(tt), T - 1), m.tvalues[0], m.pvalues[0]

for lab, (aa, ee) in (("pct-rule", (a, e)), ("round-1%", (a2, e2))):
    for per, (lo, hi) in (("dev", (1983, 2004)), ("hold", (2005, 2020)), ("full", (1983, 2020))):
        x, y = sub(aa, lo, hi).ret.values, sub(ee, lo, hi).ret.values; nyr = hi - lo + 1
        assert (sub(aa, lo, hi)[["year", "month"]].values == sub(ee, lo, hi)[["year", "month"]].values).all()
        d0, pb, pb_alt, ci = yblock(x, y, nyr, n=5000 if lab == "round-1%" else 20000)
        line = f"{lab:9s} {per:4s} SR ai {sr(x):.3f} eb {sr(y):.3f} | ret {x.mean()*12:.2f} {y.mean()*12:.2f} | JKM {jkm(x, y):.4f} | LW-HAC {lw_hac(x, y):.4f} | yblock p {pb:.4f} (centre at d0: {pb_alt:.4f}) CI {ci.round(3)}"
        if lab == "pct-rule":
            ps, cis = stationary(x, y, n=5000); sd_, tt, pt_, thac, phac = stdz_diff(x, y)
            ps6, _ = stationary(x, y, Lmean=6, n=3000); ps24, _ = stationary(x, y, Lmean=24, n=3000)
            line += f" | stationary(L=6/12/24) p {ps6:.4f}/{ps:.4f}/{ps24:.4f} | std-diff t {tt:.2f} p {pt_:.4f}, NW t {thac:.2f} p {phac:.4f}"
        print(line, flush=True)

# ---- compare against the repo pipeline (A.evaluate on P_cand)
Pall = pd.read_parquet(h.CACHE / "panel_eb.parquet"); Pall = Pall[Pall.family == fam]
keys = pd.DataFrame(dict(family=F.family.values, signalid=F.signalid.values, oos_year=D.yr)); Pc = Pall.merge(keys, on=["family", "signalid", "oos_year"])
_, ra = A.evaluate(Pc, Dfull, p_full.astype(np.float32), "T0", "sr", 1.0, ranks=(1,), schemes=("eq",))
_, re = A.evaluate(Pc, Dfull, Dfull.eb_or.astype(np.float32), "T0", "sr", 1.0, ranks=(1,), schemes=("eq",))
ra, re = ra[(1, "eq")], re[(1, "eq")]
mg = a.merge(ra, on=["year", "month"], suffixes=("", "_repo")); mge = e.merge(re, on=["year", "month"], suffixes=("", "_repo"))
print("max |own - repo| monthly ret AI:", np.abs(mg.ret - mg.ret_repo).max(), " EB:", np.abs(mge.ret - mge.ret_repo).max())
print("repo boot p (seed 0, n=2000):", nw.sharpe_diff_bootstrap_p(ra, re, 1983, 2004, n=2000)[2], " n=20000 seeds 0..4:", [round(nw.sharpe_diff_bootstrap_p(ra, re, 1983, 2004, n=20000, seed=s)[2], 4) for s in range(3)])
# per-seed and leave-one-out dispersion of top-1% dev SR
srs = [sr(sub(port((preds[s] / D.sd) * D.sg), 1983, 2004).ret.values) for s in range(10)]
print("single-seed top1 dev SR:", np.round(srs, 3), "mean", np.mean(srs).round(3))
med = port((np.median(preds, 0) / D.sd) * D.sg); x = sub(med, 1983, 2004).ret.values; y = sub(e, 1983, 2004).ret.values
print("median-ensemble top1 dev SR", sr(x).round(3), "JKM", jkm(x, y).round(4), "yblock", yblock(x, y, 22, n=5000)[1].round(4))
# per-year contribution of the difference
dd = sub(a, 1983, 2004).assign(eb=sub(e, 1983, 2004).ret.values)
yy = dd.groupby("year").agg(ai=("ret", "mean"), eb=("eb", "mean"), ai_sd=("ret", "std"), eb_sd=("eb", "std"))
print(yy.round(2).to_string())
