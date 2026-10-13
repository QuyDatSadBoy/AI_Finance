"""Within family-year IC of zero-shot forecasts (CPU). Usage: python eval_ic.py meta_acct_ew.npz [pred.npz ...]
pred.npz must hold arrays of length n (aligned with meta idx). Prints dev (1983-2004) / hold (2005-2020) mean Spearman IC with
target y (next-year mean), Sharpe-IC (x/sd vs y/sd), partial IC beyond the in-sample mean (rank residual within group), beyond\nmean + last-12-month mean, and the IC of an equal-weight within-group rank average of the signal and the in-sample mean."""
import sys, numpy as np, pandas as pd
from scipy.stats import rankdata

def ic_by_group(x, y, g, yr):
    out = {}
    df = pd.DataFrame({"x": x, "y": y, "g": g, "yr": yr}).dropna()
    r = df.groupby("g").apply(lambda d: pd.Series({"ic": np.corrcoef(rankdata(d.x), rankdata(d.y))[0, 1] if len(d) > 5 else np.nan,
                                                    "yr": d.yr.iloc[0]}), include_groups=False)
    for k, (a, b) in {"dev": (1983, 2004), "hold": (2005, 2020)}.items():
        s = r[(r.yr >= a) & (r.yr <= b)].ic
        out[k] = (s.mean(), s.std() / np.sqrt(s.notna().sum()))
    return out

def resid_rank(x, Z, g):
    """rank of x residualised on the ranks of the columns of Z within each group g"""
    Z = np.atleast_2d(np.asarray(Z).T).T if np.ndim(Z) == 1 else np.asarray(Z)
    df = pd.DataFrame({"x": x, "g": g}); zc = []
    for i in range(Z.shape[1]): df[f"z{i}"] = Z[:, i]; zc.append(f"rz{i}"); df[f"rz{i}"] = df.groupby("g")[f"z{i}"].rank(pct=True)
    df["rx"] = df.groupby("g").x.rank(pct=True)
    def f(d):
        A = np.c_[np.ones(len(d)), d[zc].values]; b = np.linalg.lstsq(A, d.rx.values, rcond=None)[0]
        return pd.Series(d.rx.values - A @ b, index=d.index)
    return df.groupby("g", group_keys=False).apply(f, include_groups=False).sort_index().values

M = np.load(sys.argv[1])
Wn = np.load("/home/phongndh/Documents/Paper/code/_cache/win240.npy", mmap_mode="r")[M["idx"]]
m12 = np.nanmean(Wn[:, -12:], 1)                                   # oriented last-12-month mean (momentum)
g = M["famcode"] * 10000 + M["year"]; yr = M["year"]; y = M["y"]; sd = M["sd"]
cols = {"EB": M["eb"], "mean_is": M["mean_or"], "absT": M["absT"], "cf_mean_small(bolt)": M["cf_mean_small"],
        "m12": m12}
for p in sys.argv[2:]:
    P = np.load(p)
    for k in P.files:
        if k in ("point12", "qmean12", "med12", "med_h1"):
            assert (P["idx"] == M["idx"]).all(), "misaligned"; cols[f"{p.split('/')[-1][:-4]}:{k}"] = P[k]
ok = np.isfinite(y)
mres = None
print(f"{'signal':48s} {'IC dev':>14s} {'IC hold':>14s} {'ShIC dev':>9s} {'ShIC hold':>9s} {'pIC|mean dev':>12s} {'pIC hold':>9s} {'pIC|mean,m12':>12s} {'hold':>6s} {'combo dev':>9s} {'hold':>6s}")
for k, x in cols.items():
    m = ok & np.isfinite(x)
    a = ic_by_group(x[m], y[m], g[m], yr[m]); b = ic_by_group(x[m] / sd[m], y[m] / sd[m], g[m], yr[m])
    c = ic_by_group(resid_rank(x[m], M["mean_or"][m], g[m]), y[m], g[m], yr[m]) if k != "mean_is" else {"dev": (np.nan, 0), "hold": (np.nan, 0)}
    rk = pd.DataFrame({"x": x[m], "z": M["mean_or"][m], "g": g[m]}).groupby("g").rank(pct=True)
    e = ic_by_group(rk.x.values + rk.z.values, y[m], g[m], yr[m])     # equal-weight rank combo with the in-sample mean (no fitting)
    mm = m & np.isfinite(m12)
    d = ic_by_group(resid_rank(x[mm], np.c_[M["mean_or"][mm], m12[mm]], g[mm]), y[mm], g[mm], yr[mm]) if k not in ("mean_is", "m12") else {"dev": (np.nan, 0), "hold": (np.nan, 0)}
    print(f"{k:48s} {a['dev'][0]:7.3f}±{a['dev'][1]:.3f} {a['hold'][0]:7.3f}±{a['hold'][1]:.3f} {b['dev'][0]:9.3f} {b['hold'][0]:9.3f} {c['dev'][0]:12.3f} {c['hold'][0]:9.3f} {d['dev'][0]:12.3f} {d['hold'][0]:6.3f} {e['dev'][0]:9.3f} {e['hold'][0]:6.3f}", flush=True)
