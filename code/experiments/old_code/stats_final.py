"""Significance tests of AI ensembles vs EB, CPU only (no GPU, no hidden states).  Usage: python stats_final.py
For each candidate: forecast accuracy (MSE vs EB, Diebold-Mariano clustered by year) and top-1/5/10% equal-weight portfolios vs EB:
Sharpe difference (Jobson-Korkie-Memmel, year-block bootstrap) and mean-return difference (Newey-West 12 lags), on dev / holdout / full."""
import sys, numpy as np, pandas as pd, ht_common as h, ht_wf as wf, ht_arch as A, ht_new as nw, ht_base as B

class LightD:                                                 # the fields A.evaluate / B.accuracy need, without the 6.7 GB of hidden states
    def __init__(self):
        F = wf.load_all(); rows = np.load(h.CACHE / "fincast_h_rows.npy")
        self.F = F.iloc[rows].reset_index(drop=True)
        self.sd, self.sg, self.yr = self.F.sd_is.values, self.F.sgn.values, self.F.oos_year.values
        self.y_or = self.F.y.values; self.eb_or = self.F.pred_ret.values * self.sg; self.nobs = self.F.nobs.values.astype(np.float32)
        self.famcode = self.F.family.astype("category").cat.codes.values

def ens(base, n=5, agg="mean"):
    return (np.median if agg == "median" else np.mean)([B.pred_of({**base, "seed": s}) for s in range(n)], 0)

def report(label, pred, D, P, eb_rets):
    acc = B.accuracy(D, pred)
    _, rets = A.evaluate(P, D, pred, "T0", "sr", 1.0, schemes=("eq",))
    rows = []
    for r in (1, 5, 10):
        a, e = rets[(r, "eq")], eb_rets[(r, "eq")]
        for nm, (lo, hi) in (("dev", nw.DEV), ("hold", nw.HOLD), ("full", (1983, 2020))):
            pa, pe = nw.perf(a, lo, hi), nw.perf(e, lo, hi)
            _, z, p_j = nw.sharpe_test_jkm(a, e, lo, hi); _, ci, p_b = nw.sharpe_diff_bootstrap_p(a, e, lo, hi)
            dm, t_m, p_m = nw.mean_diff_test(a, e, lo, hi)
            rows.append(dict(model=label, top=f"{r}%", period=nm, ret_ai=pa["mean"], ret_eb=pe["mean"], sr_ai=pa["sharpe"], sr_eb=pe["sharpe"],
                             p_JKM=p_j, p_boot=p_b, mean_diff=dm, p_mean_NW=p_m, mse_vs_eb_dev=acc["mse_vs_eb_dev"], dm_t_dev=acc["dm_t_dev"],
                             mse_vs_eb_hold=acc["mse_vs_eb_hold"], dm_t_hold=acc["dm_t_hold"]))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    D = LightD(); P = pd.read_parquet(h.CACHE / "panel_eb.parquet")
    _, eb_rets = A.evaluate(P, D, D.eb_or.astype(np.float32), "T0", "sr", 1.0, schemes=("eq",))   # EB's own forecast, same pipeline
    XB = {**B.BASE, "xs": True}
    YM = {**XB, "prep": "ydm", "pool": "mha"}
    cands = {"xs+ydm+mha mean5": ens(YM), "xs+ydm+mha median5": ens(YM, agg="median"), "xs mean5": ens(XB), "xs+ydm mean5": ens({**XB, "prep": "ydm"}), "xs+ydm median5": ens({**XB, "prep": "ydm"}, agg="median"),
             "xs+mha+F2 median5": ens({**XB, "extras": "F2", "pool": "mha"}, agg="median")}
    out = pd.concat([report(k, v, D, P, eb_rets) for k, v in cands.items()], ignore_index=True)
    out.to_csv(h.CACHE / "stats_final.csv", index=False)
    pd.set_option("display.width", 250); print(out.round(3).to_string())
