"""Queue 4 (acct_ew, base T3+hpca16 = the model with the highest within-year rank IC: dev 0.210, holdout 0.105 vs EB 0.238 / 0.070).
Root cause being tested: the 1280-dim frozen FinCast tokens add seed noise and dilute the in-sample-mean signal (hpca16 raised dev IC
0.152 -> 0.210).  Size of the bottleneck (4/8/32), cross-sectional rank-gauss of the statistical inputs (fx=rank), Wide & Deep path,
and the Sharpe-unit correlation term (which lowered holdout MSE error on T3).  5 seeds each, medians of 3 and 5.
Results: _cache/queue4_results.tsv.  Usage: python run_queue4.py"""
import traceback, pandas as pd, ht_common as h, ht_base as B
from ht_sub import make_sub
B.RES2 = h.CACHE / "queue4_results.tsv"
P = pd.read_parquet(h.CACHE / "panel_eb.parquet")
F = "acct_ew"
T3 = {**B.BASE, "xs": True, "prep": "ydms", "pool": "mha", "extras": "F2", "fams": F}
JOBS = [("T3+hpca8", {"hpca": 8}), ("T3+hpca4", {"hpca": 4}), ("T3+hpca32", {"hpca": 32}), ("T3+hpca16+fxrank", {"hpca": 16, "fx": "rank"}),
        ("T3+hpca16+wide", {"hpca": 16, "wide": True}), ("T3+hpca16+corr1.0sr", {"hpca": 16, "corrw": 1.0, "corru": "sr"})]
D = make_sub(F); print("data", F, tuple(D.H.shape), flush=True)
for nm, v in JOBS:
    cfg = {**T3, **v}
    try:
        B.walk_forward(D, cfg, years=range(1990, 1992)); print("smoke ok", nm, flush=True)
        cs = []
        for sd in range(5):
            c = {**cfg, "seed": sd}
            try: B.pred_of(c)
            except FileNotFoundError: B.run_sub(c, D, P)
            cs.append(c)
        for n in (3, 5):
            B.ensemble_sub(D, P, cs[:n], f"Q4 {nm} on {F} x{n}", agg="median")
    except Exception:
        print("JOB FAIL", nm, flush=True); traceback.print_exc()
print("QUEUE DONE", flush=True)
