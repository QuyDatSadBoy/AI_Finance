"""Queue 3 (acct_ew): seed variance dominates (single seeds of T3+hpca16: dev MSE -6.6/-5.7% vs EB, their 3-seed median +0.6%), so the
dev-selected candidates are re-run with 10 seeds and scored as 3/5/10-seed medians.  Candidates: T (xs+ydm+mha, best dev Sharpe so far),
T3 (= T2 + ydms), each with and without hpca16 (best within-year IC).  Results: _cache/queue3_results.tsv.
Usage: python run_queue3.py"""
import traceback, pandas as pd, ht_common as h, ht_base as B
from ht_sub import make_sub
B.RES2 = h.CACHE / "queue3_results.tsv"
P = pd.read_parquet(h.CACHE / "panel_eb.parquet")
F = "acct_ew"
T = {**B.BASE, "xs": True, "prep": "ydm", "pool": "mha", "fams": F}                                   # run_fam1 "xs+ydm+mha"
T3 = {**B.BASE, "xs": True, "prep": "ydms", "pool": "mha", "extras": "F2", "fams": F}               # run_queue2 base
JOBS = [("T", T), ("T+hpca16", {**T, "hpca": 16}), ("T3", T3), ("T3+hpca16", {**T3, "hpca": 16})]
D = make_sub(F); print("data", F, tuple(D.H.shape), flush=True)
for nm, cfg in JOBS:
    try:
        cs = []
        for sd in range(10):
            c = {**cfg, "seed": sd}
            try: B.pred_of(c)
            except FileNotFoundError: B.run_sub(c, D, P)
            cs.append(c)
        for n in (3, 5, 10):
            B.ensemble_sub(D, P, cs[:n], f"Q3 {nm} on {F} x{n}", agg="median")
    except Exception:
        print("JOB FAIL", nm, flush=True); traceback.print_exc()
print("QUEUE DONE", flush=True)
