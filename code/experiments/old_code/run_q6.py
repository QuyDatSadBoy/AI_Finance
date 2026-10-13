"""Queue 6 (acct_ew, 10 seeds each, cumulative, dev-only reading): fixes from the block trace + code audit (docs/KIEN_TRUC section 6.10).
A0 = xs+ydm+mha+hpca16 with the output recentred inside each family-year (rc; one set per family-year, deterministic chunks)
A1 = A0 + closed-form linear base on [m, m/sd, |t|, sd] (the network learns the residual)   A2 = A1 + level from labels <= Y-1
A3 = A2 + training hygiene (ep0, seed2, optkeep)   A4 = A3 + extras demeaned within family-year   A5 = A4 + no LayerNorm on PCA tokens
Results: _cache/q6_results.tsv.  Usage: python run_q6.py"""
import traceback, pandas as pd, ht_common as h, ht_base as B
from ht_sub import make_sub
B.RES2 = h.CACHE / "q6_results.tsv"
P = pd.read_parquet(h.CACHE / "panel_eb.parquet")
F = "acct_ew"
A0 = {**B.BASE, "xs": True, "prep": "ydm", "pool": "mha", "fams": F, "hpca": 16, "rc": True, "xset": 4096, "detev": True}
A1 = {**A0, "base": "lin4", "bK": 15}
A2 = {**A1, "lvl": "exp1"}
A3 = {**A2, "ep0": True, "seed2": True, "optkeep": True}
A4 = {**A3, "fx": "dm"}
A5 = {**A4, "hln": False}
JOBS = [("A0", A0), ("A1", A1), ("A2", A2), ("A3", A3), ("A4", A4), ("A5", A5)]
D = make_sub(F); print("data", F, tuple(D.H.shape), flush=True)
for nm, cfg in JOBS:
    try:
        cs = []
        for sd in range(10):
            c = {**cfg, "seed": sd}
            try: B.pred_of(c)
            except FileNotFoundError: B.run_sub(c, D, P)
            cs.append(c)
        B.ensemble_sub(D, P, cs, f"Q6 {nm} on {F} x10", agg="median")
        B.ensemble_sub(D, P, cs, f"Q6 {nm} on {F} x10", agg="mean")
    except Exception:
        print("JOB FAIL", nm, flush=True); traceback.print_exc()
print("QUEUE DONE", flush=True)
