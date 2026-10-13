"""Queue 5 (acct_ew): the non-pretrained counterpart required by AGENTS.md rule B.  Same configs as the 10-seed FinCast runs of queue 3,
with the encoder replaced by raw volatility-normalised return patches (enc=raw: 4 x 32, no pretrained model).
Results: _cache/queue5_results.tsv.  Usage: python run_queue5.py"""
import traceback, pandas as pd, ht_common as h, ht_base as B
from ht_sub import make_sub
B.RES2 = h.CACHE / "queue5_results.tsv"
P = pd.read_parquet(h.CACHE / "panel_eb.parquet")
F = "acct_ew"
T = {**B.BASE, "xs": True, "prep": "ydm", "pool": "mha", "fams": F, "enc": "raw"}
JOBS = [("T+hpca16 raw", {**T, "hpca": 16}), ("T raw", T)]
D = make_sub(F); print("data", F, tuple(D.H.shape), flush=True)
for nm, cfg in JOBS:
    try:
        B.use_encoder(D, "raw"); B.walk_forward(D, cfg, years=range(1990, 1992)); print("smoke ok", nm, flush=True)
        cs = []
        for sd in range(10):
            c = {**cfg, "seed": sd}
            try: B.pred_of(c)
            except FileNotFoundError: B.run_sub(c, D, P)
            cs.append(c)
        for n in (3, 10):
            B.ensemble_sub(D, P, cs[:n], f"Q5 {nm} on {F} x{n}", agg="median")
    except Exception:
        print("JOB FAIL", nm, flush=True); traceback.print_exc()
print("QUEUE DONE", flush=True)
