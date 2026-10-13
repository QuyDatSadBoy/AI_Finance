"""Queue 8 (acct_ew, 10 seeds): the time-series signal on top of A1 -- own momentum extras (E2), training rows from 1975 only, non-pretrained
log-patch tokens (lp7) instead of FinCast, 10-year level.  Results: _cache/q8_results.tsv"""
import traceback, pandas as pd, ht_common as h, ht_base as B
from ht_sub import make_sub
from run_q6_cfgs import CFGS
B.RES2 = h.CACHE / "q8_results.tsv"
P = pd.read_parquet(h.CACHE / "panel_eb.parquet")
D = make_sub("acct_ew"); print("data", tuple(D.H.shape), flush=True)
for nm in ("C1", "C2", "C3", "C4", "B2_r10"):
    try:
        cs = []
        for sd in range(10):
            c = {**CFGS[nm], "seed": sd}
            B.use_encoder(D, c.get("enc", "fincast"))
            try: B.pred_of(c)
            except FileNotFoundError: B.run_sub(c, D, P)
            cs.append(c)
        B.ensemble_sub(D, P, cs, f"Q8 {nm} on acct_ew x10", agg="mean")
    except Exception:
        print("JOB FAIL", nm, flush=True); traceback.print_exc()
print("QUEUE DONE", flush=True)
