"""Queue 7 (acct_ew, 10 seeds): grids of queue-6 steps (base window bK 10/20, level roll10) and the Sharpe-unit demeaned label (ysdm:
MSE in Sharpe units = return errors weighted by 1/sd^2; the paper ranks by predicted Sharpe).  Results: _cache/q7_results.tsv"""
import traceback, pandas as pd, ht_common as h, ht_base as B
from ht_sub import make_sub
from run_q6_cfgs import CFGS
B.RES2 = h.CACHE / "q7_results.tsv"
P = pd.read_parquet(h.CACHE / "panel_eb.parquet")
D = make_sub("acct_ew"); print("data", tuple(D.H.shape), flush=True)
for nm in ("A1_K10", "A1_K20", "A2_r10", "B2", "B3"):
    try:
        cs = []
        for sd in range(10):
            c = {**CFGS[nm], "seed": sd}
            try: B.pred_of(c)
            except FileNotFoundError: B.run_sub(c, D, P)
            cs.append(c)
        B.ensemble_sub(D, P, cs, f"Q7 {nm} on acct_ew x10", agg="mean")
    except Exception:
        print("JOB FAIL", nm, flush=True); traceback.print_exc()
print("QUEUE DONE", flush=True)
