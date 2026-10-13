"""Queue 11 (acct_ew, 10 seeds): the loss on B3 -- MSE, Huber delta 0.3 / 0.1 (robust), within-year correlation of the whole forecast (base + network) with the label as an added accuracy term.  Results: _cache/q11_results.tsv"""
import traceback, pandas as pd, ht_common as h, ht_base as B
from ht_sub import make_sub
from run_q6_cfgs import CFGS
B.RES2 = h.CACHE / "q11_results.tsv"
P = pd.read_parquet(h.CACHE / "panel_eb.parquet")
D = make_sub("acct_ew"); print("data", tuple(D.H.shape), flush=True)
NAMES = ["Dt0", "B3_hw3", "Dt0_hw3", "B3_lp7", "B3_raw", "B3_vyH", "B3_lweq4", "B3_lw4", "B3_mse", "B3_nw05", "B3_vyI", "B3_vyP", "B3_h03", "B3_c05", "B3_c2", "B3_h01"]
for nm in NAMES:
    cfg = CFGS[nm]
    try:
        B.use_encoder(D, cfg.get("enc", "fincast"))
        B.walk_forward(D, cfg, years=range(1990, 1992)); print("smoke ok", nm, flush=True)
        cs = []
        for sd in range(10):
            c = {**cfg, "seed": sd}
            try: B.pred_of(c)
            except FileNotFoundError: B.run_sub(c, D, P)
            cs.append(c)
        B.ensemble_sub(D, P, cs, f"Q11 {nm} on acct_ew x10", agg="mean")
    except Exception:
        print("JOB FAIL", nm, flush=True); traceback.print_exc()
print("QUEUE DONE", flush=True)
