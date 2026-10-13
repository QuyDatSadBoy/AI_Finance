"""Queue 16 (acct_ew, 10 seeds): the user-specified papers on Dt0 -- HyperAttention sized for a family-year, Holen L-weighting, IC-type
accuracy loss, TiRex encoder (PRETRAINED), combinations.  Results: _cache/q16_results.tsv"""
import traceback, pandas as pd, ht_common as h, ht_base as B
from ht_sub import make_sub
from run_q6_cfgs import CFGS
B.RES2 = h.CACHE / "q16_results.tsv"
P = pd.read_parquet(h.CACHE / "panel_eb.parquet")
D = make_sub("acct_ew"); print("data", tuple(D.H.shape), flush=True)
NAMES = ["Dt0_tirex2_240", "Dt0_tirex2_128", "Dt0_tirex2_240_vyH"]
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
        B.ensemble_sub(D, P, cs, f"Q16 {nm} on acct_ew x10", agg="mean")
    except Exception:
        print("JOB FAIL", nm, flush=True); traceback.print_exc()
print("QUEUE DONE", flush=True)
