"""Queue 17 (acct_ew): Chronos-2 240 encoder + HyperAttention + Holen weighting (both user papers), 10 seeds; then seeds 10-19 of the dev
leaders that use the user's papers (Dt0_chr2_240_vyH, Dt0_hyp1k_vyH, and the new config) for 20-seed tests.  Results: _cache/q17_results.tsv"""
import traceback, pandas as pd, ht_common as h, ht_base as B
from ht_sub import make_sub
from run_q6_cfgs import CFGS
B.RES2 = h.CACHE / "q17_results.tsv"
P = pd.read_parquet(h.CACHE / "panel_eb.parquet")
D = make_sub("acct_ew"); print("data", tuple(D.H.shape), flush=True)
for nm, seeds in (("Dt0_chr2_240_hyp1k_vyH", range(20)), ("Dt0_chr2_240_vyH", range(20)), ("Dt0_hyp1k_vyH", range(20))):
    cfg = CFGS[nm]
    try:
        B.use_encoder(D, cfg.get("enc", "fincast")); cs = []
        for sd in seeds:
            c = {**cfg, "seed": sd}
            try: B.pred_of(c)
            except FileNotFoundError: B.run_sub(c, D, P)
            cs.append(c)
        B.ensemble_sub(D, P, cs[:10], f"Q17 {nm} on acct_ew x10", agg="mean")
        B.ensemble_sub(D, P, cs, f"Q17 {nm} on acct_ew x20", agg="mean")
    except Exception:
        print("JOB FAIL", nm, flush=True); traceback.print_exc()
print("QUEUE DONE", flush=True)
