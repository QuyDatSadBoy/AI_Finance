"""Queue 10 (acct_ew, 10 seeds, mean ensembles): one block or technique changed at a time on the candidate B3 -- sequence block (GRU,
Transformer, none), pooling (attn, query, mean), set attention (HyperAttention, ISAB, none), Holen et al. L-/R-weighting (R EXPLORATORY),
upsampling / mixup, other pretrained encoders (Chronos-2 hidden states, 128 / 240 months).  Results: _cache/q10_results.tsv"""
import traceback, pandas as pd, ht_common as h, ht_base as B
from ht_sub import make_sub
from run_q6_cfgs import CFGS
B.RES2 = h.CACHE / "q10_results.tsv"
P = pd.read_parquet(h.CACHE / "panel_eb.parquet")
D = make_sub("acct_ew"); print("data", tuple(D.H.shape), flush=True)
for sd in range(10, 20):                                       # robustness of the dev-selected Dt0 (seeds 10-19 = separate ensemble)
    c = {**CFGS["Dt0"], "seed": sd}
    try: B.pred_of(c)
    except FileNotFoundError: B.run_sub(c, D, P)
print("extra seeds done Dt0", flush=True)
NAMES = ["Dt0_nw", "B3_gru", "B3_tf", "B3_noseq", "B3_attn", "B3_query", "B3_mean", "B3_hyper", "B3_isab", "B3_noxs", "B3_rw4",
         "B3_up2", "B3_mix2", "B3_chr2", "B3_chr2_240"]
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
        B.ensemble_sub(D, P, cs, f"Q10 {nm} on acct_ew x10", agg="mean")
    except Exception:
        print("JOB FAIL", nm, flush=True); traceback.print_exc()
print("QUEUE DONE", flush=True)
