"""Queue 12 (acct_ew, 10 seeds, mean ensembles): one block or technique changed at a time on the dev-selected Dt0 -- sequence block
(GRU, Transformer, none), pooling (attn, query, mean), set attention (HyperAttention, ISAB, none), upsampling / mixup, other pretrained
encoders (Chronos-2 hidden states 128 / 240 months), non-pretrained encoders (lp7, raw), R-weighting (EXPLORATORY).  Results: _cache/q12_results.tsv"""
import traceback, pandas as pd, ht_common as h, ht_base as B
from ht_sub import make_sub
from run_q6_cfgs import CFGS
B.RES2 = h.CACHE / "q12_results.tsv"
P = pd.read_parquet(h.CACHE / "panel_eb.parquet")
D = make_sub("acct_ew"); print("data", tuple(D.H.shape), flush=True)
NAMES = ["Dt0_lp7", "Dt0_raw", "Dt0_gru", "Dt0_tf", "Dt0_noseq", "Dt0_attn", "Dt0_query", "Dt0_mean", "Dt0_hyper", "Dt0_isab", "Dt0_noxs",
         "Dt0_up2", "Dt0_mix2", "Dt0_chr2", "Dt0_chr2_240", "Dt0_rw4"]
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
        B.ensemble_sub(D, P, cs, f"Q12 {nm} on acct_ew x10", agg="mean")
    except Exception:
        print("JOB FAIL", nm, flush=True); traceback.print_exc()
print("QUEUE DONE", flush=True)
