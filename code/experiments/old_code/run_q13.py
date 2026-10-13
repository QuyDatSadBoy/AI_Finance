"""Queue 13 (acct_ew): the dev-selected config after the block ablations (highest mean dev Sharpe: Dt0 with Chronos-2 hidden states of the
240-month window as encoder, PRETRAINED) -- seeds 10-19 as a separate ensemble for the robustness check.  Results: _cache/q13_results.tsv"""
import pandas as pd, ht_common as h, ht_base as B
from ht_sub import make_sub
from run_q6_cfgs import CFGS
B.RES2 = h.CACHE / "q13_results.tsv"
P = pd.read_parquet(h.CACHE / "panel_eb.parquet")
D = make_sub("acct_ew"); cfg = CFGS["Dt0_chr2_240"]; B.use_encoder(D, cfg["enc"])
cs = []
for sd in range(20):
    c = {**cfg, "seed": sd}
    try: B.pred_of(c)
    except FileNotFoundError: B.run_sub(c, D, P)
    cs.append(c)
B.ensemble_sub(D, P, cs[10:], "Q13 Dt0_chr2_240 seeds10-19 on acct_ew x10", agg="mean")
B.ensemble_sub(D, P, cs, "Q13 Dt0_chr2_240 on acct_ew x20", agg="mean")
print("QUEUE DONE", flush=True)
