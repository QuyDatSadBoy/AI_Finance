"""Queue 9 (acct_ew, 10 seeds, mean ensembles): fixed-prior anchor base (z(in-sample mean) + bw*z(own momentum), one slope fitted on
years Y-15..Y-1), grid bw chosen on DEV MSE only; Sharpe-unit variant; then the winner + zero-shot forecasts of pretrained time-series
models as extra inputs (PRETRAINED, paired with the same config without them).  Results: _cache/q9_results.tsv"""
import traceback, pandas as pd, ht_common as h, ht_base as B
from ht_sub import make_sub
from run_q6_cfgs import CFGS
B.RES2 = h.CACHE / "q9_results.tsv"
P = pd.read_parquet(h.CACHE / "panel_eb.parquet")
D = make_sub("acct_ew"); print("data", tuple(D.H.shape), flush=True)
def run(nm, cfg):
    try:
        cs = []
        for sd in range(10):
            c = {**cfg, "seed": sd}
            B.use_encoder(D, c.get("enc", "fincast"))
            try: B.pred_of(c)
            except FileNotFoundError: B.run_sub(c, D, P)
            cs.append(c)
        _, row = B.ensemble_sub(D, P, cs, f"Q9 {nm} on acct_ew x10", agg="mean")
        return row
    except Exception:
        print("JOB FAIL", nm, flush=True); traceback.print_exc()
rows = {b: run(f"D_bw{b}", CFGS[f"D_bw{b}"]) for b in (0.25, 0.5, 0.75, 1.0)}
best = max((b for b in rows if rows[b] is not None), key=lambda b: rows[b]["mse_vs_eb_dev"])      # dev only
print("dev-selected bw", best, flush=True)
for nm in ("B2_E2", "Ds_bw0.5", "Ds_bw0.75", "Ds_bw1.0"):                     # Sharpe-unit line
    run(nm, CFGS[nm])
W = CFGS[f"D_bw{best}"]
if (h.CACHE / "tsfm_x_acct_ew.npy").exists():
    for zc in ("chronos2_xl_dev", "chronos2_xl_dev|tirex", "chronos2_xl"):
        run(f"D_bw{best}+zx[{zc}]", {**W, "zx": "acct_ew", "zcols": zc})
    for zc in ("chronos2_xl_dev", "chronos2_xl_dev|tirex"):                      # same on the Sharpe-unit candidate B3
        run(f"B3+zx[{zc}]", {**CFGS["B3"], "zx": "acct_ew", "zcols": zc})
# robustness of the current candidate and the reference model: B3 seeds 10-19, the notebook's model with 10 seeds (paired comparison)
for nm, cfg, seeds in (("B3", CFGS["B3"], range(10, 20)), ("goc", {**B.BASE, "fams": "acct_ew"}, range(3, 10))):
    for sd in seeds:
        c = {**cfg, "seed": sd}; B.use_encoder(D, c.get("enc", "fincast"))
        try: B.pred_of(c)
        except FileNotFoundError: B.run_sub(c, D, P)
    print("extra seeds done", nm, flush=True)
print("QUEUE DONE", flush=True)
