"""Serial experiment queue (one GPU job at a time, light per-family data reused across jobs).  Each job = one change on top of a family's base,
3 seeds, median ensemble, evaluated vs EB on the same family.  Results: _cache/queue_results.tsv.  Then the ticker inference test.
Usage: python run_queue.py"""
import gc, traceback, subprocess, sys, pandas as pd, torch, ht_common as h, ht_base as B
from ht_sub import make_sub
B.RES2 = h.CACHE / "queue2_results.tsv"
P = pd.read_parquet(h.CACHE / "panel_eb.parquet")
def T2(f): return {**B.BASE, "xs": True, "prep": "ydms", "pool": "mha", "extras": "F2", "fams": f}   # T3 = T2 + ydms
JOBS = [
    # base T3 = T2 + ydms (level in Sharpe units).  Root cause next: the Sharpe ranking ignores |t| (corr 0.22 vs EB 0.87) -> ranking-aware loss in
    # Sharpe units; head normalisation (BN train/eval mismatch); then the queued block changes, all on T3.
    ("acct_ew", "T3+hnone", {"hnorm": "none"}), ("acct_ew", "T3+corr0.2sr", {"corrw": 0.2, "corru": "sr"}),
    ("acct_ew", "T3+corr1.0sr", {"corrw": 1.0, "corru": "sr"}), ("acct_ew", "T3+hnone+corr0.5sr", {"hnorm": "none", "corrw": 0.5, "corru": "sr"}),
    ("acct_ew", "T3+regime", {"regime": True}), ("acct_ew", "T3+hpca16", {"hpca": 16}), ("acct_ew", "T3+warm_sp", {"warm": "sp"}),
    ("acct_ew", "T3+setbn+xset3072+detev", {"hnorm": "setbn", "xset": 3072, "detev": True}),
    ("past_ret_vw", "T3+hnone", {"hnorm": "none"}), ("past_ret_vw", "T3+corr0.5sr", {"corrw": 0.5, "corru": "sr"}),
    ("acct_vw", "T3+hnone", {"hnorm": "none"}), ("acct_vw", "T3+corr0.5sr", {"corrw": 0.5, "corru": "sr"}),
    ("past_ret_ew", "T3+hnone", {"hnorm": "none"}), ("past_ret_ew", "T3+corr0.5sr", {"corrw": 0.5, "corru": "sr"}),
]
cur, D = None, None
for fams, nm, v in JOBS:
    if fams != cur:
        D = None; gc.collect(); torch.cuda.empty_cache(); D = make_sub(fams); cur = fams; print("data", fams, tuple(D.H.shape), flush=True)
    cfg = {**T2(fams), **v}
    try:
        B.walk_forward(D, cfg, years=range(1990, 1992)); print("smoke ok", fams, nm, flush=True)
        cs = []
        for sd in range(3):
            c = {**cfg, "seed": sd}
            try: B.pred_of(c)
            except FileNotFoundError: B.run_sub(c, D, P)
            cs.append(c)
        B.ensemble_sub(D, P, cs, f"Q {nm} on {fams} x3", agg="median")
    except Exception:
        print("JOB FAIL", fams, nm, flush=True); traceback.print_exc()
D = None; gc.collect(); torch.cuda.empty_cache()
print("QUEUE DONE", flush=True)
