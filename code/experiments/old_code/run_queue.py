"""Serial experiment queue (one GPU job at a time, light per-family data reused across jobs).  Each job = one change on top of a family's base,
3 seeds, median ensemble, evaluated vs EB on the same family.  Results: _cache/queue_results.tsv.  Then the ticker inference test.
Usage: python run_queue.py"""
import gc, traceback, subprocess, sys, pandas as pd, torch, ht_common as h, ht_base as B
from ht_sub import make_sub
B.RES2 = h.CACHE / "queue_results.tsv"
P = pd.read_parquet(h.CACHE / "panel_eb.parquet")
def T2(f): return {**B.BASE, "xs": True, "prep": "ydm", "pool": "mha", "extras": "F2", "fams": f}
JOBS = [
    # root cause (volbias check): the constant ydm level / sd makes the Sharpe ranking prefer low-volatility strategies -> level in Sharpe units
    ("acct_ew", "T2s (ydms)", {"prep": "ydms"}), ("past_ret_vw", "T2s (ydms)", {"prep": "ydms"}), ("acct_vw", "T2s (ydms)", {"prep": "ydms"}),
    ("past_ret_ew", "T2s (ydms)", {"prep": "ydms"}),
    # BatchNorm finding: per-family-year normalisation helps ranking but is inconsistent between training and prediction -> make it consistent
    ("acct_ew", "T2+setbn+xset3072+detev", {"hnorm": "setbn", "xset": 3072, "detev": True}), ("acct_ew", "T2+setbn", {"hnorm": "setbn"}),
    ("past_ret_vw", "T2+setbn+xset3072+detev", {"hnorm": "setbn", "xset": 3072, "detev": True}),
    # accounting EW: root causes from trace_blocks (FinCast tokens mostly nuisance; output < t-stat), seed drift along the warm-start chain, regime
    ("acct_ew", "T2+hpca16", {"hpca": 16}), ("acct_ew", "T2+hpca64", {"hpca": 64}),
    ("acct_ew", "T2+warm_sp", {"warm": "sp"}), ("acct_ew", "T2+warm_cold_ep5", {"warm": "cold", "epochs": 5}),
    ("acct_ew", "T2+regime", {"regime": True}), ("acct_ew", "T2+shuf+cos", {"shuf": True, "sched": "cos"}),
    ("acct_ew", "T2+detev", {"detev": True}), ("acct_ew", "T2+xset3072", {"xset": 3072}),
    # accounting VW: same base and the most promising single changes
    ("acct_vw", "T2", {}), ("acct_vw", "T2+ln", {"hnorm": "ln"}), ("acct_vw", "T2+regime", {"regime": True}),
    ("acct_vw", "T2+hpca16", {"hpca": 16}), ("acct_vw", "T2+warm_sp", {"warm": "sp"}),
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
subprocess.run([sys.executable, "run_ticker.py", "T2"], stdout=open(h.CACHE / "run_ticker_T2.log", "w"), stderr=subprocess.STDOUT)
print("QUEUE DONE", flush=True)
