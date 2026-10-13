"""Stage S1 of the research plan on one family (light data, own results file, runs next to F1): stability fixes and giving the set block the
statistics, each as ONE change on top of T2 = xs + ydm + mha + F2, 3 paired seeds, median.  Root causes from trace_blocks + block-research workflow."""
import sys, traceback, pandas as pd, ht_common as h, ht_base as B
from ht_sub import make_sub
fams = sys.argv[1] if len(sys.argv) > 1 else "acct_ew"
B.RES2 = h.CACHE / f"famfix_{fams}.tsv"
D = make_sub(fams); P = pd.read_parquet(h.CACHE / "panel_eb.parquet"); print("data", fams, tuple(D.H.shape), flush=True)
T2 = {**B.BASE, "xs": True, "prep": "ydm", "pool": "mha", "extras": "F2", "fams": fams}
A = dict(detev=True, xset=3072)
VAR = {"T2": {}, "A_detev_xset3072": A, "A+ln": {**A, "hnorm": "ln"}, "A+xfuse": {**A, "xfuse": True}, "A+shuf+cos": {**A, "shuf": True, "sched": "cos"},
       "A+wls0.5": {**A, "wls": 0.5}, "A+wls2": {**A, "wls": 2.0}}
for nm, v in VAR.items():
    try: B.walk_forward(D, {**T2, **v}, years=range(1990, 1992)); print("smoke ok", nm, flush=True)
    except Exception: print("SMOKE FAIL", nm); traceback.print_exc(); VAR[nm] = None
def seeds(cfg, n=3):
    out = []
    for sd in range(n):
        c = {**cfg, "seed": sd}
        try: B.pred_of(c)
        except FileNotFoundError: B.run_sub(c, D, P)
        out.append(c)
    return out
for nm, v in VAR.items():
    if v is None: continue
    try: B.ensemble_sub(D, P, seeds({**T2, **v}), f"fix {nm} on {fams} x3", agg="median")
    except Exception: print("RUN FAIL", nm); traceback.print_exc()
