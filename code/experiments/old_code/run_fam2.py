"""Stage S2 (one family at a time, light data, own results file): the BatchNorm fix (hnorm=ln, root cause: train BN stats of ONE family-year vs
running stats at prediction) on top of T2 = xs + ydm + mha + F2, then each remaining change one at a time on top of T2+ln.  3 paired seeds, median.
Usage: python run_fam2.py acct_ew"""
import sys, traceback, pandas as pd, ht_common as h, ht_base as B
from ht_sub import make_sub
fams = sys.argv[1]
B.RES2 = h.CACHE / f"fam2_{fams}.tsv"
D = make_sub(fams); P = pd.read_parquet(h.CACHE / "panel_eb.parquet"); print("data", fams, tuple(D.H.shape), flush=True)
T2 = {**B.BASE, "xs": True, "prep": "ydm", "pool": "mha", "extras": "F2", "fams": fams}
L = {"hnorm": "ln"}
VAR = {"T2": {}, "T2+ln": L, "T2+ln+shuf+cos": {**L, "shuf": True, "sched": "cos"}, "T2+ln+detev": {**L, "detev": True},
       "T2+ln+xset3072": {**L, "xset": 3072}, "T2+ln+hnone": {"hnorm": "none"}, "T2+ln+wls2": {**L, "wls": 2.0}}
for nm, v in VAR.items():
    try: B.walk_forward(D, {**T2, **v}, years=range(1990, 1992)); print("smoke ok", nm, flush=True)
    except Exception: print("SMOKE FAIL", nm, flush=True); traceback.print_exc(); VAR[nm] = None
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
    try: B.ensemble_sub(D, P, seeds({**T2, **v}), f"S2 {nm} on {fams} x3", agg="median")
    except Exception: print("RUN FAIL", nm, flush=True); traceback.print_exc()
