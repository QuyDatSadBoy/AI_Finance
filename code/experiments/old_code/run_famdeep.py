"""Deep block-by-block tuning on ONE family subset, next to run_fam1.py (light data, own results file).
Step 1 (root cause from trace_blocks: the output carries LESS information than the t-stat input): Wide & Deep path, statistical features F2."""
import sys, traceback, pandas as pd, ht_common as h, ht_base as B
from ht_sub import make_sub
fams = sys.argv[1] if len(sys.argv) > 1 else "acct_ew"
B.RES2 = h.CACHE / f"famdeep_{fams}.tsv"
D = make_sub(fams); P = pd.read_parquet(h.CACHE / "panel_eb.parquet"); print("data", fams, tuple(D.H.shape), flush=True)
T = {**B.BASE, "xs": True, "prep": "ydm", "pool": "mha", "fams": fams}
VAR = {"T": {}, "wide": dict(wide=True), "F2": dict(extras="F2"), "wide+F2": dict(wide=True, extras="F2"), "goc+wide": dict(xs=False, prep="none", pool="attn", wide=True)}
for nm, v in VAR.items():
    try: B.walk_forward(D, {**T, **v}, years=range(1990, 1992)); print("smoke ok", nm, flush=True)
    except Exception: print("SMOKE FAIL", nm); traceback.print_exc(); raise
def seeds(cfg, n=3):
    out = []
    for sd in range(n):
        c = {**cfg, "seed": sd}
        try: B.pred_of(c)
        except FileNotFoundError: B.run_sub(c, D, P)
        out.append(c)
    return out
for nm, v in VAR.items():
    try: B.ensemble_sub(D, P, seeds({**T, **v}), f"deep {nm} on {fams} x3", agg="median")
    except Exception: print("RUN FAIL", nm); traceback.print_exc()
