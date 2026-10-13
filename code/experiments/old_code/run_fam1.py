"""Direction of 2026-10-03, stage F1: learn only from the statistical families (accounting, past return).  For each family subset:
the reference-notebook model (BASE) and the two best configurations so far, 3 seeds each, median ensembles; evaluated vs EB on the SAME families."""
import itertools, traceback, pandas as pd, ht_common as h, ht_arch as A, ht_base as B
D = A.Data(); P = pd.read_parquet(h.CACHE / "panel_eb.parquet")
S = list(B.STAT_FAMS)
FIRST = ["acct_ew", "acct_vw", "past_ret_ew", "past_ret_vw", "+".join(S), "acct_ew+acct_vw", "past_ret_ew+past_ret_vw", "acct_ew+past_ret_ew"]
REST = ["+".join(c) for k in (2, 3) for c in itertools.combinations(S, k) if "+".join(c) not in FIRST]
CFGS = {"goc": dict(B.BASE), "xs+ydm+mha": {**B.BASE, "xs": True, "prep": "ydm", "pool": "mha"}, "xs+mha+F2": {**B.BASE, "xs": True, "extras": "F2", "pool": "mha"}}
for nm, c in CFGS.items():                                     # smoke test on two years
    try: B.walk_forward(D, {**c, "fams": "acct_ew"}, years=range(1990, 1992)); print("smoke ok", nm, flush=True)
    except Exception: print("SMOKE FAIL", nm); traceback.print_exc(); raise
def seeds(cfg, n=3):
    out = []
    for sd in range(n):
        c = {**cfg, "seed": sd}
        try: B.pred_of(c)
        except FileNotFoundError: B.run_sub(c, D, P)
        out.append(c)
    return out
for fams in FIRST + REST:
    for nm, c in CFGS.items():
        try: B.ensemble_sub(D, P, seeds({**c, "fams": fams}), f"F1 {nm} on {fams} x3", agg="median")
        except Exception: print("RUN FAIL", fams, nm); traceback.print_exc()
