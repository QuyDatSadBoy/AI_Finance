"""Seed-paired comparison of two configurations on one family, from saved predictions (CPU): per seed, dev MSE vs EB and the dev candidate
Sharpe of the top 1/5/10%, then the mean paired difference (variant - base), its standard error and t over the common seeds.
Usage: python paired_sub.py acct_ew BASE_NAME VARIANT_NAME   (names from run_q6_cfgs.CFGS; 'goc' = the reference notebook's model)"""
import sys, numpy as np, pandas as pd, ht_base as B, ht_common as h
from ht_sub import make_sub
from run_q6_cfgs import CFGS

fam, base, var = sys.argv[1:4]
CFG = {**CFGS, "goc": {**B.BASE, "fams": fam}}
D = make_sub(fam, with_X=False, with_H=False); P = pd.read_parquet(h.CACHE / "panel_eb.parquet")


def per_seed(name):
    out = {}
    for s in range(20):
        try: p = B.to_local(D, B.pred_of({**CFG[name], "seed": s}))
        except FileNotFoundError: continue
        r = B.eval_subset(D, P, p, fam)
        out[s] = dict(mse=r["mse_vs_eb_dev"], sr1=r["cand_top1_dev_sr"], sr5=r["cand_top5_dev_sr"], sr10=r["cand_top10_dev_sr"])
    return pd.DataFrame(out).T


a, b = per_seed(base), per_seed(var)
s = sorted(set(a.index) & set(b.index))
print(f"{var} - {base} on {fam}, paired seeds {s}")
for c in a.columns:
    d = b.loc[s, c] - a.loc[s, c]; se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"  {c:5s} base {a.loc[s, c].mean():+.3f}  variant {b.loc[s, c].mean():+.3f}  diff {d.mean():+.3f} (se {se:.3f}, t {d.mean() / se:+.2f})")
