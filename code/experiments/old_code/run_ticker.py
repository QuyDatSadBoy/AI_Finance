"""Ticker inference test: a model that LEARNED only from statistical families (accounting / past return) predicts ticker strategies it never
saw.  Pool the training families + ticker (candidate rows), pick the top 1/5/10% by predicted Sharpe each year, and report the share of each
family in the picks, next to EB doing the same (paper Table 3 Panel A: ticker = 0.0% of EB's top 1%).
Usage: [NSEED=10] python run_ticker.py cfgname"""
import os, sys, numpy as np, pandas as pd, ht_common as h, ht_base as B
from ht_sub import make_sub
cfgname = sys.argv[1]
TICK = "ticker_ew+ticker_vw"; S4 = "acct_ew+acct_vw+past_ret_ew+past_ret_vw"
T = {**B.BASE, "xs": True, "prep": "ydm", "pool": "mha", "hpca": 16, "infer": TICK}          # xs+ydm+mha+hpca16 (E0 extras: no family id)
T2 = {**B.BASE, "xs": True, "prep": "ydm", "pool": "mha", "extras": "F2", "infer": TICK}
_OLD = {"T2": {**T2, "fams": "acct_ew"}, "goc": {**B.BASE, "fams": "acct_ew", "infer": TICK},
       "T+hpca16": {**T, "fams": "acct_ew"},                                   # breadth features (F3) let the model see how much of a family is noise
       "T+hpca16+F3": {**T, "fams": "acct_ew", "extras": "F3"}, "T+hpca16+F3+lb": {**T, "fams": "acct_ew", "extras": "F3", "lvl": "breadth"},
       "4F T+hpca16": {**T, "fams": S4}, "4F T+hpca16+F3+lb": {**T, "fams": S4, "extras": "F3", "lvl": "breadth"},
       # ydms: level in Sharpe units, so a low-volatility family (ticker EW: median sd 1.2 vs 2.3 for accounting EW) is not lifted by level / sd
       "Ts+hpca16": {**T, "fams": "acct_ew", "prep": "ydms"}, "4F Ts+hpca16": {**T, "fams": S4, "prep": "ydms"},
       "4F Ts+hpca16+F3+lb": {**T, "fams": S4, "prep": "ydms", "extras": "F3", "lvl": "breadth"},
       "4F Ts+hpca16+F3+lb+gate": {**T, "fams": S4, "prep": "ydms", "extras": "F3", "lvl": "breadth", "gate": True},
       "4F Ts+hpca16+raw": {**T, "fams": S4, "prep": "ydms", "enc": "raw"},   # non-pretrained: are FinCast tokens of noise series off-distribution?
       # + simulated pure-noise family in training (make_null.py): the network sees what candidates of a noise family look like
       "Ts+hpca16+null": {**T, "fams": "acct_ew+null_acct_ew", "prep": "ydms"},
       "Ts+hpca16+null+F3+lb": {**T, "fams": "acct_ew+null_acct_ew", "prep": "ydms", "extras": "F3", "lvl": "breadth"},
       "T+hpca16+null+F3+lb": {**T, "fams": "acct_ew+null_acct_ew", "extras": "F3", "lvl": "breadth"}}
from run_q6_cfgs import CFGS as _Q                                      # every named acct_ew config, inferred on ticker
CFG = _OLD[cfgname] if cfgname in _OLD else {**_Q[cfgname], "infer": TICK}
NS = int(os.environ.get("NSEED", 3))
B.RES2 = h.CACHE / "ticker_test.tsv"
D = make_sub(CFG["fams"] + "+" + TICK); print("data", tuple(D.H.shape), flush=True)
preds = []
for sd in range(NS):
    c = {**CFG, "seed": sd}
    try: p = B.to_local(D, B.pred_of(c))
    except FileNotFoundError:
        B.use_encoder(D, c.get("enc", "fincast")); p = B.walk_forward(D, c, seed=sd)
        import hashlib; nm = "|".join(f"{k}={c[k]}" for k in sorted(c)); np.save(B.PRED_DIR / (hashlib.md5(nm.encode()).hexdigest() + ".npy"), B.to_full(D, p))
    preds.append(p); print("seed", sd, "done", flush=True)
p = np.median(preds, 0)
df = pd.DataFrame(dict(fam=D.F.family.values, yr=D.yr, sd=D.sd, ai=p, eb=D.eb_or))
df = df[(df.yr >= 1983) & (df.yr <= 2020) & ~df.fam.str.startswith("null")].dropna(subset=["ai"])   # simulated families: training only
for col in ("ai", "eb"):
    df[col + "_score"] = (df[col] / df.sd).abs()                     # predicted Sharpe magnitude (sign is flipped for negative forecasts)
rows = []
for r in (1, 5, 10):
    for col in ("ai", "eb"):
        sh = []
        for y, g in df.groupby("yr"):
            k = max(1, int(round(len(g) * r / 100))); top = g.nlargest(k, col + "_score")
            sh.append(top.fam.value_counts(normalize=True).reindex(sorted(df.fam.unique()), fill_value=0) * 100)
        sh = pd.DataFrame(sh); yrs = np.array(sorted(df.yr.unique()))
        rows.append(dict(model=col, top=f"{r}%", **sh.mean().round(1).to_dict(), ticker_dev=sh.loc[:, sh.columns.str.startswith("ticker")].sum(1)[yrs <= 2004].mean(),
                         ticker_hold=sh.loc[:, sh.columns.str.startswith("ticker")].sum(1)[yrs >= 2005].mean()))
res = pd.DataFrame(rows); pd.set_option("display.width", 220)
print(f"\n[{cfgname} x{NS}] share of each family in the pooled top-k (candidates of {CFG['fams'].replace('+null_acct_ew', '')} + ticker), %:\n", res.round(1).to_string(index=False))
print("\nmean abs predicted Sharpe by family:\n", df.groupby("fam")[["ai_score", "eb_score"]].mean().round(4).to_string())
print("share of ticker rows in the pool: %.1f%%" % (100 * df.fam.str.startswith("ticker").mean()))
res.to_csv(h.CACHE / f"ticker_test_{cfgname.replace(' ', '_')}_n{NS}.csv", index=False)
