"""Simulated null family for training (data augmentation, no EB anywhere): every accounting-EW strategy has each of its monthly returns
multiplied by an independent random sign (fixed per strategy-month, so a strategy stays one consistent time series across years).
The result keeps the volatility of real strategies but has zero true mean: a whole family made of noise, like the ticker strategies.
The same pipeline as the real data is applied: 20-year in-sample t-stats, top 10% |t| per year as candidates, family breadth
(share of |t|>2, mean |t|, mean t over all strategies), realised next-year return as the label, FinCast tokens of the last 128 months.
Seeing such a family in training lets the network learn what candidates of a pure-noise family look like (they are selected for a high
|t| exactly like the ticker candidates) and that their next-year return is zero.
Writes _cache/null_acct_ew_F.parquet and _cache/null_acct_ew_h.npy.  Usage: python make_null.py [--cpu-only]"""
import sys, numpy as np, pandas as pd, torch, ht_common as h

SRC, NAME = "acct_ew", "null_acct_ew"
ids, wide = h.load_wide(SRC)
rng = np.random.default_rng(12345)
wnull = wide * rng.choice(np.array([-1.0, 1.0], np.float32), size=wide.shape)
rows = []
for y in range(1966, h.YEAR_MAX + 1):
    st = h.rolling_stats(wnull, y)
    if st is None:
        continue
    ok = st["ok"]; t = st["tstat"][ok]
    r = pd.Series(np.abs(t)).rank(pct=True).values
    d = pd.DataFrame(dict(signalid=ids[ok], oos_year=y, mean_is=st["mean_is"][ok], tstat=t, nobs=st["nobs"][ok], sd_is=st["sd_is"][ok],
                          mean_oos=st["mean_oos"][ok], abs_t_rank=r, fam_frac_t2=np.mean(np.abs(t) > 2), fam_mean_abst=np.mean(np.abs(t)),
                          fam_mean_t=np.mean(t), fam_n=ok.sum()))
    rows.append(d[d.abs_t_rank >= 0.9])
F = pd.concat(rows, ignore_index=True)
F["sgn"] = np.sign(F.mean_is); F["y"] = F.mean_oos * F.sgn; F["family"] = NAME
F["thetahat"] = np.nan; F["pred_ret"] = 0.0                                   # no EB for the simulated family (never evaluated)
F.to_parquet(h.CACHE / f"{NAME}_F.parquet")
print(NAME, len(F), "rows; per year", F.groupby("oos_year").size().median(), "| frac|t|>2", F.fam_frac_t2.mean().round(3),
      "| median |t| of candidates", F.tstat.abs().median().round(2), "| mean y", F.y.mean().round(4), flush=True)
if "--cpu-only" in sys.argv:
    sys.exit()

import ht_fincast as fc
m = fc.load_encoder()
pos = pd.Series(np.arange(len(ids)), index=ids)
H = np.lib.format.open_memmap(h.CACHE / f"{NAME}_h.npy", mode="w+", dtype=np.float16, shape=(len(F), 4, 1280))
for y, idx in F.groupby("oos_year").indices.items():
    r = pos.loc[F.signalid.values[idx]].values
    b = (y - h.BASE_YEAR) * 12; lo = b - fc.CTX
    W = wnull[r, max(lo, 0):b] * F.sgn.values[idx, None]
    if lo < 0:
        W = np.pad(W, ((0, 0), (-lo, 0)), constant_values=np.nan)
    for s in range(0, len(r), 2048):
        H[idx[s:s + 2048]] = fc.hidden(m, torch.tensor(W[s:s + 2048], device="cuda", dtype=torch.float32)).half().cpu().numpy()
H.flush(); print("hidden", H.shape, flush=True)
