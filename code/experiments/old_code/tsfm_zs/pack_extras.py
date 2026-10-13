"""Pack zero-shot forecasts into one extras array aligned with ALL 655,101 candidate rows (NaN where a row was not forecast), so
ht_base can index it with D.idx_full for any family subset.  CPU only.  Usage: python pack_extras.py [acct_ew]
Per model tag (only files that exist): 3 columns, all in units of the strategy's in-sample sd (sd_IS), no EB input:
  <tag>_dev  = (med12 - in-sample mean) / sd   -- what the forecaster adds beyond the mean (the partial-IC signal)
  <tag>_lvl  = med12 / sd                        -- its 12-month Sharpe forecast
  <tag>_unc  = (q90_12m - q10_12m) / sd          -- forecast uncertainty
Writes tsfm_x_<FAM>.npy [655101, k] float32 and tsfm_x_<FAM>.json (column names, source files)."""
import sys, os, json, numpy as np
OUT = os.path.dirname(os.path.abspath(__file__)) + "/"
fam = sys.argv[1] if len(sys.argv) > 1 else "acct_ew"
N_ALL = 655101
M = np.load(OUT + f"meta_{fam}.npz"); idx = M["idx"]; sd = M["sd"]; m_is = M["mean_or"]
TAGS = ["chronos2_{f}_c240_xl", "chronos2_{f}_c240", "tirex_{f}_c240", "timesfm25_{f}_c240", "toto2-313m_{f}_c240"]
cols, names, src = [], [], []
for t in TAGS:
    p = OUT + t.format(f=fam) + ".npz"
    if not os.path.exists(p):
        print("missing", p); continue
    z = np.load(p)
    assert np.array_equal(z["idx"], idx), f"row order of {p} differs from meta_{fam}"
    short = t.split("_")[0] + ("_xl" if t.endswith("_xl") else "")
    for nm, v in ((f"{short}_dev", (z["med12"] - m_is) / sd), (f"{short}_lvl", z["med12"] / sd), (f"{short}_unc", (z["q90_12m"] - z["q10_12m"]) / sd)):
        col = np.full(N_ALL, np.nan, np.float32); col[idx] = v; cols.append(col); names.append(nm)
    src.append(p)
X = np.stack(cols, 1) if cols else np.zeros((N_ALL, 0), np.float32)
np.save(OUT + f"tsfm_x_{fam}.npy", X)
json.dump(dict(cols=names, src=src, fam=fam), open(OUT + f"tsfm_x_{fam}.json", "w"), indent=1)
print(fam, X.shape, names)
