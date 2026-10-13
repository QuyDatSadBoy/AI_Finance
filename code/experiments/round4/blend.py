"""Hậu xử lý không học lại: trộn hạng của mô hình với hạng |t| (α chọn bằng dev muộn)."""
import numpy as np, pandas as pd
from eval4 import *
F = np.nanmean([np.load(f"{HERE.parents[1]}/_cache/forecasts/final_s{s}.npy") for s in range(6)], 0)
zm = F / T.sd_is.values; at = np.abs(T.tstat.values)
g = lambda v: pd.Series(v).groupby(yr).rank(pct=True).values
rm, rt = g(zm), g(at)
rows = []
for a in [1, .75, .5, .25, 0]:
    sc = a * rm + (1 - a) * rt; zs = np.where(zm >= 0, 1.0, -1.0) * T.sgn.values
    rows.append(row(f"blend_a{a}", (zs, zs, sc), 6))
print(pd.DataFrame(rows).to_string(index=False))
