"""Thử nhanh: chỉ dev 1983-2004, so với mô hình chốt cùng seed."""
import sys, glob, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import data, evaluate as ev
T = data.load_table(); ids, wide = data.load_returns(); Q = Path(__file__).parent / "preds_quick"
seeds = sorted({int(f.split("_s")[-1][:-4]) for f in glob.glob(str(Q / "*_s*.npy"))})
groups = {"FINAL (chốt)": [data.CACHE / "forecasts" / f"final_s{s}.npy" for s in seeds]}
for f in sorted(Q.glob("*_s*.npy")): groups.setdefault(f.name.rsplit("_s", 1)[0], []).append(f)
rows = []
for k, fs in groups.items():
    p = np.mean([np.load(f) for f in fs], 0); p[T.oos_year.values > 2004] = np.nan
    R = ev.top_portfolios(T, ids, wide, ev.signal_from_oriented(T, p), years=(1983, 2004)); ic = ev.ic_years(T, p).loc[:2004]
    d = [ev.perf(R[r], *data.DEV)["sharpe"] for r in (1, 5, 10)]
    rows.append(dict(cfg=k, seeds=len(fs), dev=" / ".join(f"{x:.2f}" for x in d), dev_TB=round(np.mean(d), 3), IC_dev=round(ic.mean(), 3)))
print(pd.DataFrame(rows).sort_values("dev_TB", ascending=False).to_string(index=False))
