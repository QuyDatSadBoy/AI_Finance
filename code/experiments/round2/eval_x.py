"""Bảng: Sharpe TB top 1/5/10% dev / holdout, IC, cho mọi cấu hình đã có (trung bình các seed). Chọn chỉ bằng dev."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import data, evaluate as ev
T = data.load_table(); ids, wide = data.load_returns()
P = Path(__file__).parent / "preds"; FC = data.CACHE / "forecasts"
groups = {"final(6 seed chốt)": sorted(FC.glob("final_s*.npy"))}
for f in sorted(P.glob("*_s*.npy")):
    groups.setdefault(f.name.rsplit("_s", 1)[0], []).append(f)
rows = []
for k, fs in groups.items():
    p = np.mean([np.load(f) for f in fs], 0)
    R = ev.top_portfolios(T, ids, wide, ev.signal_from_oriented(T, p)); ic = ev.ic_years(T, p)
    d = [ev.perf(R[r], *data.DEV)["sharpe"] for r in (1, 5, 10)]; h = [ev.perf(R[r], *data.HOLD)["sharpe"] for r in (1, 5, 10)]
    rows.append(dict(cfg=k, seeds=len(fs), dev=" / ".join(f"{x:.2f}" for x in d), dev_TB=np.mean(d), hold=" / ".join(f"{x:.2f}" for x in h), hold_TB=np.mean(h),
                     IC_dev=ic.loc[1983:2004].mean(), IC_hold=ic.loc[2005:2020].mean()))
print(pd.DataFrame(rows).sort_values("dev_TB", ascending=False).round(3).to_string(index=False))
