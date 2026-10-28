"""Tổng hợp ablation: Sharpe top 1/5/10% (dev, holdout) của mỗi biến thể, trung bình (và độ lệch chuẩn) trên các seed có đủ dự báo; so với bản chốt cùng seed."""
import sys, numpy as np, pandas as pd
from pathlib import Path
CODE = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(CODE))
import data, evaluate as ev
T = data.load_table(); ids, wide = data.load_returns()
PER = {"dev": (1983, 2004), "hold": (2005, 2020)}
fd = Path(__file__).parent / "forecasts"
def sharpes(pred):
    P = ev.top_portfolios(T, ids, wide, ev.signal_from_oriented(T, pred))
    return [ev.perf(P[k], *PER[pn])["sharpe"] for pn in ("dev", "hold") for k in (1, 5, 10)]
rows = {}
SEEDS = [0, 1, 2]
rows["final"] = [sharpes(np.load(data.CACHE / "forecasts" / f"final_s{s}.npy")) for s in SEEDS]
for v in ("full", "noisab", "mse", "nowgt", "nobase", "pca8", "pca32", "pca64", "nopre"):
    fs = [fd / f"{v}_s{s}.npy" for s in SEEDS]
    if all(f.exists() for f in fs):
        rows[v] = [sharpes(np.load(f)) for f in fs]
out = []
for v, R in rows.items():
    R = np.array(R); m, sd = R.mean(0), R.std(0, ddof=1)
    out.append(dict(variant=v, n_seed=len(R), **{f"{pn}{k}": m[i] for i, (pn, k) in enumerate([(a, b) for a in ("dev", "hold") for b in (1, 5, 10)])},
                    dev_mean=m[:3].mean(), hold_mean=m[3:].mean(), dev_sd=sd[:3].mean(), hold_sd=sd[3:].mean()))
df = pd.DataFrame(out).round(3); df.to_csv(Path(__file__).parent / "ablation_results.csv", index=False); print(df.to_string())
