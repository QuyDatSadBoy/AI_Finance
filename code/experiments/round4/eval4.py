"""Đánh giá vòng 4: Sharpe top 1/5/10% ở dev, dev muộn 1995-2004 (tiêu chí chọn), holdout (chỉ báo cáo)."""
import sys, glob, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parents[1]))
import data, evaluate as ev
T = data.load_table(); ids, wide = data.load_returns(); yr = T.oos_year.values
EB = ev.signal_from_eb(T)

def sr(port, lo, hi):
    return [ev.perf(port[r], lo, hi)["sharpe"] for r in (1, 5, 10)]

def row(name, sig, n=0):
    P = ev.top_portfolios(T, ids, wide, sig)
    d, dl, h = sr(P, 1983, 2004), sr(P, 1995, 2004), sr(P, 2005, 2020)
    f = lambda v: " / ".join(f"{x:.2f}" for x in v)
    return dict(cfg=name, seeds=n, dev=f(d), dev_TB=round(np.mean(d), 3), devlate_TB=round(np.mean(dl), 3), hold=f(h), hold_TB=round(np.mean(h), 3))

def load(n):
    fs = sorted(glob.glob(str(HERE / "preds" / f"{n}_s*.npy"))); return np.nanmean([np.load(f) for f in fs], 0), len(fs)

if __name__ == "__main__":
    rows = [row("EB", EB), row("abs_t", ev.signal_abs_t(T))]
    F = [np.load(f"{HERE.parents[1]}/_cache/forecasts/final_s{s}.npy") for s in range(6)]
    rows.append(row("final6", ev.signal_from_oriented(T, np.nanmean(F, 0)), 6))
    rows.append(row("final_s01", ev.signal_from_oriented(T, np.nanmean(F[:2], 0)), 2))
    for n in sorted({Path(f).name.rsplit("_s", 1)[0] for f in glob.glob(str(HERE / "preds" / "*_s*.npy"))}):
        p, k = load(n); rows.append(row(n, ev.signal_from_oriented(T, p), k))
    print(pd.DataFrame(rows).to_string(index=False))
