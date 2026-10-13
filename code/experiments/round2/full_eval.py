"""Đủ 38 năm: so các ứng viên với EB, quy tắc |t|, bản chốt cũ (cùng seed 0-2)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import data, evaluate as ev
T = data.load_table(); ids, wide = data.load_returns(); P = Path(__file__).parent / "preds"
S = tuple(int(x) for x in __import__("os").environ.get("SEEDS", "0,1,2").split(","))
port = {"EB": ev.top_portfolios(T, ids, wide, ev.signal_from_eb(T)), "|t|": ev.top_portfolios(T, ids, wide, ev.signal_abs_t(T))}
preds = {"FINAL cũ": np.mean([np.load(data.CACHE / "forecasts" / f"final_s{s}.npy") for s in S], 0)}
for n in sys.argv[1:]:
    preds[n] = np.mean([np.load(P / f"{n}_s{s}.npy") for s in S], 0)
for k, p in preds.items(): port[k] = ev.top_portfolios(T, ids, wide, ev.signal_from_oriented(T, p))
print(ev.sharpe_table(port).to_string())
ie = ev.ic_years(T, T.eb.values * T.sgn.values)
for k, p in preds.items():
    i = ev.ic_years(T, p); print(k, "IC dev/hold", round(i.loc[:2004].mean(), 3), round(i.loc[2005:].mean(), 3), "| p vs EB dev", round(ev.ic_test(i, ie, *data.DEV)["p"], 3), "hold", round(ev.ic_test(i, ie, *data.HOLD)["p"], 3))
    for lo, hi in (data.DEV, data.HOLD):
        c = ev.calib_bins(T, p, lo, hi); print("   độ dốc", lo, hi, round(np.polyfit(c.f, c.y, 1)[0], 2))
for k in sys.argv[1:]:
    print("==", k, "vs EB"); print(ev.compare(port[k], port["EB"], k, "EB").to_string())
