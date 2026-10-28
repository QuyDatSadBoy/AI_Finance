"""Họ VW: trung bình dự báo của 6 seed (không chọn seed, không dùng holdout để chọn gì).  python ensemble_vw.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
CODE = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(CODE)); sys.path.insert(0, str(Path(__file__).parent))
import data, evaluate as ev
from run_vw import V, load_vw
ids, wide = load_vw(); T = pd.read_parquet(V / "table_acct_vw.parquet")
pe = np.mean([np.load(f) for f in sorted(V.glob("vw_s*.npy"))], 0)
Pm = ev.top_portfolios(T, ids, wide, ev.signal_from_oriented(T, pe)); Peb = ev.top_portfolios(T, ids, wide, ev.signal_from_eb(T)); Pt = ev.top_portfolios(T, ids, wide, ev.signal_abs_t(T))
PER = {"dev": (1983, 2004), "holdout": (2005, 2020), "full": (1983, 2020)}
for k in (1, 5, 10):
    for pn, (lo, hi) in PER.items():
        f = lambda R: ev.perf(R[k], lo, hi)["sharpe"]
        print(f"top{k} {pn}: ensemble {f(Pm):.2f}  EB {f(Peb):.2f}  |t| {f(Pt):.2f}  p_vs_EB {ev.sharpe_test_hac(Pm[k], Peb[k], lo, hi):.3f}")
sg = T.sgn.values; icm = ev.ic_years(T, pe); ice = ev.ic_years(T, T.eb.values * sg)
for pn, (lo, hi) in PER.items(): print("IC", pn, round(icm.loc[lo:hi].mean(), 3), round(ice.loc[lo:hi].mean(), 3), round(ev.ic_test(icm, ice, lo, hi)["p"], 3))
