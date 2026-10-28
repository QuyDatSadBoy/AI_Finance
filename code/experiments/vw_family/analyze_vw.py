"""Họ kế toán VW: chọn seed bằng Sharpe TB top 1/5/10% trong 1995-2004 (như họ EW), so với EB và |t|.  python analyze_vw.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
CODE = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(CODE))
import data, evaluate as ev
sys.path.insert(0, str(Path(__file__).parent)); from run_vw import V, load_vw

ids, wide = load_vw(); T = pd.read_parquet(V / "table_acct_vw.parquet")
PER = {"dev": (1983, 2004), "holdout": (2005, 2020), "full": (1983, 2020)}
sh = lambda R, lo, hi: ev.perf(R, lo, hi)["sharpe"]
files = sorted(V.glob("vw_s*.npy")); seeds = [int(f.stem.split("_s")[1]) for f in files]
P = {s: ev.top_portfolios(T, ids, wide, ev.signal_from_oriented(T, np.load(f))) for s, f in zip(seeds, files)}
Peb = ev.top_portfolios(T, ids, wide, ev.signal_from_eb(T)); Pt = ev.top_portfolios(T, ids, wide, ev.signal_abs_t(T))
rows = []
for s in seeds:
    r = dict(seed=s, late9504=np.mean([sh(P[s][k], 1995, 2004) for k in (1, 5, 10)]))
    for pn, (lo, hi) in PER.items():
        for k in (1, 5, 10): r[f"{pn}{k}"] = sh(P[s][k], lo, hi)
    rows.append(r)
S = pd.DataFrame(rows).round(3); print(S.to_string())
best = int(S.loc[S.late9504.idxmax(), "seed"]); print("seed chọn (dev muộn 1995-2004):", best)
out = ["# Họ kế toán value-weighted (mô hình chốt học lại, không chỉnh siêu tham số)", "", "Sharpe top 1/5/10% mỗi seed:", "", S.to_string(index=False), "", f"Seed chọn bằng 1995-2004: **{best}**", ""]
Pb = P[best]; rows2 = []
for k in (1, 5, 10):
    for pn, (lo, hi) in PER.items():
        for nm, R in [("Mô hình (seed %d)" % best, Pb), ("EB", Peb), ("|t|", Pt)]:
            d = R[k][(R[k].year >= lo) & (R[k].year <= hi)].ret.values
            r = dict(top=k, kỳ=pn, pp=nm, lãi_năm=d.mean() * 12, t=d.mean() / d.std(ddof=1) * np.sqrt(len(d)), sharpe=sh(R[k], lo, hi))
            if nm != "EB": r["p_vs_EB"] = ev.sharpe_test_hac(R[k], Peb[k], lo, hi)
            if nm.startswith("Mô"): r["p_vs_|t|"] = ev.sharpe_test_hac(R[k], Pt[k], lo, hi)
            rows2.append(r)
F = pd.DataFrame(rows2).round(3); print(F.to_string()); out += ["Bảng so với EB và |t|:", "", F.to_string(index=False), ""]
sg = T.sgn.values; icm = ev.ic_years(T, np.load(V / f"vw_s{best}.npy")); ice = ev.ic_years(T, T.eb.values * sg)
for pn, (lo, hi) in PER.items():
    line = f"IC {pn}: mô hình {icm.loc[lo:hi].mean():.3f}, EB {ice.loc[lo:hi].mean():.3f}, p={ev.ic_test(icm, ice, lo, hi)['p']:.3f}"; print(line); out.append(line + "  ")
mean6 = {pn: [np.mean([sh(P[s][k], *PER[pn]) for s in seeds]) for k in (1, 5, 10)] for pn in PER}
out += ["", "Trung bình các seed: " + "; ".join(f"{pn} " + "/".join(f"{x:.2f}" for x in v) for pn, v in mean6.items())]
(Path(__file__).parent / "result.md").write_text("\n".join(out)); print(mean6)
