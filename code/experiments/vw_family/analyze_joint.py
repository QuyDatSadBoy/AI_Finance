"""Kết quả huấn luyện chung EW+VW: Sharpe top 1/5/10% trên từng họ, chọn seed bằng dev muộn 1995-2004 của họ VW.  python analyze_joint.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
CODE = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(CODE)); sys.path.insert(0, str(Path(__file__).parent))
import data, evaluate as ev
from run_vw import V, load_vw
from joint import J, tables
T = tables(); idE, wE = data.load_returns(); idV, wV = load_vw()
FAM = {0: ("EW", idE, wE), 1: ("VW", idV, wV)}
PER = {"dev": (1983, 2004), "holdout": (2005, 2020), "full": (1983, 2020)}
sh = lambda R, lo, hi: ev.perf(R, lo, hi)["sharpe"]
files = sorted(J.glob("joint_s*.npy")); seeds = [int(f.stem.split("_s")[1]) for f in files]
out = ["# Huấn luyện chung EW + VW (cùng kiến trúc, cùng siêu tham số)", ""]
res = {}
for fam, (nm, ids, wide) in FAM.items():
    Tf = T[T.fam == fam].reset_index(drop=True); m = (T.fam == fam).values
    P = {s: ev.top_portfolios(Tf, ids, wide, ev.signal_from_oriented(Tf, np.load(f)[m])) for s, f in zip(seeds, files)}
    Peb = ev.top_portfolios(Tf, ids, wide, ev.signal_from_eb(Tf)); Pt = ev.top_portfolios(Tf, ids, wide, ev.signal_abs_t(Tf))
    res[nm] = (Tf, P, Peb, Pt)
rows = []
for s in seeds:
    r = dict(seed=s)
    for nm in ("VW", "EW"):
        P = res[nm][1][s]
        r[f"{nm}_late"] = np.mean([sh(P[k], 1995, 2004) for k in (1, 5, 10)])
        for pn in ("dev", "holdout"):
            for k in (1, 5, 10): r[f"{nm}_{pn}{k}"] = sh(P[k], *PER[pn])
    rows.append(r)
S = pd.DataFrame(rows).round(3); print(S.to_string(index=False)); out += ["Sharpe mỗi seed:", "", S.to_string(index=False), ""]
best = int(S.loc[S.VW_late.idxmax(), "seed"]); print("seed chọn (VW dev muộn):", best); out.append(f"Seed chọn bằng dev muộn 1995-2004 của họ VW: {best}\n")
for nm in ("VW", "EW"):
    Tf, P, Peb, Pt = res[nm]; rows2 = []
    for k in (1, 5, 10):
        for pn, (lo, hi) in PER.items():
            for lab, R in [(f"Chung (seed {best})", P[best]), ("EB", Peb), ("|t|", Pt)]:
                d = R[k][(R[k].year >= lo) & (R[k].year <= hi)].ret.values
                r = dict(top=k, kỳ=pn, pp=lab, lãi_năm=d.mean() * 12, t=d.mean() / d.std(ddof=1) * np.sqrt(len(d)), sharpe=sh(R[k], lo, hi))
                if lab != "EB": r["p_vs_EB"] = ev.sharpe_test_hac(R[k], Peb[k], lo, hi)
                rows2.append(r)
    F = pd.DataFrame(rows2).round(3); print(nm); print(F.to_string(index=False)); out += [f"Họ {nm}:", "", F.to_string(index=False), ""]
    m = (T.fam == {"VW": 1, "EW": 0}[nm]).values; sg = Tf.sgn.values
    icm = ev.ic_years(Tf, np.load(J / f"joint_s{best}.npy")[m]); ice = ev.ic_years(Tf, Tf.eb.values * sg)
    for pn, (lo, hi) in PER.items():
        line = f"IC {nm} {pn}: chung {icm.loc[lo:hi].mean():.3f}, EB {ice.loc[lo:hi].mean():.3f}, p={ev.ic_test(icm, ice, lo, hi)['p']:.3f}"; print(line); out.append(line + "  ")
    out.append("")
    mean = {pn: [np.mean([sh(P[s][k], *PER[pn]) for s in seeds]) for k in (1, 5, 10)] for pn in ("dev", "holdout")}
    out.append(f"TB seed {nm}: " + "; ".join(f"{pn} " + "/".join(f"{x:.2f}" for x in v) for pn, v in mean.items())); out.append("")
(Path(__file__).parent / "result_joint.md").write_text("\n".join(out))
