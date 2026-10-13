"""Số và hình cho docs/7_SO_SANH_EB.md: đánh giá mô hình chốt (seed 1) đúng các cách của paper Chen & Dim.
Table 2 (lãi, t-stat, Sharpe top 1/5/10%), Figure 1 (lãi cộng dồn), Figure 4-5 (đúng mức, 20 nhóm), Table 3 (Ticker). python make.py"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
CODE = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(CODE))
import data, evaluate as ev

FIG = CODE.parent / "docs" / "figures"; FIG.mkdir(exist_ok=True)
T = data.load_table(); ids, wide = data.load_returns()
pm = np.load(data.CACHE / "forecasts" / "final_s1.npy"); pb = np.mean([np.load(f) for f in sorted((data.CACHE / "forecasts").glob("baseline_s*.npy"))], 0)   # baseline: TB 3 seed như docs
S = {"Mô hình chốt": ev.signal_from_oriented(T, pm), "EB (paper)": ev.signal_from_eb(T),
     "Quy tắc |t|": ev.signal_abs_t(T), "Baseline": ev.signal_from_oriented(T, pb)}
COL = {"Mô hình chốt": "#d62728", "EB (paper)": "#1f77b4", "Quy tắc |t|": "#7f7f7f", "Baseline": "#bcbd22"}
P = {k: ev.top_portfolios(T, ids, wide, s) for k, s in S.items()}
PER = {"dev": (1983, 2004), "holdout": (2005, 2020), "cả kỳ": (1983, 2020)}
out = {}

# ---- Table 2
rows = []
for k in (1, 5, 10):
    for pn, (lo, hi) in PER.items():
        for nm, R in P.items():
            d = R[k][(R[k].year >= lo) & (R[k].year <= hi)].ret.values
            r = dict(top=k, kỳ=pn, pp=nm, lai_nam=d.mean() * 12, t=d.mean() / d.std(ddof=1) * np.sqrt(len(d)), sharpe=d.mean() / d.std(ddof=1) * np.sqrt(12))
            if nm != "EB (paper)":
                r["p_vs_EB"] = ev.sharpe_test_hac(R[k], P["EB (paper)"][k], lo, hi)
            rows.append(r)
tab = pd.DataFrame(rows).round(3); tab.to_csv(Path(__file__).parent / "table2.csv", index=False)
print(tab.to_string())

# ---- Hình 1: Sharpe top 1/5/10%
fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for a, pn in zip(ax, ["dev", "holdout"]):
    sub = tab[tab.kỳ == pn]; w = 0.2
    for j, nm in enumerate(P):
        v = [sub[(sub.pp == nm) & (sub.top == k)].sharpe.item() for k in (1, 5, 10)]
        b = a.bar(np.arange(3) + (j - 1.5) * w, v, w, label=nm, color=COL[nm])
        a.bar_label(b, fmt="%.2f", fontsize=7)
    a.set_xticks(range(3), ["top 1%", "top 5%", "top 10%"]); a.set_title(f"{pn} {PER[pn][0]}–{PER[pn][1]}")
ax[0].set_ylabel("Sharpe (năm)"); ax[1].legend(fontsize=8)
fig.suptitle("Sharpe danh mục top 1/5/10% (như Table 2 của paper)"); fig.tight_layout(); fig.savefig(FIG / "eb_sharpe.png", dpi=130); plt.close()

# ---- Hình 2: lãi cộng dồn top 1% (Figure 1 của paper)
fig, a = plt.subplots(figsize=(9, 4.5))
for nm, R in P.items():
    d = R[1].sort_values(["year", "month"]); x = d.year + (d.month - 0.5) / 12
    a.plot(x, np.cumsum(d.ret.values), label=nm, color=COL[nm], lw=2 if nm in ("Mô hình chốt", "EB (paper)") else 1)
a.axvspan(2005, 2021, color="0.92"); a.text(2005.5, a.get_ylim()[1] * 0.95, "holdout", fontsize=9)
a.set_ylabel("lãi cộng dồn (%)"); a.set_title("Lãi cộng dồn danh mục top 1% (như Figure 1 của paper)"); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "eb_cumret.png", dpi=130); plt.close()

# ---- Hình 3: đúng mức (Figure 4-5 của paper), dấu gốc như paper
sg = T.sgn.values
F = {"Mô hình chốt": pm * sg, "EB (paper)": T.eb.values, "Tin hẳn quá khứ": T.mean_is.values}
C3 = {**COL, "Tin hẳn quá khứ": "#2ca02c"}
Tg = T.assign(y=T.y.values * sg)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
cal = {}
for a, pn in zip(ax, ["dev", "holdout"]):
    lo, hi = PER[pn]
    for nm, f in F.items():
        c = ev.calib_bins(Tg, f, lo, hi); s = np.polyfit(c.f, c.y, 1)[0]; cal[f"{nm} {pn}"] = round(float(s), 2)
        a.plot(c.f, c.y, "o-", ms=3, color=C3[nm], label=f"{nm} (độ dốc {s:.2f})")
    m = max(abs(a.get_xlim()[0]), abs(a.get_xlim()[1])); a.plot([-m, m], [-m, m], "k--", lw=0.8, label="đúng mức (45°)")
    a.set_xlabel("dự báo TB của nhóm (%/tháng)"); a.set_ylabel("lãi thật TB của nhóm (%/tháng)"); a.set_title(f"{pn} {lo}–{hi}"); a.legend(fontsize=7)
fig.suptitle("Dự báo có đúng mức không: 20 nhóm theo dự báo (như Figure 4–5 của paper)"); fig.tight_layout(); fig.savefig(FIG / "eb_calibration.png", dpi=130); plt.close()
out["calib_slope"] = cal

# ---- Hình 4: tỉ lệ Ticker (Table 3 của paper), từ experiments/ticker_test
share = {"Mô hình chốt": [0.19, 3.43, 9.30], "EB (paper)": [0.00, 0.18, 1.25], "Quy tắc |t|": [0.00, 0.52, 2.88]}
fig, a = plt.subplots(figsize=(7, 4)); w = 0.25
for j, (nm, v) in enumerate(share.items()):
    b = a.bar(np.arange(3) + (j - 1) * w, v, w, label=nm, color=COL[nm]); a.bar_label(b, fmt="%.2f", fontsize=7)
a.axhline(43.5, ls="--", color="k", lw=0.8); a.text(0, 40, "chọn bừa: 43,5%", fontsize=8)
a.set_xticks(range(3), ["top 1%", "top 5%", "top 10%"]); a.set_ylabel("% Ticker trong danh mục"); a.set_ylim(0, 48)
a.set_title("Họ nhiễu Ticker lọt vào danh mục bao nhiêu (như Table 3 của paper)"); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "eb_ticker.png", dpi=130); plt.close()

# ---- IC
ic = {nm: ev.ic_years(T, p) for nm, p in [("Mô hình chốt", pm), ("EB (paper)", T.eb.values * sg)]}
for pn, (lo, hi) in PER.items():
    out[f"IC {pn}"] = {nm: round(float(v.loc[lo:hi].mean()), 3) for nm, v in ic.items()}
    out[f"IC {pn} p"] = round(float(ev.ic_test(ic["Mô hình chốt"], ic["EB (paper)"], lo, hi)["p"]), 3)
json.dump(out, open(Path(__file__).parent / "extra.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
