"""Bảng kết quả của mô hình chốt từ dự báo đã lưu (không huấn luyện lại).  python report.py  ->  results/*.csv

Mô hình chốt = seed 1 (chọn trong 6 seed bằng Sharpe trung bình giai đoạn 1995-2004). Mốc so sánh: EB mining (Chen & Dim),
naive mining (xếp theo |t|), baseline model (trung bình dự báo 3 lần chạy). Mọi phương pháp dùng cùng tập chiến lược,
cùng luật chọn top r% và cùng cách tính lãi danh mục.
"""
from pathlib import Path
import numpy as np, pandas as pd
import data, evaluate as ev

RES = Path(__file__).resolve().parents[1] / "results"
FC = RES / "forecasts"          # dự báo đã lưu của mô hình đề xuất (seed 1) và baseline (3 lần chạy)
FINAL_SEED = 1
PER = (("dev", data.DEV), ("holdout", data.HOLD), ("full", (1983, 2020)))


def tstat(ret, lo, hi):
    d = ret[(ret.year >= lo) & (ret.year <= hi)].ret.values
    return d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))


def main():
    RES.mkdir(exist_ok=True)
    T = data.load_table(); ids, wide = data.load_returns()
    final = np.load(FC / f"final_s{FINAL_SEED}.npy")
    base = np.mean([np.load(f) for f in sorted(FC.glob("baseline_s*.npy"))], 0)
    P = {"Baseline model": ev.top_portfolios(T, ids, wide, ev.signal_from_oriented(T, base)),
         "Naive mining (|t|)": ev.top_portfolios(T, ids, wide, ev.signal_abs_t(T)),
         "EB mining": ev.top_portfolios(T, ids, wide, ev.signal_from_eb(T)),
         "Proposed model": ev.top_portfolios(T, ids, wide, ev.signal_from_oriented(T, final))}

    sh = pd.DataFrame({m: {f"{n} top{k}%": ev.perf(R[k], *p)["sharpe"] for n, p in PER for k in (1, 5, 10)} for m, R in P.items()}).T.round(2)
    sh.to_csv(RES / "sharpe.csv"); print(sh.to_string(), "\n")

    eb, nv, fm = P["EB mining"], P["Naive mining (|t|)"], P["Proposed model"]
    rows = []
    for k in (1, 5, 10):
        for n, (lo, hi) in PER:
            rows.append({"portfolio": f"top {k}%", "period": n,
                         "EB return (%/yr)": ev.perf(eb[k], lo, hi)["mean"], "EB t-stat": tstat(eb[k], lo, hi),
                         "Proposed return (%/yr)": ev.perf(fm[k], lo, hi)["mean"], "Proposed t-stat": tstat(fm[k], lo, hi),
                         "p vs EB (HAC)": ev.sharpe_test_hac(fm[k], eb[k], lo, hi), "p vs naive (HAC)": ev.sharpe_test_hac(fm[k], nv[k], lo, hi)})
    sel = pd.DataFrame(rows).round(3); sel.to_csv(RES / "selection.csv", index=False); print(sel.to_string(), "\n")

    ic_f, ic_e = ev.ic_years(T, final), ev.ic_years(T, T.eb.values * T.sgn.values)
    ic = pd.DataFrame([{"method": "EB mining", **{n: ic_e.loc[lo:hi].mean() for n, (lo, hi) in PER}},
                       {"method": "Proposed model", **{n: ic_f.loc[lo:hi].mean() for n, (lo, hi) in PER}}]).round(3)
    ic.to_csv(RES / "ic.csv", index=False); print(ic.to_string())


if __name__ == "__main__":
    main()
