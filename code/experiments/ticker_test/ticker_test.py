"""Kiểm tra trên họ nhiễu Ticker (docs/4: "suy luận trên Ticker mà không huấn luyện trên đó").
Mô hình chốt seed 1 (checkpoint code/checkpoints/final_s1, KHÔNG học lại) dự báo cho kế toán EW + Ticker EW gộp mỗi năm.
Mô hình tốt phải gần như không chọn Ticker vào top 1/5/10% (paper: Ticker 0% top 1%).
So với EB (fit trên t-stat gộp hai họ mỗi năm) và quy tắc |t|.   python ticker_test.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
CODE = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(CODE))
import data, evaluate as ev
from model import FinalNet
from train import Static, BASE_YEARS

OUT = Path(__file__).resolve().parent
OFF = 10_000_000          # signalid Ticker = OFF + id gốc (tránh trùng id kế toán)


def ticker_wide(ncol):
    f = data.CACHE / "wide_ticker_ew.npz"
    if not f.exists():
        df = pd.read_csv(data.RAW / "TickerSignalsLongShort.csv", usecols=["signalid", "date", "ret_ew"])
        d = pd.to_datetime(df.date)
        m = (d.dt.year.values - data.BASE_YEAR) * 12 + d.dt.month.values - 1
        ids, inv = np.unique(df.signalid.values, return_inverse=True)
        wide = np.full((len(ids), ncol), np.nan, np.float32)
        k = m < ncol; wide[inv[k], m[k]] = df.ret_ew.values[k]
        np.savez(f, ids=ids + OFF, wide=wide)
    z = np.load(f); return z["ids"], z["wide"]


def main():
    ida, wa = data.load_returns(); idt, wt = ticker_wide(wa.shape[1])
    Ta = data.load_table(); Ta = Ta[Ta.oos_year >= 1983 - BASE_YEARS - 1].reset_index(drop=True)
    ft = data.CACHE / "table_ticker_ew.parquet"
    if not ft.exists():
        data.build_table(idt, wt, years=range(1983, 2021)).to_parquet(ft)
    Tt = pd.read_parquet(ft)
    # token Ticker: FinCast -> cùng PCA (fit lại đúng như cũ trên kế toán <= 1982, rng 0)
    fh = data.CACHE / "ticker_pca16.npy"
    m = None
    if not fh.exists():
        m = data.load_fincast(); TA = data.load_table()
        mu, V, _ = data.fit_pca(TA, ida, wa, m)
        Hc = np.load(data.CACHE / "full_acct_pca16.npy", mmap_mode="r")
        chk = np.arange(0, len(TA), 50000)[:20]
        diff = np.abs(data.project(data.hidden_of(TA, ida, wa, chk, m), mu, V).cpu().numpy() - Hc[chk].astype(np.float32)).max()
        print("kiểm PCA dựng lại vs cache, sai lệch max:", diff, flush=True)
        Ht = np.zeros((len(Tt), 4, 16), np.float16)
        for s in range(0, len(Tt), 20000):
            sel = np.arange(s, min(s + 20000, len(Tt)))
            Ht[sel] = data.project(data.hidden_of(Tt, idt, wt, sel, m), mu, V).half().cpu().numpy()
        np.save(fh, Ht); del m; torch.cuda.empty_cache()
    Ht = np.load(fh)
    # bảng gộp
    Ta["fam"], Tt["fam"] = "acct", "ticker"
    full_acct = data.load_table(); Ha_all = np.load(data.CACHE / "full_acct_pca16.npy", mmap_mode="r")
    keep = (full_acct.oos_year >= 1983 - BASE_YEARS - 1).values
    Ha = np.asarray(Ha_all[np.where(keep)[0]])
    T = pd.concat([Ta, Tt], ignore_index=True); H = np.concatenate([Ha, Ht])
    # EB gộp hai họ mỗi năm (paper gộp các họ khi xếp hạng)
    T["eb_pool"] = np.nan
    for Y in range(1983, 2021):
        i = np.where(T.oos_year.values == Y)[0]; t = T.tstat.values[i]
        T.loc[i, "eb_pool"] = data.eb_posterior_mean(data.fit_eb(t), t) * T.mean_is.values[i] / t
    S = Static(T)        # zt = |t| chuẩn hoá trong năm GỘP
    Hg = torch.tensor(H, device="cuda")
    pred = np.full(len(T), np.nan, np.float32)
    net = FinalNet(16, 2).cuda().eval()
    for Y in range(1983, 2021):
        c = torch.load(CODE / "checkpoints" / "final_s1" / f"Y{Y}.pt", weights_only=False, map_location="cuda")
        net.load_state_dict(c["net"])
        fb = np.where((S.yr <= Y - 1) & (S.yr >= Y - BASE_YEARS) & np.isfinite(S.ytar) & (T.fam.values == "acct"))[0]
        sb = S.zt[fb].std() + 1e-9          # gần 1 (zt chuẩn hoá trong năm)
        te = np.where(S.yr == Y)[0]
        E = np.nan_to_num(np.clip((S.E[te] - c["e_mu"]) / c["e_sd"], -10, 10)).astype(np.float32)
        with torch.no_grad():
            o = net(Hg[te].float(), torch.tensor(E, device="cuda")); o = (o - o.mean()).cpu().numpy()
        pred[te] = (np.clip(o, c["lo"], c["hi"]) + c["level"] + S.zt[te] / sb * c["b"][0]) * S.sd[te]
    T["pred"] = pred
    T[["signalid", "oos_year", "fam", "pred", "eb_pool"]].to_parquet(OUT / "preds.parquet")

    # ---- đánh giá
    E_ = T[T.oos_year >= 1983].reset_index(drop=True); p = E_.pred.values
    ids = np.concatenate([ida, idt]); wide = np.concatenate([wa, wt])
    sigs = {"Mô hình (seed 1)": ev.signal_from_oriented(E_, p),
            "EB (gộp)": (lambda z: (z, np.where(z >= 0, 1., -1.), np.abs(z)))(E_.eb_pool.values / E_.sd_is.values),
            "Quy tắc |t|": ev.signal_abs_t(E_)}
    rows, port = [], {}
    tick = (E_.fam == "ticker").values; yr = E_.oos_year.values
    for nm, sg in sigs.items():
        pct = pd.Series(sg[2]).groupby(yr).rank(pct=True).values * 100
        r = {"phương pháp": nm}
        for k in (1, 5, 10):
            sel = pct >= 100 - k
            r[f"% Ticker top {k}%"] = round(100 * tick[sel].mean(), 2)
        r["Sharpe dự báo TB: Ticker / kế toán"] = f"{np.nanmean(sg[2][tick]):.3f} / {np.nanmean(sg[2][~tick]):.3f}"
        rows.append(r)
        port[nm + " (gộp)"] = ev.top_portfolios(E_, ids, wide, sg)
    share = pd.DataFrame(rows)
    # mô hình chỉ trên kế toán (đã báo cáo) để thấy gộp Ticker có làm hỏng không
    A = E_[~tick].reset_index(drop=True)
    port["Mô hình, chỉ kế toán"] = ev.top_portfolios(A, ida, wa, ev.signal_from_oriented(A, A.pred.values))
    sh = ev.sharpe_table(port)
    # IC trên riêng Ticker: dự báo có đoán được lãi Ticker không?
    Tk = E_[tick].reset_index(drop=True)
    ic = pd.DataFrame({"Mô hình": ev.ic_years(Tk, Tk.pred.values), "EB": ev.ic_years(Tk, Tk.eb_pool.values * Tk.sgn.values)})
    # lãi thật của các Ticker mô hình chọn vào top 1%
    pct = pd.Series(sigs["Mô hình (seed 1)"][2]).groupby(yr).rank(pct=True).values * 100
    picked = tick & (pct >= 99)
    with open(OUT / "result.md", "w") as f:
        f.write("## Tỉ lệ Ticker trong danh mục (1983-2020, gộp kế toán EW + Ticker EW)\n\n" + share.to_string(index=False) + "\n\n")
        f.write(f"Ticker chiếm {100 * tick.mean():.1f}% số dòng gộp.\n\n")
        f.write("## Sharpe danh mục\n\n" + sh.to_string(index=False) + "\n\n")
        f.write("## IC trên riêng Ticker (TB theo năm)\n\n" + ic.mean().round(4).to_frame("IC").to_string() + "\n\n")
        f.write(f"Ticker được mô hình chọn vào top 1%: {picked.sum()} dòng; lãi thật TB (nhân dấu) {np.nanmean(E_.y.values[picked]) if picked.any() else float('nan'):.3f} %/tháng\n")
    print(open(OUT / "result.md").read())


if __name__ == "__main__":
    main()
