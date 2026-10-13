"""Thêm dữ liệu từ các họ có tính thống kê khác (Accounting VW, Past Return EW/VW) để học, vẫn CHẤM trên Accounting EW.
Mỗi họ: CSV -> ma trận lãi -> bảng (chiến lược, năm) như data.build_table -> FinCast -> CÙNG phép chiếu PCA16 của Accounting EW.
python families.py  -> fam_<họ>_table.parquet, fam_<họ>_pca16.npy, fam_<họ>_wide.npz trong thư mục này."""
import sys, time, numpy as np, pandas as pd, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import data

FAMS = ("acct_vw", "past_ret_ew", "past_ret_vw")


def wide_of(fam):
    f = HERE / f"fam_{fam}_wide.npz"
    if not f.exists():
        if fam == "acct_vw":
            df = pd.read_csv(data.RAW / "DataMinedLongShortReturnsVW.csv", usecols=["signalid", "year", "month", "ret"])
            y, m, r = df.year.values, df.month.values, df.ret.values
        else:
            df = pd.read_csv(data.RAW / "PastReturnSignalsLongShort.csv")
            df = df[(df.nshort >= 2) & (df.nlong >= 2)]                         # cùng bộ lọc với mã của tác giả
            d = pd.to_datetime(df.date); y, m = d.dt.year.values, d.dt.month.values
            r = df["ret_" + fam[-2:]].values
        mi = (y.astype(np.int32) - data.BASE_YEAR) * 12 + (m.astype(np.int32) - 1)
        ids, inv = np.unique(df.signalid.values, return_inverse=True)
        wide = np.full((len(ids), int(mi.max()) + 1), np.nan, np.float32); wide[inv, mi] = r
        np.savez(f, ids=ids, wide=wide)
    z = np.load(f)
    return z["ids"], z["wide"]


if __name__ == "__main__":
    t0 = time.time()
    T0 = data.load_table(); ids0, w0 = data.load_returns()
    m = data.load_fincast()
    mu, V, _ = data.fit_pca(T0, ids0, w0, m)                                 # phép chiếu của Accounting EW (fit <= 1982)
    for fam in FAMS:
        ids, wide = wide_of(fam)
        T = data.build_table(ids, wide); T.to_parquet(HERE / f"fam_{fam}_table.parquet")
        H = np.lib.format.open_memmap(HERE / f"fam_{fam}_pca16.npy", mode="w+", dtype=np.float16, shape=(len(T), 4, 16))
        for s in range(0, len(T), 20000):
            sel = np.arange(s, min(s + 20000, len(T)))
            H[sel] = data.project(data.hidden_of(T, ids, wide, sel, m), mu, V).half().cpu().numpy()
        H.flush(); print(fam, len(T), "dòng", round(time.time() - t0), "s", flush=True)
    print("FAMDONE")
