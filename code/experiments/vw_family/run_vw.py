"""Họ Accounting VALUE-WEIGHTED (cùng 29.314 tín hiệu, lãi theo vốn hoá): học lại mô hình chốt từ đầu, KHÔNG chỉnh siêu tham số.
Bước 1 bảng + EB đối thủ, bước 2 FinCast -> PCA16 (fit lại trên VW <= 1982), bước 3 walk-forward 6 seed.
Kết quả: _cache/vw/*.  python run_vw.py   (bỏ qua bước đã có)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
CODE = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(CODE))
import data, train

V = data.CACHE / "vw"; V.mkdir(exist_ok=True)

def load_vw():
    f = V / "wide_acct_vw.npz"
    if not f.exists():
        df = pd.read_csv(data.RAW / "DataMinedLongShortReturnsVW.csv", usecols=["signalid", "year", "month", "ret"],
                         dtype={"signalid": "int32", "year": "int16", "month": "int8", "ret": "float32"})
        m = (df.year.values.astype(np.int32) - data.BASE_YEAR) * 12 + (df.month.values.astype(np.int32) - 1)
        ids, inv = np.unique(df.signalid.values, return_inverse=True)
        wide = np.full((len(ids), int(m.max()) + 1), np.nan, dtype=np.float32)
        wide[inv, m] = df.ret.values
        np.savez(f, ids=ids, wide=wide)
    z = np.load(f); return z["ids"], z["wide"]

if __name__ == "__main__":
    ids, wide = load_vw()
    ft = V / "table_acct_vw.parquet"
    if not ft.exists():
        data.build_table(ids, wide).to_parquet(ft)
    T = pd.read_parquet(ft); print("bảng", len(T), flush=True)
    ftok = V / "full_acct_vw_pca16.npy"
    if not ftok.exists():
        data.extract_tokens(T, ids, wide, out=ftok); print("token xong", flush=True)
    import torch
    H = torch.from_numpy(np.load(ftok)).to("cuda")
    for seed in range(6):
        f = V / f"vw_s{seed}.npy"
        if f.exists():
            continue
        np.save(f, train.walk_forward(T, H, seed=seed, verbose=False)); print("seed", seed, "xong", flush=True)
