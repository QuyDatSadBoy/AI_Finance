"""Chạy mô hình chốt cho các seed và lưu dự báo:  python run.py 0 1 2 3 4 5   ->  _cache/forecasts/final_s<seed>.npy
Mỗi seed ~7 phút trên GPU (38 năm 1983-2020). Cần _cache/full_acct_pca16.npy (python -c "import data; data.extract_tokens(...)")."""
import sys, numpy as np
import data, train

if __name__ == "__main__":
    T, H = data.load_table(), data.load_tokens()
    out = data.CACHE / "forecasts"; out.mkdir(exist_ok=True)
    for seed in map(int, sys.argv[1:] or [0]):
        np.save(out / f"final_s{seed}.npy", train.walk_forward(T, H, seed=seed, verbose=False, ckpt_dir=data.ROOT / "code" / "checkpoints" / f"final_s{seed}"))
        print("seed", seed, "xong", flush=True)
