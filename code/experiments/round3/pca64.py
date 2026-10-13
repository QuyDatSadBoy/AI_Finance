"""FinCast 128 tháng -> PCA 64 thành phần (fit <= 1982) -> fc_pca64.npy [N, 4, 64]; cột cuối = thành phần lớn nhất."""
import sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parents[1]))
import data
T = data.load_table(); ids, wide = data.load_returns(); m = data.load_fincast()
mu, V, sh = data.fit_pca(T, ids, wide, m, k=64); print("giữ", sh)
H = np.lib.format.open_memmap(HERE / "fc_pca64.npy", mode="w+", dtype=np.float16, shape=(len(T), 4, 64))
for s in range(0, len(T), 20000):
    sel = np.arange(s, min(s + 20000, len(T))); H[sel] = data.project(data.hidden_of(T, ids, wide, sel, m), mu, V).half().cpu().numpy()
H.flush(); print("xong")
