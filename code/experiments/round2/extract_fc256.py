"""FinCast trên 256 tháng (8 đoạn × 32) -> PCA16 (fit <= 1982) -> fc256_pca16.npy [N, 8, 16]."""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import data
T = data.load_table(); ids, wide = data.load_returns(); m = data.load_fincast()
def hid(sel, b=1024):
    out = []
    for s in range(0, len(sel), b):
        W = torch.tensor(data.windows(T.iloc[sel[s:s + b]], ids, wide, ctx=256), device="cuda")
        out.append(data.fincast_hidden(m, W).float())
    return torch.cat(out)
fit = np.where(T.oos_year.values <= 1982)[0]; fit = fit[np.random.default_rng(0).permutation(len(fit))[:50000]]
Z = hid(fit).reshape(-1, 1280); mu = Z.mean(0)
ev, V = torch.linalg.eigh(torch.cov((Z - mu).T)); V = V[:, -16:] / ev[-16:].clamp(min=1e-6).sqrt(); del Z
H = np.lib.format.open_memmap(Path(__file__).parent / "fc256_pca16.npy", mode="w+", dtype=np.float16, shape=(len(T), 8, 16))
for s in range(0, len(T), 20000):
    sel = np.arange(s, min(s + 20000, len(T))); H[sel] = data.project(hid(sel), mu, V).half().cpu().numpy()
H.flush(); print("xong")
