"""Thử nhanh khối đầu ra (FC) của mô hình chốt: dev 1983-2004, mỗi biến thể một số seed.  python heads.py <tên> <seed> ..."""
import sys, time, numpy as np, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import data, train
from model import FinalNet


class SwiGLU(nn.Module):            # Shazeer 2020, "GLU Variants Improve Transformer"
    def __init__(self, d, h, drop):
        super().__init__(); self.a, self.b = nn.Linear(d, h), nn.Linear(d, h); self.n = nn.LayerNorm(h); self.dp = nn.Dropout(drop); self.o = nn.Linear(h, 1)
    def forward(self, x): return self.o(self.dp(self.n(nn.functional.silu(self.a(x)) * self.b(x))))


class GRN(nn.Module):               # Lim et al. 2021, Temporal Fusion Transformers: gated residual network
    def __init__(self, d, h, drop):
        super().__init__(); self.skip = nn.Linear(d, h); self.l1 = nn.Linear(d, h); self.l2 = nn.Linear(h, h); self.g = nn.Linear(h, 2 * h)
        self.dp = nn.Dropout(drop); self.n = nn.LayerNorm(h); self.o = nn.Linear(h, 1)
    def forward(self, x):
        e = self.dp(self.l2(nn.functional.elu(self.l1(x))))
        return self.o(self.n(self.skip(x) + nn.functional.glu(self.g(e), -1)))


def make(kind):
    def f(d_in, n_extra):
        net = FinalNet(d_in, n_extra); d = 256 + n_extra
        if kind == "gelu":   net.head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.BatchNorm1d(64), nn.Dropout(0.3), nn.Linear(64, 1))
        if kind == "ln":     net.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.LayerNorm(64), nn.Dropout(0.3), nn.Linear(64, 1))
        if kind == "drop01": net.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.1), nn.Linear(64, 1))
        if kind == "swiglu": net.head = nn.Sequential(SwiGLU(d, 64, 0.3))
        if kind == "grn":    net.head = nn.Sequential(GRN(d, 64, 0.3))
        return net
    return f


if __name__ == "__main__":
    kind, seeds = sys.argv[1], list(map(int, sys.argv[2:]))
    T, H = data.load_table(), data.load_tokens(); out = Path(__file__).parent / "preds"; out.mkdir(exist_ok=True)
    for s in seeds:
        t0 = time.time()
        p = train.walk_forward(T, H, seed=s, years=range(1983, 2005), verbose=False, net_fn=None if kind == "base" else make(kind))
        np.save(out / f"{kind}_s{s}.npy", p); print(kind, "seed", s, "xong", round(time.time() - t0), "s", flush=True)
