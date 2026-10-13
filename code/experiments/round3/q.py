"""Thử nhanh từng khối của mô hình chốt (dev 1983-2004).  python q.py <tên> <seed> ...
Mỗi tên = đúng MỘT thay đổi so với mô hình chốt (code/model.py, code/train.py)."""
import sys, time, numpy as np, torch, torch.nn as nn
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
import data, train, model
from heads import make as head_make


class SeqTransformer(nn.Module):        # Transformer encoder (Vaswani et al., NeurIPS 2017) thay Bi-LSTM, cùng đầu ra [n, T, 256]
    def __init__(self, d_in, hid=128):
        super().__init__(); self.p = nn.Linear(d_in, 2 * hid); self.pos = nn.Parameter(torch.zeros(1, 4, 2 * hid))
        self.e = nn.TransformerEncoder(nn.TransformerEncoderLayer(2 * hid, 4, 4 * hid, 0.1, batch_first=True, norm_first=True), 2)
    def forward(self, x): return self.e(self.p(x) + self.pos[:, :x.shape[1]]), None


class PoolSimple(nn.Module):            # gộp 4 đoạn: chỉ attention pooling (như baseline) hoặc trung bình
    def __init__(self, d, kind):
        super().__init__(); self.kind = kind; self.proj, self.v = nn.Linear(d, d), nn.Linear(d, 1, bias=False)
    def forward(self, x, return_weights=False):
        a = torch.full(x.shape[:2], 1 / x.shape[1], device=x.device) if self.kind == "mean" else torch.softmax(self.v(torch.tanh(self.proj(x))).squeeze(-1), 1)
        o = (a.unsqueeze(-1) * x).sum(1); return (o, a) if return_weights else o


CFG = {
    "base": {}, "gelu": {"head": "gelu"}, "ln": {"head": "ln"}, "swiglu": {"head": "swiglu"}, "grn": {"head": "grn"}, "drop01": {"head": "drop01"},
    "lr3e-4": {"lr": 3e-4}, "lr3e-3": {"lr": 3e-3}, "wd1e-3": {"wd": 1e-3}, "wd1e-1": {"wd": 1e-1},
    "b1024": {"sub": 1024}, "b2048": {"sub": 2048}, "b8192": {"sub": 8192},
    "pca8": {"pca": 8}, "pca32": {"pca": 32}, "pca64": {"pca": 64}, "noln": {"noln": True},
    "gru": {"seq": "gru"}, "transformer": {"seq": "tf"}, "pool_attn": {"pool": "attn"}, "pool_mean": {"pool": "mean"},
    "win300k": {"max_train": 300_000}, "winall": {"max_train": 2_000_000},
}
CFG["sw_noln_pca8"] = {"head": "swiglu", "noln": True, "pca": 8}
for k in ["pca8", "pca32", "pca64", "pool_mean", "lr3e-4", "lr3e-3", "wd1e-3", "wd1e-1", "b2048", "b8192", "gru", "pool_attn", "win300k", "winall", "noln", "transformer"]:
    CFG["sw_" + k] = {**CFG[k], "head": "swiglu"}


def net_fn(c):
    def f(d_in, n_extra):
        net = head_make(c["head"])(d_in, n_extra) if c.get("head") else model.FinalNet(d_in, n_extra)
        if c.get("noln"): net.norm = nn.Identity()
        if c.get("seq") == "gru": net.seq = nn.GRU(d_in, 128, batch_first=True, bidirectional=True)
        if c.get("seq") == "tf": net.seq = SeqTransformer(d_in)
        if c.get("pool"): net.pool = PoolSimple(256, c["pool"])
        return net
    return f


if __name__ == "__main__":
    name, seeds = sys.argv[1], list(map(int, sys.argv[2:])); c = CFG[name]
    T = data.load_table(); H = data.load_tokens()
    if c.get("pca"):
        H = torch.from_numpy(np.ascontiguousarray(np.load(HERE / "fc_pca64.npy")[:, :, -c["pca"]:])).to("cuda")   # k thành phần lớn nhất
    train.LR, train.WD, train.SUB = c.get("lr", 1e-3), c.get("wd", 1e-2), c.get("sub", 4096)
    out = HERE / "preds"; out.mkdir(exist_ok=True)
    for s in seeds:
        t0 = time.time()
        p = train.walk_forward(T, H, seed=s, years=range(1983, 2005), verbose=False, max_train=c.get("max_train", 500_000), net_fn=net_fn(c))
        np.save(out / f"{name}_s{s}.npy", p); print(name, "seed", s, "xong", round(time.time() - t0), "s", flush=True)
