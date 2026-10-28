"""Ablation leave-one-out trên cấu hình CUỐI. Mỗi biến thể chạy seed 0,1,2 (như bản chốt), lưu dự báo vào forecasts/.
python run_ablation.py <biến_thể> <seed...>
biến thể: noisab | mse | nowgt | nobase | pca8 | pca32 | pca64 | nopre
Mọi thứ khác (đáp án, nền |t|, tối ưu, 3 vòng, validation) giữ đúng như bản chốt. Không sửa mã lõi: dùng monkeypatch trong tiến trình."""
import sys, numpy as np, torch, torch.nn as nn
from pathlib import Path
CODE = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(CODE))
import data, train
from model import FinalNet

OUT = Path(__file__).parent / "forecasts"; OUT.mkdir(exist_ok=True)


def main(variant, seeds):
    T = data.load_table()
    H, net_fn = None, None
    if variant in ("pca8", "pca32", "pca64"):
        k = int(variant[3:]); H = torch.from_numpy(np.load(data.CACHE / "abl_pca64.npy")[:, :, -k:].copy()).to("cuda")
    elif variant == "nopre":
        H = torch.from_numpy(np.load(data.CACHE / "abl_rand16.npy")).to("cuda")
    else:
        H = data.load_tokens()
    if variant == "noisab":
        def net_fn(d, n):
            m = FinalNet(d, n); m.xs = nn.Identity(); return m
    if variant == "mse":
        F = nn.functional; orig = F.huber_loss
        F.huber_loss = lambda a, b, delta=1.0: 0.5 * F.mse_loss(a, b)        # 0,5 u^2 = đúng vùng bậc hai của Huber
    if variant == "nowgt":
        orig_prep = train.prepare_year
        def prep(S, Y, max_train=500_000):
            W = orig_prep(S, Y, max_train)
            if W is not None: W.year_weight = np.ones_like(W.year_weight)
            return W
        train.prepare_year = prep
    if variant == "nobase":                                               # bỏ nền |t|: đáp án = Sharpe - TB năm (cắt 1/99%)
        orig_prep2 = train.prepare_year
        def prep2(S, Y, max_train=500_000):
            W = orig_prep2(S, Y, max_train)
            if W is None: return None
            W.base = np.zeros_like(W.base)
            W.lo, W.hi = np.percentile(S.ytar[W.trn], [1, 99])
            W.target = np.clip(S.ytar, W.lo, W.hi).astype(np.float32)
            return W
        train.prepare_year = prep2
    for seed in seeds:
        p = train.walk_forward(T, H, seed=seed, verbose=False, net_fn=net_fn)
        np.save(OUT / f"{variant}_s{seed}.npy", p); print(variant, "seed", seed, "xong", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], list(map(int, sys.argv[2:])) or [0, 1, 2])
