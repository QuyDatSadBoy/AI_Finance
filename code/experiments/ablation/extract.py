"""Tạo token cho ablation: (a) FinCast pretrained + PCA 64 (cắt lấy 8/16/32/64 chiều), (b) FinCast KHỞI TẠO NGẪU NHIÊN + PCA 16.
python extract.py pca64 | rand16"""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import data


def load_random_fincast(seed=0, device="cuda"):
    """Cùng kiến trúc FinCast, trọng số khởi tạo ngẫu nhiên (không pretrained)."""
    import importlib.util
    sys.path.insert(0, str(data.FINCAST_SRC))
    spec = importlib.util.spec_from_file_location("ppd", data.FINCAST_SRC / "ffm" / "pytorch_patched_decoder_MOE.py")
    ppd = importlib.util.module_from_spec(spec); sys.modules["ppd"] = ppd; spec.loader.exec_module(ppd)
    cfg = ppd.FFMConfig(num_layers=50, num_heads=16, hidden_size=1280, intermediate_size=1280, patch_len=32, horizon_len=128, head_dim=80,
                        quantiles=ppd.create_quantiles(), use_positional_embedding=False, num_experts=4, gating_top_n=2,
                        threshold_train=0.2, threshold_eval=0.2)
    torch.manual_seed(seed)
    return ppd.PatchedTimeSeriesDecoder_MOE(cfg).to(device=device).eval()


def run(mode):
    T = data.load_table(); ids, wide = data.load_returns()
    m = data.load_fincast() if mode == "pca64" else load_random_fincast()
    k = 64 if mode == "pca64" else 16
    mu, V, ev = data.fit_pca(T, ids, wide, m, k=k)
    print(mode, "varianza giữ lại:", ev, flush=True)
    out = data.CACHE / f"abl_{mode}.npy"
    H = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(len(T), 4, k))
    for s in range(0, len(T), 20000):
        sel = np.arange(s, min(s + 20000, len(T)))
        H[sel] = data.project(data.hidden_of(T, ids, wide, sel, m), mu, V).half().cpu().numpy()
    H.flush(); print("xong", mode, flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
