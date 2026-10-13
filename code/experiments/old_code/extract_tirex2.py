"""TiRex-2 (NX-AI, 2026) hidden states for the candidate rows of one family -> [N, 4, D] float16 (the normalised output of the last
recurrent block, averaged over 4 equal segments of the context tokens).  Run in ~/.local/share/research-kernel/venv_tsfm with a CUDA toolkit
for the flashrnn kernels:
  export CUDA_HOME=~/.local/share/research-kernel/cuda-12.8 PATH=~/.local/share/research-kernel/venv_tsfm/bin:$CUDA_HOME/bin:$PATH
  export NVCC_APPEND_FLAGS="-ccbin ~/.local/share/research-kernel/mamba/gcc14/bin/x86_64-conda-linux-gnu-g++" (host GCC 14; system GCC 16 is
  too new for CUDA 12.8)
Usage: python extract_tirex2.py 240 acct_ew [nrows]   -> _cache/tirex2_<ctx>_sub.npy (+ tirex2_<ctx>_sub_idx.npy)"""
import sys, time, numpy as np, torch
from tirex2 import TimeseriesType, load_model

ctx, fam = int(sys.argv[1]), sys.argv[2]
nmax = int(sys.argv[3]) if len(sys.argv) > 3 else None
C = "/home/phongndh/Documents/Paper/code/_cache/"
ix = np.load(C + f"rows_{fam}.npy")[:nmax]
W = np.asarray(np.load(C + f"win{ctx}.npy", mmap_mode="r")[ix], dtype=np.float32)
m = load_model("NX-AI/TiRex-2", device="cuda"); net = m.model if hasattr(m, "model") else m
cap = []
hook = net.stack_out_norm.register_forward_hook(lambda mod, i, o: cap.append(o.detach()))
out = None; t0 = time.time(); bs = 512
for s in range(0, len(W), bs):
    x = W[s:s + bs]
    ts = []
    for r in x:                                                      # full window, missing months as NaN (TiRex-2 embeds a presence mask)
        ts.append(TimeseriesType(target=torch.tensor(r[None, :]), past_covariates=None, future_covariates=None))
    cap.clear()
    with torch.no_grad():
        net._predict_once(ts, 12)
    H = cap[0].float()                                               # [B, L, D] (first pass; no sign-flip augmentation here)
    L = H.shape[1]; seg = np.array_split(np.arange(L), 4)
    e = torch.stack([H[:, g].mean(1) for g in seg], 1)               # [B, 4, D]
    if out is None:
        out = np.lib.format.open_memmap(C + f"tirex2_{ctx}_sub.npy", mode="w+", dtype=np.float16, shape=(len(W), 4, e.shape[-1]))
        np.save(C + f"tirex2_{ctx}_sub_idx.npy", ix)
    out[s:s + len(x)] = e.cpu().numpy().astype(np.float16)
    if s % (bs * 40) == 0: print(s, len(W), f"{time.time() - t0:.0f}s", tuple(H.shape), flush=True)
out.flush(); hook.remove(); print("done", out.shape, f"{time.time() - t0:.0f}s")
