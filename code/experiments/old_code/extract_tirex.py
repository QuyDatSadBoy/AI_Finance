"""Run in ~/.local/share/research-kernel/venv_tsfm (TiRex, Auer et al. arXiv 2505.23719).  Embeds the saved windows with TiRex and keeps the
last-layer hidden state of every 32-month patch token -> [N, tokens, 512] float16.  Usage: python extract_tirex.py 128 [cpu|cuda] [acct_ew]   (optional: only the rows of that family -> tirex<ctx>_sub.npy + _sub_idx.npy)"""
import sys, time, numpy as np, torch
from tirex import load_model
ctx = int(sys.argv[1]); dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"
torch.set_num_threads(16)
C = "/home/phongndh/Documents/Paper/code/_cache/"
W = np.load(C + f"win{ctx}.npy", mmap_mode="r"); tag = ""
if len(sys.argv) > 3:                                                  # restrict to one family (row positions among all candidates)
    ix = np.load(C + f"rows_{sys.argv[3]}.npy")                     # positions among all candidate rows (written from the main venv)
    np.save(C + f"tirex{ctx}_sub_idx.npy", ix); W = W[ix]; tag = "_sub"
m = load_model("NX-AI/TiRex", device=dev)
out = None; t0 = time.time(); bs = 2048
for s in range(0, len(W), bs):
    x = torch.tensor(np.asarray(W[s:s + bs]), device=dev)
    with torch.no_grad():
        e = m._embed_context(x)[:, :, -1, :].float()                 # last layer, [b, tokens, 512]
    if out is None:
        out = np.lib.format.open_memmap(C + f"tirex{ctx}{tag}.npy", mode="w+", dtype=np.float16, shape=(len(W), e.shape[1], e.shape[2]))
    out[s:s + len(x)] = e.cpu().numpy().astype(np.float16)
    if s % (bs * 20) == 0: print(s, len(W), f"{time.time() - t0:.0f}s", flush=True)
out.flush(); print("done", out.shape, f"{time.time() - t0:.0f}s")
