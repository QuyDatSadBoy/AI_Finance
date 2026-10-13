"""Zero-shot 12-month forecasts from TiRex (NX-AI/TiRex, 35M xLSTM, NX-AI community licence; Auer et al. NeurIPS 2025,
arXiv 2505.23719). PRETRAINED.  Run in ~/.local/share/research-kernel/venv_tsfm (tirex-ts 1.4.2).
TiRex masks NaN natively; its 'mean' output is the median; 9 quantiles 0.1..0.9.
Usage: python fc_tirex.py FAM CTX [--rows N] [--cum] [--bs B]
Output: <OUT>/tirex_<FAM>_c<CTX>[_cum][_smokeN].npz  (see tsfm_common)."""
import os, sys, time, argparse, numpy as np, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import tsfm_common as C
from tirex import load_model

ap = argparse.ArgumentParser(); ap.add_argument("fam"); ap.add_argument("ctx", type=int)
ap.add_argument("--rows", type=int); ap.add_argument("--cum", action="store_true"); ap.add_argument("--bs", type=int, default=2048)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.set_num_threads(int(os.environ.get("NTHREADS", 16)))
m = load_model("NX-AI/TiRex", device=dev)
levels = list(m.config.quantiles)
idx, W = C.load_rows(a.fam, a.ctx, a.rows)
n = len(idx); Q = np.full((n, C.H, len(levels)), np.nan, np.float32); nvalid = np.zeros(n, np.int32); cT = np.zeros(n, np.float32)
t0 = time.time()
for s, X in C.chunks(idx, W, a.bs):
    ok = (~np.isnan(X)).sum(1) >= 3; nvalid[s:s + len(X)] = (~np.isnan(X)).sum(1); cT[s:s + len(X)] = np.nansum(X, 1)
    if a.cum: X = C.to_cum(X)[0]
    if ok.any():
        q, _ = m.forecast(torch.tensor(X[ok]), prediction_length=C.H, batch_size=a.bs, output_type="numpy")
        Q[s + np.where(ok)[0]] = q
    print(s, n, f"{time.time() - t0:.0f}s", flush=True)
cT = cT if a.cum else None
cols = C.summarise(Q, levels, cT=cT)
tag = f"tirex_{a.fam}_c{a.ctx}" + ("_cum" if a.cum else "") + (f"_smoke{a.rows}" if a.rows else "")
print(C.save(tag, idx, cols, Q, levels, nvalid, cT)); C.report(tag, cols, nvalid, t0)
