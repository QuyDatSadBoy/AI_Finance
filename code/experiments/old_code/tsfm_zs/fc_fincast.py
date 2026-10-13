"""Zero-shot 12-month forecasts from FinCast's own forecasting head (Zhu et al., CIKM 2025, arXiv 2508.19609; weights
HF Vincent05R/FinCast v1.pth, Apache-2.0, ~1B params, 50 layers, MoE). PRETRAINED.
Run in the main venv. Reuses code/ht_fincast.load_encoder (same config: patch 32, head horizon 128, 9 quantiles + mean).
One forward pass: the output of the LAST input patch gives the next 128 months; we keep h = 1..12. Output index 0 is the
point (mean) head, 1..9 are quantiles 0.1..0.9. The model's instance normalisation (first patch with >= 3 valid values)
is inverted inside _postprocess_output, so forecasts are in %/month. freq id 1 = monthly (FinCast freq_reader_inference).
Usage: python fc_fincast.py FAM CTX [--rows N] [--cum] [--bs B]     CTX 128 -> 4 patches; 240 -> left-padded (NaN) to 256 = 8 patches."""
import os, sys, time, argparse, numpy as np, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0]); sys.path.insert(0, "/home/phongndh/Documents/Paper/code")
import tsfm_common as C
import ht_fincast as FC

ap = argparse.ArgumentParser(); ap.add_argument("fam"); ap.add_argument("ctx", type=int)
ap.add_argument("--rows", type=int); ap.add_argument("--bs", type=int, default=512); ap.add_argument("--cum", action="store_true")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.set_num_threads(int(os.environ.get("NTHREADS", 16)))
m = FC.load_encoder(device=dev)
levels = list(m.config.quantiles)                                  # 0.1 ... 0.9
L = -(-a.ctx // FC.PATCH) * FC.PATCH                               # round up to a multiple of 32
idx, W = C.load_rows(a.fam, a.ctx, a.rows)
n = len(idx); Q = np.full((n, C.H, len(levels)), np.nan, np.float32); P = np.full((n, C.H), np.nan, np.float32)
nvalid = np.zeros(n, np.int32); cT = np.zeros(n, np.float32); t0 = time.time()


@torch.no_grad()
def run(X):
    x = torch.tensor(np.pad(X, ((0, 0), (L - X.shape[1], 0)), constant_values=np.nan), device=dev)
    pad = torch.isnan(x).float(); x = torch.nan_to_num(x, nan=0.0)
    freq = torch.ones(len(x), 1, dtype=torch.long, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
        out, _ = m(x, pad, freq)                                   # [b, patches, 128, 10]
    o = out[:, -1, :C.H, :].float().cpu().numpy()
    return o[:, :, 0], np.sort(o[:, :, 1:], axis=-1)               # point head; quantiles (sorted: no crossing)


for s, X in C.chunks(idx, W, a.bs):
    ok = (~np.isnan(X)).sum(1) >= 3; nvalid[s:s + len(X)] = (~np.isnan(X)).sum(1); cT[s:s + len(X)] = np.nansum(X, 1)
    if a.cum: X = C.to_cum(X)[0]
    if ok.any():
        p, q = run(X[ok]); j = s + np.where(ok)[0]; P[j] = p; Q[j] = q
    print(s, n, f"{time.time() - t0:.0f}s", flush=True)

cT = cT if a.cum else None
cols = C.summarise(Q, levels, point=P, cT=cT)
tag = f"fincast_{a.fam}_c{a.ctx}" + ("_cum" if a.cum else "") + (f"_smoke{a.rows}" if a.rows else "")
print(C.save(tag, idx, cols, Q, levels, nvalid, cT)); C.report(tag, cols, nvalid, t0)
