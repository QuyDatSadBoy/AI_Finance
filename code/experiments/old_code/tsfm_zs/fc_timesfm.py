"""Zero-shot 12-month forecasts from TimesFM 2.5 (google/timesfm-2.5-200m-pytorch, Apache-2.0, 231M; Das et al. ICML 2024,
arXiv 2310.10688). PRETRAINED.  Run in ~/.local/share/research-kernel/venv_tsfm (pip timesfm[torch] 3.0.2).
TimesFM strips leading NaN and linearly interpolates inner NaN; output = mean head + quantiles 0.1..0.9.
(TimesFM 3.0 exists but is under a non-commercial licence; not used here.)
Usage: python fc_timesfm.py FAM CTX [--rows N] [--cum] [--bs B]
Output: <OUT>/timesfm25_<FAM>_c<CTX>[_cum][_smokeN].npz  (see tsfm_common)."""
import os, sys, time, argparse, numpy as np, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import tsfm_common as C
import timesfm

ap = argparse.ArgumentParser(); ap.add_argument("fam"); ap.add_argument("ctx", type=int)
ap.add_argument("--rows", type=int); ap.add_argument("--cum", action="store_true"); ap.add_argument("--bs", type=int, default=1024)
a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("NTHREADS", 16)))
m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch", torch_compile=False)
idx, W = C.load_rows(a.fam, a.ctx, a.rows)
bs = min(a.bs, len(idx))
m.compile(timesfm.ForecastConfig(max_context=-(-a.ctx // 32) * 32, max_horizon=128, normalize_inputs=True, per_core_batch_size=bs,
                                 use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=False,
                                 fix_quantile_crossing=True))
levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
n = len(idx); Q = np.full((n, C.H, 9), np.nan, np.float32); P = np.full((n, C.H), np.nan, np.float32)
nvalid = np.zeros(n, np.int32); cT = np.zeros(n, np.float32); t0 = time.time()
for s, X in C.chunks(idx, W, 8 * bs):
    ok = (~np.isnan(X)).sum(1) >= 3; nvalid[s:s + len(X)] = (~np.isnan(X)).sum(1); cT[s:s + len(X)] = np.nansum(X, 1)
    if a.cum: X = C.to_cum(X)[0]
    if ok.any():
        p, q = m.forecast(horizon=C.H, inputs=list(X[ok].astype(np.float64)))
        j = s + np.where(ok)[0]; P[j] = p[:, :C.H]; Q[j] = q[:, :C.H, 1:]
    print(s, n, f"{time.time() - t0:.0f}s", flush=True)
cT = cT if a.cum else None
cols = C.summarise(Q, levels, point=P, cT=cT)
tag = f"timesfm25_{a.fam}_c{a.ctx}" + ("_cum" if a.cum else "") + (f"_smoke{a.rows}" if a.rows else "")
print(C.save(tag, idx, cols, Q, levels, nvalid, cT)); C.report(tag, cols, nvalid, t0)
