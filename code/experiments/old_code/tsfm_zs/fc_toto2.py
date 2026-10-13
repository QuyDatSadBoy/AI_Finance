"""Zero-shot 12-month forecasts from Toto 2.0 (Datadog, Apache-2.0; Toto 1: Cohen et al. arXiv 2505.14766; Toto 2.0 family
4m..2.5B on HF Datadog/Toto-2.0-*). PRETRAINED.  Run in ~/.local/share/research-kernel/venv_tsfm (pip toto-models 1.0.0).
Toto masks missing values (target_mask=False); output = 9 quantiles 0.1..0.9 (no separate mean head -> median).
Usage: python fc_toto2.py FAM CTX [--size 22m|313m|1B|2.5B] [--rows N] [--cum] [--bs B]
Output: <OUT>/toto2-<size>_<FAM>_c<CTX>[_cum][_smokeN].npz  (see tsfm_common)."""
import os, sys, time, argparse, numpy as np, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import tsfm_common as C
from toto2 import Toto2Model

ap = argparse.ArgumentParser(); ap.add_argument("fam"); ap.add_argument("ctx", type=int); ap.add_argument("--size", default="313m")
ap.add_argument("--rows", type=int); ap.add_argument("--cum", action="store_true"); ap.add_argument("--bs", type=int, default=1024)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.set_num_threads(int(os.environ.get("NTHREADS", 16)))
m = Toto2Model.from_pretrained(f"Datadog/Toto-2.0-{a.size}").to(dev).eval()
levels = list(m.output_head.knots); P = m.config.patch_size
L = -(-a.ctx // P) * P                                              # left-pad (masked) to a multiple of the patch size
idx, W = C.load_rows(a.fam, a.ctx, a.rows)
n = len(idx); Q = np.full((n, C.H, len(levels)), np.nan, np.float32); nvalid = np.zeros(n, np.int32); cT = np.zeros(n, np.float32)
t0 = time.time()
for s, X in C.chunks(idx, W, a.bs):
    ok = (~np.isnan(X)).sum(1) >= 3; nvalid[s:s + len(X)] = (~np.isnan(X)).sum(1); cT[s:s + len(X)] = np.nansum(X, 1)
    if a.cum: X = C.to_cum(X)[0]
    if ok.any():
        x = torch.tensor(np.pad(X[ok], ((0, 0), (L - X.shape[1], 0)), constant_values=np.nan), device=dev)[:, None, :]
        msk = ~torch.isnan(x)
        with torch.inference_mode():
            q = m.forecast({"target": torch.nan_to_num(x), "target_mask": msk,
                            "series_ids": torch.zeros(len(x), 1, dtype=torch.long, device=dev)}, horizon=C.H, has_missing_values=True,
                           scaler_fallback_min_obs=8, quantile_real_cap_k=1e4)   # defaults of the official GluonTS wrapper
        Q[s + np.where(ok)[0]] = q[:, :, 0, :C.H].permute(1, 2, 0).float().cpu().numpy()     # (9,b,1,h) -> (b,h,9)
    print(s, n, f"{time.time() - t0:.0f}s", flush=True)
cT = cT if a.cum else None
cols = C.summarise(Q, levels, cT=cT)
tag = f"toto2-{a.size}_{a.fam}_c{a.ctx}" + ("_cum" if a.cum else "") + (f"_smoke{a.rows}" if a.rows else "")
print(C.save(tag, idx, cols, Q, levels, nvalid, cT)); C.report(tag, cols, nvalid, t0)
