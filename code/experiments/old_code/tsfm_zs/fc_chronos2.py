"""Zero-shot 12-month forecasts from Chronos-2 (amazon/chronos-2, Apache-2.0, 120M; Ansari et al. arXiv 2510.15821). PRETRAINED.
Run in the main venv (chronos-forecasting 2.3.2).
Usage: python fc_chronos2.py FAM CTX [--rows N] [--xl] [--cum] [--bs B]
  FAM   acct_ew | all | any prepared idx_<FAM>.npy        CTX  128 | 240 (months before the forecast year)
  --xl  cross-learning (group attention) inside each family-year, random groups of 100 series (Chronos-2 report's group size).
        Only series of the same family-year are grouped, all contexts end before Y, so no look-ahead.
  --cum feed the cumulative-return path instead of returns (see tsfm_common).
Output: <OUT>/chronos2_<FAM>_c<CTX>[_xl][_cum][_smokeN].npz  (see tsfm_common)."""
import os, sys, time, argparse, numpy as np, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import tsfm_common as C
from chronos import Chronos2Pipeline

ap = argparse.ArgumentParser(); ap.add_argument("fam"); ap.add_argument("ctx", type=int)
ap.add_argument("--rows", type=int); ap.add_argument("--xl", action="store_true"); ap.add_argument("--cum", action="store_true"); ap.add_argument("--bs", type=int, default=1024)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.set_num_threads(int(os.environ.get("NTHREADS", 16)))
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=dev)
levels = pipe.quantiles                                      # 21 levels 0.01 ... 0.99
idx, W = C.load_rows(a.fam, a.ctx, a.rows)
n = len(idx); Q = np.full((n, C.H, len(levels)), np.nan, np.float32); nvalid = np.zeros(n, np.int32); cT = np.zeros(n, np.float32)
t0 = time.time()


def run(X, cross):
    """X: [b, ctx] with NaN = missing (leading NaN = not yet listed). Returns [b, H, K]."""
    ok = (~np.isnan(X)).sum(1) >= 3
    if a.cum: X = C.to_cum(X)[0]
    out = np.full((len(X), C.H, len(levels)), np.nan, np.float32)
    if ok.any():
        q, _ = pipe.predict_quantiles(torch.tensor(X[ok])[:, None, :], prediction_length=C.H, quantile_levels=levels,
                                      batch_size=100 if cross else a.bs, cross_learning=cross)
        out[ok] = torch.stack([t[0] for t in q]).float().numpy()
    return out


with torch.inference_mode():
    if not a.xl:
        for s, X in C.chunks(idx, W, 16 * a.bs):
            Q[s:s + len(X)] = run(X, False); nvalid[s:s + len(X)] = (~np.isnan(X)).sum(1); cT[s:s + len(X)] = np.nansum(X, 1)
            print(s, n, f"{time.time() - t0:.0f}s", flush=True)
    else:
        M = np.load(C.OUT + f"meta_{a.fam}.npz"); pos = np.searchsorted(M["idx"], idx)
        g = M["famcode"][pos] * 10000 + M["year"][pos]; rng = np.random.default_rng(0)
        for k, gv in enumerate(np.unique(g)):
            j = rng.permutation(np.where(g == gv)[0])
            X = np.asarray(W[idx[j]], np.float32)
            Q[j] = run(X, True); nvalid[j] = (~np.isnan(X)).sum(1); cT[j] = np.nansum(X, 1)
            if k % 20 == 0: print(k, gv, f"{time.time() - t0:.0f}s", flush=True)

cT = cT if a.cum else None
cols = C.summarise(Q, levels, cT=cT)
tag = f"chronos2_{a.fam}_c{a.ctx}" + ("_xl" if a.xl else "") + ("_cum" if a.cum else "") + (f"_smoke{a.rows}" if a.rows else "")
print(C.save(tag, idx, cols, Q, levels, nvalid, cT)); C.report(tag, cols, nvalid, t0)
