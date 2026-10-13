"""Shared helpers for the zero-shot forecast scripts (numpy only, so it works in both venvs).

Rows: idx_<fam>.npy holds positions among the 655,101 candidate rows (= rows of code/_cache/win{128,240}.npy).
Output <tag>.npz, aligned with idx order:
  idx      row positions (copy of the index file, for alignment checks)
  point12  forecast of the next-12-month mean return from the model's point output (FinCast: its mean head;
           Chronos-2 / TiRex: the median, because their 'mean' output IS the median)
  med12    from the 0.5 quantile;  qmean12  from the quantile-integrated mean sum_i w_i q_i (w = mid-point spacing)
  q10_12m, q90_12m  10% / 90% quantiles of the 12-month mean
  sdm      implied monthly sd (predicted Sharpe = qmean12 / sdm)
  med_h1   one-month-ahead median return
  nvalid   number of non-NaN months in the context
  Q        float16 [n, 12, K] raw per-step model quantiles (returns, or cumulative level in --cum mode), levels in `levels`
Two input modes:
  returns (default)  the model sees monthly returns. 12-month figures = average over h=1..12 of the per-step outputs;
                     q10_12m/q90_12m use an independence approximation med12 -/+ (per-step spread averaged)/sqrt(12).
  --cum              the model sees the cumulative-return index c_t = sum of past returns (a log-price-like path, the kind of
                     series price-trained models such as FinCast were pretrained on). The forecast of the 12-month mean is
                     (c_{T+12} - c_T)/12, so q10_12m/q90_12m are the model's own quantiles of the 12-month mean (exact).
All numbers are in %/month, oriented (sign of the in-sample mean), i.e. the same scale as D.y_or."""
import os, time, numpy as np

CACHE = "/home/phongndh/Documents/Paper/code/_cache/"
OUT = os.path.dirname(os.path.abspath(__file__)) + "/"
H = 12


def load_rows(fam, ctx, nrows=None):
    idx = np.load(OUT + f"idx_{fam}.npy")
    if nrows:                                              # smoke test: rows spread evenly over the subset
        idx = idx[np.linspace(0, len(idx) - 1, nrows).astype(int)]
    W = np.load(CACHE + f"win{ctx}.npy", mmap_mode="r")
    return idx, W


def chunks(idx, W, size):
    for s in range(0, len(idx), size):
        j = idx[s:s + size]
        yield s, np.asarray(W[j], dtype=np.float32)       # idx is sorted, so this is a forward read of the memmap


def to_cum(X):
    """Cumulative-return path; NaN months stay NaN (masked by the models) and add nothing to the level. Returns (C, c_T)."""
    Cm = np.nancumsum(X, 1)
    Cm[np.isnan(X)] = np.nan
    return Cm, np.nansum(X, 1)


def summarise(Q, levels, point=None, cT=None):
    """Q: [n, H, K] quantiles (levels ascending). point: [n, H] model point forecast or None (-> median).
    cT: None for returns mode, else [n] last cumulative level (cum mode)."""
    lv = np.asarray(levels, float)
    e = np.r_[0.0, (lv[1:] + lv[:-1]) / 2, 1.0]; w = np.diff(e)            # sums to 1
    i10, i50, i90 = (int(np.argmin(np.abs(lv - q))) for q in (0.1, 0.5, 0.9))
    if point is None: point = Q[:, :, i50]
    if cT is None:
        med = Q[:, :, i50]; spread = (Q[:, :, i90] - Q[:, :, i10]).mean(1)
        out = dict(point12=point.mean(1), med12=med.mean(1), qmean12=(Q @ w).mean(1),
                   q10_12m=med.mean(1) - (med - Q[:, :, i10]).mean(1) / np.sqrt(H),
                   q90_12m=med.mean(1) + (Q[:, :, i90] - med).mean(1) / np.sqrt(H),
                   sdm=spread / 2.5631, med_h1=med[:, 0])
    else:
        c = cT[:, None]; QH = (Q[:, -1, :] - c) / H                           # quantiles of the 12-month mean
        out = dict(point12=(point[:, -1] - cT) / H, med12=QH[:, i50], qmean12=QH @ w, q10_12m=QH[:, i10], q90_12m=QH[:, i90],
                   sdm=(QH[:, i90] - QH[:, i10]) / 2.5631 * np.sqrt(H), med_h1=Q[:, 0, i50] - cT)
    return out


def save(tag, idx, cols, Q, levels, nvalid, cT=None):
    """In cum mode Q is stored relative to c_T (cumulative return from the end of the context) to keep float16 precision."""
    path = OUT + f"{tag}.npz"
    Qs = Q if cT is None else Q - cT[:, None, None]
    np.savez(path, idx=idx, levels=np.asarray(levels, np.float32), Q=Qs.astype(np.float16), nvalid=nvalid, cum=cT is not None,
             **{k: v.astype(np.float32) for k, v in cols.items()})
    return path


def report(tag, cols, nvalid, t0):
    ok = nvalid >= 3
    print(f"[{tag}] n={len(nvalid)} ok={ok.sum()} time={time.time() - t0:.1f}s")
    for k, v in cols.items():
        v = v[ok]
        print(f"  {k:8s} mean={np.nanmean(v):8.4f} sd={np.nanstd(v):7.4f} p5={np.nanpercentile(v, 5):8.4f} "
              f"p95={np.nanpercentile(v, 95):8.4f} nan={np.isnan(v).sum()}")
