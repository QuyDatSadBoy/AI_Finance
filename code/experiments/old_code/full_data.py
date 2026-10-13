"""Data object with ALL Accounting-EW strategy-years (not only the top-10% |t| candidates), for ht_base.walk_forward and the evaluation scripts.
The frozen FinCast tokens are already projected to the whitened PCA16 bottleneck (extract_full_acct.py), so configs use hpca=0."""
import numpy as np, pandas as pd, torch
import ht_common as h, ht_wf as wf, ht_base as B, ht_arch as A

_orig_window240 = B.window240
B.window240 = lambda D: np.asarray(D.win240, dtype=np.float64) if hasattr(D, "win240") else _orig_window240(D)    # windows of the rows of D


class Full:
    pass


def make_full(with_H=True, with_win=True, encf=None, with_X=False):
    F = wf.load_all(); rows = np.load(h.CACHE / "full_acct_rows.npy"); Fc = F.iloc[rows].reset_index(drop=True)
    D = Full(); D.F = Fc
    D.sd, D.sg, D.yr = Fc.sd_is.values, Fc.sgn.values, Fc.oos_year.values
    D.y_or = Fc.y.values; D.eb_or = Fc.pred_ret.values * D.sg; D.nobs = Fc.nobs.values.astype(np.float32)
    D.famcode = np.zeros(len(Fc), np.int64)               # acct_ew has code 0 in the 6-family coding
    D.idx_full = np.arange(len(Fc))
    if with_X:                                          # financial features of the strategy (design matrix of ht_wf; EB / zero-shot columns are dropped by extras_free)
        D.X_all = wf.design(Fc)[0].reset_index(drop=True)
    if with_win:
        D.win240 = A.oriented_windows(D, ctx=240)
    if with_H:
        if encf in ("lp7", "raw"):                                            # NON-pretrained controls (AGENTS.md rule D)
            D.H = (B.logpatch_tokens(D) if encf == "lp7" else A.raw_patch_hidden(D)).to("cuda"); return D
        fin = np.load(h.CACHE / "full_acct_pca16.npy")                       # FinCast tokens, PCA16
        if encf == "exaone": fin = np.load(h.CACHE / "full_acct_exaone_pca16.npy")      # EXAONE Forecast for Finance tokens, PCA16 (PRETRAINED, financial synthetic data)
        elif encf == "chr2": fin = np.load(h.CACHE / "full_acct_chr2_pca16.npy")          # Chronos-2 (240 months) tokens, PCA16 (PRETRAINED)
        elif encf == "both": fin = np.concatenate([fin, np.load(h.CACHE / "full_acct_exaone_pca16.npy")], 1)    # 8 tokens: FinCast 4 + EXAONE 4
        D.H = torch.from_numpy(fin).to("cuda")
    return D
