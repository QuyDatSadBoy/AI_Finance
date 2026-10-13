"""Light data object restricted to some families: only their FinCast hidden states go to the GPU (~1.2 GB per family instead of 6.3 GB),
so single-family experiments can run next to a full-data job.  Has every field ht_base.walk_forward / eval_subset need, incl. X_all (F1/F2)."""
import numpy as np, pandas as pd, torch, ht_common as h, ht_wf as wf, ht_base as B

class Sub:
    pass

NULLS = ("null_acct_ew",)                                              # simulated pure-noise families (make_null.py), training only


def make_sub(fams, with_X=True, with_H=True):
    nulls = [f for f in fams.split("+") if f in NULLS]; fams = "+".join(f for f in fams.split("+") if f not in NULLS)
    F = wf.load_all(); rows = np.load(h.CACHE / "fincast_h_rows.npy"); Fc = F.iloc[rows].reset_index(drop=True)
    m = np.isin(Fc.family.values, fams.split("+")); idx = np.where(m)[0]
    D = Sub(); D.F = Fc.iloc[idx].reset_index(drop=True)
    if nulls:
        return _add_nulls(D, Fc, idx, nulls, with_X, with_H)
    D.sd, D.sg, D.yr = D.F.sd_is.values, D.F.sgn.values, D.F.oos_year.values
    D.y_or = D.F.y.values; D.eb_or = D.F.pred_ret.values * D.sg; D.nobs = D.F.nobs.values.astype(np.float32)
    D.famcode = Fc.family.astype("category").cat.codes.values[idx].astype(np.int64)      # same codes as the full data (6 families)
    if with_X:
        X = wf.design(F)[0]; D.X_all = X.iloc[rows].reset_index(drop=True).iloc[idx].reset_index(drop=True)
    D.idx_full = idx                                                    # positions of these rows among all candidate rows (pred files)
    if with_H:
        H = np.load(h.CACHE / "fincast_h.npy", mmap_mode="r")
        D.H = torch.from_numpy(np.ascontiguousarray(H[idx])).to("cuda")
    return D


def _add_nulls(D, Fc, idx, nulls, with_X, with_H):
    """Real rows of the requested families + simulated null-family rows appended at the end (family codes 6, 7, ...; idx_full = -1)."""
    Fn = [pd.read_parquet(h.CACHE / f"{n}_F.parquet") for n in nulls]
    D.F = pd.concat([D.F] + Fn, ignore_index=True)
    D.sd, D.sg, D.yr = D.F.sd_is.values, D.F.sgn.values, D.F.oos_year.values
    D.y_or = D.F.y.values; D.eb_or = D.F.pred_ret.values * D.sg; D.nobs = D.F.nobs.values.astype(np.float32)
    code = Fc.family.astype("category").cat.codes.values[idx].astype(np.int64)
    D.famcode = np.r_[code, np.concatenate([np.full(len(f), 6 + i, np.int64) for i, f in enumerate(Fn)])]
    D.idx_full = np.r_[idx, np.full(sum(len(f) for f in Fn), -1)]
    if with_X:
        X = wf.design(wf.load_all())[0]; rows = np.load(h.CACHE / "fincast_h_rows.npy")
        D.X_all = pd.concat([X.iloc[rows].reset_index(drop=True).iloc[idx].reset_index(drop=True)] + [pd.DataFrame(index=range(len(f)), columns=X.columns, dtype=float) for f in Fn], ignore_index=True)
    if with_H:
        H = np.load(h.CACHE / "fincast_h.npy", mmap_mode="r")
        D.H = torch.from_numpy(np.concatenate([np.ascontiguousarray(H[idx])] + [np.load(h.CACHE / f"{n}_h.npy") for n in nulls])).to("cuda")
    return D
