"""Feature engineering for the walk-forward learner (notebook 2).

Everything for (strategy, year Y) is computed from returns strictly BEFORE Y (months < 12*(Y-1960)).
Returns-based features are *oriented*: multiplied by sgn = sign(IS mean), so the learner always sees a
strategy that "looks long-positive". The target is the oriented next-year mean return. The flip-symmetry
(f(-strategy) = -f(strategy)) is therefore built in.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import ht_common as h

TRAIN_START_YEAR = 1966       # first pseudo-OOS year used only for training (>= 36 IS months exist)


# ------------------------------------------------------------------------------------ strategy metadata
def _groups_acct(ids):
    sl = pd.read_csv(h.RAW / "DataMinedSignalList.csv").set_index("signalid").reindex(ids)
    return {"form": sl.signal_form.astype(str).values, "v1": sl.v1.astype(str).values, "v2": sl.v2.astype(str).values,
            "fv1": (sl.signal_form.astype(str) + "|" + sl.v1.astype(str)).values,
            "fv2": (sl.signal_form.astype(str) + "|" + sl.v2.astype(str)).values}


def _groups_past(ids):
    nm = pd.read_csv(h.RAW / "PastReturnSignalNames.csv").set_index("signalid").reindex(ids)
    tok = nm.signalname.str.split("_")
    return {"root": nm.root.astype(str).values,
            "tok": [[f"{r}:{x}" for x in t[1:]] for r, t in zip(nm.root, tok)]}


def _groups_ticker(ids):
    nm = pd.read_csv(h.RAW / "TickerSignalNames.csv").set_index("signalid").reindex(ids)
    t = nm.signalname.str.split("_")                       # L1 lng a b sht c d
    return {"L": t.str[0].values,
            "tok": [[f"lng:{x[2]}", f"lng:{x[3]}", f"sht:{x[5]}", f"sht:{x[6]}"] for x in t],
            "pair_l": ("l" + t.str[2] + "_" + t.str[3]).values, "pair_s": ("s" + t.str[5] + "_" + t.str[6]).values}


def membership_matrices(fam):
    """List of sparse (n_strategies x n_groups) 0/1 matrices, one per grouping of the strategy definition."""
    ids, _ = h.load_wide(fam)
    kind = fam.rsplit("_", 1)[0]
    g = {"acct": _groups_acct, "past_ret": _groups_past, "ticker": _groups_ticker}[kind](ids)
    mats = []
    for key, v in g.items():
        if isinstance(v, list):                             # multi-membership (tokens)
            rows = np.repeat(np.arange(len(v)), [len(x) for x in v])
            lab = np.concatenate([np.array(x, dtype=object) for x in v])
        else:
            rows, lab = np.arange(len(v)), np.asarray(v, dtype=object)
        codes, _ = pd.factorize(lab)
        mats.append(sp.csr_matrix((np.ones(len(rows), np.float32), (rows, codes)), shape=(len(ids), codes.max() + 1)))
    return mats


def peer_features(mats, v, valid, absv=False):
    """Leave-one-out mean of v among peers sharing a group, averaged over each strategy's memberships."""
    out = []
    v = np.where(valid, v, 0.0).astype(np.float64)
    w = valid.astype(np.float64)
    for M in mats:
        S, C = M.T @ v, M.T @ w
        inv = np.where(C > 1, 1.0 / np.maximum(C - 1, 1), 0.0)
        k = np.asarray(M.sum(1)).ravel()
        num = M @ (S * inv) - v * (M @ inv)                 # sum over memberships of LOO group mean
        out.append(num / np.maximum(k, 1))
    return np.stack(out, 1)


# ------------------------------------------------------------------------------------ per family-year features
def _nanstats(x):
    n = np.sum(~np.isnan(x), 1)
    with np.errstate(all="ignore"):
        m = np.nanmean(x, 1); s = np.nanstd(x, 1)
    return n, m, s


def family_year_features(fam, year, mats=None):
    ids, wide = h.load_wide(fam)
    st = h.rolling_stats(wide, year) if year + 1 - h.BASE_YEAR <= wide.shape[1] / 12 else None
    if st is None:
        return None
    ok = st["ok"]
    b = (year - h.BASE_YEAR) * 12
    t = st["tstat"]
    sg = np.where(st["mean_is"] >= 0, 1.0, -1.0).astype(np.float32)
    W = wide[:, max(b - 240, 0):b] * sg[:, None]            # oriented IS window (n x <=240)
    T = W.shape[1]
    feat = {"signalid": ids, "oos_year": year, "sgn": sg}
    # --- full-window statistics + EB (the paper's information set)
    tt = t[ok]
    par = h.fit_eb(tt)
    th = np.full(len(ids), np.nan); th[ok] = h.eb_posterior_mean(par, tt)
    feat.update(mean_is=st["mean_is"], tstat=t, nobs=st["nobs"], sd_is=st["sd_is"], thetahat=th,
                pred_ret=th * st["mean_is"] / t)
    # --- horizons
    for k in (6, 12, 24, 36, 60, 120):
        if T >= 1:
            n, m, s = _nanstats(W[:, max(T - k, 0):])
            feat[f"m{k}"] = m
            if k >= 24:
                feat[f"s{k}"] = s
                with np.errstate(all="ignore"):
                    feat[f"t{k}"] = m / (s / np.sqrt(n))
    # --- annual means of the last 5 years (a1 = most recent) and first/second half
    for j in range(5):
        lo, hi = T - 12 * (j + 1), T - 12 * j
        feat[f"a{j+1}"] = _nanstats(W[:, max(lo, 0):hi])[1] if hi > 0 else np.nan
    h1 = _nanstats(W[:, :T // 2])[1]; h2 = _nanstats(W[:, T // 2:])[1]
    feat["decay"] = h2 - h1
    with np.errstate(all="ignore"):
        feat["hit60"] = np.nanmean(np.where(np.isnan(W[:, max(T - 60, 0):]), np.nan, (W[:, max(T - 60, 0):] > 0)), 1)
    # --- cross-section context (rank of own strength within family-year, family-level summary of the t distribution)
    absr = pd.Series(np.where(ok, np.abs(t), np.nan)).rank(pct=True).values
    feat["abs_t_rank"] = absr
    feat["fam_frac_t2"] = np.mean(np.abs(tt) > 2)
    feat["fam_mean_abst"] = np.mean(np.abs(tt))
    feat["fam_mean_t"] = np.mean(tt)
    feat["fam_n"] = ok.sum()
    p, mua, sa, mub, sb = h._unpack(par)[2], *[h._unpack(par)[i] for i in (0, 1, 3, 4)]
    feat.update(eb_p=p, eb_mua=mua, eb_sa=sa, eb_mub=mub, eb_sb=sb)
    # --- peers sharing definition tokens (signed by own orientation so "agreeing" peers are positive)
    mats = mats if mats is not None else membership_matrices(fam)
    pa = peer_features(mats, t, ok)
    pabs = peer_features(mats, np.abs(t), ok)
    for i in range(pa.shape[1]):
        feat[f"peer_t{i}"] = pa[:, i] * sg
        feat[f"peer_abs{i}"] = pabs[:, i]
    # --- target
    feat["mean_oos"] = st["mean_oos"]
    df = pd.DataFrame(feat)
    df["family"] = fam
    return df[ok].reset_index(drop=True)


def build_feature_panel(families=h.FAMILIES, y0=TRAIN_START_YEAR, y1=h.YEAR_MAX, verbose=True):
    parts = []
    for fam in families:
        mats = membership_matrices(fam)
        for y in range(y0, y1 + 1):
            d = family_year_features(fam, y, mats)
            if d is not None and len(d):
                parts.append(d)
        if verbose:
            print(fam, "done", sum(len(p) for p in parts))
    return pd.concat(parts, ignore_index=True)
