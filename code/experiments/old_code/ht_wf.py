"""Walk-forward learners (notebook 2): residual-on-EB targets, LightGBM (CPU) and MLP on pretrained embeddings (GPU).

Target: what EB gets wrong.  r = realised_oriented_SR_next_year - EB_oriented_SR,  demeaned within family-year
(removing the common year shock that no strategy-level feature can know).  Everything is volatility-normalised
(divided by the strategy's IS standard deviation) so that families with different volatility are comparable.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, lightgbm as lgb
import ht_common as h

CAND_Q = 0.85            # learners only see / score strategies whose |t| is in the top 15% of their family-year


def load_all():
    """Feature panel + Chronos zero-shot forecasts. Row order = features.parquet (family, year blocks)."""
    F = pd.read_parquet(h.CACHE / "features.parquet")
    C = pd.read_parquet(h.CACHE / "chronos_small.parquet")
    return pd.concat([F, C], axis=1)


def design(F, with_chronos=True):
    sd, sg = F.sd_is.values, F.sgn.values
    ret = ["m6", "m12", "m24", "m36", "m60", "a1", "a2", "a3", "a4", "a5", "decay"]
    X = {"tstat": F.tstat * sg, "thetahat": F.thetahat * sg / np.sqrt(F.nobs), "nobs": F.nobs, "abs_t_rank": F.abs_t_rank,
         "sdis": sd, "mean_is": F.mean_is * sg / sd}
    X |= {c: F[c] / sd for c in ret}
    X |= {c: F[c] / sd for c in ("s24", "s36", "s60")}
    X |= {"t24": F.t24, "t36": F.t36, "t60": F.t60, "hit60": F.hit60}
    X |= {f"peer_t{i}": F[f"peer_t{i}"] for i in range(5)} | {f"peer_abs{i}": F[f"peer_abs{i}"] for i in range(5)}
    X["fam"] = F.family.astype("category").cat.codes
    if with_chronos:
        X |= {"cf": F.cf_mean_small / sd, "cfmed": F.cf_med_small / sd, "cfq": (F.cf_q90_small - F.cf_q10_small) / sd}
    X = pd.DataFrame(X)
    eb = F.pred_ret.values * sg / sd                       # EB oriented monthly SR forecast
    ysr = F.y.values / sd                                  # realised oriented monthly SR
    r = ysr - eb
    r = r - pd.Series(r).groupby([F.family.values, F.oos_year.values]).transform("mean").values
    return X, eb, ysr, r


def _split(F, cand, y, Y, max_train):
    yr = F.oos_year.values
    tr = np.where((yr <= Y - 1) & cand & np.isfinite(y))[0]
    te = np.where((yr == Y) & cand)[0]
    if len(tr) > max_train:
        tr = np.random.default_rng(Y).choice(tr, max_train, replace=False)
    return tr, te


def walk_forward_lgbm(F, X, r, cand, years, rounds=120, max_train=1_500_000, verbose=False):
    params = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=3000, feature_fraction=0.8,
                  bagging_fraction=0.7, bagging_freq=1, lambda_l2=100, num_threads=16, verbose=-1, seed=0)
    out = np.full(len(F), np.nan)
    for Y in years:
        tr, te = _split(F, cand, r, Y, max_train)
        if len(tr) < 5000:
            continue
        lo, hi = np.percentile(r[tr], [1, 99])
        m = lgb.train(params, lgb.Dataset(X.iloc[tr], np.clip(r[tr], lo, hi)), rounds)
        out[te] = m.predict(X.iloc[te])
        if verbose:
            print(Y, end=" ", flush=True)
    return out


class ResMLP(nn.Module):
    def __init__(self, d_in, d_h=256, p=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_h), nn.GELU(), nn.Dropout(p), nn.Linear(d_h, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def walk_forward_mlp(F, Xtab, E, rows_E, r, cand, years, use_emb=True, epochs=4, bs=4096, lr=1e-3, wd=1e-2, seed=0,
                     max_train=1_000_000, device="cuda", verbose=False):
    """GPU MLP on [tabular features | pretrained Chronos embedding]. E[k] is the embedding of row rows_E[k] of F."""
    out = np.full(len(F), np.nan)
    row2k = np.full(len(F), -1); row2k[rows_E] = np.arange(len(rows_E))
    for Y in years:
        tr, te = _split(F, cand & (row2k >= 0), r, Y, max_train)
        if len(tr) < 5000:
            continue
        xt = Xtab.iloc[tr].values.astype(np.float32); med = np.nanmedian(xt, 0)
        mu, sd = np.nanmean(xt, 0), np.nanstd(xt, 0) + 1e-6
        prep = lambda a: np.where(np.isnan(a), med, a)
        A = (prep(xt) - mu) / sd; B = (prep(Xtab.iloc[te].values.astype(np.float32)) - mu) / sd
        if use_emb:
            Etr, Ete = E[row2k[tr]].astype(np.float32), E[row2k[te]].astype(np.float32)
            em, es = Etr.mean(0), Etr.std(0) + 1e-6
            A = np.c_[A, (Etr - em) / es]; B = np.c_[B, (Ete - em) / es]
        lo, hi = np.percentile(r[tr], [1, 99])
        ya = np.clip(r[tr], lo, hi).astype(np.float32)
        A_t, B_t, y_t = (torch.tensor(a, device=device) for a in (A, B, ya))
        torch.manual_seed(seed + Y)
        net = ResMLP(A.shape[1]).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        for _ in range(epochs):
            net.train()
            perm = torch.randperm(len(A_t), device=device)
            for i in range(0, len(perm), bs):
                j = perm[i:i + bs]
                loss = ((net(A_t[j]) - y_t[j]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            out[te] = net(B_t).cpu().numpy()
        if verbose:
            print(Y, end=" ", flush=True)
    return out
