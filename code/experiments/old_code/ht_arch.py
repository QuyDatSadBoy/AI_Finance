"""Notebook 3: the FinCast -> Bi-LSTM -> attention-pooling -> MLP architecture (user's diagram) and its block replacements.

Everything is walk-forward: for target year Y the net is trained on years <= Y-2, early-stopped on year Y-1, and
predicts year Y. Inputs are oriented by sgn(IS mean) (sign-flip symmetry); the final long/short sign is sgn * sign(pred).
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
import ht_common as h

DEV = "cuda"


# ------------------------------------------------------------------------------------------ data on the GPU
class Data:
    """Candidate rows (top-10% |t| per family-year) with hidden sequences, extras and targets, all as GPU/NumPy arrays."""

    def __init__(self, hidden_path=None, rows_path=None, hidden_tokens=None):
        import ht_wf as wf
        F = wf.load_all()
        rows = np.load(rows_path or (h.CACHE / "fincast_h_rows.npy"))
        self.F = F.iloc[rows].reset_index(drop=True)
        self.rows = rows
        hp = hidden_path or (h.CACHE / "fincast_h.npy")
        self.H = torch.from_numpy(np.load(hp, mmap_mode="r")[:]).to(DEV)           # [N, T, D] float16 on GPU
        X, eb_sr, ysr, r = wf.design(F)
        self.X_all = X.iloc[rows].reset_index(drop=True)
        sd, sg = self.F.sd_is.values, self.F.sgn.values
        self.sd, self.sg = sd, sg
        self.yr = self.F.oos_year.values
        self.y_or = self.F.y.values                                                  # oriented realised monthly return (%)
        self.eb_or = self.F.pred_ret.values * sg                                     # EB oriented monthly prediction (%)
        self.ysr = ysr[rows]
        self.eb_sr = eb_sr[rows]
        rr = r[rows]                                                                  # residual SR (family-year demeaned over all rows)
        self.resid = rr
        self.famcode = self.F.family.astype("category").cat.codes.values
        self.nobs = self.F.nobs.values.astype(np.float32)
        self.t_or = (self.F.tstat.values * sg).astype(np.float32)                   # oriented IS t-stat (>0 for candidates)
        self.t_oos = (self.y_or * np.sqrt(self.nobs) / sd).astype(np.float32)       # OOS mean in IS-t units, noise var = nobs/12

    def extras(self, kind):
        X, F, sg, sd = self.X_all, self.F, self.sg, self.sd
        base = pd.DataFrame({"tstat_is": F.tstat.values * sg, "mean_ret_is": F.mean_is.values * sg})   # the 2 extras of the diagram
        if kind == "E0":
            return base.values.astype(np.float32)
        if kind == "E1":                                                              # + EB output
            return np.c_[base.values, self.eb_or, F.thetahat.values * sg / np.sqrt(F.nobs.values)].astype(np.float32)
        if kind == "E2":                                                              # + recent-return / peer / family context
            cols = [c for c in X.columns if c not in ("tstat", "fam", "cf", "cfmed", "cfq")]
            return np.c_[base.values, self.eb_or, X[cols].values].astype(np.float32)
        if kind == "E3":                                                              # + family-year context (what EB uses), no EB output
            ctx = ["fam_frac_t2", "fam_mean_abst", "fam_mean_t", "eb_p", "eb_mua", "eb_sa", "eb_mub", "eb_sb", "nobs", "abs_t_rank"]
            onehot = pd.get_dummies(F.family).reindex(columns=h.FAMILIES, fill_value=0).values
            return np.c_[base.values, F[ctx].values, onehot].astype(np.float32)
        raise ValueError(kind)

    def target(self, kind):
        if kind == "T0":   return self.y_or                                          # raw oriented mean OOS return (diagram)
        if kind == "T1":   return self.y_or / self.sd                                # oriented OOS Sharpe (vol-normalised)
        if kind == "T2":   return self.resid                                         # what EB misses
        if kind in ("T3", "T4", "T5", "T6"): return self.t_oos                                   # neural EB: learned prior, Bayes update
        raise ValueError(kind)


# ------------------------------------------------------------------------------------------ model blocks
class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj, self.v = nn.Linear(d, d), nn.Linear(d, 1, bias=False)

    def forward(self, x):                                    # x [B, T, d]
        a = torch.softmax(self.v(torch.tanh(self.proj(x))).squeeze(-1), 1)
        return (a.unsqueeze(-1) * x).sum(1), a


class Net(nn.Module):
    """seq: bilstm (diagram) | gru | mean | transformer | none(mlp on flattened tokens);  pool: attn | mean | last."""

    def __init__(self, d_in, n_extra, seq="bilstm", pool="attn", hid=128, fc=64, drop=0.3, norm="bn", in_norm=False, in_drop=0.0, n_out=1):
        super().__init__()
        self.in_norm = nn.LayerNorm(d_in) if in_norm else nn.Identity()
        self.in_drop = nn.Dropout(in_drop)
        self.seq_kind, self.pool_kind = seq, pool
        if seq == "bilstm":
            self.seq = nn.LSTM(d_in, hid, batch_first=True, bidirectional=True); d = 2 * hid
        elif seq == "gru":
            self.seq = nn.GRU(d_in, hid, batch_first=True, bidirectional=True); d = 2 * hid
        elif seq == "transformer":
            self.proj = nn.Linear(d_in, 2 * hid)
            self.seq = nn.TransformerEncoder(nn.TransformerEncoderLayer(2 * hid, 4, 4 * hid, 0.1, batch_first=True, norm_first=True), 1); d = 2 * hid
        elif seq == "mean":
            self.proj = nn.Linear(d_in, 2 * hid); d = 2 * hid
        else:
            raise ValueError(seq)
        self.mha = nn.MultiheadAttention(d, 4, batch_first=True) if pool == "mha" else None      # self-attention over the 4 patches, then attention pooling
        self.attn = AttnPool(d) if pool in ("attn", "mha") else None
        layers = [nn.Linear(d + n_extra, fc), nn.ReLU()]
        if norm == "bn": layers.append(nn.BatchNorm1d(fc))
        if norm == "ln": layers.append(nn.LayerNorm(fc))
        layers += [nn.Dropout(drop), nn.Linear(fc, n_out)]
        self.n_out = n_out
        self.head = nn.Sequential(*layers)

    def forward(self, h_seq, extra):
        x = self.in_drop(self.in_norm(h_seq))
        if self.seq_kind in ("bilstm", "gru"):
            x, _ = self.seq(x)
        elif self.seq_kind == "transformer":
            x = self.seq(self.proj(x))
        else:
            x = torch.relu(self.proj(x))
        if self.pool_kind in ("attn", "mha"):
            if self.mha is not None:
                x = x + self.mha(x, x, x, need_weights=False)[0]
            c, _ = self.attn(x)
        elif self.pool_kind == "mean":
            c = x.mean(1)
        else:
            c = x[:, -1]
        o = self.head(torch.cat([c, extra], 1))
        return o.squeeze(-1) if self.n_out == 1 else o


# ------------------------------------------------------------------------------------------ walk-forward training
def walk_forward(D, extras="E0", target="T0", years=range(h.OOS_START, h.YEAR_MAX + 1), loss="huber", epochs=3, lr=1e-3, wd=1e-2,
                 bs=4096, seed=0, max_train=500_000, verbose=False, **net_kw):
    """Returns array pred (len = D rows) in the units of `target`, NaN for years not predicted."""
    E = D.extras(extras); y = D.target(target).astype(np.float32)
    yr = D.yr
    pred = np.full(len(yr), np.nan, np.float32)
    net = None
    for Y in years:
        trn = np.where((yr <= Y - 2) & np.isfinite(y))[0]
        val = np.where(yr == Y - 1)[0]
        te = np.where(yr == Y)[0]
        if len(trn) < 20000 or len(te) == 0:
            continue
        if len(trn) > max_train:                                   # keep the most recent rows (regime relevance)
            trn = trn[np.argsort(-yr[trn], kind="stable")[:max_train]]
        mu, sd_ = np.nanmean(E[trn], 0), np.nanstd(E[trn], 0) + 1e-6
        lo, hi = np.percentile(y[trn], [1, 99])
        Et = torch.tensor(np.nan_to_num(np.clip((E - mu) / sd_, -10, 10)).astype(np.float32), device=DEV)   # missing -> train mean
        yt = torch.tensor(np.clip(y, lo, hi).astype(np.float32), device=DEV)
        torch.manual_seed(seed + Y)
        if net is None:
            net = Net(D.H.shape[2], E.shape[1], n_out={"T3": 2, "T4": 5, "T5": 2, "T6": 5}.get(target, 1), **net_kw).to(DEV)   # warm start afterwards
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        lf0 = nn.MSELoss() if loss == "mse" else nn.HuberLoss(delta=1.0)
        if target in ("T3", "T5"):
            use_is = target == "T3"          # T5: drop the IS marginal term (biased on the |t|-selected sample)
            t_is = torch.tensor(D.t_or, device=DEV); n12 = torch.tensor(D.nobs / 12.0, device=DEV)

            def post(o, j):                                   # prior N(m, tau2) from the net, Bayes update with the IS t-stat
                m, tau2 = o[:, 0], nn.functional.softplus(o[:, 1]) + 1e-3
                return (tau2 * t_is[j] + m) / (tau2 + 1), tau2 / (tau2 + 1), m, tau2

            def lf(o, target_vals, j):
                pm, pv, m, tau2 = post(o, j)
                v = pv + n12[j]
                nll_oos = 0.5 * (torch.log(v) + (target_vals - pm) ** 2 / v)                 # OOS outcome | IS t-stat
                vi = tau2 + 1
                nll_is = 0.5 * (torch.log(vi) + (t_is[j] - m) ** 2 / vi)                     # EB marginal likelihood of the IS t
                return (nll_oos + nll_is).mean() if use_is else nll_oos.mean()

            def out_pred(o, j):
                return post(o, j)[0]
        elif target in ("T4", "T6"):                         # neural EB with a 2-component mixture prior (as the paper's mixnorm)
            use_is = target == "T4"
            t_is = torch.tensor(D.t_or, device=DEV); n12 = torch.tensor(D.nobs / 12.0, device=DEV)
            LOG2PI = float(np.log(2 * np.pi))

            def comps(o, j):
                lp = torch.stack([nn.functional.logsigmoid(o[:, 0]), nn.functional.logsigmoid(-o[:, 0])], 1)
                m = torch.stack([o[:, 1], o[:, 3]], 1)
                tau2 = torch.stack([nn.functional.softplus(o[:, 2]), nn.functional.softplus(o[:, 4])], 1) + 1e-3
                t = t_is[j][:, None]
                lmarg = lp - 0.5 * (LOG2PI + torch.log(tau2 + 1) + (t - m) ** 2 / (tau2 + 1))   # log pi_k N(t; m_k, tau_k^2+1)
                lw = lmarg - torch.logsumexp(lmarg, 1, keepdim=True)
                mu = (tau2 * t + m) / (tau2 + 1); v = tau2 / (tau2 + 1)
                return lmarg, lw, mu, v

            def lf(o, target_vals, j):
                lmarg, lw, mu, v = comps(o, j)
                vv = v + n12[j][:, None]
                lpred = lw - 0.5 * (LOG2PI + torch.log(vv) + (target_vals[:, None] - mu) ** 2 / vv)
                return (-torch.logsumexp(lpred, 1) - (torch.logsumexp(lmarg, 1) if use_is else 0)).mean()

            def out_pred(o, j):
                _, lw, mu, _ = comps(o, j)
                return (lw.exp() * mu).sum(1)
        else:
            def lf(o, target_vals, j):
                return lf0(o, target_vals)

            def out_pred(o, j):
                return o
        best, best_state = 1e18, None
        trn_t, val_t, te_t = (torch.tensor(a, device=DEV) for a in (trn, val, te))
        for ep in range(epochs):
            net.train()
            if loss.startswith("list"):                       # ListNet: rank strategies against each other within one year
                yrs_tr = np.unique(yr[trn]); by_year = {u: trn_t[torch.tensor(yr[trn] == u, device=DEV)] for u in yrs_tr}
                for _ in range(max(1, len(trn) // bs)):
                    g = by_year[yrs_tr[np.random.randint(len(yrs_tr))]]
                    j = g[torch.randint(len(g), (min(bs, len(g)),), device=DEV)]
                    p_ = net(D.H[j].float(), Et[j]); t_ = yt[j]
                    l = -(torch.softmax(t_ / t_.std().clamp(min=1e-6), 0) * torch.log_softmax(p_, 0)).sum()
                    if loss == "list+huber":
                        l = l + nn.functional.huber_loss(p_, t_)
                    opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            else:
                perm = trn_t[torch.randperm(len(trn_t), device=DEV)]
                for i in range(0, len(perm), bs):
                    j = perm[i:i + bs]
                    if len(j) < 8: continue
                    l = lf(net(D.H[j].float(), Et[j]), yt[j], j)
                    opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(sum(lf(net(D.H[val_t[i:i + 8192]].float(), Et[val_t[i:i + 8192]]), yt[val_t[i:i + 8192]], val_t[i:i + 8192]) * len(val_t[i:i + 8192])
                               for i in range(0, len(val_t), 8192)) / max(len(val_t), 1))
            if vl < best:
                best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            pred[te] = torch.cat([out_pred(net(D.H[te_t[i:i + 8192]].float(), Et[te_t[i:i + 8192]]), te_t[i:i + 8192])
                                  for i in range(0, len(te_t), 8192)]).cpu().numpy()
        if verbose: print(Y, end=" ", flush=True)
    return pred


# ------------------------------------------------------------------------------------------ evaluation with the paper's protocol
def to_signal(D, pred, target, rank="sr", kappa=1.0):
    """Turn a network output into (oriented total forecast in SR units, long/short sign multiplier, sd) per candidate row."""
    if target == "T0":   z = pred / D.sd                      # oriented %/month -> monthly SR
    elif target == "T1": z = pred
    elif target in ("T3", "T4", "T5", "T6"): z = pred / np.sqrt(D.nobs)              # posterior theta (t units) -> monthly Sharpe, like EB's thetahat/sqrt(n)
    else:                z = D.eb_sr + kappa * pred            # EB + what the net learned EB misses
    if rank == "ret":    z_rank = z * D.sd                    # rank by predicted return instead of predicted Sharpe
    else:                z_rank = z
    return z, z_rank


def hybrid(P, D, pred, target, rank="sr", kappa=1.0):
    """Forecast for EVERY row of P: the network's forecast on candidate rows, EB's forecast elsewhere.
    Returns (z_signed, zr_signed, ok): monthly-SR forecast in the strategy's own sign, ranking score, candidate mask."""
    z, zr = to_signal(D, pred, target, rank, kappa)
    key = pd.DataFrame(dict(family=D.F.family.values, signalid=D.F.signalid.values, oos_year=D.yr, z=z, zr=zr, sgn=D.sg))
    M = P[["family", "signalid", "oos_year", "pred_ret", "sd_is"]].merge(key, on=["family", "signalid", "oos_year"], how="left")
    ok = M.z.notna().values
    sd = M.sd_is.values
    eb_signed = M.pred_ret.values / sd                       # EB forecast in monthly SR units (signed)
    sgn_is = M.sgn.fillna(1).values                          # sign(IS mean) used to orient the inputs
    z_signed = np.where(ok, M.z.fillna(0).values * sgn_is, eb_signed)
    zr_signed = np.where(ok, M.zr.fillna(0).values * sgn_is, eb_signed * (sd if rank == "ret" else 1.0))
    return z_signed, zr_signed, ok


def evaluate(P, D, pred, target, rank="sr", kappa=1.0, ranks=(1, 5, 10), schemes=("eq", "mv"), delta=0.9, fill="eb"):
    """Paper-style evaluation over the FULL universe of P.
    The network only scores the candidate rows (top 10% |t| per family-year); every other strategy keeps its EB forecast
    (fill='eb'), so the AI re-ranks the head of the distribution while the rest is exactly the paper's EB.
    With pred = 0 and target 'T2' this reproduces the paper's EB numbers exactly."""
    import ht_new as nw
    z_signed, zr_signed, ok = hybrid(P, D, pred, target, rank, kappa)
    sd = P.sd_is.values
    sign = np.where(z_signed >= 0, 1.0, -1.0)
    score = np.abs(zr_signed) if fill == "eb" else np.where(ok, np.abs(zr_signed), -1e9)
    mu = np.abs(z_signed) * sd
    out, rets = {}, {}
    for r in ranks:
        for sc in schemes:
            ret = nw.portfolio_returns(P, sign, score, mu, r, sc, delta=delta)
            rets[(r, sc)] = ret
            for nm, (lo, hi) in (("dev", nw.DEV), ("hold", nw.HOLD), ("full", (1983, 2020))):
                p = nw.perf(ret, lo, hi)
                out[(r, sc, nm)] = (p["mean"], p["sharpe"])
    return out, rets


def ic_report(D, pred, target, kappa=1.0):
    """Mean within-family-year Spearman IC of the total forecast vs realised oriented SR (candidate rows), dev / holdout."""
    z, _ = to_signal(D, pred, target, "sr", kappa)
    d = pd.DataFrame(dict(fam=D.F.family.values, yr=D.yr, z=z, y=D.ysr)).dropna()
    s = d.groupby(["fam", "yr"]).apply(lambda g: g.z.corr(g.y, method="spearman") if len(g) > 50 else np.nan).groupby(level=1).mean()
    return s.loc[:2004].mean(), s.loc[2005:].mean()


# ------------------------------------------------------------------------------------------ alternative encoders (controls)
def oriented_windows(D, ctx=128):
    """[N, ctx] float32 oriented monthly returns before the forecast year (NaN = missing)."""
    out = np.full((len(D.yr), ctx), np.nan, np.float32)
    for (fam, y), idx in D.F.groupby(["family", "oos_year"]).indices.items():
        pos, wide = h.wide_of(fam)
        r = pos.loc[D.F.signalid.values[idx]].values
        b = (y - h.BASE_YEAR) * 12
        lo = b - ctx
        W = wide[r, max(lo, 0):b]
        if lo < 0:
            W = np.pad(W, ((0, 0), (-lo, 0)), constant_values=np.nan)
        out[idx] = W * D.sg[idx, None]
    return out


def raw_patch_hidden(D):
    """No pretrained encoder: the same 128 months cut into 4 patches of 32, vol-normalised, NaN -> 0."""
    W = oriented_windows(D) / D.sd[:, None]
    return torch.tensor(np.nan_to_num(W, nan=0.0).reshape(len(W), 4, 32), dtype=torch.float16, device=DEV)


@torch.no_grad()
def chronos_hidden(D, name="amazon/chronos-bolt-small", batch=8192):
    """Chronos-Bolt encoder tokens for the same 128 months, averaged to 4 tokens (same shape role as FinCast's 4 x 1280)."""
    import ht_pretrained as pt
    pipe = pt.load_pipeline(name)
    W = oriented_windows(D)
    outs = []
    for s in range(0, len(W), batch):
        e, _ = pipe.embed(torch.tensor(W[s:s + batch]))
        e = e.float()
        T = e.shape[1] - (e.shape[1] % 4)
        e = e[:, -T:].reshape(e.shape[0], 4, T // 4, e.shape[2]).mean(2)
        outs.append(e.half().to(DEV))
    return torch.cat(outs)


def family_calibrate(D, z, K=10, lam=0.5):
    """Walk-forward per-family linear calibration of a forecast z (monthly SR units, oriented):
    realised_SR ~ a_f + b_f * z fitted on candidate rows of the K previous OOS years only (shrunk toward a=0, b=1)."""
    yr, fam = D.yr, D.F.family.values
    y = D.y_or / D.sd if hasattr(D, "y_or") else D.F.y.values / D.sd
    out = np.full(len(z), np.nan, np.float32)
    for f in np.unique(fam):
        mf = fam == f
        for Y in np.unique(yr[mf]):
            te = mf & (yr == Y)
            tr = mf & (yr <= Y - 1) & (yr >= Y - K) & np.isfinite(y) & np.isfinite(z)
            if tr.sum() < 200:
                out[te] = z[te]; continue
            X = np.c_[np.ones(tr.sum()), z[tr]]
            n = tr.sum()
            # ridge toward prior (a=0, b=1) with weight lam * n
            A_ = X.T @ X + lam * n * np.diag([1.0, np.mean(z[tr] ** 2)])
            bvec = X.T @ y[tr] + lam * n * np.array([0.0, np.mean(z[tr] ** 2)])
            a, b = np.linalg.solve(A_, bvec)
            out[te] = a + b * z[te]
    return out
