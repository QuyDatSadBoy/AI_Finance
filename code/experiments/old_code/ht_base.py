"""BASE MODEL (the user's diagram) with swappable blocks.  One file: network, losses, preprocessing, training, evaluation.

    FinCast (frozen) -> sequence block -> attention block -> [ctx (+) extras] -> FC head -> pred_ret

What it predicts
    pred_ret = mean monthly return (%) of a strategy in year Y, DIRECTLY from the strategy's own history before Y.
    No Empirical-Bayes output is used as input or as anchor (columns thetahat, eb_*, ctx0..6 are excluded, see `extras_free`).

Blocks that can be swapped (all selected on dev 1983-2004 only; holdout 2005-2020 is reported, never used to choose):
    seq    : bilstm (diagram) | gru | transformer | mean
    pool   : attn (diagram) | mha (self-attention over the 4 patches, then attention pooling) | query (multi-head learned-query pooling) | mean
    enc    : fincast (diagram) | chronos | raw (the 128 months cut into 4 patches of 32, no pretrained model)   -- see use_encoder
    loss   : mse | huber | wmse (error-weighted MSE, cf. arXiv 2304.12778 L-weighting, weights carry no gradient)
             | mae (median regression) | studentt (heteroscedastic Student-t likelihood of the return, df=4)    -- all are forecast-error / likelihood losses
    extras : E0 (tstat_IS, mean_ret_IS = diagram) | E2 (E0 + own momentum: months 2-12, 6, 12 / sd, 12-month hit rate, drawdown / sd) | F1 (+ recent-return / persistence / peer features, EB-free) | F2 (F1 + family-year context) | F3 (E0 + family breadth, no family id)
    xs     : False | True  (cross-sectional set attention: every strategy attends to the other strategies of its family-year,
             so the network can learn the shrinkage that EB hard-codes; batches are drawn from ONE family-year)
    aug    : none | noise (Gaussian noise on the hidden states) | recent (rows of the last 8 train years seen twice)
             | up2 (2x data: every batch is also seen with noisy hidden states and extras) | mix2 (2x data: every batch is also seen as a mixup with a shuffled copy of itself)
    fx     : none | rank (rank-gauss of each continuous extra feature within its family-year)
    hpca   : 0 (off) | k: whitened PCA of the encoder tokens to k dims, fitted on pre-1983 rows of the training families
    warm   : chain (default: year Y starts from year Y-1) | cold (re-initialise every year) | sp (shrink-and-perturb, sp_lam, sp_sig)
    regime : False | True (2 extra features: lagged realised within-family IC of |t| with the return, EWMA and 5-year mean)
    detev  : False | True (prediction chunks of a family-year are fixed instead of randomly split)
    hnorm  : bn (default) | ln | none | setbn (batch statistics in training AND prediction: with xs every batch is one family-year, so the head
             is normalised within the family-year consistently; use with xset >= family-year size and detev)
    xfuse  : False | True (the extra statistics are added to the set tokens before xs, zero-initialised)
    sched  : const | cos (5% warm-up then cosine, per year)
    wide   : False | True (Wide & Deep: a linear path from the extra features to the output, added to the deep head)
    zx/zcols: zero-shot forecasts of pretrained TS models as extras (file _cache/tsfm_x_<zx>.npy, columns matching zcols)
    trmin  : first label year used for training (default all; the 1966-82 rows have short in-sample windows)
    enc    : ... | lp7 (no pretrained model: 7 log-spaced patches x [mean/sd, sd/sd, t, hit, coverage])
    fyw    : none | vH | v | inv (family-year loss weight 0.5(1+v/mean v) / v/mean v / mean v/v, v = mean squared TRAINING label of the
             family-year: Holen et al. L-weighting with the loss replaced by its expectation)
    hw     : 0 | w (rows in the top 10% |t| of their family-year weigh 1+w in the loss: the top-k rule only uses the head)
    nw     : 0 | tau2 (row weight 1/((s24/sd)^2/12 + tau2): heteroscedastic Gaussian likelihood)   lwm: l (Holen) | eq (control)
    rc     : False | True (recentre the network output within each family-year in training, validation and prediction: with a demeaned label the
             output's family-year mean is not identified -- under BatchNorm it is a random year shift at prediction time; needs xs and xset >= size)
    base   : none | blend (one slope on z(m) + bw*z(momentum), z(|t|) for Sharpe labels) | lin4 (closed-form ridge of the demeaned label on demeaned [m, m/sd, |t|, sd] of the strategy, fitted on years Y-bK..Y-1;
             the network learns the residual, its last layer starts at zero, prediction = level + base + network)
    lvl    : fam (mean label of years <= Y-2) | exp1 (years <= Y-1) | roll10 / roll20 (last 10 / 20 years <= Y-1) | breadth |
             ratio<w> (Sharpe labels: c * mean in-sample Sharpe of the family-year, c = realised/in-sample over the last w years)
             mom (family momentum: family-year mean label ~ mean last-12-month return of the candidates, fitted on family-years <= Y-1)
    fx     : none | rank | dm (extras demeaned within the family-year, magnitudes kept)
    ep0 / seed2 / hln / optkeep : score the warm-started state before training / seed = seed*1000+Y / LayerNorm on the tokens (default True) /
             keep the AdamW state across years
    gate   : False | True (needs extras F3: the deviation output is multiplied by 2*sigmoid(linear(family breadth)), so a family whose
             strategies are mostly noise can have its forecasts shrunk toward the level, learned across the training families)
    shuf   : False | True (visit family-year chunks in random order each epoch)
    lw     : 0 (off) | K>1: L-weighted merge of K family-years per optimiser step (loss_i/sum(loss)+1/K, Holen et al. arXiv 2304.12778)
    wls    : 0 (off) | tau^2: inverse-variance weighted loss 1/(sd^2/12 + tau^2)
    corrw  : 0 (off) | lam: add lam*(1 - corr(pred, y)) inside each batch (= family-year with xs)
    corru  : ret (default) | sr: the correlation is computed in Sharpe units ((forecast incl. level)/sd vs y/sd), as the paper ranks by predicted Sharpe | tot: base + network vs the demeaned label
    rw     : 0 (off) | K>1: R-Weighted merge (reward = top-10% realised return of training labels) -- EXPLORATORY, violates rule A
    hdelta : Huber threshold in %/month (default 1)
    ema    : 0 (off) | decay in (0,1): exponential moving average of the weights, used for validation and prediction
    head   : fc (diagram: Linear-ReLU-norm-Dropout-Linear) | grn (gated residual network of Temporal Fusion Transformers)
    hblock / hm: HyperAttention exact block size and number of sampled keys (defaults 256 / 256; block >= set size = exact attention)
    xk     : full (self-attention over the set) | hyper (HyperAttention: sortLSH blocks + uniformly sampled keys, Han et al. ICLR 2024) | isab (induced set attention with xm inducing points, O(n*m)); with isab xset can be the whole family-year (e.g. 16384)
    xset   : size of the family-year set the xs attention works on (default 2048)
    cal    : none | val (after early stopping, fit pred -> a*pred + b by least squares on the validation year Y-1, a clipped to [0, 1.5])
    prep   : none | ydms (as ydm, but the level added back is in SHARPE units: family mean of y/sd times each strategy's sd, so the level
             does not push the Sharpe ranking toward low-volatility strategies) | ydm (train on the return minus its own family-year mean, i.e. without the market-wide year effect; the family mean of the training labels is added back at prediction) | clip3 (clip standardised extras at +-3 instead of +-10) | ysd (target = return / sd_IS, prediction rescaled back) | ysdm (ysd demeaned within the family-year, level = mean training Sharpe: an MSE in Sharpe units, i.e. return errors weighted by 1/sd^2 -- the paper ranks by predicted Sharpe)
No Sharpe or portfolio term is ever in the loss (AGENTS.md rule A).
"""
import copy, json, time
import numpy as np, pandas as pd, torch, torch.nn as nn
import ht_common as h
import ht_arch as A

DEV = "cuda"
EB_COLS = ("thetahat", "cf", "cfmed", "cfq", "fam")                       # EB output / pretrained-Chronos columns / integer family code
RES = h.CACHE / "base_results.tsv"
RES2 = h.CACHE / "fam_results.tsv"            # results of the family-subset direction (2026-10-03)


def append_row(row):
    """Append one result row; if the columns differ from the file (new metrics added later) rewrite the file with the union of columns."""
    new = pd.DataFrame([row])
    if RES.exists():
        old = pd.read_csv(RES, sep="\t")
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(RES, sep="\t", index=False)

PRED_DIR = h.CACHE / "base_preds"


# ------------------------------------------------------------------------------------------ preprocessing
def window240(D):
    """Oriented monthly returns of the 240 months before the forecast year for the rows of D (cached array, NaN = missing)."""
    W = np.load(h.CACHE / "win240.npy", mmap_mode="r")
    return np.asarray(W[D.idx_full] if hasattr(D, "idx_full") else W, dtype=np.float64)


def ts_momentum(D):
    """[mean of months t-12..t-2, mean t-6..t-1, mean t-12..t-1] / sd_IS, 12-month hit rate, current drawdown / sd_IS (all before Y)."""
    W = window240(D); sd = D.sd[:, None]
    with np.errstate(all="ignore"):
        m2_12, m6, m12 = np.nanmean(W[:, -12:-1], 1), np.nanmean(W[:, -6:], 1), np.nanmean(W[:, -12:], 1)
        hit = np.nanmean(np.where(np.isnan(W[:, -12:]), np.nan, W[:, -12:] > 0), 1)
        c = np.nancumsum(W, 1); dd = c[:, -1] - np.nanmax(c, 1)
    return np.nan_to_num(np.c_[m2_12 / D.sd, m6 / D.sd, m12 / D.sd, hit, dd / D.sd])


def logpatch_tokens(D):
    """No pretrained model: 7 patches on a log scale back from Y (months 1, 2-6, 7-12, 13-24, 25-60, 61-120, 121-240), each token =
    [patch mean / sd_IS, patch sd / sd_IS, patch t, hit rate, share of months present].  Scaled by the whole-window sd only (the level is
    the signal; FinCast's per-series normalisation removes it)."""
    W = window240(D); sd = D.sd[:, None]; edges = [(239, 240), (234, 239), (228, 234), (216, 228), (180, 216), (120, 180), (0, 120)]
    toks = []
    with np.errstate(all="ignore"):
        for a, b in edges[::-1]:                                       # oldest first, like a time series
            P = W[:, a:b]; n = np.sum(~np.isnan(P), 1); m = np.nanmean(P, 1); s_ = np.nanstd(P, 1)
            toks.append(np.c_[m / D.sd, s_ / D.sd, m / (s_ / np.sqrt(np.maximum(n, 1)) + 1e-9) * (b - a > 1), np.nanmean(np.where(np.isnan(P), np.nan, P > 0), 1), n / (b - a)])
    T = np.nan_to_num(np.stack(toks, 1)).clip(-10, 10)
    return torch.tensor(T, dtype=torch.float16, device=DEV)


def extras_free(D, kind):
    """Extra inputs concatenated to the attention context.  Oriented by sign(IS mean); no EB-derived column."""
    F, sg, sd = D.F, D.sg, D.sd
    base = np.c_[F.tstat.values * sg, F.mean_is.values * sg]              # tstat_IS, mean_ret_IS (diagram)
    if kind == "E0":
        return base.astype(np.float32)
    if kind == "E2":                                                   # E0 + time-series momentum of the strategy itself (the extra signal found
        return np.c_[base, ts_momentum(D)].astype(np.float32)          # beyond the in-sample mean: partial IC ~0.08-0.09 in dev AND holdout)
    cols = [c for c in D.X_all.columns if c not in EB_COLS and not c.startswith("eb_")]
    feat = D.X_all[cols].values
    if kind == "F1":
        return np.c_[base, feat].astype(np.float32)
    if kind == "F3":                                                   # E0 + breadth of the family-year (share of |t|>2, mean |t|, mean t over
        return np.c_[base, F[["fam_frac_t2", "fam_mean_abst", "fam_mean_t"]].values].astype(np.float32)   # all its strategies): no family id
    ctx = ["fam_frac_t2", "fam_mean_abst", "fam_mean_t", "nobs", "abs_t_rank"]               # family-year breadth of signals, still EB-free
    onehot = pd.get_dummies(F.family).reindex(columns=h.FAMILIES, fill_value=0).values
    return np.c_[base, feat, F[ctx].values, onehot].astype(np.float32)


# ------------------------------------------------------------------------------------------ network
class Pool(nn.Module):
    def __init__(self, d, kind):
        super().__init__()
        self.kind = kind
        if kind in ("attn", "mha"):
            self.proj, self.v = nn.Linear(d, d), nn.Linear(d, 1, bias=False)
        if kind == "mha":
            self.mha = nn.MultiheadAttention(d, 4, batch_first=True)
        if kind == "query":
            self.q = nn.Parameter(torch.randn(1, 1, d) * d ** -0.5); self.mq = nn.MultiheadAttention(d, 4, batch_first=True)

    def forward(self, x):                                                  # x [B, T, d] -> [B, d]
        if self.kind == "mean":
            return x.mean(1)
        if self.kind == "query":
            return self.mq(self.q.expand(len(x), -1, -1), x, x, need_weights=False)[0][:, 0]
        if self.kind == "mha":
            x = x + self.mha(x, x, x, need_weights=False)[0]
        a = torch.softmax(self.v(torch.tanh(self.proj(x))).squeeze(-1), 1)
        return (a.unsqueeze(-1) * x).sum(1)


class ISAB(nn.Module):
    """Induced set attention block (Lee et al., Set Transformer): m learned inducing points first summarise the whole set, then every element
    reads that summary.  Cost O(n*m) instead of O(n^2), so one batch can hold a whole family-year (a low-rank attention, cf. Nystrom landmarks)."""

    def __init__(self, d, heads=4, m=32):
        super().__init__()
        self.ind = nn.Parameter(torch.randn(1, m, d) * d ** -0.5)
        self.a1 = nn.MultiheadAttention(d, heads, batch_first=True); self.a2 = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1, self.n2, self.n3, self.n4 = (nn.LayerNorm(d) for _ in range(4))
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):                                                  # x [1, n, d]
        h = self.n1(self.ind + self.a1(self.ind, x, x, need_weights=False)[0])         # [1, m, d] summary of the set
        y = self.n2(x + self.a2(x, h, h, need_weights=False)[0])                        # each element reads the summary
        return self.n3(y + self.ff(self.n4(y)))


class GRN(nn.Module):
    """Gated residual network (Lim et al., Temporal Fusion Transformers, IJF 2021): ELU MLP, GLU gate, skip connection, LayerNorm."""

    def __init__(self, d_in, d, drop=0.3):
        super().__init__()
        self.skip = nn.Linear(d_in, d); self.l2 = nn.Linear(d_in, d); self.l1 = nn.Linear(d, d); self.drop = nn.Dropout(drop)
        self.g, self.v = nn.Linear(d, d), nn.Linear(d, d); self.norm = nn.LayerNorm(d)

    def forward(self, a):
        e = self.drop(self.l1(nn.functional.elu(self.l2(a))))
        return self.norm(self.skip(a) + torch.sigmoid(self.g(e)) * self.v(e))


class HyperSetAttention(nn.Module):
    """HyperAttention-style self-attention over a set (Han et al., ICLR 2024): sortLSH puts similar elements next to each other and attention is
    computed exactly inside blocks of size `block` (the large entries); the remaining keys are estimated from m uniformly sampled keys scaled by
    (n - block)/m.  Cost O(n*(block+m)) per head, so a batch can be a whole family-year.  With block >= n it is exact softmax attention."""

    def __init__(self, d, heads=4, block=256, m=256, bits=8):
        super().__init__()
        self.h, self.dh, self.block, self.m = heads, d // heads, block, m
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.register_buffer("planes", torch.randn(heads, self.dh, bits))
        self.register_buffer("pow2", 2 ** torch.arange(bits))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def attend(self, x):                                                  # x [n, d] -> [n, d]
        n = x.shape[0]
        q, k, v = self.qkv(x).view(n, 3, self.h, self.dh).unbind(1)          # each [n, h, dh]
        q, k, v = (t.transpose(0, 1) * (1.0 if i else self.dh ** -0.5) for i, t in enumerate((q, k, v)))  # [h, n, dh]; scale q
        b = min(self.block, n); nb = -(-n // b); pad = nb * b - n
        code = ((torch.einsum("hnd,hdr->hnr", k, self.planes) > 0).long() * self.pow2).sum(-1)       # SimHash bucket of each key/query
        perm = torch.argsort(code, dim=1)                                    # sortLSH: same permutation for q and k (self-attention)
        g = lambda t: torch.gather(t, 1, perm[..., None].expand(-1, -1, t.shape[-1]))
        qs, ks, vs = g(q), g(k), g(v)
        if pad:
            qs, ks, vs = (torch.cat([t, t[:, :pad]], 1) for t in (qs, ks, vs))
        qb, kb, vb = (t.view(self.h, nb, b, self.dh) for t in (qs, ks, vs))
        s_blk = qb @ kb.transpose(-1, -2)                                   # [h, nb, b, b] exact large entries
        if pad:                                                             # mask the padded keys of the last block
            valid = torch.arange(nb * b, device=x.device).view(nb, b) < n
            s_blk = s_blk.masked_fill(~valid[None, :, None, :], float("-inf"))
        rest = n - b
        if self.m > 0 and rest > 0:
            idx = torch.randint(0, n, (self.m,), device=x.device)            # uniform sample of keys (shared by all queries)
            ksm, vsm = k[:, idx], v[:, idx]                                  # [h, m, dh]
            s_smp = qs.view(self.h, nb, b, self.dh) @ ksm.transpose(-1, -2)[:, None]       # [h, nb, b, m]
            blk_of_key = torch.empty(n, dtype=torch.long, device=x.device)
            blk_of_key[perm[0]] = torch.arange(n, device=x.device) // b         # block id of every key (head 0 permutation)
            same = (torch.arange(nb, device=x.device)[:, None] == blk_of_key[idx][None, :])   # sampled key already counted in the block
            s_smp = s_smp.masked_fill(same[None, :, None, :], float("-inf"))
            w = torch.log(torch.tensor(rest / self.m, device=x.device))          # each sample stands for rest/m keys
            mx = torch.maximum(s_blk.amax(-1, keepdim=True), (s_smp + w).amax(-1, keepdim=True)).detach()
            e1, e2 = torch.exp(s_blk - mx), torch.exp(s_smp + w - mx)
            out = (e1 @ vb + e2 @ vsm[:, None]) / (e1.sum(-1, keepdim=True) + e2.sum(-1, keepdim=True))
        else:
            out = torch.softmax(s_blk, -1) @ vb
        out = out.reshape(self.h, nb * b, self.dh)[:, :n]
        inv = torch.empty_like(perm); inv.scatter_(1, perm, torch.arange(n, device=x.device).expand(self.h, -1))
        out = torch.gather(out, 1, inv[..., None].expand(-1, -1, self.dh))   # undo the permutation
        return self.o(out.transpose(0, 1).reshape(n, -1))

    def forward(self, x):                                                  # x [1, n, d]
        y = x[0] + self.attend(self.n1(x[0]))
        return (y + self.ff(self.n2(y)))[None]


class BaseNet(nn.Module):
    """[B,4,1280] hidden states + [B,n_extra] extras -> [B] (or [B,2] for studentt: mean, log-scale)."""

    def __init__(self, d_in, n_extra, seq="bilstm", pool="attn", hid=128, fc=64, drop=0.3, n_out=1, xs=False, xl=1, xk="full", xm=32, wide=False, hnorm="bn", xfuse=False, gate=False, hln=True, hblock=256, hm=256, head="fc"):
        super().__init__()
        self.gate = nn.Linear(2, 1) if gate else None                # family reliability gate on the breadth columns of F3 (share of |t|>2,
        if gate: nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)   # mean |t|): scales the forecast deviation, starts at 1
        self.wide = nn.Linear(n_extra, n_out) if wide else None          # Wide & Deep: linear path from the statistical inputs straight to the output
        if not xs:
            self.xs = None
        elif xk == "hyper":
            self.xs = nn.Sequential(*[HyperSetAttention(2 * hid, 4, hblock, hm) for _ in range(xl)])
        elif xk == "isab":
            self.xs = nn.Sequential(*[ISAB(2 * hid, 4, xm) for _ in range(xl)]) if xl > 1 else ISAB(2 * hid, 4, xm)
        else:
            self.xs = nn.TransformerEncoder(nn.TransformerEncoderLayer(2 * hid, 4, 4 * hid, 0.1, batch_first=True, norm_first=True), xl)
        self.norm = nn.LayerNorm(d_in) if hln else nn.Identity()
        self.seq_kind = seq
        if seq in ("bilstm", "gru"):
            self.seq = (nn.LSTM if seq == "bilstm" else nn.GRU)(d_in, hid, batch_first=True, bidirectional=True); d = 2 * hid
        elif seq == "transformer":
            self.proj = nn.Linear(d_in, 2 * hid)
            self.seq = nn.TransformerEncoder(nn.TransformerEncoderLayer(2 * hid, 4, 4 * hid, 0.1, batch_first=True, norm_first=True), 1); d = 2 * hid
        else:
            self.proj = nn.Linear(d_in, 2 * hid); d = 2 * hid
        self.pool = Pool(d, pool)
        hn = {"bn": nn.BatchNorm1d(fc), "ln": nn.LayerNorm(fc), "none": nn.Identity(),
              "setbn": nn.BatchNorm1d(fc, track_running_stats=False)}[hnorm]   # setbn: always the statistics of the current set (= one family-year)   # bn: train stats of ONE family-year vs running stats at eval
        self.head = nn.Sequential(nn.Linear(d + n_extra, fc), nn.ReLU(), hn, nn.Dropout(drop), nn.Linear(fc, n_out)) if head == "fc" else \
            nn.Sequential(GRN(d + n_extra, fc, drop), nn.Linear(fc, n_out))     # head=grn: gated residual network
        self.xf = None
        if xfuse:                                                         # statistics (t-stat, mean, features) enter the set tokens before xs
            self.xf = nn.Sequential(nn.Linear(n_extra, d), nn.GELU(), nn.Linear(d, d))
            nn.init.zeros_(self.xf[2].weight); nn.init.zeros_(self.xf[2].bias)
        self.n_out = n_out

    def forward(self, hs, extra):
        x = self.norm(hs)
        if self.seq_kind in ("bilstm", "gru"):
            x, _ = self.seq(x)
        elif self.seq_kind == "transformer":
            x = self.seq(self.proj(x))
        else:
            x = torch.relu(self.proj(x))
        c = self.pool(x)
        if self.xf is not None:
            c = c + self.xf(extra)
        if self.xs is not None:                                            # the batch is one family-year: set attention over its strategies
            c = self.xs(c[None])[0]
        o = self.head(torch.cat([c, extra], 1))
        if self.wide is not None:
            o = o + (self.wide(extra).squeeze(-1) if o.dim() == 1 else self.wide(extra))
        if self.gate is not None:
            o = o * 2 * torch.sigmoid(self.gate(extra[:, 2:4]))
        return o.squeeze(-1) if self.n_out == 1 else o


def loss_fn(kind, o, y, delta=1.0):
    """Forecast-error / likelihood losses of the return y (%/month, oriented). `o` is the network output."""
    if kind == "studentt":                                                 # mean m, scale exp(s), df 4
        m, s = o[:, 0], o[:, 1].clamp(-4, 4)
        z2 = ((y - m) / s.exp()) ** 2
        return (s + 2.5 * torch.log1p(z2 / 4)).mean()
    e2 = (o - y) ** 2
    if kind == "mse":
        return e2.mean()
    if kind == "mae":
        return (o - y).abs().mean()
    if kind == "huber":
        return nn.functional.huber_loss(o, y, delta=delta)
    if kind == "wmse":
        return (e2 * (e2 / e2.mean().clamp(min=1e-8)).detach().sqrt().clamp(0.2, 5.0)).mean()
    raise ValueError(kind)


# ------------------------------------------------------------------------------------------ family subsets
STAT_FAMS = ("acct_ew", "acct_vw", "past_ret_ew", "past_ret_vw")      # sources with statistical content; ticker is only for the final inference test


def fam_mask(D, spec):
    """spec: 'all' | 'none' | 'acct_ew+past_ret_ew' ... -> boolean mask over the candidate rows."""
    fam = D.F.family.values
    if spec == "all": return np.ones(len(fam), bool)
    if spec in ("none", "", None): return np.zeros(len(fam), bool)
    return np.isin(fam, spec.split("+"))


# ------------------------------------------------------------------------------------------ walk-forward training
XS_SET = 2048


def batches(idx, grp, size, train, by_group=False, shuf=False, det_eval=False):
    """Mini-batches of row indices.  Without xs (grp is used only for xs): random rows.  For xs the batch is a subset of ONE family-year.
    shuf=True: the family-year chunks are visited in random order (otherwise always in sorted order, so the last updates of every epoch
    always come from the same family-year)."""
    if by_group:
        g = grp[idx]; chunks = []
        for u in torch.unique(g):
            rows = idx[g == u]
            if train or not det_eval:                                  # det_eval: prediction chunks fixed (same set seen every time)
                rows = rows[torch.randperm(len(rows), device=rows.device)]
            for i in range(0, len(rows), size):
                if len(rows[i:i + size]) >= 64 or not train: chunks.append(rows[i:i + size])
        order = torch.randperm(len(chunks)).tolist() if (train and shuf) else range(len(chunks))
        for k in order: yield chunks[k]
        return
    perm = idx[torch.randperm(len(idx), device=idx.device)]
    for i in range(0, len(perm), size):
        if len(perm[i:i + size]) >= 8 or not train: yield perm[i:i + size]

def walk_forward(D, cfg, years=range(h.OOS_START, h.YEAR_MAX + 1), bs=4096, max_train=500_000, seed=0):
    """Train on years <= Y-2, early-stop on Y-1, predict Y (oriented pred_ret, %/month); warm start across years."""
    E = extras_free(D, cfg["extras"]); yr = D.yr
    if cfg.get("zx"):                                            # zero-shot forecasts of pretrained time-series models (code/tsfm_zs, PRETRAINED:
        import json                                              # labelled), as extra inputs only: e.g. zcols="chronos2_xl_dev|tirex_dev"
        Z = np.load(h.CACHE / f"tsfm_x_{cfg['zx']}.npy", mmap_mode="r"); zc = json.load(open(h.CACHE / f"tsfm_x_{cfg['zx']}.json"))["cols"]
        sel = [i for i, c in enumerate(zc) if any(z in c for z in cfg.get("zcols", "").split("|"))]
        E = np.c_[E, np.asarray(Z[D.idx_full][:, sel], dtype=np.float32)]
    Hx = D.H
    if cfg.get("hpca", 0):                                       # whitened PCA bottleneck of the frozen tokens, fitted on pre-1983 rows of the
        k = cfg["hpca"]                                          # training families only (before the first forecast year: no look-ahead)
        fit = np.where((yr <= 1982) & fam_mask(D, cfg.get("fams", "all")))[0]
        fit = fit[np.random.default_rng(0).permutation(len(fit))[:50000]]
        Z = D.H[torch.tensor(fit, device=DEV)].float().reshape(-1, D.H.shape[2]); mu_h = Z.mean(0)
        ev, V = torch.linalg.eigh(torch.cov((Z - mu_h).T)); V = V[:, -k:] / ev[-k:].clamp(min=1e-6).sqrt()
        Hx = torch.cat([(((D.H[s0:s0 + 65536].float() - mu_h) @ V).clamp(-5, 5)).half() for s0 in range(0, len(D.H), 65536)])
        del Z
    if cfg.get("regime", False):                                 # lagged realised predictability of each family: within-family-year Spearman IC of
        g = pd.DataFrame(dict(f=D.famcode, yr=yr, t=np.abs(D.F.tstat.values), y=D.y_or))   # |t_IS| with the realised return, EWMA over the years
        ic = g.groupby(["f", "yr"]).apply(lambda d: d.t.corr(d.y, method="spearman"), include_groups=False).rename("ic").reset_index()
        feat = {}
        for f, d in ic.groupby("f"):
            d = d.sort_values("yr").set_index("yr").ic
            for t_ in range(int(yr.min()), int(yr.max()) + 1):     # value used for rows of year t_: realised ICs of years <= t_-1 only
                past = d[d.index <= t_ - 1]
                feat[(f, t_)] = (past.ewm(halflife=3).mean().iloc[-1], past.tail(5).mean()) if len(past) else (0.0, 0.0)
        R = np.array([feat[(f, t_)] for f, t_ in zip(D.famcode, yr)], np.float32)
        E = np.c_[E, R]
    if cfg.get("fx", "none") == "rank":                       # cross-sectional rank-gauss of each continuous feature inside its family-year
        from scipy.stats import norm
        gk0 = D.famcode.astype(np.int64) * 10000 + yr
        cols = [j for j in range(E.shape[1]) if len(np.unique(E[:2000, j][np.isfinite(E[:2000, j])])) > 2]
        r = pd.DataFrame(E[:, cols]).groupby(gk0).rank(pct=True).values
        E = E.copy(); E[:, cols] = norm.ppf(np.clip(r, 0.001, 0.999)).astype(np.float32)
    if cfg.get("fx", "none") == "dm":                         # remove the year drift of the extras (e.g. the oriented in-sample mean averages
        gk0 = D.famcode.astype(np.int64) * 10000 + yr               # 0.63 in 1983, 0.83 in 2003): demeaned within the family-year, magnitudes kept
        E = (E - pd.DataFrame(E).groupby(gk0).transform("mean").values).astype(np.float32)
    ysc = (D.sd if cfg["prep"] in ("ysd", "ysdm") else np.ones(len(yr))).astype(np.float32); y = (D.y_or / ysc).astype(np.float32)
    clipv = 3 if cfg["prep"] == "clip3" else 10
    gk = D.famcode.astype(np.int64) * 10000 + yr
    ytar = (y - pd.Series(y).groupby(gk).transform("mean").values).astype(np.float32) if cfg["prep"] in ("ydm", "ydms", "ysdm") else y
    pred = np.full(len(yr), np.nan, np.float32); net = None; prev_state = None; opt_prev = None
    parts = {}; D._parts = parts
    rc = cfg.get("rc", False)
    if rc:
        assert cfg["xs"] and cfg["loss"] != "studentt" and cfg.get("xset", XS_SET) >= np.bincount(gk - gk.min()).max(), "rc needs xs batches holding a whole family-year"
    if cfg.get("base", "none") == "blend":                    # fixed-prior anchor: z(in-sample mean) + bw * z(own momentum) within the
        zf = lambda v: (v - pd.Series(v).groupby(gk).transform("mean").values) / (pd.Series(v).groupby(gk).transform("std").values + 1e-9)
        mom = ts_momentum(D)[:, :4]; zm = np.mean([zf(mom[:, k]) for k in range(4)], 0)             # family-year (fixed weights: learned
        lead = np.abs(D.F.tstat.values) if cfg["prep"] in ("ysd", "ysdm") else D.F.mean_is.values * D.sg   # weights overfit the
        Xb = (zf(lead) + cfg.get("bw", 0.5) * zf(zm))[:, None]                                       # year-varying momentum payoff)
    if cfg.get("base", "none") == "lin4":
        m_or = D.F.mean_is.values * D.sg
        Xb = np.c_[m_or, m_or / D.sd, np.abs(D.F.tstat.values), D.sd].astype(np.float64)
        Xb = Xb - pd.DataFrame(Xb).groupby(gk).transform("mean").values      # within family-year: the base only ranks, the level is separate
    _rc = (lambda o: o - o.mean()) if rc else (lambda o: o)
    n_out = 2 if cfg["loss"] == "studentt" else 1
    xset = cfg.get("xset", XS_SET)
    NF = int(D.famcode.max()) + 1                                    # 6 real families (+ simulated null families)
    fam_tr = fam_mask(D, cfg.get("fams", "all"))                      # families the model LEARNS from
    fam_pr = fam_tr | fam_mask(D, cfg.get("infer", "none"))          # families it predicts (infer: predicted, never trained on)
    loss_fn_ = lambda kind, o, yy_: loss_fn(kind, o, yy_, cfg.get("hdelta", 1.0))
    sd_lab = torch.tensor((D.sd / ysc).astype(np.float32), device=DEV)    # in-sample sd of each strategy, in label units
    nv = (D.F.s24.values / D.sd) ** 2 / 12 if "s24" in D.F else np.ones(len(yr)) / 12
    rk = pd.Series(np.abs(D.F.tstat.values)).groupby(D.famcode.astype(np.int64) * 10000 + yr).rank(pct=True).values   # |t| rank in the family-year
    hw_t = torch.tensor((1.0 + cfg.get("hw", 0) * (rk >= 0.9)).astype(np.float32), device=DEV)
    nv_t = torch.tensor(np.where(np.isfinite(nv), nv, np.nanmedian(nv)).astype(np.float32), device=DEV)

    def batch_loss(o, j, yy=None):
        """Training loss of one batch.  wls: inverse-variance weights 1/(sd^2/12 + tau^2) -- the label is a 12-month mean, so its noise variance
        is sd^2/12 and noisy strategies should count less (validation loss stays unweighted).  corrw: + lam*(1 - corr(pred, y)) inside the batch,
        i.e. inside one family-year when xs is on, so the loss also rewards ranking the strategies correctly."""
        yy = yt[j] if yy is None else yy
        if cfg.get("hw", 0) and cfg["loss"] in ("huber", "mse") and o.dim() == 1:   # head weight: the paper keeps only the top of the ranking,
            el = nn.functional.huber_loss(o, yy, delta=cfg.get("hdelta", 1.0), reduction="none") if cfg["loss"] == "huber" else (o - yy) ** 2
            w = hw_t[j]; base = (el * w).sum() / w.sum()           # so errors on the strategies most likely to be picked count (1 + hw) times
        elif cfg.get("nw", 0) and cfg["loss"] in ("huber", "mse") and o.dim() == 1:   # heteroscedastic Gaussian likelihood: noise variance of a
            el = nn.functional.huber_loss(o, yy, delta=cfg.get("hdelta", 1.0), reduction="none") if cfg["loss"] == "huber" else (o - yy) ** 2
            w = 1.0 / (nv_t[j] + cfg["nw"]); base = (el * w / w.mean()).mean()   # 12-month mean in Sharpe units = (s24/sd_IS)^2/12, known before Y
        elif cfg.get("wls", 0) and cfg["loss"] in ("huber", "mse") and o.dim() == 1:
            el = nn.functional.huber_loss(o, yy, delta=cfg.get("hdelta", 1.0), reduction="none") if cfg["loss"] == "huber" else (o - yy) ** 2
            w = 1.0 / (sd_lab[j] ** 2 / 12.0 + cfg["wls"]); base = (el * w / w.mean()).mean()
        else:
            base = loss_fn_(cfg["loss"], o, yy)
        if cfg.get("fyw", "none") != "none":
            base = base * wfy_t[j[0]]                               # one xs batch = one family-year
        if cfg.get("corrw", 0):
            p = o if o.dim() == 1 else o[:, 0]
            if cfg.get("corru", "ret") == "tot":                    # correlation of the WHOLE within-year forecast (base + network) with the
                p, yy = p + base_t[j], ytot_t[j]                    # demeaned label (the network alone only forecasts the residual of the base)
            elif cfg.get("corru", "ret") == "sr":                     # correlation in SHARPE units (forecast and label / in-sample sd): the paper ranks by predicted Sharpe
                p, yy = (p + lvl_t[j]) / sd_lab[j], (yt0[j]) / sd_lab[j]
            pc, yc = p - p.mean(), yy - yy.mean()
            base = base + cfg["corrw"] * (1 - (pc * yc).sum() / (pc.norm() * yc.norm() + 1e-8))
        return base
    for Y in years:
        nval = cfg.get("nval", 1)                                  # validation = the last `nval` years before Y (default 1: Y-1); training years <= Y-1-nval
        trn = np.where((yr <= Y - 1 - nval) & (yr >= cfg.get("trmin", 0)) & np.isfinite(y) & fam_tr)[0]; val = np.where((yr >= Y - nval) & (yr <= Y - 1) & fam_tr)[0]; te = np.where((yr == Y) & fam_pr)[0]
        if len(trn) < cfg.get("minrows", 20000) or len(te) == 0:
            continue
        if len(trn) > max_train:
            trn = trn[np.argsort(-yr[trn], kind="stable")[:max_train]]
        frozen = bool(cfg.get("freeze")) and Y > cfg["freeze"] and prev_state is not None      # freeze=Y0: no learning after Y0 (train-once control)
        if not frozen:
            mu, sd_ = np.nanmean(E[trn], 0), np.nanstd(E[trn], 0) + 1e-6                 # train-only statistics
            lo, hi = np.percentile(ytar[trn], [1, 99])
        Et = torch.tensor(np.nan_to_num(np.clip((E - mu) / sd_, -clipv, clipv)).astype(np.float32), device=DEV)
        yt = torch.tensor(np.clip(ytar, lo, hi).astype(np.float32), device=DEV)
        yt0 = torch.tensor(np.clip(y, *np.percentile(y[trn], [1, 99])).astype(np.float32), device=DEV)
        cfam = np.array([y[trn][D.famcode[trn] == f].mean() if (D.famcode[trn] == f).any() else 0.0 for f in range(NF)], np.float32) if cfg["prep"] in ("ydm", "ysdm") else np.zeros(NF, np.float32)
        srl = (y / ysc / D.sd).astype(np.float32)                    # realised Sharpe of each row (label / in-sample sd)
        csr = np.array([srl[trn][D.famcode[trn] == f].mean() if (D.famcode[trn] == f).any() else 0.0 for f in range(NF)], np.float32) if cfg["prep"] == "ydms" else np.zeros(NF, np.float32)
        unseen = ~np.isin(np.arange(NF), D.famcode[trn])              # inferred families: pooled level of the training families, not 0
        if cfg["prep"] in ("ydm", "ysdm"): cfam[unseen] = y[trn].mean()          # (a level of 0 would push them out of the top-k by construction)
        if cfg["prep"] == "ydms": csr[unseen] = srl[trn].mean()
        lvl = cfam[D.famcode] + csr[D.famcode] * D.sd / ysc
        if cfg.get("lvl", "fam") == "breadth" and cfg["prep"] in ("ydm", "ydms"):   # level = linear fit of the family-year mean label on the
            Bf = D.F[["fam_frac_t2", "fam_mean_abst"]].values         # family's breadth of significant signals (share of |t|>2, mean |t| over ALL
            lab = y if cfg["prep"] == "ydm" else srl                    # its strategies), so an unseen family gets a level from its own statistics
            g = pd.DataFrame(dict(k=gk[trn], y=lab[trn], a=Bf[trn, 0], b=Bf[trn, 1])).groupby("k").mean()   # (ydms: in Sharpe units, times sd)
            w_l = np.linalg.lstsq(np.c_[g.a, g.b, np.ones(len(g))], g.y.values, rcond=None)[0]
            lvl = (np.c_[Bf, np.ones(len(Bf))] @ w_l).astype(np.float32) * (1.0 if cfg["prep"] == "ydm" else D.sd / ysc)
        lm = cfg.get("lvl", "fam")
        if lm in ("exp1", "roll10", "roll20") and cfg["prep"] in ("ydm", "ysdm"):   # the level may use the labels of Y-1 (known before Y)
            okl = (yr <= Y - 1) & np.isfinite(y) & fam_tr & ((yr >= Y - int(lm[4:])) if lm.startswith("roll") else True)
            cf = np.array([y[okl & (D.famcode == f)].mean() if (okl & (D.famcode == f)).any() else y[okl].mean() for f in range(NF)], np.float32)
            lvl = cf[D.famcode]
        if lm.startswith("ratio") and cfg["prep"] == "ysdm":       # level in Sharpe units = c * mean in-sample Sharpe of the family-year's candidates,
            isr = D.F.mean_is.values * D.sg / D.sd                     # c = sum of realised / sum of in-sample family-year mean Sharpe over the last
            fyr = pd.DataFrame(dict(k=gk, f=D.famcode, yr=yr, y=y, x=isr)).groupby("k").agg(f=("f", "first"), yr=("yr", "first"), y=("y", "mean"), x=("x", "mean"))
            w_ = int(lm[5:]) if len(lm) > 5 else 999                     # `ratio<w>` years <= Y-1 (EB-free shrinkage of the in-sample level, re-estimated
            pst = fyr[(fyr.yr <= Y - 1) & (fyr.yr >= Y - w_) & fyr.f.isin(np.unique(D.famcode[trn]))].dropna()   # every year: tracks the post-2004 decay)
            lvl = ((pst.y.sum() / pst.x.sum()) * fyr.x.reindex(gk).values * D.sd / ysc).astype(np.float32)
        if lm == "mom" and cfg["prep"] in ("ydm", "ysdm"):            # family momentum: the family-year mean label regressed on the mean return of
            fy = pd.DataFrame(dict(k=gk, f=D.famcode, yr=yr, y=y, m=D.F.m12.values / ysc)).groupby("k").agg(f=("f", "first"), yr=("yr", "first"), y=("y", "mean"), m=("m", "mean"))
            p_ = fy[(fy.yr <= Y - 1) & fy.f.isin(np.unique(D.famcode[trn]))].dropna()   # its candidates over the last 12 months (family-years <= Y-1)
            w_m = np.polyfit(p_.m.values, p_.y.values, 1)
            lvl = np.polyval(w_m, fy.m.reindex(gk).values).astype(np.float32)
        base = np.zeros(len(yr), np.float32)
        if cfg.get("base", "none") in ("lin4", "blend"):
            fb = np.where((yr <= Y - 1) & (yr >= Y - cfg.get("bK", 15)) & np.isfinite(ytar) & fam_tr)[0]
            sb = Xb[fb].std(0) + 1e-9; A_ = Xb[fb] / sb; yb = np.clip(ytar[fb], *np.percentile(ytar[fb], [1, 99]))
            wb = np.linalg.solve(A_.T @ A_ + cfg.get("bL", 1.0) * np.eye(A_.shape[1]), A_.T @ yb)
            base = ((Xb / sb) @ wb).astype(np.float32)
            r_ = ytar - base; lo, hi = np.percentile(r_[trn], [1, 99])
            yt = torch.tensor(np.clip(r_, lo, hi).astype(np.float32), device=DEV)
        lvl_t = torch.tensor(lvl.astype(np.float32), device=DEV)
        base_t = torch.tensor(base, device=DEV); ytot_t = torch.tensor(np.clip(ytar, *np.percentile(ytar[trn], [1, 99])).astype(np.float32), device=DEV)
        if cfg.get("fyw", "none") != "none":               # family-year weight from the dispersion of its TRAINING labels (Holen et al. L-weighting
            v_ = pd.Series(np.clip(ytar, *np.percentile(ytar[trn], [1, 99])) ** 2).groupby(gk).transform("mean").values   # with the loss replaced by
            r_ = v_ / pd.Series(v_[trn]).groupby(gk[trn]).first().mean()                                      # its expectation v: no model feedback)
            w0 = {"vH": 0.5 * (1 + r_), "v": r_, "inv": 1 / r_}[cfg["fyw"]]                                  # vH = E[l/mean(l) + 1/K] scaled to mean 1
            wfy_t = torch.tensor((w0 / pd.Series(w0[trn]).groupby(gk[trn]).first().mean()).astype(np.float32), device=DEV)
        torch.manual_seed(seed * 1000 + Y if cfg.get("seed2", False) else seed + Y)
        if net is not None and cfg.get("warm", "chain") == "cold":
            net = None
        elif net is not None and cfg.get("warm", "chain") == "sp":
            with torch.no_grad():
                for p_ in net.parameters(): p_.mul_(cfg.get("sp_lam", 0.8)).add_(cfg.get("sp_sig", 0.01) * torch.randn_like(p_))
        if net is None:
            net = BaseNet(Hx.shape[2], E.shape[1], cfg["seq"], cfg["pool"], cfg["hid"], 64, cfg["drop"], n_out, cfg["xs"], cfg.get("xl", 1), cfg.get("xk", "full"), cfg.get("xm", 32), cfg.get("wide", False), cfg.get("hnorm", "bn"), cfg.get("xfuse", False), cfg.get("gate", False), cfg.get("hln", True), cfg.get("hblock", 256), cfg.get("hm", 256), cfg.get("head", "fc")).to(DEV)
            if cfg.get("base", "none") != "none":                     # residual on a closed-form base: start from "no correction"
                with torch.no_grad(): net.head[-1].weight.zero_(); net.head[-1].bias.zero_()
            opt_prev = None
        opt = opt_prev if (cfg.get("optkeep", False) and opt_prev is not None) else torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        opt_prev = opt
        sch = None
        ema = copy.deepcopy(net).eval() if cfg.get("ema", 0) else None

        Klw0 = (cfg.get("lw", 0) or cfg.get("rw", 0)) if cfg["xs"] else 0

        def ema_step():
            if sch is not None:
                sch.step()
            if ema is not None:
                with torch.no_grad():
                    for pe, p_ in zip(ema.parameters(), net.parameters()): pe.mul_(cfg["ema"]).add_(p_.detach(), alpha=1 - cfg["ema"])
                    for be, b_ in zip(ema.buffers(), net.buffers()): be.copy_(b_)
        if cfg["aug"] == "recent":
            trn = np.r_[trn, trn[yr[trn] > Y - 10]]
        trn_t, val_t, te_t = (torch.tensor(a, device=DEV) for a in (trn, val, te))
        grp = torch.tensor(D.famcode.astype(np.int64) * 10000 + yr, device=DEV)
        fwd = lambda j, noise=0.0: (ema if ema is not None else net)((Hx[j].float() if not noise else Hx[j].float() + noise * torch.randn_like(Hx[j].float())), Et[j])
        best, best_state = 1e18, None
        val_loss = lambda: float(sum(loss_fn_(cfg["loss"], _rc(fwd(j)), yt[j]) * len(j) for j in batches(val_t, grp, xset if cfg["xs"] else 8192, train=False, by_group=cfg["xs"], det_eval=cfg.get("detev", False))) / max(len(val_t), 1))
        if cfg.get("ep0", False) and prev_state is not None:          # the warm-started state is a candidate too (no forced update every year)
            net.eval()
            with torch.no_grad(): best = val_loss()
            best_state = {k: v.clone() for k, v in (ema if ema is not None else net).state_dict().items()}
        if frozen:
            best_state = prev_state
        for ep in range(0 if frozen else cfg["epochs"]):
            net.train()
            bl = list(batches(trn_t, grp, xset if cfg["xs"] else bs, train=True, by_group=cfg["xs"], shuf=cfg.get("shuf", False)))
            if cfg.get("sched", "const") == "cos" and sch is None:          # 5% linear warm-up then cosine to 0 over this year's steps
                tot = max(1, cfg["epochs"] * -(-len(bl) // max(Klw0, 1)) * (2 if cfg["aug"] in ("up2", "mix2") else 1)); wu = max(1, tot // 20)
                sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / wu) * 0.5 * (1 + np.cos(np.pi * min(1.0, k / tot))))
            Klw = (cfg.get("lw", 0) or cfg.get("rw", 0)) if cfg["xs"] else 0
            for ci in range(0, len(bl), max(Klw, 1)):
                blk = bl[ci:ci + max(Klw, 1)]
                if Klw > 1 and len(blk) > 1:        # weighted gradient merge of K family-years ("workers"), Holen et al. arXiv 2304.12778
                    outs = [_rc(net(Hx[j].float(), Et[j])) for j in blk]
                    ls = [batch_loss(o_, j) for o_, j in zip(outs, blk)]
                    if cfg.get("rw", 0):            # R-Weighted: reward = realised mean return of the top 10% by |prediction| (signed), TRAINING labels only.
                        r = []                      # EXPLORATORY: puts a portfolio-type quantity into the loss weights (violates rule A).
                        for o_, j in zip(outs, blk):
                            p = (o_ if o_.dim() == 1 else o_[:, 0]).detach(); kk = max(1, len(p) // 10); top = torch.topk(p.abs(), kk).indices
                            r.append(float((torch.sign(p[top]) * yt[j][top]).mean()))
                        r = torch.tensor(r); r = r - r.min(); wts = r / r.sum().clamp(min=1e-12) + 1.0 / len(blk)
                    else:                           # L-Weighted: weight = loss_i / sum(loss) + 1/K
                        lt = [l_.detach() if cfg.get("lwm", "l") == "l" else torch.ones_like(l_.detach()) for l_ in ls]   # lwm=eq: same K-merge, equal weights
                        tot = sum(lt).clamp(min=1e-12); wts = [x / tot + 1.0 / len(blk) for x in lt]
                    l = sum(w_ * l_ for w_, l_ in zip(wts, ls))
                    opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                    ema_step()
                    continue
                for j in blk:
                  for pas in range(2 if cfg["aug"] in ("up2", "mix2") else 1):
                    hs, ex, yy = Hx[j].float(), Et[j], yt[j]
                    if cfg["aug"] == "noise" or (cfg["aug"] == "up2" and pas == 1):
                        hs = hs + 0.1 * torch.randn_like(hs)
                        if cfg["aug"] == "up2": ex = ex + 0.05 * torch.randn_like(ex)
                    elif cfg["aug"] == "mix2" and pas == 1:
                        lam = torch.distributions.Beta(0.4, 0.4).sample((len(j), 1)).to(DEV); p = torch.randperm(len(j), device=DEV)
                        hs = lam[:, :, None] * hs + (1 - lam[:, :, None]) * hs[p]; ex = lam * ex + (1 - lam) * ex[p]; yy = lam[:, 0] * yy + (1 - lam[:, 0]) * yy[p]
                    l = batch_loss(_rc(net(hs, ex)), j, yy)
                    opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                    ema_step()
            net.eval()
            with torch.no_grad():
                vl = val_loss() if len(val_t) else -float(ep)           # nval=0: no validation year, the last epoch is kept
            if vl < best:
                best, best_state = vl, {k: v.clone() for k, v in (ema if ema is not None else net).state_dict().items()}
        if best_state is None:                      # training diverged (NaN validation loss): fall back to last year's weights
            assert prev_state is not None, f"training diverged in the first year {Y}"
            print(f"  [warn] year {Y}: non-finite validation loss, reusing the previous year's weights", flush=True)
            best_state = prev_state
        net.load_state_dict(best_state); net.eval()
        prev_state = {k: v.clone() for k, v in best_state.items()}
        a_cal, b_cal = 1.0, 0.0
        if cfg["cal"] == "val":
            with torch.no_grad():
                jl = list(batches(val_t, grp, xset if cfg["xs"] else 8192, train=False, by_group=cfg["xs"], det_eval=cfg.get("detev", False)))
                ov = torch.cat([_rc(fwd(j)) for j in jl]); jv = torch.cat(jl)
                pv = (ov[:, 0] if n_out == 2 else ov).double(); yv = yt[jv].double()
                a_cal = float(((pv - pv.mean()) * (yv - yv.mean())).mean() / pv.var(unbiased=False).clamp(min=1e-12)); a_cal = min(max(a_cal, 0.0), 1.5)
                b_cal = float(yv.mean() - a_cal * pv.mean())
        with torch.no_grad():
            js = list(batches(te_t, grp, xset if cfg["xs"] else 8192, train=False, by_group=cfg["xs"], det_eval=cfg.get("detev", False)))
            o = torch.cat([_rc(fwd(j)) for j in js]); o = o[:, 0] if n_out == 2 else o
            ti = torch.cat(js).cpu().numpy(); pred[ti] = (a_cal * np.clip(o.cpu().numpy(), lo, hi) + b_cal + lvl[ti] + base[ti]) * ysc[ti]      # winsorise the output at the train-label 1%/99% bounds
            if cfg.get("parts", False):                       # keep the three additive parts (label units x ysc) so the network correction can be rescaled afterwards
                if "res" not in parts: parts.update(res=np.full(len(yr), np.nan, np.float32), base=np.full(len(yr), np.nan, np.float32), lvl=np.full(len(yr), np.nan, np.float32))
                parts["res"][ti] = (a_cal * np.clip(o.cpu().numpy(), lo, hi) + b_cal) * ysc[ti]; parts["base"][ti] = base[ti] * ysc[ti]; parts["lvl"][ti] = lvl[ti] * ysc[ti]
    D._last = dict(net=net, mu=mu, sd=sd_, clipv=clipv, E=E, Hx=Hx)           # last trained network of the walk (used by trace_blocks.py)
    return pred


# ------------------------------------------------------------------------------------------ evaluation
def accuracy(D, pred):
    """MSE of the forecast vs EB on candidate rows (oriented %/month): % reduction and Diebold-Mariano t (year-clustered), dev and holdout."""
    m = np.isfinite(pred); d = pd.DataFrame(dict(yr=D.yr[m], e_ai=(pred[m] - D.y_or[m]) ** 2, e_eb=(D.eb_or[m] - D.y_or[m]) ** 2))
    out = {}
    for nm, lo, hi in (("dev", 1983, 2004), ("hold", 2005, 2020)):
        g = d[(d.yr >= lo) & (d.yr <= hi)].groupby("yr")[["e_ai", "e_eb"]].mean()
        diff = g.e_ai - g.e_eb
        out[f"mse_vs_eb_{nm}"] = 100 * (1 - g.e_ai.sum() / g.e_eb.sum())
        out[f"dm_t_{nm}"] = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))))
    return out


def make_encoder(D, name):
    """4- or 8-token representation of the oriented return window.  raw: no pretrained model; chronos*: Chronos-Bolt; chronos2*: Chronos-2;
    the suffix _240 feeds the whole 240-month in-sample window (what EB uses) instead of the last 128 months."""
    if name == "lp7":
        return logpatch_tokens(D)
    if name == "raw":
        return A.raw_patch_hidden(D)
    if name == "raw240":                                      # 240 months -> 8 patches of 30, volatility-normalised
        W = A.oriented_windows(D, ctx=240) / D.sd[:, None]
        return torch.tensor(np.nan_to_num(W, nan=0.0).reshape(len(W), 8, 30), dtype=torch.float16, device=DEV)
    if name in ("chronos", "chronos_base"):
        return A.chronos_hidden(D, "amazon/chronos-bolt-base" if name == "chronos_base" else "amazon/chronos-bolt-small")
    if name.startswith("sub:"):                                   # any hidden-state file extracted for a family's rows: sub:<stem> ->
        f = h.CACHE / f"{name[4:]}.npy"; Z = np.load(f, mmap_mode="r"); ix = np.load(h.CACHE / f"{name[4:]}_idx.npy")   # _cache/<stem>.npy
        pos = pd.Series(np.arange(len(ix)), index=ix).reindex(D.idx_full).values; assert not np.isnan(pos).any(), "rows missing"
        return torch.from_numpy(np.ascontiguousarray(Z[pos.astype(int)])).to(DEV)
    if name in ("tirex_sub", "tirex240_sub"):                      # TiRex hidden states extracted for the rows of D only (extract_tirex.py ... sub)
        f = h.CACHE / ("tirex240_sub.npy" if "240" in name else "tirex128_sub.npy"); Z = np.load(f, mmap_mode="r"); ix = np.load(str(f).replace(".npy", "_idx.npy"))
        pos = pd.Series(np.arange(len(ix)), index=ix).reindex(D.idx_full).values; assert not np.isnan(pos).any(), "TiRex rows missing"
        return torch.from_numpy(np.ascontiguousarray(Z[pos.astype(int)])).to(DEV)
    if name in ("tirex", "tirex240"):                          # TiRex (xLSTM, Auer et al. arXiv 2505.23719), embedded offline by extract_tirex.py
        f = h.CACHE / ("tirex240.npy" if name.endswith("240") else "tirex128.npy")
        return torch.from_numpy(np.load(f)).to(DEV)
    if name in ("chronos2", "chronos2_240"):
        from chronos import Chronos2Pipeline
        pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
        W = A.oriented_windows(D, ctx=240 if name.endswith("_240") else 128)
        outs = []
        with torch.no_grad():
            for st in range(0, len(W), 4096):
                embs, _ = pipe.embed([torch.tensor(w) for w in W[st:st + 4096]], batch_size=1024)
                e = torch.cat([x.float() for x in embs], 0)
                T = e.shape[1] - (e.shape[1] % 4)
                outs.append(e[:, -T:].reshape(e.shape[0], 4, T // 4, e.shape[2]).mean(2).half())
        del pipe; torch.cuda.empty_cache()
        return torch.cat(outs)
    raise ValueError(name)


def use_encoder(D, name):
    """Swap D.H for another encoder's 4-token representation.  Only the active encoder lives on the GPU; the others are parked on the CPU.
    fincast is the diagram."""
    if not hasattr(D, "_enc"):
        D._enc, D._cur = {"fincast": D.H}, "fincast"
    if name == D._cur:
        return
    D._enc[D._cur] = D._enc[D._cur].cpu()
    D.H = None
    torch.cuda.empty_cache()
    if name not in D._enc:
        D._enc[name] = make_encoder(D, name)
    D.H = D._enc[name].to(DEV)
    D._cur = name


def ref_window_ic(D, pred):
    """Mean yearly Spearman IC of pred vs realised oriented return on 2014-2020, per family, over the candidate rows we score
    (the reference notebook scores every strategy of the family, so this is NOT identical to its IC)."""
    out = {}
    for fam in ("acct_ew", "past_ret_ew"):
        m = np.isfinite(pred) & (D.F.family.values == fam) & (D.yr >= 2014)
        d = pd.DataFrame(dict(yr=D.yr[m], p=pred[m], y=D.y_or[m]))
        out[f"ic_{fam}_1420"] = float(d.groupby("yr").apply(lambda g: g.p.corr(g.y, method="spearman")).mean())
    return out


def run(cfg, D, P):
    t0 = time.time()
    use_encoder(D, cfg["enc"])
    pred = walk_forward(D, cfg, seed=cfg.get("seed", 0))
    import hashlib
    nm = "|".join(f"{k}={cfg[k]}" for k in sorted(cfg)); PRED_DIR.mkdir(exist_ok=True)
    np.save(PRED_DIR / (hashlib.md5(nm.encode()).hexdigest() + ".npy"), pred)
    res, _ = A.evaluate(P, D, pred, "T0", "sr", 1.0, schemes=("eq",))
    row = dict(name="|".join(f"{k}={cfg[k]}" for k in sorted(cfg)), secs=round(time.time() - t0), **accuracy(D, pred), **ref_window_ic(D, pred))
    for (r, sc, per), (m, s) in res.items():
        row[f"top{r}_{per}_mean"], row[f"top{r}_{per}_sr"] = m, s
    append_row(row)
    print(f"[{row['secs']}s] dev MSE vs EB {row['mse_vs_eb_dev']:+.1f}% (DM {row['dm_t_dev']:.2f}) | top1 SR dev {row['top1_dev_sr']:.2f} hold {row['top1_hold_sr']:.2f} :: {row['name']}", flush=True)
    return pred, row


BASE = dict(extras="E0", seq="bilstm", pool="attn", loss="huber", aug="none", xs=False, enc="fincast", prep="none", cal="none", hid=128, drop=0.3, lr=1e-3, wd=1e-2, epochs=3, seed=0)


def pred_of(cfg):
    import hashlib
    nm = "|".join(f"{k}={cfg[k]}" for k in sorted(cfg))
    return np.load(PRED_DIR / (hashlib.md5(nm.encode()).hexdigest() + ".npy"))


def ensemble_row(D, P, cfgs, label, agg="mean"):
    """Average the forecasts of several already-trained configs (typically seeds of one config) and score the average like any other run."""
    pred = (np.median if agg == "median" else np.mean)([pred_of(c) for c in cfgs], 0)
    res, _ = A.evaluate(P, D, pred, "T0", "sr", 1.0, schemes=("eq",))
    row = dict(name=f"ENSEMBLE{len(cfgs)}{agg[:3]}|{label}", secs=0, **accuracy(D, pred), **ref_window_ic(D, pred))
    for (r, sc, per), (m, s) in res.items():
        row[f"top{r}_{per}_mean"], row[f"top{r}_{per}_sr"] = m, s
    append_row(row)
    print(f"[ens] dev MSE vs EB {row['mse_vs_eb_dev']:+.1f}% (DM {row['dm_t_dev']:.2f}) hold {row['mse_vs_eb_hold']:+.1f}% | top1 SR dev {row['top1_dev_sr']:.2f} hold {row['top1_hold_sr']:.2f} :: {row['name']}", flush=True)
    return pred, row


# ------------------------------------------------------------------------------------------ evaluation on a family subset (direction of 2026-10-03)
_EB_CACHE = {}


N_FULL = 655101                                                      # number of candidate rows of the full data (length of every saved prediction file)


def to_full(D, pred):
    """Prediction of a light (family-restricted) data object -> full-length array (NaN elsewhere), so every saved file has the same layout."""
    if not hasattr(D, "idx_full") or len(pred) == N_FULL: return pred
    k = D.idx_full >= 0                                              # simulated null rows (idx_full = -1) are not saved
    out = np.full(N_FULL, np.nan, np.float32); out[D.idx_full[k]] = pred[k]; return out


def to_local(D, pred):
    if not (hasattr(D, "idx_full") and len(pred) == N_FULL): return pred
    return np.where(D.idx_full >= 0, pred[np.maximum(D.idx_full, 0)], np.nan)


def eval_subset(D, P, pred, fams, label=""):
    """Forecast accuracy and top-k portfolios of `pred` vs EB on the SAME families.  Two universes:
    cand = only the candidate rows of these families (the AI decides every pick); all = every strategy of these families (paper style, EB fills
    the non-candidates).  Returns a flat dict."""
    import ht_new as nw
    m = fam_mask(D, fams)
    p = np.where(m, pred, np.nan).astype(np.float32)
    row = dict(**accuracy(D, p))
    keys = pd.DataFrame(dict(family=D.F.family.values[m], signalid=D.F.signalid.values[m], oos_year=D.yr[m]))
    P_all = P[P.family.isin(fams.split("+") if fams != "all" else P.family.unique())]
    P_cand = P_all.merge(keys, on=["family", "signalid", "oos_year"], how="inner")
    for uni, PU in (("cand", P_cand), ("all", P_all)):
        ck = (fams, uni)
        if ck not in _EB_CACHE:
            _EB_CACHE[ck] = A.evaluate(PU, D, np.where(m, D.eb_or, np.nan).astype(np.float32), "T0", "sr", 1.0, schemes=("eq",))
        eb_out, eb_rets = _EB_CACHE[ck]
        ai_out, ai_rets = A.evaluate(PU, D, p, "T0", "sr", 1.0, schemes=("eq",))
        for (r, sc, per), (mn, sr) in ai_out.items():
            row[f"{uni}_top{r}_{per}_sr"], row[f"{uni}_top{r}_{per}_mean"] = sr, mn
            row[f"{uni}_top{r}_{per}_sr_eb"], row[f"{uni}_top{r}_{per}_mean_eb"] = eb_out[(r, sc, per)][1], eb_out[(r, sc, per)][0]
    return row


def run_sub(cfg, D, P):
    """Train on cfg['fams'], evaluate on the same families; append to fam_results.tsv."""
    import hashlib
    t0 = time.time(); use_encoder(D, cfg.get("enc", "fincast"))
    pred = walk_forward(D, cfg, seed=cfg.get("seed", 0))
    nm = "|".join(f"{k}={cfg[k]}" for k in sorted(cfg)); PRED_DIR.mkdir(exist_ok=True)
    np.save(PRED_DIR / (hashlib.md5(nm.encode()).hexdigest() + ".npy"), to_full(D, pred))
    row = dict(name=nm, secs=round(time.time() - t0), **eval_subset(D, P, pred, cfg.get("fams", "all")))
    new = pd.DataFrame([row])
    if RES2.exists(): new = pd.concat([pd.read_csv(RES2, sep="\t"), new], ignore_index=True)
    new.to_csv(RES2, sep="\t", index=False)
    print(f"[{row['secs']}s] {cfg.get('fams')} | dev MSE vs EB {row['mse_vs_eb_dev']:+.1f}% (DM {row['dm_t_dev']:.2f}) hold {row['mse_vs_eb_hold']:+.1f}% | "
          f"cand top1 SR dev {row['cand_top1_dev_sr']:.2f} (EB {row['cand_top1_dev_sr_eb']:.2f}) hold {row['cand_top1_hold_sr']:.2f} (EB {row['cand_top1_hold_sr_eb']:.2f}) :: {nm}", flush=True)
    return pred, row


def ensemble_sub(D, P, cfgs, label, agg="median"):
    pred = (np.median if agg == "median" else np.mean)([to_local(D, pred_of(c)) for c in cfgs], 0)
    fams = cfgs[0].get("fams", "all")
    row = dict(name=f"ENSEMBLE{len(cfgs)}{agg[:3]}|{label}", secs=0, **eval_subset(D, P, pred, fams))
    new = pd.DataFrame([row])
    if RES2.exists(): new = pd.concat([pd.read_csv(RES2, sep="\t"), new], ignore_index=True)
    new.to_csv(RES2, sep="\t", index=False)
    print(f"[ens] {fams} | dev MSE vs EB {row['mse_vs_eb_dev']:+.1f}% (DM {row['dm_t_dev']:.2f}) hold {row['mse_vs_eb_hold']:+.1f}% | "
          + " ".join(f"cand top{r} SR full {row[f'cand_top{r}_full_sr']:.2f}/EB {row[f'cand_top{r}_full_sr_eb']:.2f}" for r in (1, 5, 10)) + f" :: {row['name']}", flush=True)
    return pred, row
