"""Thử nghiệm: train.walk_forward của mô hình chốt + đúng MỘT thay đổi theo cấu hình (chỉ dùng để tìm cấu hình tốt hơn).
Mọi thứ không ghi trong cfg giống hệt mô hình chốt (code/train.py)."""
import sys
from pathlib import Path
CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
import numpy as np, pandas as pd, torch, torch.nn as nn
import data, train
from model import FinalNet, ISAB
from blocks_x import HyperSetAttention

DEV = "cuda"


class Lion(torch.optim.Optimizer):
    """Lion (Chen et al., NeurIPS 2023, 'Symbolic Discovery of Optimization Algorithms'): cập nhật = dấu của trung bình trượt gradient."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None: continue
                st = self.state[p]
                if "m" not in st: st["m"] = torch.zeros_like(p)
                m = st["m"]
                p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(torch.sign(m * b1 + p.grad * (1 - b1)), alpha=-g["lr"])
                m.mul_(b2).add_(p.grad, alpha=1 - b2)


class Multi:
    """Gộp nhiều optimizer (Muon cho ma trận 2 chiều, AdamW cho phần còn lại)."""

    def __init__(self, opts): self.opts = opts
    def zero_grad(self): [o.zero_grad() for o in self.opts]
    def step(self): [o.step() for o in self.opts]


def make_opt(net, cfg):
    kind, lr, wd = cfg.get("opt", "adamw"), cfg.get("lr", train.LR), cfg.get("wd", train.WD)
    if kind == "lion":
        return Lion(net.parameters(), lr=lr / 3, weight_decay=wd * 3)          # khuyến nghị của bài: lr nhỏ hơn 3-10 lần, wd lớn hơn
    if kind == "muon":                                                          # Muon (Jordan et al. 2024; Liu et al., 'Muon is Scalable', 2025)
        mats = [p for n, p in net.named_parameters() if p.ndim == 2 and "head.4" not in n and "ind" not in n]
        ids = {id(p) for p in mats}; rest = [p for p in net.parameters() if id(p) not in ids]
        return Multi([torch.optim.Muon(mats, lr=0.02, weight_decay=wd), torch.optim.AdamW(rest, lr=lr, weight_decay=wd)])
    if kind == "sf":                                                            # Schedule-Free AdamW (Defazio et al., NeurIPS 2024; thắng AlgoPerf 2024)
        import schedulefree
        return schedulefree.AdamWScheduleFree(net.parameters(), lr=lr, weight_decay=wd, warmup_steps=0)
    if kind == "radam":                                                         # RAdam (Liu et al., ICLR 2020)
        return torch.optim.RAdam(net.parameters(), lr=lr, weight_decay=wd, decoupled_weight_decay=True)
    return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)


def extra_features(T, kinds, ids, wide):
    """Đặc trưng thêm (đều tính từ dữ liệu trước Y, nhân dấu): nobs, sd (log), t-stat và mean của 60 / 12 tháng gần nhất (/sd)."""
    cols = []
    if not kinds:
        return np.zeros((len(T), 0), np.float32)
    W = None
    if any(k in kinds for k in ("t60", "m12", "t24")):
        W = data.windows(T, ids, wide, ctx=60).astype(np.float64)
    sd = T.sd_is.values
    with np.errstate(all="ignore"):
        for k in kinds:
            if k == "nobs": cols.append(T.nobs.values / 240.0)
            elif k == "lsd": cols.append(np.log(sd))
            elif k in ("t60", "t24"):
                w = W[:, -int(k[1:]):]; n = np.sum(~np.isnan(w), 1); m = np.nanmean(w, 1); s = np.nanstd(w, 1)
                cols.append(np.nan_to_num(m / (s / np.sqrt(np.maximum(n, 1)) + 1e-9)))
            elif k == "m12": cols.append(np.nan_to_num(np.nanmean(W[:, -12:], 1) / sd))
            elif k in ("zt", "tsd", "f2"):          # vị trí t-stat trong phân phối t của CÙNG năm (thông tin EB dùng, nhưng không dùng đầu ra EB)
                at = np.abs(T.tstat.values); g = pd.Series(at).groupby(T.oos_year.values)
                if k == "zt": cols.append((at - g.transform("mean").values) / g.transform("std").values)
                if k == "tsd": cols.append(g.transform("std").values)                    # độ phân tán t của năm
                if k == "f2": cols.append(g.transform(lambda v: (v > 2).mean()).values)  # tỉ lệ chiến lược |t| > 2 trong năm
    return np.c_[tuple(cols)].astype(np.float32)


def aux_loss(kind, o, y):
    """Thành phần xếp hạng trong năm (cộng thêm vào Huber):
    listnet: cross-entropy giữa phân phối top-one softmax(y/τ) và softmax(o/τ) (Cao et al., ICML 2007);
    corr: 1 - tương quan Pearson trong năm."""
    if kind == "listnet":
        tau = y.std() + 1e-6
        return -(torch.softmax(y / tau, 0) * torch.log_softmax(o / tau, 0)).sum()
    if kind == "corr":
        oc, yc = o - o.mean(), y - y.mean()
        return 1 - (oc * yc).sum() / (oc.norm() * yc.norm() + 1e-8)
    if kind == "srank":                               # Rank IC khả vi (mục tiêu của LambdaRankIC 2026; xếp hạng mềm kiểu Blondel et al. ICML 2020)
        n = len(o); k = torch.randperm(n, device=o.device)[:2048]; oo, yy = o[k], y[k]
        r = torch.sigmoid((oo[:, None] - oo[None, :]) / (oo.std() * 0.1 + 1e-6)).sum(1)          # thứ hạng mềm của dự báo
        ry = (yy[:, None] > yy[None, :]).float().sum(1)                                          # thứ hạng thật
        rc_, yc_ = r - r.mean(), ry - ry.mean()
        return 1 - (rc_ * yc_).sum() / (rc_.norm() * yc_.norm() + 1e-8)
    if kind == "pair":                                # Feng et al. (TOIS 2019, RSR): hinge theo cặp, phạt khi thứ tự dự báo ngược thứ tự thật
        n = len(o); i = torch.randint(0, n, (50000,), device=o.device); j = torch.randint(0, n, (50000,), device=o.device)
        return torch.relu(-(o[i] - o[j]) * (y[i] - y[j])).mean()
    raise ValueError(kind)


def eval_batches(idx, grp, size):
    """Khi dự báo: chia mỗi năm thành các lô cỡ `size` (thứ tự ngẫu nhiên cố định), để cỡ tập attention lúc dự báo = lúc học."""
    g = grp[idx]; gen = torch.Generator(device="cpu").manual_seed(0)
    for u in torch.unique(g):
        rows = idx[g == u]; rows = rows[torch.randperm(len(rows), generator=gen).to(rows.device)]
        for i in range(0, len(rows), size):
            yield rows[i:i + size]


def walk_forward(T, H, cfg, seed=0, years=range(1983, 2021), max_train=500_000, ids=None, wide=None):
    S = train.Static(T)
    if cfg.get("label") == "raw":                  # học THẲNG lãi (nhân dấu), không đổi Sharpe, không trừ TB năm, không nền
        S.y = T.y.values.astype(np.float32); S.ytar = S.y.copy(); S.zt = np.zeros_like(S.zt); S.sd = np.ones(len(T))
    elif cfg.get("label") == "sharpe":             # học thẳng Sharpe, không trừ TB năm, không nền
        S.ytar = S.y.copy(); S.zt = np.zeros_like(S.zt)
    elif cfg.get("label") in ("rank", "rank_pos"):               # học THỨ HẠNG trong năm (0..1, trừ 0,5), không nền (Poh et al., learning to rank)
        S.ytar = (pd.Series(S.y).groupby(S.yr).rank(pct=True).values - 0.5).astype(np.float32); S.zt = np.zeros_like(S.zt)
    elif cfg.get("label") == "nobase":             # Sharpe trừ TB năm, không nền |t|
        S.zt = np.zeros_like(S.zt)
    if cfg.get("xsrank"):                           # đầu vào t-stat, mean -> thứ hạng trong năm, đưa về [-1, 1] (Gu-Kelly-Xiu 2020; Freyberger et al. 2020)
        S.E = ((pd.DataFrame(S.E).groupby(S.yr).rank(pct=True).values - 0.5) * 2).astype(np.float32)
    alt = None
    if cfg.get("lh"):                                 # Label Horizon Paradox (Song et al. 2026): học bằng nhãn kỳ hạn khác (chỉ trên tập học)
        h = cfg["lh"]; pos = pd.Series(np.arange(len(ids)), index=ids); r = pos.loc[T.signalid.values].values
        b = (T.oos_year.values - data.BASE_YEAR) * 12
        M = wide[r[:, None], np.minimum(b[:, None] + np.arange(h)[None, :], wide.shape[1] - 1)]
        with np.errstate(all="ignore"):
            ya = (np.nanmean(M, 1) * T.sgn.values / S.sd).astype(np.float32)
        alt = (ya - pd.Series(ya).groupby(S.yr).transform("mean").values).astype(np.float32)
    X = extra_features(T, cfg.get("feats", []), ids, wide)
    if X.shape[1]:
        S.E = np.c_[S.E, X].astype(np.float32)
    grp = torch.tensor(S.yr.astype(np.int64), device=DEV)
    rc = (lambda o: o) if cfg.get("label") in ("raw", "sharpe") else (lambda o: o - o.mean())
    pred = np.full(len(S.yr), np.nan, np.float32)
    net = opt = prev_state = ema = None
    lam, aux = cfg.get("lam", 0.0), cfg.get("aux")
    global LOG
    LOG = []
    EB_ = (lambda ix: eval_batches(ix, grp, cfg["sub"])) if cfg.get("evsub") else (lambda ix: train.year_batches(ix, grp, False))
    for Y in years:
        W = train.prepare_year(S, Y, max_train)
        if W is None:
            continue
        if cfg.get("label") in ("raw", "sharpe"):
            W.level = np.float32(0.0)                 # nhãn không trừ TB năm -> không cộng lại mức năm
        if alt is not None:                                     # nhãn học = kỳ hạn khác; nhãn validation giữ 12 tháng
            lo_, hi_ = np.nanpercentile(alt[W.trn], [1, 99]); W.target = W.target.copy()
            W.target[W.trn] = np.nan_to_num(np.clip(alt[W.trn], lo_, hi_))
        Et, yt = torch.tensor(W.E, device=DEV), torch.tensor(W.target, device=DEV)
        wfy_t = torch.tensor(W.year_weight if cfg.get("holen", True) else np.ones_like(W.year_weight), device=DEV)
        ytot = torch.tensor(np.clip(S.ytar, *np.percentile(S.ytar[W.trn], [1, 99])).astype(np.float32), device=DEV)
        base_t = torch.tensor(W.base, device=DEV)
        torch.manual_seed(seed * 1000 + Y)
        if net is None:
            net = (FuseNet if cfg.get("fuse") else TokNet if cfg.get("tok_x") else InitNet if cfg.get("tftinit") else FinalNet)(H.shape[2], S.E.shape[1], hid=cfg.get("hid", 128), drop=cfg.get("drop", 0.3), isab_points=cfg.get("m", 64)).to(DEV)
            if cfg.get("xk") == "hyper":                          # HyperAttention thay ISAB
                net.xs = HyperSetAttention(2 * cfg.get("hid", 128), 4, cfg.get("hblock", 512), cfg.get("hm", 256)).to(DEV)
            if cfg.get("xk") == "hyper_g":                        # HyperAttention có cổng khởi tạo 0, thay ISAB
                net.xs = Gated(HyperSetAttention(2 * cfg.get("hid", 128), 4, cfg.get("hblock", 512), cfg.get("hm", 256))).to(DEV)
            if cfg.get("xk") == "isab_hyper":                     # ISAB rồi HyperAttention có cổng khởi tạo 0
                net.xs = nn.Sequential(net.xs, Gated(HyperSetAttention(2 * cfg.get("hid", 128), 4, cfg.get("hblock", 512), cfg.get("hm", 256)))).to(DEV)
            if cfg.get("layers", 1) > 1:
                net.xs = nn.Sequential(*[ISAB(2 * cfg.get("hid", 128), 4, cfg.get("m", 64)) for _ in range(cfg["layers"])]).to(DEV)
            opt = make_opt(net, cfg)
            if cfg.get("ema"):
                import copy
                ema = copy.deepcopy(net)
        trn_t, val_t, te_t = (torch.tensor(a, device=DEV) for a in (W.trn, W.val, W.te))
        use = lambda: ema if ema is not None else net
        fwd = lambda j, m=None: (m or use())(H[j].float(), Et[j])

        def val_loss():
            with torch.no_grad():
                return float(sum(nn.functional.huber_loss(rc(fwd(j)), yt[j], delta=1.0) * len(j) for j in EB_(val_t)) / max(len(val_t), 1))
        best, best_state, best_ep = 1e18, None, -1
        if prev_state is not None:
            use().eval(); (opt.eval() if hasattr(opt, "eval") else None); best = val_loss(); best_state = {k: v.clone() for k, v in use().state_dict().items()}; best_ep = 0
        for ep in range(cfg.get("epochs", train.EPOCHS)):
            net.train(); (opt.train() if hasattr(opt, "train") else None)
            sub = cfg.get("sub", 0)
            bl = [jj for j0 in train.year_batches(trn_t, grp, True) for jj in (j0.split(sub) if sub else [j0]) if len(jj) >= 64]
            K = cfg.get("lw", 1)
            for c0 in range(0, len(bl), K):
                blk = bl[c0:c0 + K]
                losses = []
                for j in blk:
                    for pas in range(2 if cfg.get("aug") in ("up2", "mix2") else 1):
                        hs, ex, yy = H[j].float(), Et[j], yt[j]
                        if pas == 1 and cfg["aug"] == "up2":            # upsampling: thấy lại lô với nhiễu nhỏ
                            hs = hs + 0.1 * torch.randn_like(hs); ex = ex + 0.05 * torch.randn_like(ex)
                        if pas == 1 and cfg["aug"] == "mix2":           # mixup (Zhang et al., ICLR 2018) trong lô
                            lam_ = torch.distributions.Beta(0.4, 0.4).sample((len(j), 1)).to(DEV); p = torch.randperm(len(j), device=DEV)
                            hs = lam_[:, :, None] * hs + (1 - lam_[:, :, None]) * hs[p]; ex = lam_ * ex + (1 - lam_) * ex[p]; yy = lam_[:, 0] * yy + (1 - lam_[:, 0]) * yy[p]
                        o = rc(net(hs, ex))
                        l = nn.functional.huber_loss(o, yy, delta=1.0)
                        if aux:
                            l = l + lam * aux_loss(aux, o + base_t[j], ytot[j])
                        l = l * wfy_t[j[0]]
                        if K == 1:
                            opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                        else:
                            losses.append(l)
                if K > 1 and losses:                                   # Holen et al.: gộp gradient K lô, trọng số = loss_i / tổng + 1/K
                    lt = torch.stack([x.detach() for x in losses]) if cfg.get("lwm", "l") == "l" else torch.ones(len(losses), device=DEV)
                    w = lt / lt.sum() + 1.0 / len(losses)
                    l = (w * torch.stack(losses)).sum()
                    opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                if ema is not None:
                    with torch.no_grad():
                        for pe, p_ in zip(ema.parameters(), net.parameters()): pe.mul_(cfg["ema"]).add_(p_.detach(), alpha=1 - cfg["ema"])
                        for be, b_ in zip(ema.buffers(), net.buffers()): be.copy_(b_)
            use().eval(); (opt.eval() if hasattr(opt, "eval") else None)
            vl = val_loss()
            if vl < best:
                best, best_state, best_ep = vl, {k: v.clone() for k, v in use().state_dict().items()}, ep + 1
        if best_state is None:
            best_state = prev_state
        use().load_state_dict(best_state)
        if ema is not None:
            net.load_state_dict(best_state)
        use().eval(); prev_state = {k: v.clone() for k, v in best_state.items()}
        with torch.no_grad():
            js = list(EB_(te_t))
            o = torch.cat([rc(fwd(j)) for j in js]).cpu().numpy(); ti = torch.cat(js).cpu().numpy()
        if cfg.get("cal"):                             # hiệu chỉnh tuyến tính trên năm validation Y-1 (recalibration, Kuleshov et al. ICML 2018)
            with torch.no_grad():
                jv = list(EB_(val_t)); ov = torch.cat([rc(fwd(j)) for j in jv]).cpu().numpy(); vi = torch.cat(jv).cpu().numpy()
            a_ = float(np.clip(np.polyfit(ov, S.ytar[vi], 1)[0], 0.0, 3.0)); o = a_ * o
        if cfg.get("label") == "rank_pos":             # thứ hạng: luôn giữ chiều đã nhân dấu, xếp theo thứ hạng dự báo
            pred[ti] = (np.clip(o, -0.5, 0.5) + 1.0) * S.sd[ti]
        elif cfg.get("label") == "rank":                 # đổi thứ hạng dự báo ra Sharpe bằng đường thẳng fit trên năm Y-1
            with torch.no_grad():
                jv = list(EB_(val_t))
                ov = torch.cat([rc(fwd(j)) for j in jv]).cpu().numpy(); vi = torch.cat(jv).cpu().numpy()
            bb, aa = np.polyfit(ov, S.y[vi], 1)
            pred[ti] = (aa + bb * o) * S.sd[ti]
        else:
            pred[ti] = (np.clip(o, W.lo, W.hi) + W.level + W.base[ti]) * S.sd[ti]
        LOG.append(dict(year=Y, best_ep=best_ep, net_sd=float(np.std(np.clip(o, W.lo, W.hi))), base_sd=float(np.std(W.base[ti]))))
    global NET, LASTW
    NET, LASTW = net, W
    return pred


# ---------------------------------------------------------------- học thêm từ nhiều họ, chấm trên Accounting EW
class FuseNet(FinalNet):
    """FinalNet + đưa t-stat, mean (2 số) vào vector của mỗi chiến lược TRƯỚC ISAB, để attention giữa các chiến lược
    nhìn thấy t-stat của cả năm (giống EB co t-stat theo phân phối t của năm). Lớp cuối của nhánh này khởi tạo 0."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.xf = nn.Sequential(nn.Linear(2, 256), nn.GELU(), nn.Linear(256, 256))
        with torch.no_grad(): self.xf[2].weight.zero_(); self.xf[2].bias.zero_()

    def forward(self, tokens, extra, trace=None):
        x, _ = self.seq(self.norm(tokens))
        c = self.pool(x) + self.xf(extra[:, :2])
        c = self.xs(c[None])[0]
        return self.head(torch.cat([c, extra], 1)).squeeze(-1)


class Gated(nn.Module):
    """Khối attention có cổng khởi tạo 0 (ReZero, Bachlechner et al., UAI 2021): ra = x + a*(khối(x) - x), a học được, bắt đầu = 0."""

    def __init__(self, blk):
        super().__init__(); self.blk = blk; self.a = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.a * (self.blk(x) - x)


class TokNet(FinalNet):
    """t-stat, mean ghép vào MỌI token (4 đoạn) trước Bi-LSTM: [n,4,16] -> [n,4,18]; mạng xử lý chúng cùng hình dạng chuỗi."""

    def __init__(self, d_in, n_extra, **k):
        super().__init__(d_in + n_extra, n_extra, **k)

    def forward(self, tokens, extra, trace=None):
        t = torch.cat([tokens, extra[:, None, :].expand(-1, tokens.shape[1], -1)], -1)
        return super().forward(t, extra)


class InitNet(FinalNet):
    """Temporal Fusion Transformer (Lim et al., IJF 2021): biến tĩnh (t-stat, mean) sinh trạng thái khởi đầu (h0, c0) của Bi-LSTM."""

    def __init__(self, d_in, n_extra, hid=128, **k):
        super().__init__(d_in, n_extra, hid=hid, **k)
        self.h0 = nn.Linear(n_extra, 2 * hid); self.c0 = nn.Linear(n_extra, 2 * hid)

    def forward(self, tokens, extra, trace=None):
        n = len(tokens); hid = self.seq.hidden_size
        h0 = self.h0(extra).view(n, 2, hid).transpose(0, 1).contiguous(); c0 = self.c0(extra).view(n, 2, hid).transpose(0, 1).contiguous()
        x, _ = self.seq(self.norm(tokens), (h0, c0))
        c = self.xs(self.pool(x)[None])[0]
        return self.head(torch.cat([c, extra], 1)).squeeze(-1)


class StaticMulti:
    """Như train.Static, nhưng mọi phép 'trong năm' làm theo (họ, năm)."""

    def __init__(self, T, fam):
        self.yr, self.sd, self.sg, self.fam = T.oos_year.values, T.sd_is.values, T.sgn.values, fam
        self.gk = fam.astype(np.int64) * 10000 + self.yr
        self.y = (T.y.values / self.sd).astype(np.float32)
        self.ytar = (self.y - pd.Series(self.y).groupby(self.gk).transform("mean").values).astype(np.float32)
        self.E = np.c_[T.tstat.values * self.sg, T.mean_is.values * self.sg].astype(np.float32)
        at = np.abs(T.tstat.values); g = pd.Series(at).groupby(self.gk)
        self.zt = ((at - g.transform("mean").values) / (g.transform("std").values + 1e-9))[:, None]


def prepare_year_multi(S, Y, max_train):
    yr, y, ytar, gk = S.yr, S.y, S.ytar, S.gk
    trn = np.where((yr <= Y - 2) & np.isfinite(y))[0]
    val = np.where((yr == Y - 1) & (S.fam == 0))[0]
    te = np.where((yr == Y) & (S.fam == 0))[0]
    if len(trn) < 20000 or len(te) == 0:
        return None
    if len(trn) > max_train:
        trn = trn[np.argsort(-yr[trn], kind="stable")[:max_train]]
    W = train.Year(); W.trn, W.val, W.te = trn, val, te
    W.e_mu, W.e_sd = S.E[trn].mean(0), S.E[trn].std(0) + 1e-6
    W.E = np.nan_to_num(np.clip((S.E - W.e_mu) / W.e_sd, -10, 10)).astype(np.float32)
    fb = np.where((yr <= Y - 1) & (yr >= Y - train.BASE_YEARS) & np.isfinite(ytar) & (S.fam == 0))[0]   # nền: fit trên họ chấm
    sb = S.zt[fb].std(0) + 1e-9; A_ = S.zt[fb] / sb
    W.b = np.linalg.solve(A_.T @ A_ + train.BASE_RIDGE * np.eye(1), A_.T @ np.clip(ytar[fb], *np.percentile(ytar[fb], [1, 99])))
    W.base = ((S.zt / sb) @ W.b).astype(np.float32)
    res = ytar - W.base
    W.lo, W.hi = np.percentile(res[trn], [1, 99])
    W.target = np.clip(res, W.lo, W.hi).astype(np.float32)
    W.level = np.float32(y[(yr <= Y - 1) & np.isfinite(y) & (S.fam == 0)].mean())
    v_ = pd.Series(np.clip(ytar, *np.percentile(ytar[trn], [1, 99])) ** 2).groupby(gk).transform("mean").values
    r_ = v_ / pd.Series(v_[trn]).groupby(gk[trn]).first().mean()
    w0 = 0.5 * (1 + r_)
    W.year_weight = (w0 / pd.Series(w0[trn]).groupby(gk[trn]).first().mean()).astype(np.float32)
    return W


def walk_forward_multi(T, H, fam, cfg, seed=0, years=range(1983, 2021), max_train=1_500_000):
    """T, H, fam: bảng / token / mã họ ghép (Accounting EW = 0, đứng đầu). Trả dự báo cho các dòng Accounting EW (theo thứ tự đầu bảng)."""
    S = StaticMulti(T, fam)
    grp = torch.tensor(S.gk, device=DEV)
    rc = lambda o: o - o.mean()
    n0 = int((fam == 0).sum()); pred = np.full(n0, np.nan, np.float32)
    net = opt = prev_state = None
    for Y in years:
        W = prepare_year_multi(S, Y, max_train)
        if W is None:
            continue
        Et, yt = torch.tensor(W.E, device=DEV), torch.tensor(W.target, device=DEV)
        wfy_t = torch.tensor(W.year_weight, device=DEV)
        torch.manual_seed(seed * 1000 + Y)
        if net is None:
            net = FinalNet(H.shape[2], S.E.shape[1]).to(DEV)
            opt = torch.optim.AdamW(net.parameters(), lr=train.LR, weight_decay=train.WD)
        trn_t, val_t, te_t = (torch.tensor(a, device=DEV) for a in (W.trn, W.val, W.te))
        fwd = lambda j: net(H[j].float(), Et[j])

        def val_loss():
            with torch.no_grad():
                return float(sum(nn.functional.huber_loss(rc(fwd(j)), yt[j], delta=1.0) * len(j) for j in train.year_batches(val_t, grp, False)) / max(len(val_t), 1))
        best, best_state = 1e18, None
        if prev_state is not None:
            net.eval(); best = val_loss(); best_state = {k: v.clone() for k, v in net.state_dict().items()}
        for ep in range(train.EPOCHS):
            net.train()
            sub = cfg.get("sub", 0)
            for j in [jj for j0 in train.year_batches(trn_t, grp, True) for jj in (j0.split(sub) if sub else [j0]) if len(jj) >= 64]:
                l = nn.functional.huber_loss(rc(fwd(j)), yt[j], delta=1.0) * wfy_t[j[0]]
                opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            net.eval(); vl = val_loss()
            if vl < best:
                best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state or prev_state); net.eval(); prev_state = {k: v.clone() for k, v in net.state_dict().items()}
        with torch.no_grad():
            js = list(train.year_batches(te_t, grp, False))
            o = torch.cat([rc(fwd(j)) for j in js]).cpu().numpy(); ti = torch.cat(js).cpu().numpy()
        pred[ti] = (np.clip(o, W.lo, W.hi) + W.level + W.base[ti]) * S.sd[ti]
    return pred
