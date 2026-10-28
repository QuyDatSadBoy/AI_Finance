"""Huấn luyện CHUNG hai họ kế toán EW + VW, giữ nguyên kiến trúc FinalNet và mọi siêu tham số.
Khác train.py ở chỗ DUY NHẤT: mọi thứ tính "theo năm" (trung bình nhãn, |t| chuẩn hoá, mức năm, hệ số nền b, trọng số năm, lô) tính theo (họ, năm),
vì cross-section của hai họ khác nhau; mỗi lô vẫn là một (họ, năm) nên ISAB chỉ thấy một cross-section. Trần dữ liệu học nâng 500k -> 1.000k dòng (cùng độ phủ năm cho mỗi họ).
PCA16 fit lại trên mẫu <= 1982 gộp hai họ (cùng một không gian đặc trưng cho cả hai).  python joint.py [seed ...]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
CODE = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(CODE)); sys.path.insert(0, str(Path(__file__).parent))
import data
from train import year_batches, DEV, EPOCHS, LR, WD, BASE_YEARS, BASE_RIDGE, sharpe_label
from model import FinalNet
from run_vw import V, load_vw

J = data.CACHE / "joint"; J.mkdir(exist_ok=True)

def tables():
    Te = data.load_table().assign(fam=0); Tv = pd.read_parquet(V / "table_acct_vw.parquet").assign(fam=1)
    return pd.concat([Te, Tv], ignore_index=True)

def tokens(T):
    f = J / "tokens_joint.npy"
    if not f.exists():
        ide, we = data.load_returns(); idv, wv = load_vw(); m = data.load_fincast()
        parts = {0: (ide, we), 1: (idv, wv)}
        Z = []
        for fam, (ids, wide) in parts.items():
            Tf = T[T.fam == fam].reset_index(drop=True)
            fit = np.where(Tf.oos_year.values <= 1982)[0]; fit = fit[np.random.default_rng(0).permutation(len(fit))[:25000]]
            Z.append(data.hidden_of(Tf, ids, wide, fit, m).reshape(-1, 1280))
        Z = torch.cat(Z); mu = Z.mean(0); ev, Vv = torch.linalg.eigh(torch.cov((Z - mu).T)); Vv = Vv[:, -16:] / ev[-16:].clamp(min=1e-6).sqrt()
        H = np.lib.format.open_memmap(f, mode="w+", dtype=np.float16, shape=(len(T), 4, 16)); off = 0
        for fam, (ids, wide) in parts.items():
            Tf = T[T.fam == fam].reset_index(drop=True)
            for s in range(0, len(Tf), 20000):
                sel = np.arange(s, min(s + 20000, len(Tf)))
                H[off + sel] = data.project(data.hidden_of(Tf, ids, wide, sel, m), mu, Vv).half().cpu().numpy()
            off += len(Tf); print("token họ", fam, "xong", flush=True)
        H.flush()
    return torch.from_numpy(np.load(f)).to(DEV)

class S:
    def __init__(self, T):
        self.yr, self.fam, self.sd, self.sg = T.oos_year.values, T.fam.values, T.sd_is.values, T.sgn.values
        self.g = self.yr * 2 + self.fam
        self.y = (T.y.values / T.sd_is.values).astype(np.float32)
        gb = pd.Series(self.y).groupby(self.g)
        self.ytar = (self.y - gb.transform("mean").values).astype(np.float32)
        self.E = np.c_[T.tstat.values * self.sg, T.mean_is.values * self.sg].astype(np.float32)
        at = np.abs(T.tstat.values); gt = pd.Series(at).groupby(self.g)
        self.zt = (at - gt.transform("mean").values) / (gt.transform("std").values + 1e-9)

def prepare(S, Y, max_train=1_000_000):
    yr, fam, y, ytar = S.yr, S.fam, S.y, S.ytar
    trn = np.where((yr <= Y - 2) & np.isfinite(y))[0]; val = np.where(yr == Y - 1)[0]; te = np.where(yr == Y)[0]
    if len(trn) < 40000 or len(te) == 0: return None
    if len(trn) > max_train: trn = trn[np.argsort(-yr[trn], kind="stable")[:max_train]]
    W = type("W", (), {})(); W.trn, W.val, W.te = trn, val, te
    W.e_mu, W.e_sd = S.E[trn].mean(0), S.E[trn].std(0) + 1e-6
    W.E = np.nan_to_num(np.clip((S.E - W.e_mu) / W.e_sd, -10, 10)).astype(np.float32)
    W.base = np.zeros(len(yr), np.float32); W.level = np.zeros(len(yr), np.float32); W.b = []
    for f in (0, 1):
        fb = np.where((yr <= Y - 1) & (yr >= Y - BASE_YEARS) & np.isfinite(ytar) & (fam == f))[0]
        sb = S.zt[fb].std() + 1e-9; A = S.zt[fb] / sb; yb = np.clip(ytar[fb], *np.percentile(ytar[fb], [1, 99]))
        b = (A @ yb) / (A @ A + BASE_RIDGE); W.b.append(b); m = fam == f
        W.base[m] = S.zt[m] / sb * b
        W.level[m] = y[(yr <= Y - 1) & np.isfinite(y) & m].mean()
    res = ytar - W.base; W.lo, W.hi = np.percentile(res[trn], [1, 99]); W.target = np.clip(res, W.lo, W.hi).astype(np.float32)
    v_ = pd.Series(np.clip(ytar, *np.percentile(ytar[trn], [1, 99])) ** 2).groupby(S.g).transform("mean").values
    r_ = v_ / pd.Series(v_[trn]).groupby(S.g[trn]).first().mean(); w0 = 0.5 * (1 + r_)
    W.year_weight = (w0 / pd.Series(w0[trn]).groupby(S.g[trn]).first().mean()).astype(np.float32)
    return W

def walk(T, H, seed, years=range(1983, 2021)):
    s = S(T); grp = torch.tensor(s.g.astype(np.int64), device=DEV); rc = lambda o: o - o.mean()
    pred = np.full(len(s.yr), np.nan, np.float32); net = opt = prev = None
    for Y in years:
        W = prepare(s, Y)
        if W is None: continue
        Et, yt, wf = (torch.tensor(a, device=DEV) for a in (W.E, W.target, W.year_weight))
        torch.manual_seed(seed * 1000 + Y)
        if net is None:
            net = FinalNet(H.shape[2], s.E.shape[1]).to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        trn_t, val_t, te_t = (torch.tensor(a, device=DEV) for a in (W.trn, W.val, W.te))
        fwd = lambda j: net(H[j].float(), Et[j])
        def vloss():
            with torch.no_grad():
                return float(sum(nn.functional.huber_loss(rc(fwd(j)), yt[j], delta=1.0) * len(j) for j in year_batches(val_t, grp, False)) / max(len(val_t), 1))
        best, bs = 1e18, None
        if prev is not None:
            net.eval(); best = vloss(); bs = {k: v.clone() for k, v in net.state_dict().items()}
        for ep in range(EPOCHS):
            net.train()
            for j in list(year_batches(trn_t, grp, True)):
                l = nn.functional.huber_loss(rc(fwd(j)), yt[j], delta=1.0) * wf[j[0]]
                opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            net.eval(); vl = vloss()
            if vl < best: best, bs = vl, {k: v.clone() for k, v in net.state_dict().items()}
        if bs is None: bs = prev
        net.load_state_dict(bs); net.eval(); prev = {k: v.clone() for k, v in bs.items()}
        with torch.no_grad():
            js = list(year_batches(te_t, grp, False)); o = torch.cat([rc(fwd(j)) for j in js]).cpu().numpy(); ti = torch.cat(js).cpu().numpy()
        pred[ti] = (np.clip(o, W.lo, W.hi) + W.level[ti] + W.base[ti]) * s.sd[ti]
        print(Y, flush=True)
    return pred

if __name__ == "__main__":
    T = tables(); H = tokens(T); print("bảng", len(T), flush=True)
    for seed in map(int, sys.argv[1:] or [0, 1, 2]):
        f = J / f"joint_s{seed}.npy"
        if not f.exists():
            np.save(f, walk(T, H, seed)); print("seed", seed, "xong", flush=True)
