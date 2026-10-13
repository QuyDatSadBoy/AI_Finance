"""Huấn luyện walk-forward của mô hình chốt. Dự báo năm Y:
    học   : các dòng có năm <= Y-2 (tối đa 500.000 dòng gần nhất), mỗi bước cập nhật = 4.096 chiến lược ngẫu nhiên của CÙNG một năm
    kiểm  : năm Y-1 (chọn vòng học tốt nhất, không cập nhật trọng số)
    dự báo: năm Y (cả năm một lần); trọng số năm Y-1 làm điểm xuất phát cho năm Y. 38 năm 1983-2020.

Đáp án (Bước 6 của docs): e = y / s - TB_năm(y / s)  (Sharpe năm sau trừ Sharpe trung bình của năm), cắt ở phân vị 1/99% của tập học.
Dự báo (%/tháng, theo dấu đã nhân) = (đáp án dự báo, cắt cùng khoảng + mức năm) x s;  mức năm = Sharpe trung bình mọi năm <= Y-1.
Loss: Huber(delta = 1) x trọng số năm kiểu Holen et al. (arXiv 2304.12778).
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
from model import FinalNet

DEV = "cuda"
BATCH_ROWS = 27000          # >= số chiến lược lớn nhất của một năm (26.541): khi dự báo, một lô = cả một năm
SUB = 4096                  # khi học: một lô = 4.096 chiến lược ngẫu nhiên của một năm
EPOCHS, LR, WD = 3, 1e-3, 1e-2


def sharpe_label(T):
    """Nhãn thô theo đơn vị Sharpe: lãi thật nhân dấu / sd in-sample, và nhãn đã trừ trung bình năm."""
    y = (T.y.values / T.sd_is.values).astype(np.float32)
    ytar = (y - pd.Series(y).groupby(T.oos_year.values).transform("mean").values).astype(np.float32)
    return y, ytar


def year_batches(idx, grp, train):
    """Mỗi batch = một năm. Khi học: xáo thứ tự dòng trong năm; khi dự báo: giữ nguyên thứ tự."""
    g = grp[idx]
    for u in torch.unique(g):
        rows = idx[g == u]
        if train:
            rows = rows[torch.randperm(len(rows), device=rows.device)]
        for i in range(0, len(rows), BATCH_ROWS):
            if len(rows[i:i + BATCH_ROWS]) >= 64 or not train:
                yield rows[i:i + BATCH_ROWS]


class Static:
    """Mảng không phụ thuộc năm Y: nhãn Sharpe và 5 số thêm."""

    def __init__(self, T):
        self.yr, self.sd, self.sg = T.oos_year.values, T.sd_is.values, T.sgn.values
        self.y, self.ytar = sharpe_label(T)
        # 5 số thêm (đều từ 20 năm trước Y): t-stat, lãi TB (nhân dấu), số tháng có dữ liệu, Sharpe in-sample, log độ dao động
        self.E = np.c_[T.tstat.values * self.sg, T.mean_is.values * self.sg, T.nobs.values,
                       T.mean_is.values * self.sg / T.sd_is.values, np.log(T.sd_is.values)].astype(np.float32)


class Year:
    """Mọi thứ để học và dự báo năm Y (chỉ dùng dữ liệu của năm <= Y-1)."""


def prepare_year(S, Y, max_train=500_000):
    """Chuẩn bị năm Y: chia học/kiểm/dự báo, chuẩn hoá 2 số, đáp án học, mức năm, trọng số năm. None nếu thiếu dữ liệu."""
    yr, y, ytar = S.yr, S.y, S.ytar
    trn = np.where((yr <= Y - 2) & np.isfinite(y))[0]
    val = np.where(yr == Y - 1)[0]
    te = np.where(yr == Y)[0]
    if len(trn) < 20000 or len(te) == 0:
        return None
    n_before = len(trn)
    if len(trn) > max_train:
        trn = trn[np.argsort(-yr[trn], kind="stable")[:max_train]]
    W = Year(); W.trn, W.val, W.te, W.n_trn_all = trn, val, te, n_before
    # đầu vào số: chuẩn hoá bằng thống kê tập học
    W.e_mu, W.e_sd = S.E[trn].mean(0), S.E[trn].std(0) + 1e-6
    W.E = np.nan_to_num(np.clip((S.E - W.e_mu) / W.e_sd, -10, 10)).astype(np.float32)
    # đáp án học = Sharpe trừ trung bình năm, cắt ở phân vị 1/99 của tập học (khoảng của nhãn huấn luyện; đầu ra cũng cắt theo)
    W.lo, W.hi = np.percentile(ytar[trn], [1, 99])
    W.target = np.clip(ytar, W.lo, W.hi).astype(np.float32)
    # mức năm: trung bình nhãn Sharpe của mọi dòng năm <= Y-1 (biết trước Y)
    W.level = np.float32(y[(yr <= Y - 1) & np.isfinite(y)].mean())
    # trọng số năm kiểu Holen: v = trung bình bình phương nhãn của năm; w = 0,5 (1 + v / v_trung_bình_tập_học), chuẩn hoá trung bình 1
    v_ = pd.Series(np.clip(ytar, *np.percentile(ytar[trn], [1, 99])) ** 2).groupby(yr).transform("mean").values
    r_ = v_ / pd.Series(v_[trn]).groupby(yr[trn]).first().mean()
    w0 = 0.5 * (1 + r_)
    W.year_weight = (w0 / pd.Series(w0[trn]).groupby(yr[trn]).first().mean()).astype(np.float32)
    return W


def walk_forward(T, H, seed=0, years=range(1983, 2021), max_train=500_000, verbose=True, keep_last=False, ckpt_dir=None, net_fn=None):
    """T: bảng (data.load_table), H: token [N,4,16] trên GPU. Trả về dự báo [N] (NaN ở năm không dự báo).
    ckpt_dir: nếu có, lưu checkpoint mỗi năm: <ckpt_dir>/Y<năm>.pt (trọng số mạng, optimizer, mức năm, khoảng cắt, chuẩn hoá)."""
    S = Static(T)
    grp = torch.tensor(S.yr.astype(np.int64), device=DEV)
    rc = lambda o: o - o.mean()                                                          # khử mức dịch theo năm của BatchNorm
    pred = np.full(len(S.yr), np.nan, np.float32)
    net = opt = prev_state = None
    for Y in years:
        W = prepare_year(S, Y, max_train)
        if W is None:
            continue
        Et, yt = torch.tensor(W.E, device=DEV), torch.tensor(W.target, device=DEV)
        wfy_t = torch.tensor(W.year_weight, device=DEV)
        # mạng: năm đầu tạo mới; các năm sau tiếp tục từ trọng số năm trước (cùng optimizer)
        torch.manual_seed(seed * 1000 + Y)
        if net is None:
            net = (net_fn or FinalNet)(H.shape[2], S.E.shape[1]).to(DEV)
            opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        trn_t, val_t, te_t = (torch.tensor(a, device=DEV) for a in (W.trn, W.val, W.te))
        fwd = lambda j: net(H[j].float(), Et[j])

        def val_loss():
            with torch.no_grad():
                return float(sum(nn.functional.huber_loss(rc(fwd(j)), yt[j], delta=1.0) * len(j) for j in year_batches(val_t, grp, False)) / max(len(val_t), 1))
        best, best_state = 1e18, None
        if prev_state is not None:                                    # trạng thái năm trước cũng là một ứng viên (epoch 0)
            net.eval(); best = val_loss(); best_state = {k: v.clone() for k, v in net.state_dict().items()}
        for ep in range(EPOCHS):
            net.train()
            for j in [jj for j0 in year_batches(trn_t, grp, True) for jj in j0.split(SUB) if len(jj) >= 64]:
                l = nn.functional.huber_loss(rc(fwd(j)), yt[j], delta=1.0) * wfy_t[j[0]]     # một lô thuộc một năm -> một trọng số
                opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            net.eval()
            vl = val_loss()
            if vl < best:
                best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
        if best_state is None:
            best_state = prev_state
        net.load_state_dict(best_state); net.eval(); prev_state = {k: v.clone() for k, v in best_state.items()}
        with torch.no_grad():
            js = list(year_batches(te_t, grp, False))
            o = torch.cat([rc(fwd(j)) for j in js]).cpu().numpy(); ti = torch.cat(js).cpu().numpy()
        pred[ti] = (np.clip(o, W.lo, W.hi) + W.level) * S.sd[ti]
        if ckpt_dir is not None:
            from pathlib import Path
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            torch.save(dict(year=Y, seed=seed, net=best_state, opt=opt.state_dict(), level=W.level, lo=W.lo, hi=W.hi,
                            e_mu=W.e_mu, e_sd=W.e_sd, val_loss=best), Path(ckpt_dir) / f"Y{Y}.pt")
        if verbose:
            print(f"{Y}: học {len(W.trn):,} dòng, valid {len(W.val):,}, dự báo {len(W.te):,} | val loss {best:.4f}", flush=True)
    return (pred, net) if keep_last else pred
