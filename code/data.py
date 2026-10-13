"""Dữ liệu: từ file CSV gốc đến bảng huấn luyện (mỗi dòng = một chiến lược trong một năm) và dấu vân tay FinCast.

Bước 1  load_returns        CSV gốc -> ma trận lãi tháng [chiến lược x tháng]
Bước 2  year_stats          20 năm trước Y -> mean, sd, t-stat; năm Y -> lãi thật
Bước 3  fit_eb              EB của paper (chỉ dùng làm ĐỐI THỦ so sánh, không bao giờ là đầu vào)
Bước 4  build_table         ghép các năm 1966-2020 thành bảng 1.189.707 dòng
Bước 5  windows             128 tháng trước Y, nhân dấu
Bước 6  fincast_hidden      FinCast (pretrained, đóng băng): 128 tháng -> [4, 1280]
Bước 7  extract_tokens      PCA16 (fit chỉ trên năm <= 1982) -> [4, 16], lưu cache
Mọi thứ của năm Y chỉ dùng tháng < tháng 1 của Y.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
CACHE = Path(__file__).resolve().parent / "_cache"
CACHE.mkdir(exist_ok=True, parents=True)

BASE_YEAR = 1960      # tháng 0 = 1960-01
IS_YEARS = 20         # cửa sổ in-sample: 20 năm trước Y
IS_MIN_OBS = 36       # tối thiểu 36 tháng in-sample
FIRST_YEAR = 1966     # năm đầu tiên dùng để học (đủ >= 36 tháng in-sample)
OOS_START, YEAR_MAX = 1983, 2020
DEV = (1983, 2004)    # chọn cấu hình
HOLD = (2005, 2020)   # chỉ báo cáo
CTX = 128             # số tháng đưa vào FinCast (4 patch x 32 tháng)


# ---------------------------------------------------------------- Bước 1
def load_returns():
    """Họ Accounting EW: (signalid [n], lãi tháng % [n, tháng], NaN = thiếu). Cache ở _cache/wide_acct_ew.npz."""
    f = CACHE / "wide_acct_ew.npz"
    if not f.exists():
        df = pd.read_csv(RAW / "DataMinedLongShortReturnsEW.csv", usecols=["signalid", "year", "month", "ret"],
                         dtype={"signalid": "int32", "year": "int16", "month": "int8", "ret": "float32"})
        m = (df.year.values.astype(np.int32) - BASE_YEAR) * 12 + (df.month.values.astype(np.int32) - 1)
        ids, inv = np.unique(df.signalid.values, return_inverse=True)
        wide = np.full((len(ids), int(m.max()) + 1), np.nan, dtype=np.float32)
        wide[inv, m] = df.ret.values
        np.savez(f, ids=ids, wide=wide)
    z = np.load(f)
    return z["ids"], z["wide"]


# ---------------------------------------------------------------- Bước 2
def year_stats(wide, Y):
    """Thống kê 20 năm TRƯỚC Y (in-sample) và lãi trung bình của năm Y (cái cần dự báo). t = mean / (sd / sqrt(n))."""
    a, b = max((Y - IS_YEARS - BASE_YEAR) * 12, 0), (Y - BASE_YEAR) * 12
    if b + 12 > wide.shape[1]:
        return None
    IS, OOS = wide[:, a:b], wide[:, b:b + 12]
    n_is, n_oos = (~np.isnan(IS)).sum(1), (~np.isnan(OOS)).sum(1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # hàng toàn NaN
        m, s = np.nanmean(IS, 1), np.nanstd(IS, 1)          # ddof = 0
        t = m / (s / np.sqrt(n_is))
        y = np.nanmean(OOS, 1)
    ok = (n_is >= IS_MIN_OBS) & (n_oos >= 12) & np.isfinite(t)
    return dict(mean_is=m, sd_is=s, tstat=t, nobs=n_is, y_raw=y, ok=ok)


# ---------------------------------------------------------------- Bước 3: EB của paper (đối thủ)
def _eb_par(par):
    mua, lsa, logit_p, mub, lsb = par
    return mua, np.exp(lsa), np.clip(expit(logit_p), 1e-9, 1 - 1e-9), mub, np.exp(lsb)


def _eb_nll(par, t):
    mua, sa, p, mub, sb = _eb_par(par)
    lf = np.logaddexp(np.log(p) - 0.5 * np.log(2 * np.pi * (sa ** 2 + 1)) - 0.5 * (t - mua) ** 2 / (sa ** 2 + 1),
                      np.log(1 - p) - 0.5 * np.log(2 * np.pi * (sb ** 2 + 1)) - 0.5 * (t - mub) ** 2 / (sb ** 2 + 1))
    return -np.sum(lf)


def fit_eb(t, starts=((0.5, 0.0, 0.0, -0.5, 0.0), (1.0, 0.0, 0.0, -1.0, -0.5), (0.0, 0.5, 1.0, 0.0, -1.0))):
    """theta ~ p N(mua, sa^2) + (1-p) N(mub, sb^2), t = theta + N(0,1): ước lượng chỉ từ phân phối t của năm đó (không dùng lãi thật)."""
    best = None
    for x0 in starts:
        r = minimize(_eb_nll, np.array(x0, float), args=(t,), method="Nelder-Mead", options=dict(maxiter=2000, xatol=1e-4, fatol=1e-6))
        best = r if best is None or r.fun < best.fun else best
    return best.x


def eb_posterior_mean(par, t):
    mua, sa, p, mub, sb = _eb_par(par)
    la = np.log(p) - 0.5 * np.log(sa ** 2 + 1) - 0.5 * (t - mua) ** 2 / (sa ** 2 + 1)
    lb = np.log(1 - p) - 0.5 * np.log(sb ** 2 + 1) - 0.5 * (t - mub) ** 2 / (sb ** 2 + 1)
    w = expit(la - lb)
    return w * (mua + sa ** 2 / (sa ** 2 + 1) * (t - mua)) + (1 - w) * (mub + sb ** 2 / (sb ** 2 + 1) * (t - mub))


# ---------------------------------------------------------------- Bước 4
def build_table(ids, wide, years=range(FIRST_YEAR, YEAR_MAX + 1)):
    """Bảng huấn luyện: mỗi dòng = (chiến lược, năm Y). Cột:
    signalid, oos_year, sgn (dấu của mean in-sample), mean_is, sd_is, tstat, nobs,
    y (lãi thật năm Y, NHÂN DẤU), eb (dự báo EB, theo dấu gốc của chiến lược; chỉ để so sánh)."""
    parts = []
    for Y in years:
        st = year_stats(wide, Y)
        if st is None:
            continue
        ok = st["ok"]
        t = st["tstat"][ok]
        th = eb_posterior_mean(fit_eb(t), t)
        sg = np.where(st["mean_is"][ok] >= 0, 1.0, -1.0)
        parts.append(pd.DataFrame(dict(signalid=ids[ok], oos_year=Y, sgn=sg, mean_is=st["mean_is"][ok], sd_is=st["sd_is"][ok], tstat=t,
                                       nobs=st["nobs"][ok], y=st["y_raw"][ok] * sg, eb=th * st["mean_is"][ok] / t)))
    return pd.concat(parts, ignore_index=True)


def load_table():
    f = CACHE / "table_acct_ew.parquet"
    if not f.exists():
        build_table(*load_returns()).to_parquet(f)
    return pd.read_parquet(f)


# ---------------------------------------------------------------- Bước 5
def windows(T, ids, wide, ctx=CTX):
    """[n, ctx] lãi tháng của `ctx` tháng ngay trước Y, nhân dấu sgn (NaN = chưa có dữ liệu)."""
    pos = pd.Series(np.arange(len(ids)), index=ids)
    r = pos.loc[T.signalid.values].values
    b = (T.oos_year.values - BASE_YEAR) * 12
    W = np.full((len(T), ctx), np.nan, np.float32)
    for i in range(len(T)):
        w = wide[r[i], max(b[i] - ctx, 0):b[i]]
        W[i, ctx - len(w):] = w
    return W * T.sgn.values[:, None].astype(np.float32)


# ---------------------------------------------------------------- Bước 6-7
FINCAST_SRC = ROOT / "Research" / "tools" / "FinCast-fts" / "src"


def load_fincast(device="cuda"):
    """FinCast (CIKM 2025, Zhu et al.), PRETRAINED, trọng số HF Vincent05R/FinCast v1.pth. Chỉ chạy tới lớp decoder cuối."""
    import importlib.util
    sys.path.insert(0, str(FINCAST_SRC))
    import torch
    from huggingface_hub import hf_hub_download
    spec = importlib.util.spec_from_file_location("ppd", FINCAST_SRC / "ffm" / "pytorch_patched_decoder_MOE.py")
    ppd = importlib.util.module_from_spec(spec); sys.modules["ppd"] = ppd; spec.loader.exec_module(ppd)
    cfg = ppd.FFMConfig(num_layers=50, num_heads=16, hidden_size=1280, intermediate_size=1280, patch_len=32, horizon_len=128, head_dim=80,
                        quantiles=ppd.create_quantiles(), use_positional_embedding=False, num_experts=4, gating_top_n=2,
                        threshold_train=0.2, threshold_eval=0.2)
    m = ppd.PatchedTimeSeriesDecoder_MOE(cfg)
    sd = torch.load(hf_hub_download("Vincent05R/FinCast", "v1.pth"), map_location="cpu", weights_only=True)
    sd = {k.replace("_orig_mod.module.", "").replace("_orig_mod.", "").replace("module.", "", 1) if k.startswith(("module.", "_orig_mod")) else k: v
          for k, v in sd.items()}
    m.load_state_dict(sd, strict=True)
    return m.to(device=device).eval()


def fincast_hidden(m, W):
    """W: tensor [B, 128] (NaN = thiếu) -> [B, 4, 1280]: 4 patch x 32 tháng, đầu ra lớp transformer cuối."""
    import torch
    with torch.no_grad():
        dev = W.device
        pad = torch.isnan(W).float(); x = torch.nan_to_num(W, nan=0.0).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mi, pp, _, _ = m._preprocess_input(x, pad)
            mi = mi + m.freq_emb(torch.ones(x.shape[0], 1, dtype=torch.long, device=dev))
            out, _ = m.stacked_transformer(mi, pp)
    return out


def hidden_of(T, ids, wide, sel, m, batch=2048):
    """FinCast [len(sel), 4, 1280] (float32, GPU) cho các dòng `sel` của bảng T."""
    import torch
    res = []
    for s in range(0, len(sel), batch):
        W = torch.tensor(windows(T.iloc[sel[s:s + batch]], ids, wide), device="cuda")
        res.append(fincast_hidden(m, W).float())
    return torch.cat(res)


def fit_pca(T, ids, wide, m, k=16, n_fit=50000):
    """PCA trắng hoá k chiều của các vector 1280 chiều, fit trên <= n_fit dòng NGẪU NHIÊN của năm <= 1982 (trước dự báo đầu tiên: không nhìn trước)."""
    import torch
    fit = np.where(T.oos_year.values <= 1982)[0]
    fit = fit[np.random.default_rng(0).permutation(len(fit))[:n_fit]]
    Z = hidden_of(T, ids, wide, fit, m).reshape(-1, 1280); mu = Z.mean(0)
    ev, V = torch.linalg.eigh(torch.cov((Z - mu).T)); V = V[:, -k:] / ev[-k:].clamp(min=1e-6).sqrt()
    return mu, V, float(ev[-k:].sum() / ev.sum())


def project(Hid, mu, V):
    """[B, 4, 1280] -> [B, 4, 16], kẹp [-5, 5]."""
    return ((Hid - mu) @ V).clamp(-5, 5)


def extract_tokens(T, ids, wide, out=CACHE / "full_acct_pca16.npy"):
    """FinCast -> PCA16 -> float16 [N, 4, 16] cho TOÀN BỘ bảng, lưu vào cache (~9 phút trên GPU)."""
    m = load_fincast()
    mu, V, _ = fit_pca(T, ids, wide, m)
    H = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(len(T), 4, 16))
    for s in range(0, len(T), 20000):
        sel = np.arange(s, min(s + 20000, len(T)))
        H[sel] = project(hidden_of(T, ids, wide, sel, m), mu, V).half().cpu().numpy()
    H.flush()


def load_tokens():
    """[N, 4, 16] float16 trên GPU, cùng thứ tự dòng với load_table()."""
    import torch
    return torch.from_numpy(np.load(CACHE / "full_acct_pca16.npy")).to("cuda")
