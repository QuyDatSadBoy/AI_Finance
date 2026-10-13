"""python run_x.py <tên cấu hình> <seed> ... -> preds/<tên>_s<seed>.npy"""
import sys, time, numpy as np
from pathlib import Path
import train_x as X
import data
CFGS = {
    "final":      {},                                   # đối chứng: đúng mô hình chốt (seed 6+ để tăng ensemble)
    "isab_m128":  {"m": 128},                           # Set Transformer: nhiều điểm cảm ứng hơn
    "isab_2lop":  {"layers": 2},                        # Set Transformer: 2 khối ISAB chồng nhau
    "listnet05":  {"aux": "listnet", "lam": 0.5},       # + ListNet (Cao et al., ICML 2007), xếp hạng trong năm
    "corr05":     {"aux": "corr", "lam": 0.5},          # + (1 - tương quan) trong năm, trên toàn bộ dự báo
    "ema09":      {"ema": 0.9},                         # trung bình trượt trọng số (Polyak; SWA, Izmailov et al. UAI 2018)
    "feat_t60":   {"feats": ["t60", "t24", "m12"]},      # đặc trưng: t-stat 60/24 tháng gần nhất, mean 12 tháng
    "feat_nobs":  {"feats": ["nobs", "lsd"]},           # đặc trưng: số tháng in-sample, log sd
    "hid64":      {"hid": 64},                          # mạng nhỏ hơn
    "drop05":     {"drop": 0.5},
    "sundial":    {"tok": "sundial"},                   # encoder Sundial (ICML 2025 Oral, pretrained) thay FinCast
    "fc_sundial": {"tok": "both"},                      # FinCast 4 token + Sundial 4 token
    "diag":       {},                                   # = mô hình chốt, ghi nhật ký vòng được chọn mỗi năm
    "sub4096":    {"sub": 4096},                        # nhiều bước cập nhật hơn: mỗi năm chia lô 4.096 dòng (~6 lần số bước)
    "ep10":       {"epochs": 10},                       # học 10 vòng thay vì 3
    "fam_vw":     {"multi": ["acct_vw"]},               # học thêm Accounting VW
    "fam_pr":     {"multi": ["past_ret_ew", "past_ret_vw"]},   # học thêm Past Return EW/VW
    "fam_all":    {"multi": ["acct_vw", "past_ret_ew", "past_ret_vw"]},   # học thêm cả 3 họ
    "fam_all_sub": {"multi": ["acct_vw", "past_ret_ew", "past_ret_vw"], "sub": 4096},
    "sub_sharpe": {"sub": 4096, "label": "sharpe"},
    "label_rank": {"label": "rank"},                  # học thứ hạng trong năm
    "label_nobase": {"label": "nobase"},              # Sharpe trừ TB năm, KHÔNG nền |t| (bỏ phép 3)
    "pair1":      {"aux": "pair", "lam": 1.0},         # + loss xếp hạng theo cặp (Feng et al., TOIS 2019)
    "pair10":     {"aux": "pair", "lam": 10.0},
    "label_rank_pos": {"label": "rank_pos"},          # học thứ hạng, không đảo chiều
    "sharpe_fuse": {"label": "sharpe", "fuse": True},  # học thẳng Sharpe + t-stat/mean vào ISAB
    "fuse":       {"fuse": True},
    "sharpe_fuse_sub": {"label": "sharpe", "fuse": True, "sub": 4096},   # + nhiều bước cập nhật hơn
    "sharpe_sub": {"label": "sharpe", "sub": 4096},
    "nobase_rank": {"label": "nobase", "xsrank": True},
    "nobase_sub": {"label": "nobase", "sub": 4096},
    "nobase_rank_sub": {"label": "nobase", "xsrank": True, "sub": 4096},
    "nobase_fuse_sub": {"label": "nobase", "fuse": True, "sub": 4096},
    "N_pair1": {"label": "nobase", "sub": 4096, "aux": "pair", "lam": 1.0},          # Feng et al. TOIS 2019
    "N_pair10": {"label": "nobase", "sub": 4096, "aux": "pair", "lam": 10.0},
    "N_calib01": {"label": "nobase", "sub": 4096, "aux": "corr", "lam": 0.1},         # CalibRank (Wan et al. 2026): MSE - lam*corr cắt ngang
    "N_calib1": {"label": "nobase", "sub": 4096, "aux": "corr", "lam": 1.0},
    "N_listnet": {"label": "nobase", "sub": 4096, "aux": "listnet", "lam": 0.5},      # ListNet (Cao et al. ICML 2007)
    "N_tok": {"label": "nobase", "sub": 4096, "tok_x": True},                          # t-stat, mean ghép vào mọi token
    "N_tftinit": {"label": "nobase", "sub": 4096, "tftinit": True},                    # t-stat, mean khởi tạo LSTM (TFT)
    "N_fuse": {"label": "nobase", "sub": 4096, "fuse": True},                          # t-stat, mean vào ISAB
    "N_srank05": {"label": "nobase", "sub": 4096, "aux": "srank", "lam": 0.5},     # + Rank IC khả vi (LambdaRankIC 2026)
    "N_srank01": {"label": "nobase", "sub": 4096, "aux": "srank", "lam": 0.1},
    "N_feat_nobs": {"label": "nobase", "sub": 4096, "feats": ["nobs", "lsd"]},       # + số tháng in-sample, log độ dao động
    "N_feat_zt": {"label": "nobase", "sub": 4096, "feats": ["zt"]},                      # + |t| chuẩn hoá trong năm
    "N_feat_ydist": {"label": "nobase", "sub": 4096, "feats": ["zt", "tsd", "f2"]},       # + |t| trong năm + phân phối t của năm
    "H": {"label": "nobase", "sub": 4096, "xk": "hyper"},                                          # Bước 6 mới + HyperAttention (Han et al., ICLR 2024)
    "H_feat_zt": {"label": "nobase", "sub": 4096, "xk": "hyper", "feats": ["zt"]},
    "H_feat_ydist": {"label": "nobase", "sub": 4096, "xk": "hyper", "feats": ["zt", "tsd", "f2"]},
    "H_feat_nobs": {"label": "nobase", "sub": 4096, "xk": "hyper", "feats": ["nobs", "lsd"]},
    "H_srank01": {"label": "nobase", "sub": 4096, "xk": "hyper", "aux": "srank", "lam": 0.1},
    "H_calib01": {"label": "nobase", "sub": 4096, "xk": "hyper", "aux": "corr", "lam": 0.1},
    "H_ev": {"label": "nobase", "sub": 4096, "xk": "hyper", "evsub": True},                       # dự báo cũng theo lô 4.096 (khớp lúc học)
    "H_full": {"label": "nobase", "xk": "hyper"},          # HyperAttention, cả năm một lô (học và dự báo)
    "H_m0": {"label": "nobase", "sub": 4096, "xk": "hyper", "hm": 0},                              # HyperAttention chỉ phần khối sortLSH, không lấy mẫu ngẫu nhiên
    "H_exact": {"label": "nobase", "sub": 4096, "xk": "hyper", "hblock": 4096},                     # khối >= lô: HyperAttention = attention đầy đủ
    "Hg": {"label": "nobase", "sub": 4096, "xk": "hyper_g"},
    "IH": {"label": "nobase", "sub": 4096, "xk": "isab_hyper"},
    "IH_zt": {"label": "nobase", "sub": 4096, "xk": "isab_hyper", "feats": ["zt"]},
    "IH_ydist": {"label": "nobase", "sub": 4096, "xk": "isab_hyper", "feats": ["zt", "tsd", "f2"]},
    "IH_nobs": {"label": "nobase", "sub": 4096, "xk": "isab_hyper", "feats": ["nobs", "lsd"]},
    "IH_srank": {"label": "nobase", "sub": 4096, "xk": "isab_hyper", "aux": "srank", "lam": 0.1},
    "IH_calib": {"label": "nobase", "sub": 4096, "xk": "isab_hyper", "aux": "corr", "lam": 0.1},
    "IH_lion": {"label": "nobase", "sub": 4096, "xk": "isab_hyper", "opt": "lion"},
    "IH_muon": {"label": "nobase", "sub": 4096, "xk": "isab_hyper", "opt": "muon"},
    "IH_radam": {"label": "nobase", "sub": 4096, "xk": "isab_hyper", "opt": "radam"},
    "IH_sf": {"label": "nobase", "sub": 4096, "xk": "isab_hyper", "opt": "sf"},
    "IH_fmom": {"label": "nobase", "sub": 4096, "xk": "isab_hyper", "feats": ["m12"]},                 # factor momentum: lãi 12 tháng gần nhất / độ dao động (Ehsani & Linnainmaa, JF 2022)
    "IH_lh6": {"label": "nobase", "sub": 4096, "xk": "isab_hyper", "lh": 6},                            # học bằng nhãn 6 tháng (Label Horizon Paradox 2026)
    "IH_lh24": {"label": "nobase", "sub": 4096, "xk": "isab_hyper", "lh": 24},                          # học bằng nhãn 24 tháng
    "IH_b256": {"label": "nobase", "sub": 4096, "xk": "isab_hyper", "hblock": 256},                       # HyperAttention: khối 256
    "IH_m512": {"label": "nobase", "sub": 4096, "xk": "isab_hyper", "hm": 512},                           # HyperAttention: 512 khoá lấy mẫu
    "IH_cal": {"label": "nobase", "sub": 4096, "xk": "isab_hyper", "cal": True},                        # hiệu chỉnh độ dốc dự báo trên năm Y-1
    "IH_nobs_srank": {"label": "nobase", "sub": 4096, "xk": "isab_hyper", "feats": ["nobs", "lsd"], "aux": "srank", "lam": 0.1},
    "IH_lh24_nobs": {"label": "nobase", "sub": 4096, "xk": "isab_hyper", "lh": 24, "feats": ["nobs", "lsd"]},
    "IH_lh24_srank": {"label": "nobase", "sub": 4096, "xk": "isab_hyper", "lh": 24, "aux": "srank", "lam": 0.1},
    "IH_lh24_nobs_srank": {"label": "nobase", "sub": 4096, "xk": "isab_hyper", "lh": 24, "feats": ["nobs", "lsd"], "aux": "srank", "lam": 0.1},
    "IH_lh36": {"label": "nobase", "sub": 4096, "xk": "isab_hyper", "lh": 36},
    "IH_256": {"label": "nobase", "sub": 4096, "xk": "isab_hyper", "tok": "fc256"},                     # FinCast 256 tháng (8 đoạn)
    "H_lion": {"label": "nobase", "sub": 4096, "xk": "hyper", "opt": "lion"},
    "H_muon": {"label": "nobase", "sub": 4096, "xk": "hyper", "opt": "muon"},
    "H_sf": {"label": "nobase", "sub": 4096, "xk": "hyper", "opt": "sf"},
    "H_radam": {"label": "nobase", "sub": 4096, "xk": "hyper", "opt": "radam"},
    "nobase_fuse": {"label": "nobase", "fuse": True},  # Sharpe trừ TB năm (centering) + t-stat vào ISAB, không nền công thức                     # mô hình chốt + t-stat/mean vào ISAB
    "label_raw":  {"label": "raw"},                     # học THẲNG lãi
    "label_sharpe": {"label": "sharpe"},                # học thẳng Sharpe (không trừ TB năm, không nền)
    "hyper":      {"xk": "hyper"},                      # HyperAttention (ICLR 2024) thay ISAB
    "lw4":        {"lw": 4},                            # Holen et al.: gộp gradient 4 năm, trọng số theo loss
    "up2":        {"aug": "up2"},                       # upsampling có nhiễu
    "mix2":       {"aug": "mix2"},                      # mixup
}
if __name__ == "__main__":
    name, seeds = sys.argv[1], list(map(int, sys.argv[2:]))
    T, H = data.load_table(), data.load_tokens(); ids, wide = data.load_returns()
    tok = CFGS[name].get("tok")
    if tok:
        import torch
        if tok == "fc256":
            H = torch.from_numpy(np.load(Path(__file__).parent / "fc256_pca16.npy")).to("cuda")
        else:
            S_ = torch.from_numpy(np.load(Path(__file__).parent / "sundial_pca16.npy")).to("cuda")
            H = S_ if tok == "sundial" else torch.cat([H, S_], 1)
    import os
    out = Path(__file__).parent / ("preds_quick" if os.environ.get("QUICK") else "preds"); out.mkdir(exist_ok=True)
    for s in seeds:
        t0 = time.time()
        cfg = CFGS[name]
        if cfg.get("multi"):
            import pandas as pd, torch
            Ts, Hs, fs = [T], [H], [np.zeros(len(T), np.int64)]
            for k, f in enumerate(cfg["multi"], 1):
                Ts.append(pd.read_parquet(Path(__file__).parent / f"fam_{f}_table.parquet"))
                Hs.append(torch.from_numpy(np.load(Path(__file__).parent / f"fam_{f}_pca16.npy")).to("cuda")); fs.append(np.full(len(Ts[-1]), k))
            p = X.walk_forward_multi(pd.concat(Ts, ignore_index=True), torch.cat(Hs), np.concatenate(fs), cfg, seed=s)
        else:
            import os
            yrs = range(1983, 2005) if os.environ.get("QUICK") else range(1983, 2021)
            p = X.walk_forward(T, H, cfg, seed=s, ids=ids, wide=wide, years=yrs)
        np.save(out / f"{name}_s{s}.npy", p)
        import json; json.dump(getattr(X, "LOG", []) if not cfg.get("multi") else [], open(out / f"{name}_s{s}.log.json", "w"))
        print(name, "seed", s, "xong", round(time.time() - t0), "s", flush=True)
