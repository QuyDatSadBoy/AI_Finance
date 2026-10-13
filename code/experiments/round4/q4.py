"""Vòng 4: thử để giữ được ở holdout. python q4.py <tên> <seed>...  (38 năm; chọn bằng dev, đặc biệt dev muộn 1995-2004)"""
import sys, time, numpy as np, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
import data, train4, model, torch.nn as nn
sys.path.insert(0, str(HERE.parent / "round3"))
CFG = {"base": {}, "rec5": {"rec_half": 5}, "rec10": {"rec_half": 10}, "ema": {"ema": 0.99}, "corr": {"corr": 0.1},
       "corr05": {"corr": 0.5}, "delta05": {"delta": 0.5}, "gate01": {"gate": 0.1}, "lev5": {"level_n": 5}, "feat": {"feat": True}, "tbase": {"tbase": True}, "tbase_rec5": {"tbase": True, "rec_half": 5},
       "tbase_fc": {"tbase": True, "fc": True}, "tbase_nohyper": {"tbase": True, "nohyper": True}, "tbase_old": {"tbase": True, "fc": True, "nohyper": True},
       "tbase_rec5_corr": {"tbase": True, "rec_half": 5, "corr": 0.1}, "tbase_old_rec5": {"tbase": True, "fc": True, "nohyper": True, "rec_half": 5}}

def net_fn(c):
    def f(d_in, n_extra):
        net = model.FinalNet(d_in, n_extra)
        if c.get("fc"): net.head = nn.Sequential(nn.Linear(256 + n_extra, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.3), nn.Linear(64, 1))
        if c.get("nohyper"): net.hyper = model.Gated(nn.Identity()); net.hyper.a.requires_grad_(False)
        return net
    return f
if __name__ == "__main__":
    name, seeds = sys.argv[1], list(map(int, sys.argv[2:])); train4.OPT.update(CFG[name])
    T = data.load_table(); H = data.load_tokens(); (HERE / "preds").mkdir(exist_ok=True)
    for s in seeds:
        t0 = time.time(); p = train4.walk_forward(T, H, seed=s, verbose=False, net_fn=net_fn(CFG[name]))
        np.save(HERE / "preds" / f"{name}_s{s}.npy", p); print(name, s, "xong", round(time.time() - t0), "s", flush=True)
