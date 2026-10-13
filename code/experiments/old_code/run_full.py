"""python run_full.py <cfg name> <seed0> <seed1> ... : full-data walk-forward 1983-2020, saves the forecast of each seed."""
import sys, time, hashlib, numpy as np, torch
import ht_base as B, ht_common as h
from full_data import make_full
from full_cfgs import FCFGS
name, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
encf = FCFGS[name].get('encf'); D = make_full(encf=encf, with_X=FCFGS[name].get('extras') in ('F1', 'F2'))
for s in seeds:
    cfg = {**FCFGS[name], "seed": s}; t0 = time.time()
    pred = B.walk_forward(D, cfg, seed=s)
    nm = "|".join(f"{k}={cfg[k]}" for k in sorted(cfg)); B.PRED_DIR.mkdir(exist_ok=True)
    np.save(B.PRED_DIR / ("full_" + hashlib.md5(nm.encode()).hexdigest() + ".npy"), pred)
    if cfg.get("parts"): np.savez(B.PRED_DIR / ("fullparts_" + hashlib.md5(nm.encode()).hexdigest() + ".npz"), **D._parts)
    print(f"{name} seed {s}: {round(time.time() - t0)}s, {int(np.isfinite(pred).sum())} forecasts", flush=True)
