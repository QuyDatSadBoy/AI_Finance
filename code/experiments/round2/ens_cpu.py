"""Ensemble các cấu hình đã có dự báo (chỉ CPU). Chọn theo dev bằng ensemble selection tham lam (Caruana et al., ICML 2004)."""
import sys, hashlib, numpy as np, pandas as pd
from pathlib import Path
E = Path(__file__).resolve().parent; sys.path.insert(0, str(E.parents[1])); sys.path.insert(1, str(E.parent / "old_code"))
import data, evaluate as ev
T = data.load_table(); ids, wide = data.load_returns()
import importlib.util
spec = importlib.util.spec_from_file_location("fc", E.parent / "old_code" / "full_cfgs.py")
# full_cfgs imports ht_base -> needs old_code on path (done); only the dict is used
from full_cfgs import FCFGS
PD = E.parent / "config_forecasts"
def mp(n):
    ps = []
    for s in range(20):
        cfg = {**FCFGS[n], "seed": s}; nm = "|".join(f"{k}={cfg[k]}" for k in sorted(cfg))
        f = PD / ("full_" + hashlib.md5(nm.encode()).hexdigest() + ".npy")
        if f.exists(): ps.append(np.load(f))
    return (np.mean(ps, 0), len(ps)) if ps else (None, 0)
def score(p):
    R = ev.top_portfolios(T, ids, wide, ev.signal_from_oriented(T, p.astype(np.float32)))
    return np.mean([ev.perf(R[r], *data.DEV)["sharpe"] for r in (1, 5, 10)]), np.mean([ev.perf(R[r], *data.HOLD)["sharpe"] for r in (1, 5, 10)])
P = {}
for n in FCFGS:
    p, k = mp(n)
    if p is not None and k >= 3 and np.isfinite(p[T.oos_year.values >= 1983]).all(): P[n] = p
FIN = np.mean([np.load(f) for f in sorted((data.CACHE / "forecasts").glob("final_s*.npy"))], 0); P["final_new"] = FIN
sc = {n: score(p) for n, p in P.items()}
print(pd.DataFrame(sc, index=["dev", "hold"]).T.sort_values("dev", ascending=False).round(3).head(15).to_string())
# greedy ensemble selection on dev (with replacement), z-scored within year so scales match
cur, members, best = None, [], -1
for it in range(8):
    cand = []
    for n, p in P.items():
        q = p if cur is None else (cur * len(members) + p) / (len(members) + 1)
        cand.append((score(q)[0], n, q))
    d, n, q = max(cand, key=lambda x: x[0])
    if d <= best + 1e-4: break
    best, cur = d, q; members.append(n); print(it, n, "dev", round(d, 3), "hold", round(score(q)[1], 3), flush=True)
np.save(E / "preds" / "ens_greedy.npy", cur); print("members", members)
