"""Future-destruction test of walk_forward with the exact B3 config on CPU (2 forecast years)."""
import sys, time, numpy as np, torch
import ht_base as B, run_q6_cfgs as Q
from ht_sub import make_sub
B.DEV = "cpu"; torch.set_num_threads(16)
Y1 = int(sys.argv[1]) if len(sys.argv) > 1 else 1990
yrs = [Y1, Y1 + 1]
cfg = {**Q.CFGS["B3"], "seed": 0}; print(cfg)
D = make_sub("acct_ew", with_X=False, with_H=False)
H = np.load(B.h.CACHE / "fincast_h.npy", mmap_mode="r"); D.H = torch.from_numpy(np.ascontiguousarray(H[D.idx_full]))
keep = D.yr <= Y1 + 1                                  # drop later rows to save CPU time? no: keep all rows so the test covers them
def run(tag):
    t0 = time.time(); p = B.walk_forward(D, cfg, years=yrs, seed=0); print(tag, "secs", round(time.time() - t0), flush=True); return p.copy()
pA = run("A original")
pA2 = pA
rng = np.random.default_rng(123)
# B: destroy labels of years >= Y1+1 and ALL inputs of years >= Y1+2
m = D.yr >= Y1 + 1; D.y_or = D.y_or.copy(); D.y_or[m] = rng.normal(0, 5, m.sum()).astype(D.y_or.dtype)
f = D.yr >= Y1 + 2; fi = np.where(f)[0]
D.H[torch.tensor(fi)] = torch.randn(len(fi), *D.H.shape[1:]).half()
D.F.loc[f, "tstat"] = rng.normal(0, 5, f.sum()).astype(D.F.tstat.dtype); D.F.loc[f, "mean_is"] = rng.normal(0, 1, f.sum()).astype(D.F.mean_is.dtype)
D.sd = D.sd.copy(); D.sd[f] = rng.uniform(0.5, 5, f.sum()).astype(D.sd.dtype); D.sg = D.sg.copy(); D.sg[f] = rng.choice([-1., 1.], f.sum())
pB = run("B future destroyed")
# C: positive control, also destroy labels of year Y1 (legitimately used for year Y1+1)
m0 = D.yr == Y1; D.y_or[m0] = rng.normal(0, 5, m0.sum()).astype(D.y_or.dtype)
pC = run("C + labels of Y1 destroyed")
for Y in yrs:
    k = D.yr == Y
    d = lambda a, b: np.nanmax(np.abs(a[k] - b[k]))
    print(Y, "n", k.sum(), "finite", np.isfinite(pA[k]).sum(), "| A vs A2", d(pA, pA2), "| A vs B", d(pA, pB), "| A vs C", d(pA, pC), "| sd(pred)", np.nanstd(pA[k]))
