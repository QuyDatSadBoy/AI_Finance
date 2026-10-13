"""Selection-bias check (from the adversarial verification): build.py rebuilds the dev top-k monthly returns of every acct_ew config
(10-seed mean) into _cache/verify/rets.pkl; maxstat.py runs a joint year-block bootstrap max-statistic / Romano-Wolf correction and a
deflated-Sharpe check for the config named by TARGET (default B3).  Usage (from code/): PYTHONPATH=. python verify/build.py; TARGET=Dt0 PYTHONPATH=. python verify/maxstat.py"""
import pickle, numpy as np, pandas as pd, ht_new as nw
from scipy.stats import norm
OUT = "/home/phongndh/Documents/Paper/code/_cache/verify/"
res, meta = pickle.load(open(OUT + "rets.pkl", "rb"))
import os, sys; sys.path.insert(0, ".."); from run_q6_cfgs import CFGS
TGT = os.environ.get("TARGET", "B3"); _c = CFGS[TGT]; B3 = "|".join(f"{k}={_c[k]}" for k in sorted(_c) if k != "seed")   # the selected config (name in CFGS)
assert B3 in res, ("target not built", TGT)
print("target", TGT)
names = [k for k in res if k != "EB"]
def mat(ret):
    r = ret[(ret.year >= 1983) & (ret.year <= 2004)].pivot(index="year", columns="month", values="ret")
    return r.values  # 22 x 12
EBm = {r: mat(res["EB"][r]) for r in (1, 5, 10)}
M = {(k, r): mat(res[k][r]) for k in names for r in (1, 5, 10)}
sh = lambda X: X.mean() / X.std(ddof=1) * np.sqrt(12)
cols = list(M)                                     # (config, topk)
d0 = np.array([sh(M[c]) - sh(EBm[c[1]]) for c in cols])
ib3 = {r: cols.index((B3, r)) for r in (1, 5, 10)}
print("K configs", len(names), "tests", len(cols))
print("B3 diffs", {r: round(d0[ib3[r]], 3) for r in (1, 5, 10)})
# JKM z for every test
def jkm(A, Bm):
    A, Bm = A.ravel(), Bm.ravel(); T = len(A); s1, s2 = A.mean()/A.std(ddof=1), Bm.mean()/Bm.std(ddof=1); rho = np.corrcoef(A, Bm)[0,1]
    V = (2-2*rho+0.5*(s1**2+s2**2-2*s1*s2*rho**2))/T; return (s1-s2)/np.sqrt(V)
z0 = np.array([jkm(M[c], EBm[c[1]]) for c in cols])
rank = pd.DataFrame(dict(cfg=[c[0][:0] + str(names.index(c[0])) for c in cols], top=[c[1] for c in cols], d=d0, z=z0))
print("top by d (dev):"); print(rank.sort_values("d", ascending=False).head(8).to_string())
# joint year-block bootstrap
rng = np.random.default_rng(0); n = 5000; Y = 22
DB = np.empty((n, len(cols)))
for b in range(n):
    i = rng.integers(0, Y, Y)
    ebs = {r: sh(EBm[r][i]) for r in (1, 5, 10)}
    DB[b] = [sh(M[c][i]) - ebs[c[1]] for c in cols]
se = DB.std(0, ddof=1); Cc = DB - DB.mean(0)
def report(sel, tag):
    sel = np.array(sel)
    t0 = d0[sel] / se[sel]
    # single-test p for B3 top1 (same rule as stats_sub: two-sided centred)
    j = list(sel).index(ib3[1])
    p_single = np.mean(np.abs(Cc[:, ib3[1]]) >= abs(d0[ib3[1]]))
    # White RC / Romano-Wolf single-step, raw and studentised, one-sided (H1: AI > EB) and two-sided
    mx_raw = Cc[:, sel].max(1); mx_st = (Cc[:, sel] / se[sel]).max(1); mxa_st = np.abs(Cc[:, sel] / se[sel]).max(1)
    p_raw = np.mean(mx_raw >= d0[ib3[1]]); p_st = np.mean(mx_st >= t0[j]); p_st2 = np.mean(mxa_st >= abs(t0[j]))
    # best-observed stat (what the search actually does): max over sel of t0
    p_best = np.mean(mx_st >= t0.max())
    print(f"[{tag}] m={len(sel)}  B3 top1 d={d0[ib3[1]]:.3f} t={t0[j]:.2f} | single two-sided p={p_single:.3f} | maxstat one-sided raw p={p_raw:.3f} stud p={p_st:.3f} | stud two-sided p={p_st2:.3f} | best t0 in family={t0.max():.2f} p={p_best:.3f}")
    # Romano-Wolf stepdown on studentised for top tests
    order = np.argsort(-t0); padj = []; prev = 0
    remaining = list(range(len(sel)))
    for o in order:
        p = np.mean((Cc[:, sel][:, remaining] / se[sel][remaining]).max(1) >= t0[o]); p = max(p, prev); prev = p; padj.append((o, p)); remaining.remove(o)
    return dict(padj)
all_idx = list(range(len(cols)))
top1_idx = [cols.index((k, 1)) for k in names]
sh_line = [i for i, c in enumerate(cols) if "prep=ysdm" in c[0]]
pa = report(all_idx, "all configs x top1/5/10")
pb = report(top1_idx, "all configs, top1 only")
pc = report(sh_line, "Sharpe-unit line x top1/5/10")
q67 = [i for i, c in enumerate(cols) if ("rc=True" in c[0])]
pd_ = report(q67, "q6-q9 (rc=True) x top1/5/10")
for tag, p, sel in (("all", pa, all_idx), ("top1", pb, top1_idx), ("rc", pd_, q67)):
    print(tag, "RW stepdown adj p for B3 top1/5/10:", {r: round(p[list(sel).index(ib3[r])], 3) for r in (1, 5, 10) if ib3[r] in sel})
# correlation structure / effective number of trials
R = np.array([np.r_[M[(k,1)].ravel() - EBm[1].ravel()] for k in names])   # dev monthly diff returns, top1
C = np.corrcoef(R); lam = np.linalg.eigvalsh(C); neff = lam.sum()**2/(lam**2).sum()
rho_bar = C[np.triu_indices(len(names), 1)].mean()
Rr = np.array([M[(k,1)].ravel() for k in names]); Cr = np.corrcoef(Rr); rbr = Cr[np.triu_indices(len(names),1)].mean()
print(f"top1 dev excess-return corr: mean rho={rho_bar:.3f}, eigen N_eff={neff:.1f}; raw-return corr mean={rbr:.3f}")
# Deflated-Sharpe style: expected max of N iid N(0,1) (Bailey & Lopez de Prado 2014 eq.), applied to the z of the SR difference
g = 0.5772156649
for K in (55, 3*55, int(round(neff)), 100):
    K = max(K, 2)
    Emax = (1-g)*norm.ppf(1-1/K) + g*norm.ppf(1-1/(K*np.e))
    zb = z0[ib3[1]]; zboot = d0[ib3[1]]/se[ib3[1]]
    print(f"N={K}: E[max z]~{Emax:.2f}; B3 top1 JKM z={zb:.2f} (bootstrap t={zboot:.2f}); deflated one-sided p (JKM) = {1-norm.cdf(zb-Emax):.3f}, (boot t) = {1-norm.cdf(zboot-Emax):.3f}")
# variance of trial SR diffs to scale (DSR uses SD of trial SRs)
sd_trials = np.std([d0[cols.index((k,1))] for k in names], ddof=1); print("SD across configs of top1 dev SR diff", round(sd_trials,3), "mean", round(np.mean([d0[cols.index((k,1))] for k in names]),3))
