"""Selection-bias check (from the adversarial verification): build.py rebuilds the dev top-k monthly returns of every acct_ew config
(10-seed mean) into _cache/verify/rets.pkl; maxstat.py runs a joint year-block bootstrap max-statistic / Romano-Wolf correction and a
deflated-Sharpe check for the config named by TARGET (default B3).  Usage (from code/): PYTHONPATH=. python verify/build.py; TARGET=Dt0 PYTHONPATH=. python verify/maxstat.py"""
import os, glob, hashlib, pickle, numpy as np, pandas as pd, ht_common as h, ht_base as B, ht_arch as A, ht_new as nw
from ht_sub import make_sub
OUT = str(h.CACHE / "verify") + "/"
rows = []
for f in sorted(glob.glob(str(h.CACHE / "*.tsv"))):
    if "base_results" in f: continue
    d = pd.read_csv(f, sep="\t", usecols=["name"]); d["file"] = os.path.basename(f); rows.append(d)
R = pd.concat(rows); R = R[~R.name.str.startswith("ENSEMBLE")]
R = R[R.name.str.contains(r"fams=acct_ew(?:\||$)", regex=True)]
R["key"] = R.name.map(lambda n: "|".join(p for p in n.split("|") if not p.startswith("seed=")))
R["seed"] = R.name.str.extract(r"seed=(\d+)").astype(int)
R = R.drop_duplicates("name")
R["path"] = R.name.map(lambda n: B.PRED_DIR / (hashlib.md5(n.encode()).hexdigest() + ".npy"))
R["exists"] = R.path.map(lambda p: p.exists())
print("acct_ew single rows", len(R), "with pred file", R.exists.sum(), "distinct cfg", R.key.nunique())
D = make_sub("acct_ew", with_X=False, with_H=False); P = pd.read_parquet(h.CACHE / "panel_eb.parquet")
keys = pd.DataFrame(dict(family=D.F.family.values, signalid=D.F.signalid.values, oos_year=D.yr))
P_c = P[P.family == "acct_ew"].merge(keys, on=["family", "signalid", "oos_year"])
_, eb = A.evaluate(P_c, D, D.eb_or.astype(np.float32), "T0", "sr", 1.0, schemes=("eq",))
res = {"EB": {r: eb[(r, "eq")] for r in (1, 5, 10)}}; meta = {}
for k, g in R[R.exists].groupby("key"):
    g = g[g.seed < 10]
    ps = []
    for p in g.path:
        x = np.load(p); x = B.to_local(D, x) if len(x) == B.N_FULL else x
        if len(x) != len(D.yr): print("len mismatch", len(x)); break
        ps.append(x)
    else:
        pred = np.mean(ps, 0).astype(np.float32)
        _, rets = A.evaluate(P_c, D, pred, "T0", "sr", 1.0, schemes=("eq",))
        res[k] = {r: rets[(r, "eq")] for r in (1, 5, 10)}; meta[k] = dict(nseed=len(ps), files=",".join(sorted(set(g.file))))
        print(len(ps), {r: round(nw.perf(res[k][r], 1983, 2004)["sharpe"], 3) for r in (1, 5, 10)}, k[:150], flush=True)
pickle.dump((res, meta), open(OUT + "rets.pkl", "wb"))
