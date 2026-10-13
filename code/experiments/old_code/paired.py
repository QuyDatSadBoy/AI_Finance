"""Seed-blocked comparison of a variant against its base (experimental-design: the replicate is the training run, blocked by seed).
Usage: python paired.py 'pool=mha|prep=ydm|xs=True' 'pool=mha|prep=ydm|shuf=True|xs=True'   (keys that differ from BASE)
Prints per-seed dev MSE-vs-EB difference (variant - base), its mean, standard error and t over the paired seeds."""
import sys, numpy as np, pandas as pd, ht_base as B
NUM = dict(drop=float, lr=float, wd=float, hid=int, epochs=int, seed=int, xl=int, xset=int, lw=int, ema=float, hdelta=float, xm=int,
           xs=lambda v: v == "True", shuf=lambda v: v == "True")
def parse(name):
    return {**B.BASE, **{k: NUM.get(k, str)(v) for k, v in (p.split("=", 1) for p in name.split("|") if p)}}
def key(c): return tuple(sorted((k, str(v)) for k, v in c.items() if k != "seed" and not (k in ("xl",) and str(v) == "1")))
R = pd.read_csv(B.RES, sep="\t"); R = R[~R.name.str.startswith("ENSEMBLE")]
R["cfg"] = [key(parse(n)) for n in R.name]; R["seed"] = [parse(n)["seed"] for n in R.name]
R = R.drop_duplicates(["cfg", "seed"], keep="last")
def runs(spec): return R[R.cfg == key(parse(spec))].set_index("seed")
def compare(base, var, col="mse_vs_eb_dev"):
    a, b = runs(base), runs(var); s = sorted(set(a.index) & set(b.index))
    if len(s) < 2: return print(f"  {var}: only {len(s)} paired seeds")
    d = b.loc[s, col] - a.loc[s, col]; se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"  {var:60s} seeds {s} | diff {d.mean():+.2f} pts (se {se:.2f}, t {d.mean() / se:+.2f}) | per seed {np.round(d.values, 2).tolist()}")
if __name__ == "__main__":
    for col in ("mse_vs_eb_dev", "top5_dev_sr", "top10_dev_sr"):
        print(col); [compare(sys.argv[1], v, col) for v in sys.argv[2:]]
