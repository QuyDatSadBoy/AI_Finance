"""Read one configuration of a family the way the plan prescribes (CPU only, saved predictions): per seed and for the 10-seed mean,
dev / holdout MSE vs EB with the year-clustered DM t, the split of the squared error into the family-year LEVEL part and the WITHIN-year part
(each against EB's), within-year Spearman IC, within-year slope of the realised return on the forecast, and the seed spread.
Usage: python diag_acct.py acct_ew A0 A1 ...   (names from run_q6.py)  or  python diag_acct.py acct_ew 'name=k:v,k:v' ..."""
import sys, numpy as np, pandas as pd, ht_base as B
from ht_sub import make_sub
import run_q6_cfgs as Q

fam = sys.argv[1]
D = make_sub(fam, with_X=False, with_H=False)
gk = D.famcode * 10000 + D.yr


def parts(p):
    """dev/hold dict: total MSE vs EB, DM, level and within parts vs EB, IC, slope."""
    out = {}
    for nm, lo, hi in (("dev", 1983, 2004), ("hold", 2005, 2020)):
        m = (D.yr >= lo) & (D.yr <= hi) & np.isfinite(p)
        d = pd.DataFrame(dict(g=gk[m], yr=D.yr[m], p=p[m], e=D.eb_or[m], y=D.y_or[m]))
        for c in ("p", "e", "y"):
            d[c + "m"] = d.groupby("g")[c].transform("mean")
        lv = d.groupby("yr").apply(lambda x: pd.Series({"a": (x.pm.iloc[0] - x.ym.iloc[0]) ** 2, "b": (x.em.iloc[0] - x.ym.iloc[0]) ** 2}), include_groups=False)
        wi = d.assign(a=((d.p - d.pm) - (d.y - d.ym)) ** 2, b=((d.e - d.em) - (d.y - d.ym)) ** 2).groupby("yr")[["a", "b"]].mean()
        tot = lv + wi; diff = tot.a - tot.b
        out[nm] = dict(mse=100 * (1 - tot.a.sum() / tot.b.sum()), dm=diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))),
                       lvl=100 * (lv.b.sum() - lv.a.sum()) / tot.b.sum(), within=100 * (wi.b.sum() - wi.a.sum()) / tot.b.sum(),
                       ic=d.groupby("g").apply(lambda x: x.p.corr(x.y, method="spearman"), include_groups=False).mean(),
                       ic_eb=d.groupby("g").apply(lambda x: x.e.corr(x.y, method="spearman"), include_groups=False).mean(),
                       slope=((d.p - d.pm) * (d.y - d.ym)).sum() / ((d.p - d.pm) ** 2).sum())
    return out


def show(name, cfg, n=10):
    ps = []
    for s in range(n):
        try: ps.append(B.to_local(D, B.pred_of({**cfg, "seed": s})))
        except FileNotFoundError: break
    if not ps:
        print(name, "no predictions"); return
    single = [parts(p)["dev"]["mse"] for p in ps]
    for agg, p in (("mean", np.mean(ps, 0)), ("median", np.median(ps, 0))):
        r = parts(p); d, hd = r["dev"], r["hold"]
        print(f"{name:10s} x{len(ps)} {agg:6s} | DEV mse {d['mse']:+5.1f}% (DM {d['dm']:+.2f}) = level {d['lvl']:+5.1f} + within {d['within']:+5.1f} | IC {d['ic']:.3f} (EB {d['ic_eb']:.3f}) slope {d['slope']:.2f}"
              f" || hold mse {hd['mse']:+5.1f}% (DM {hd['dm']:+.2f}) IC {hd['ic']:.3f} (EB {hd['ic_eb']:.3f})")
    print(f"{'':10s} single-seed dev mse: mean {np.mean(single):+.1f}  sd {np.std(single):.1f}")


if __name__ == "__main__":
    for a in sys.argv[2:]:
        if "=" in a:
            nm, kv = a.split("=", 1); c = {**B.BASE, "fams": fam}
            for p_ in filter(None, kv.split(",")):
                k, v = p_.split(":"); c[k] = {"True": True, "False": False}.get(v, v)
            show(nm, c)
        else:
            show(a, Q.CFGS[a])
