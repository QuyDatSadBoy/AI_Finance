"""Trace every block of a trained model, step by step (inputs/outputs, shapes, distributions) and measure how much forecasting information
survives at each block with a linear probe: ridge regression fitted on year Y-1 (validation year, labels known before Y) from the block's output,
evaluated on year Y (mean within-family-year Spearman IC with the realised return).  Loads only the chosen families' hidden states (light).
Usage: python trace_blocks.py acct_ew T+hpca16 1990,1995,2000,2004"""
import sys, numpy as np, pandas as pd, torch, ht_common as h, ht_base as B
fams = sys.argv[1]; cfgname = sys.argv[2]; years = [int(y) for y in sys.argv[3].split(",")]
T = {**B.BASE, "xs": True, "prep": "ydm", "pool": "mha"}
CFG = {"goc": dict(B.BASE), "xs": {**B.BASE, "xs": True}, "xs+ydm+mha": T, "T+hpca16": {**T, "hpca": 16}, "Ts+hpca16": {**T, "hpca": 16, "prep": "ydms"}}[cfgname]
cfg = {**CFG, "fams": fams}
from ht_sub import make_sub
D = make_sub(fams); print("hidden of", fams, tuple(D.H.shape), flush=True)
def ic(p, y, g):
    d = pd.DataFrame(dict(p=p, y=y, g=g)); return d.groupby("g").apply(lambda t: t.p.corr(t.y, method="spearman"), include_groups=False).mean()
rows = []
for Y in years:
    pred = B.walk_forward(D, cfg, years=[Y - 1, Y], seed=0)          # trains for Y-1 then (warm start) for Y; keeps the net of year Y
    L = D._last; net = L["net"].eval()
    Et = torch.tensor(np.nan_to_num(np.clip((L["E"] - L["mu"]) / L["sd"], -L["clipv"], L["clipv"])).astype(np.float32), device="cuda")
    acts = {}
    hooks = [net.norm.register_forward_hook(lambda mod, i, o: acts.__setitem__("1 encoder (LayerNorm of FinCast)", o.mean(1))),
             net.pool.register_forward_hook(lambda mod, i, o: acts.__setitem__("3 attention pooling", o)),
             net.head[2].register_forward_hook(lambda mod, i, o: acts.__setitem__("5 FC 64 (after BN)", o))]
    if hasattr(net, "seq") and isinstance(net.seq, torch.nn.LSTM):
        hooks.append(net.seq.register_forward_hook(lambda mod, i, o: acts.__setitem__("2 Bi-LSTM", o[0].mean(1))))
    if net.xs is not None:
        hooks.append(net.xs.register_forward_hook(lambda mod, i, o: acts.__setitem__("4 set attention xs", o[0])))
    out = {}
    for yy in (Y - 1, Y):
        sel = np.where(D.yr == yy)[0]; gk = D.famcode[sel] * 10000 + yy
        reps = {}
        with torch.no_grad():
            for g in np.unique(gk):                                    # one family-year per forward (as xs expects)
                j = torch.tensor(sel[gk == g], device="cuda")
                for s0 in range(0, len(j), 2048):
                    jj = j[s0:s0 + 2048]; o = net(L["Hx"][jj].float(), Et[jj]); acts["6 output"] = o[:, None]
                    for k, v in acts.items(): reps.setdefault(k, []).append(v.float().cpu().numpy())
                    reps.setdefault("_rows", []).append(jj.cpu().numpy())
        r_ = np.concatenate(reps.pop("_rows")); out[yy] = (r_, {k: np.concatenate(v) for k, v in reps.items()})
    for hk in hooks: hk.remove()
    (rv, Av), (rt, At) = out[Y - 1], out[Y]
    yv, yt_ = D.y_or[rv], D.y_or[rt]; gv, gt = D.famcode[rv] * 10000 + Y - 1, D.famcode[rt] * 10000 + Y
    yv_dm = yv - pd.Series(yv).groupby(gv).transform("mean").values
    for k in sorted(At):
        Xv, Xt = Av[k], At[k]; mu_, sd0 = Xv.mean(0), Xv.std(0) + 1e-6; Xv, Xt = (Xv - mu_) / sd0, (Xt - mu_) / sd0
        lam = 10.0 * Xv.shape[1]; w = np.linalg.solve(Xv.T @ Xv + lam * np.eye(Xv.shape[1]), Xv.T @ yv_dm)
        rows.append(dict(year=Y, block=k, dim=Xv.shape[1], std=float(At[k].std()), probe_IC=ic(Xt @ w, yt_, gt),
                         raw_IC=ic(At[k][:, 0], yt_, gt) if k.startswith("6") else np.nan))
    rows.append(dict(year=Y, block="EB forecast", dim=1, std=float(D.eb_or[rt].std()), probe_IC=np.nan, raw_IC=ic(D.eb_or[rt], yt_, gt)))
    rows.append(dict(year=Y, block="0 extras tstat_IS", dim=1, std=np.nan, probe_IC=np.nan, raw_IC=ic(np.abs(D.F.tstat.values[rt]), yt_, gt)))
    rows.append(dict(year=Y, block="0 extras mean_IS (oriented)", dim=1, std=np.nan, probe_IC=np.nan, raw_IC=ic(D.F.mean_is.values[rt] * D.sg[rt], yt_, gt)))
    p_ = pred[rt]; pc = p_ - pd.Series(p_).groupby(gt).transform("mean").values; yc = yt_ - pd.Series(yt_).groupby(gt).transform("mean").values
    rows.append(dict(year=Y, block="7 final forecast (slope of y on it, within family-year)", dim=1, std=float(np.std(p_)), probe_IC=float((pc * yc).sum() / (pc ** 2).sum()), raw_IC=ic(p_, yt_, gt)))
res = pd.DataFrame(rows); pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
print("\nmean over years:\n", res.groupby("block")[["dim", "std", "probe_IC", "raw_IC"]].mean().round(3).to_string())
res.to_csv(h.CACHE / f"trace_{fams}_{cfgname}.csv", index=False)
