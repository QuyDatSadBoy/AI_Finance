from eval4 import *
E=dict(d=[1.549,1.913,1.862],h=[0.818,0.778,0.855])
L=lambda d:[np.load(f"{d}/final_s{s}.npy") for s in range(6)]
groups={"old":L("../old_final/forecasts"),"first":L("../final_first_run"),"cur":L("../../_cache/forecasts"),"nosw":L("../old_final2")}
for n in sorted({Path(f).name.rsplit("_s",1)[0] for f in glob.glob(str(HERE/"preds"/"*_s*.npy"))}):
    groups[n]=[np.load(f) for f in sorted(glob.glob(str(HERE/"preds"/f"{n}_s*.npy")))]
rows=[]
for n,P in groups.items():
    p=np.nanmean(P,0); Pt=ev.top_portfolios(T,ids,wide,ev.signal_from_oriented(T,p))
    d,h=sr(Pt,1983,2004),sr(Pt,2005,2020)
    win=sum(a>b for a,b in zip(d,E['d']))+sum(a>b for a,b in zip(h,E['h']))
    nseed=sum(1 for _ in P)
    rows.append(dict(cfg=n,seeds=nseed,dev=" / ".join(f"{x:.2f}" for x in d),hold=" / ".join(f"{x:.2f}" for x in h),win6=win))
print(pd.DataFrame(rows).sort_values("win6",ascending=False).to_string(index=False))
