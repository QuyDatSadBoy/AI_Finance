from eval4 import *
E=dict(d=[1.549,1.913,1.862],h=[0.818,0.778,0.855])
items=[]
for nm,d in [("old","../old_final/forecasts"),("first","../final_first_run"),("cur","../../_cache/forecasts"),("nosw","../old_final2")]:
    for s in range(6): items.append((f"{nm}_s{s}",np.load(f"{d}/final_final_s{s}.npy".replace("final_final","final"))))
for f in sorted(glob.glob(str(HERE/"preds"/"*_s*.npy"))): items.append((Path(f).stem,np.load(f)))
rows=[]
for nm,p in items:
    P=ev.top_portfolios(T,ids,wide,ev.signal_from_oriented(T,p))
    d,dl,h=sr(P,1983,2004),sr(P,1995,2004),sr(P,2005,2020)
    win=sum(a>b for a,b in zip(d,E['d']))+sum(a>b for a,b in zip(h,E['h']))
    rows.append(dict(cfg=nm,dev=" / ".join(f"{x:.2f}" for x in d),devlate=round(np.mean(dl),3),hold=" / ".join(f"{x:.2f}" for x in h),hold_TB=round(np.mean(h),3),win6=win))
D=pd.DataFrame(rows).sort_values("devlate",ascending=False)
print(D.to_string(index=False)); D.to_csv(HERE/"all_seeds.csv",index=False)
