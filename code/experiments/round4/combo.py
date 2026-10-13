from eval4 import *
L=lambda d:[np.load(f"{d}/final_s{s}.npy") for s in range(6)]
old,cur,first,nosw=L("../old_final/forecasts"),L("../../_cache/forecasts"),L("../final_first_run"),L("../old_final2")
m=lambda P:np.nanmean(P,0)
rows=[row(n,ev.signal_from_oriented(T,m(P)),len(P)) for n,P in [("old6",old),("first6",first),("old+first12",old+first),("old+cur12",old+cur),("old+nosw12",old+nosw),("all24",old+cur+first+nosw)]]
for s in range(6): rows.append(row(f"old_s{s}+cur_s{s}",ev.signal_from_oriented(T,m([old[s],cur[s]])),2))
print(pd.DataFrame(rows).to_string(index=False))
