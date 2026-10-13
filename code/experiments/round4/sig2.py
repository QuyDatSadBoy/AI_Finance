from eval4 import *
L=lambda d,s:np.load(f"{d}/final_s{s}.npy")
for nm,p in [("old_s5+cur_s5",np.nanmean([L("../old_final/forecasts",5),L("../../_cache/forecasts",5)],0)),("old+cur12",np.nanmean([L(d,s) for d in ["../old_final/forecasts","../../_cache/forecasts"] for s in range(6)],0))]:
    A=ev.top_portfolios(T,ids,wide,ev.signal_from_oriented(T,p)); B=ev.top_portfolios(T,ids,wide,EB)
    print(nm); print(ev.compare(A,B,"AI","EB").to_string(index=False))
