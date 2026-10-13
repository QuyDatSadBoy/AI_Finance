from eval4 import *
p=np.nanmean([np.load(f"preds/feat_s{s}.npy") for s in (0,1)],0)
A=ev.top_portfolios(T,ids,wide,ev.signal_from_oriented(T,p)); B=ev.top_portfolios(T,ids,wide,EB)
print(ev.compare(A,B,"AI","EB").to_string(index=False))
