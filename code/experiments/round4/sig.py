from eval4 import *
p=np.load("../old_final/forecasts/final_s1.npy"); A=ev.top_portfolios(T,ids,wide,ev.signal_from_oriented(T,p)); B=ev.top_portfolios(T,ids,wide,EB)
print(ev.compare(A,B,"AI","EB").to_string(index=False))
print("IC", ev.ic_test(ev.ic_years(T,p), ev.ic_years(T,EB[0]*T.sd_is.values), 1983, 2004), ev.ic_test(ev.ic_years(T,p), ev.ic_years(T,EB[0]*T.sd_is.values), 2005, 2020))
