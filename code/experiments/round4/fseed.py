from eval4 import *
for s in range(6):
    f=HERE.parents[1]/"_cache/forecasts"/f"final_s{s}.npy"
    if f.exists(): print(row(f"final_s{s}", ev.signal_from_oriented(T,np.load(f)),1))
