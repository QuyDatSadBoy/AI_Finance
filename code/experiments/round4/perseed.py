from eval4 import *
rows=[]
for nm,d in [("old","../old_final/forecasts"),("cur","../../_cache/forecasts"),("first","../final_first_run"),("nosw","../old_final2")]:
    for s in range(6):
        p=np.load(f"{d}/final_s{s}.npy"); rows.append(row(f"{nm}_s{s}", ev.signal_from_oriented(T,p),1))
print(pd.DataFrame(rows).to_string(index=False))
