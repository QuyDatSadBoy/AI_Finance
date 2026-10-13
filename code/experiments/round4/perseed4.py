from eval4 import *
rows=[]
for f in sorted(glob.glob(str(HERE/"preds"/"*_s*.npy"))):
    rows.append(row(Path(f).stem, ev.signal_from_oriented(T,np.load(f)),1))
print(pd.DataFrame(rows).to_string(index=False))
