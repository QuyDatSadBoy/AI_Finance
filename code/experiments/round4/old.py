from eval4 import *
for nm,d in [("old_final(nền |t|, ISAB)","old_final/forecasts"),("old_final2(no SwiGLU)","old_final2")]:
    F=[np.load(f) for f in sorted(glob.glob(f"{HERE.parent}/{d}/final_s*.npy"))]
    print(row(nm, ev.signal_from_oriented(T,np.nanmean(F,0)),len(F)))
