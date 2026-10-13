"""More seeds (up to 10) for the best configurations on one family, light data, own results file -- seed averaging is the strongest lever."""
import sys, pandas as pd, ht_common as h, ht_base as B
from ht_sub import make_sub
fams = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
B.RES2 = h.CACHE / f"famseeds_{fams}.tsv"
D = make_sub(fams); P = pd.read_parquet(h.CACHE / "panel_eb.parquet")
C = {"T": {**B.BASE, "xs": True, "prep": "ydm", "pool": "mha", "fams": fams}, "F2": {**B.BASE, "xs": True, "extras": "F2", "pool": "mha", "fams": fams}}
for nm, c in C.items():
    cs = []
    for sd in range(n):
        cc = {**c, "seed": sd}
        try: B.pred_of(cc)
        except FileNotFoundError: B.run_sub(cc, D, P)
        cs.append(cc)
    B.ensemble_sub(D, P, cs, f"seeds {nm} on {fams} x{n}", agg="median")
