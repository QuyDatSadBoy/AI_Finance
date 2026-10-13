import sys, glob, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parents[1]))
import data, evaluate as ev
T = data.load_table(); ids, wide = data.load_returns(); yr = T.oos_year.values
rows = []
for n in sorted({Path(f).name.rsplit("_s", 1)[0] for f in glob.glob(str(HERE / "preds" / "*_s*.npy"))}):
    fs = sorted(glob.glob(str(HERE / "preds" / f"{n}_s*.npy"))); p = np.mean([np.load(f) for f in fs], 0); p[yr > 2004] = np.nan
    R = ev.top_portfolios(T, ids, wide, ev.signal_from_oriented(T, p), years=(1983, 2004)); d = [ev.perf(R[r], *data.DEV)["sharpe"] for r in (1, 5, 10)]
    rows.append(dict(cfg=n, seeds=len(fs), dev=" / ".join(f"{x:.2f}" for x in d), dev_TB=round(np.mean(d), 3), IC_dev=round(ev.ic_years(T, p).loc[:2004].mean(), 3)))
print(pd.DataFrame(rows).sort_values("dev_TB", ascending=False).to_string(index=False))
