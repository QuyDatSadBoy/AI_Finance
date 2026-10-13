from eval4 import *
sh = np.abs(T.mean_is.values / T.sd_is.values); sg = np.sign(T.mean_is.values)
print(T.nobs.describe())
rows = [row("abs_sharpe_is", (sg, sg, sh))]
for c in [2, 3, 4]:
    rows.append(row(f"t>{c}_then_sharpe", (sg, sg, sh * (np.abs(T.tstat.values) > c))))
print(pd.DataFrame(rows).to_string(index=False))
