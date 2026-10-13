"""FinCast (frozen) hidden states for ALL Accounting-EW strategy-years (not only the top-10% |t| candidates), already projected to the whitened
PCA16 bottleneck that ht_base.walk_forward(hpca=16) uses (fitted on <=1982 rows only: no look-ahead).  Writes
_cache/full_acct_pca16.npy [N,4,16] float16 and _cache/full_acct_rows.npy (positions of the rows in the feature panel F, same order)."""
import sys, time, numpy as np, pandas as pd, torch
import ht_common as h, ht_wf as wf, ht_fincast as fc

F = wf.load_all()
rows = np.where(F.family.values == "acct_ew")[0]
G = F.iloc[rows].reset_index(drop=True)
ids, wide = h.load_wide("acct_ew"); pos = pd.Series(np.arange(len(ids)), index=ids)
m = fc.load_encoder()


def windows(sel):
    """sel: positions in G (one block). Oriented 128-month windows."""
    g = G.iloc[sel]
    r = pos.loc[g.signalid.values].values; b = (g.oos_year.values - h.BASE_YEAR) * 12
    W = np.full((len(sel), fc.CTX), np.nan, np.float32)
    for i in range(len(sel)):
        lo = b[i] - fc.CTX
        w = wide[r[i], max(lo, 0):b[i]]
        W[i, fc.CTX - len(w):] = w
    return W * g.sgn.values[:, None].astype(np.float32)


@torch.no_grad()
def hid(sel, batch=2048):
    out = []
    for s in range(0, len(sel), batch):
        W = torch.tensor(windows(sel[s:s + batch]), device="cuda")
        out.append(fc.hidden(m, W).float())            # [B,4,1280]
    return torch.cat(out)

t0 = time.time()
yr = G.oos_year.values
fit = np.where(yr <= 1982)[0]; fit = fit[np.random.default_rng(0).permutation(len(fit))[:50000]]
Z = hid(fit).reshape(-1, 1280); mu = Z.mean(0)
ev, V = torch.linalg.eigh(torch.cov((Z - mu).T)); V = V[:, -16:] / ev[-16:].clamp(min=1e-6).sqrt()
del Z; print("PCA fitted", round(time.time() - t0), "s; kept variance share", float(ev[-16:].sum() / ev.sum()), flush=True)
out = np.lib.format.open_memmap(h.CACHE / "full_acct_pca16.npy", mode="w+", dtype=np.float16, shape=(len(G), 4, 16))
CH = 20000
for s in range(0, len(G), CH):
    sel = np.arange(s, min(s + CH, len(G)))
    H = hid(sel)
    out[sel] = (((H - mu) @ V).clamp(-5, 5)).half().cpu().numpy()
    print(f"{s + len(sel)}/{len(G)} {round(time.time() - t0)}s", flush=True)
out.flush(); np.save(h.CACHE / "full_acct_rows.npy", rows)
print("done", round(time.time() - t0), "s")
