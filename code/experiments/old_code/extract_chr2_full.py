"""Chronos-2 (amazon/chronos-2, PRETRAINED) encoder tokens of the oriented 240 months before the forecast year, for ALL Accounting-EW strategy-years, projected to the
whitened PCA16 bottleneck fitted on <=1982 rows (no look-ahead).  Writes _cache/full_acct_chr2_pca16.npy [N,4,16].  Run: python extract_chr2_full.py"""
import time, numpy as np, torch
import ht_common as h
from chronos import Chronos2Pipeline
C = h.CACHE
W = np.load(C / "full_acct_win240.npy", mmap_mode="r"); yr = np.load(C / "full_acct_meta.npz")["yr"]
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")


@torch.no_grad()
def tokens(rows, bs=4096):
    out = []
    for s in range(0, len(rows), bs):
        w = np.ascontiguousarray(W[rows[s:s + bs]])
        embs, _ = pipe.embed([torch.tensor(x) for x in w], batch_size=1024)
        e = torch.cat([x.float() for x in embs], 0)
        T = e.shape[1] - (e.shape[1] % 4)
        out.append(e[:, -T:].reshape(e.shape[0], 4, T // 4, e.shape[2]).mean(2))
    return torch.cat(out)

t0 = time.time()
fit = np.where(yr <= 1982)[0]; fit = fit[np.random.default_rng(0).permutation(len(fit))[:50000]]; fit.sort()
Z = tokens(fit).reshape(-1, tokens(fit[:2]).shape[-1]); mu = Z.mean(0)
ev, V = torch.linalg.eigh(torch.cov((Z - mu).T)); V = V[:, -16:] / ev[-16:].clamp(min=1e-6).sqrt(); del Z
print("PCA fitted", round(time.time() - t0), "s; variance share kept", float(ev[-16:].sum() / ev.sum()), flush=True)
out = np.lib.format.open_memmap(C / "full_acct_chr2_pca16.npy", mode="w+", dtype=np.float16, shape=(len(W), 4, 16))
CH = 20000
for s in range(0, len(W), CH):
    sel = np.arange(s, min(s + CH, len(W))); out[sel] = (((tokens(sel) - mu) @ V).clamp(-5, 5)).half().cpu().numpy()
    if (s // CH) % 5 == 0: print(f"{sel[-1] + 1}/{len(W)} {round(time.time() - t0)}s", flush=True)
out.flush(); print("done", round(time.time() - t0), "s")
