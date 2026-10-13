"""EXAONE Forecast for Finance (LG AI Research, arXiv 2609.04239; PRETRAINED on synthetic financial data; non-commercial licence) encoder tokens for ALL Accounting-EW
strategy-years.  Input: the oriented 240 monthly returns before the forecast year (NaN = missing; full_acct_win240.npy).  Tokens = [mean of context patches 0-4, 5-9,
10-14, the forecast-patch embedding] (4 x 1024) -> whitened PCA16 fitted on <=1982 rows (no look-ahead), like the other encoders.  Run with venv_tsfm:
python extract_exaone_full.py [device] [n_rows_for_test]"""
import sys, time, numpy as np, torch
from pathlib import Path
from exaone_forecast.finance import from_pretrained
C = Path(__file__).resolve().parent / "_cache"
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"; nmax = int(sys.argv[2]) if len(sys.argv) > 2 else None
W = np.load(C / "full_acct_win240.npy", mmap_mode="r"); yr = np.load(C / "full_acct_meta.npz")["yr"]
if nmax: W = W[:nmax]; yr = yr[:nmax]
fc = from_pretrained(device=dev); m = fc.model.eval()


@torch.no_grad()
def tokens(rows, bs=1024):
    out = []
    for s in range(0, len(rows), bs):
        cb = torch.from_numpy(np.ascontiguousarray(W[rows[s:s + bs]])).to(dev)
        enc = m.encode(context=cb, group_ids=torch.arange(cb.shape[0], device=dev), num_output_patches=1)[0]
        h = enc[0] if isinstance(enc, tuple) else enc.last_hidden_state if hasattr(enc, "last_hidden_state") else enc[0]
        n = h.shape[1]; ctx = h[:, :15]
        out.append(torch.stack([ctx[:, 0:5].mean(1), ctx[:, 5:10].mean(1), ctx[:, 10:15].mean(1), h[:, -1]], 1).float())
    return torch.cat(out)

t0 = time.time()
fit = np.where(yr <= 1982)[0]; fit = fit[np.random.default_rng(0).permutation(len(fit))[:50000]]; fit.sort()
Z = tokens(fit).reshape(-1, 1024); mu = Z.mean(0)
ev, V = torch.linalg.eigh(torch.cov((Z - mu).T)); V = V[:, -16:] / ev[-16:].clamp(min=1e-6).sqrt()
print("PCA fitted", round(time.time() - t0), "s; variance share kept", float(ev[-16:].sum() / ev.sum()), flush=True)
out = np.lib.format.open_memmap(C / ("full_acct_exaone_pca16%s.npy" % ("_test" if nmax else "")), mode="w+", dtype=np.float16, shape=(len(W), 4, 16))
CH = 20000
for s in range(0, len(W), CH):
    sel = np.arange(s, min(s + CH, len(W)))
    out[sel] = (((tokens(sel) - mu) @ V).clamp(-5, 5)).half().cpu().numpy()
    if (s // CH) % 5 == 0: print(f"{sel[-1] + 1}/{len(W)} {round(time.time() - t0)}s", flush=True)
out.flush(); print("done", round(time.time() - t0), "s")
