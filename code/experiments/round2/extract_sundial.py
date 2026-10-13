"""Sundial (Liu et al., ICML 2025 Oral, arXiv 2502.00816; HF thuml/sundial-base-128m), PRETRAINED, đóng băng.
128 tháng (nhân dấu, NaN -> 0) -> 8 token x 768 (patch 16 tháng) -> gộp đôi thành 4 token -> PCA16 trắng hoá (fit <= 1982) -> sundial_pca16.npy."""
import sys, time, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import data
from transformers import AutoModelForCausalLM
T = data.load_table(); ids, wide = data.load_returns()
m = AutoModelForCausalLM.from_pretrained("thuml/sundial-base-128m", trust_remote_code=True).cuda().eval()

@torch.no_grad()
def hid(sel, b=4096):
    out = []
    for s in range(0, len(sel), b):
        W = torch.tensor(np.nan_to_num(data.windows(T.iloc[sel[s:s + b]], ids, wide)), device="cuda")
        h = m.model(input_ids=W, use_cache=False).last_hidden_state.float()      # [B, 8, 768]
        out.append(h.reshape(len(W), 4, 2, -1).mean(2))
    return torch.cat(out)
t0 = time.time()
fit = np.where(T.oos_year.values <= 1982)[0]; fit = fit[np.random.default_rng(0).permutation(len(fit))[:50000]]
Z = hid(fit).reshape(-1, 768); mu = Z.mean(0)
ev, V = torch.linalg.eigh(torch.cov((Z - mu).T)); V = V[:, -16:] / ev[-16:].clamp(min=1e-6).sqrt()
print("PCA giữ", float(ev[-16:].sum() / ev.sum()), flush=True)
H = np.lib.format.open_memmap(Path(__file__).parent / "sundial_pca16.npy", mode="w+", dtype=np.float16, shape=(len(T), 4, 16))
for s in range(0, len(T), 20000):
    sel = np.arange(s, min(s + 20000, len(T))); H[sel] = ((hid(sel) - mu) @ V).clamp(-5, 5).half().cpu().numpy()
H.flush(); print("xong", round(time.time() - t0), "s")
