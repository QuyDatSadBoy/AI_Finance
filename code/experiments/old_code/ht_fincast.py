"""FinCast (CIKM 2025, Zhu et al.) frozen encoder: 128 monthly returns -> hidden sequence 4 x 1280 (4 patches of 32).

Uses the authors' public code (cloned to Research/tools/FinCast-fts) and weights (HF: Vincent05R/FinCast, v1.pth).
We only run the decoder stack up to the last transformer layer; the forecasting heads are not used.
"""
import sys, numpy as np, pandas as pd, torch
import ht_common as h

FINCAST_SRC = h.ROOT / "Research" / "tools" / "FinCast-fts" / "src"
CTX = 128          # context length in months -> 4 patches
PATCH = 32


def load_encoder(device="cuda"):
    import importlib.util
    sys.path.insert(0, str(FINCAST_SRC))               # st_moe_pytorch is imported as a top-level package
    from huggingface_hub import hf_hub_download
    spec = importlib.util.spec_from_file_location("ppd", FINCAST_SRC / "ffm" / "pytorch_patched_decoder_MOE.py")
    ppd = importlib.util.module_from_spec(spec); sys.modules["ppd"] = ppd; spec.loader.exec_module(ppd)   # skip the jax-dependent package __init__
    cfg = ppd.FFMConfig(num_layers=50, num_heads=16, hidden_size=1280, intermediate_size=1280, patch_len=PATCH,
                        horizon_len=128, head_dim=80, quantiles=ppd.create_quantiles(), use_positional_embedding=False,
                        num_experts=4, gating_top_n=2, threshold_train=0.2, threshold_eval=0.2)
    m = ppd.PatchedTimeSeriesDecoder_MOE(cfg)
    sd = torch.load(hf_hub_download("Vincent05R/FinCast", "v1.pth"), map_location="cpu", weights_only=True)
    sd = {k.replace("_orig_mod.module.", "").replace("_orig_mod.", "").replace("module.", "", 1) if k.startswith(("module.", "_orig_mod")) else k: v
          for k, v in sd.items()}
    m.load_state_dict(sd, strict=True)
    return m.to(device=device).eval()          # fp32 weights (4 GB); bf16 autocast at run time


@torch.no_grad()
def hidden(m, x):
    """x: float tensor [B, 128] on device (NaN = missing). Returns [B, 4, 1280] hidden sequence (last decoder layer)."""
    dev = x.device
    pad = torch.isnan(x).float()
    x = torch.nan_to_num(x, nan=0.0).float()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        mi, pp, _, _ = m._preprocess_input(x, pad)
        mi = mi + m.freq_emb(torch.ones(x.shape[0], 1, dtype=torch.long, device=dev))      # freq id 1: monthly/low-freq
        out, _ = m.stacked_transformer(mi, pp)
    return out


CAND_Q_FC = 0.90      # hidden sequences are stored for the top 10% |t| strategies of every family-year


@torch.no_grad()
def extract(F, out_path, batch=2048, q=CAND_Q_FC):
    """F: feature panel (family, signalid, oos_year, sgn, abs_t_rank). Writes float16 memmap [n, 4, 1280] + row index."""
    m = load_encoder()
    mask = (F.abs_t_rank >= q).values
    rows = np.where(mask)[0]
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(len(rows), 4, 1280))
    sub = F.iloc[rows]
    k0 = 0
    for (fam, y), g in sub.groupby(["family", "oos_year"], sort=False):
        ids, wide = h.load_wide(fam)
        pos = pd.Series(np.arange(len(ids)), index=ids)
        r = pos.loc[g.signalid.values].values
        b = (y - h.BASE_YEAR) * 12
        lo = b - CTX
        W = wide[r, max(lo, 0):b] * g.sgn.values[:, None]
        if lo < 0:
            W = np.pad(W, ((0, 0), (-lo, 0)), constant_values=np.nan)
        idx = sub.index.get_indexer(g.index)
        for s in range(0, len(r), batch):
            o = hidden(m, torch.tensor(W[s:s + batch], device="cuda", dtype=torch.float32)).half().cpu().numpy()
            mm[idx[s:s + batch]] = o
    mm.flush()
    np.save(str(out_path).replace(".npy", "_rows.npy"), rows)
    return rows
