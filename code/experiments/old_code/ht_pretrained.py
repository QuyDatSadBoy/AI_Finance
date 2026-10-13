"""Zero-shot forecasts from a pretrained time-series foundation model (Chronos-Bolt, Amazon) on the GPU.

For every (strategy, year Y) the context is the last 240 months of returns *before* Y, oriented by sgn.
The model never sees Y or later, and it is not fine-tuned here (zero-shot), so no look-ahead is possible.
"""
import numpy as np, pandas as pd, torch
import ht_common as h

QL = [0.1, 0.5, 0.9]


def load_pipeline(name="amazon/chronos-bolt-small"):
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline.from_pretrained(name, device_map="cuda", torch_dtype=torch.bfloat16)


@torch.no_grad()
def forecast_family_year(pipe, fam, year, sgn, rows_ok, batch=8192, context=240):
    ids, wide = h.load_wide(fam)
    b = (year - h.BASE_YEAR) * 12
    W = wide[:, max(b - context, 0):b] * sgn[:, None]
    out = np.full((len(ids), 4), np.nan, dtype=np.float32)
    idx = np.where(rows_ok)[0]
    for s in range(0, len(idx), batch):
        j = idx[s:s + batch]
        x = torch.tensor(W[j])
        q, mean = pipe.predict_quantiles(x, prediction_length=12, quantile_levels=QL)
        q, mean = q.float().cpu().numpy(), mean.float().cpu().numpy()
        out[j, 0] = mean.mean(1)                    # mean forecast, averaged over the 12 horizons
        out[j, 1] = q[:, :, 1].mean(1)              # median
        out[j, 2] = q[:, :, 0].mean(1)              # q10
        out[j, 3] = q[:, :, 2].mean(1)              # q90
    return out


def build(F, name="amazon/chronos-bolt-small", tag="small"):
    """F = feature panel (family, signalid, oos_year, sgn). Returns F-aligned DataFrame of forecasts."""
    pipe = load_pipeline(name)
    res = np.full((len(F), 4), np.nan, dtype=np.float32)
    for (fam, y), g in F.groupby(["family", "oos_year"]):
        ids, wide = h.load_wide(fam)
        pos = pd.Series(np.arange(len(ids)), index=ids)
        r = pos.loc[g.signalid.values].values
        sgn_full = np.ones(len(ids), np.float32); sgn_full[r] = g.sgn.values
        ok = np.zeros(len(ids), bool); ok[r] = True
        o = forecast_family_year(pipe, fam, y, sgn_full, ok)
        res[F.index.get_indexer(g.index)] = o[r]
    return pd.DataFrame(res, columns=[f"cf_mean_{tag}", f"cf_med_{tag}", f"cf_q10_{tag}", f"cf_q90_{tag}"], index=F.index)


@torch.no_grad()
def embed_rows(F, mask, name="amazon/chronos-bolt-small", batch=4096, context=240):
    """Mean-pooled Chronos encoder embedding (512-d) of the oriented 240-month history for rows where mask is True.
    Returns float16 array [mask.sum(), d] in the row order of F[mask]."""
    pipe = load_pipeline(name)
    sub = F[mask]
    out = None
    for (fam, y), g in sub.groupby(["family", "oos_year"], sort=False):
        ids, wide = h.load_wide(fam)
        pos = pd.Series(np.arange(len(ids)), index=ids)
        r = pos.loc[g.signalid.values].values
        b = (y - h.BASE_YEAR) * 12
        W = wide[r, max(b - context, 0):b] * g.sgn.values[:, None]
        for s in range(0, len(r), batch):
            e, _ = pipe.embed(torch.tensor(W[s:s + batch]))
            e = e.float().mean(1).cpu().numpy().astype(np.float16)
            if out is None:
                out = np.zeros((len(sub), e.shape[1]), np.float16)
            out[sub.index.get_indexer(g.index[s:s + batch])] = e
    return out
