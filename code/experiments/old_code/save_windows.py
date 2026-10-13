"""Save the oriented return windows of the candidate rows (CPU only) so that encoders living in another environment (TiRex) can embed them."""
import numpy as np, ht_common as h, ht_arch as A
from stats_final import LightD
D = LightD()
for ctx in (128, 240):
    W = A.oriented_windows(D, ctx=ctx).astype(np.float32); np.save(h.CACHE / f"win{ctx}.npy", W); print(ctx, W.shape, np.isnan(W).mean(), flush=True)
