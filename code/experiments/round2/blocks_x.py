"""HyperAttention (Han et al., ICLR 2024) dạng set attention, chép nguyên từ mã thử nghiệm cũ (old_code/ht_base.py)."""
import torch, torch.nn as nn


class HyperSetAttention(nn.Module):
    """HyperAttention-style self-attention over a set (Han et al., ICLR 2024): sortLSH puts similar elements next to each other and attention is
    computed exactly inside blocks of size `block` (the large entries); the remaining keys are estimated from m uniformly sampled keys scaled by
    (n - block)/m.  Cost O(n*(block+m)) per head, so a batch can be a whole family-year.  With block >= n it is exact softmax attention."""

    def __init__(self, d, heads=4, block=256, m=256, bits=8):
        super().__init__()
        self.h, self.dh, self.block, self.m = heads, d // heads, block, m
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.register_buffer("planes", torch.randn(heads, self.dh, bits))
        self.register_buffer("pow2", 2 ** torch.arange(bits))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def attend(self, x):                                                  # x [n, d] -> [n, d]
        n = x.shape[0]
        q, k, v = self.qkv(x).view(n, 3, self.h, self.dh).unbind(1)          # each [n, h, dh]
        q, k, v = (t.transpose(0, 1) * (1.0 if i else self.dh ** -0.5) for i, t in enumerate((q, k, v)))  # [h, n, dh]; scale q
        b = min(self.block, n); nb = -(-n // b); pad = nb * b - n
        code = ((torch.einsum("hnd,hdr->hnr", k, self.planes) > 0).long() * self.pow2).sum(-1)       # SimHash bucket of each key/query
        perm = torch.argsort(code, dim=1)                                    # sortLSH: same permutation for q and k (self-attention)
        g = lambda t: torch.gather(t, 1, perm[..., None].expand(-1, -1, t.shape[-1]))
        qs, ks, vs = g(q), g(k), g(v)
        if pad:
            qs, ks, vs = (torch.cat([t, t[:, :pad]], 1) for t in (qs, ks, vs))
        qb, kb, vb = (t.view(self.h, nb, b, self.dh) for t in (qs, ks, vs))
        s_blk = qb @ kb.transpose(-1, -2)                                   # [h, nb, b, b] exact large entries
        if pad:                                                             # mask the padded keys of the last block
            valid = torch.arange(nb * b, device=x.device).view(nb, b) < n
            s_blk = s_blk.masked_fill(~valid[None, :, None, :], float("-inf"))
        rest = n - b
        if self.m > 0 and rest > 0:
            idx = torch.randint(0, n, (self.m,), device=x.device)            # uniform sample of keys (shared by all queries)
            ksm, vsm = k[:, idx], v[:, idx]                                  # [h, m, dh]
            s_smp = qs.view(self.h, nb, b, self.dh) @ ksm.transpose(-1, -2)[:, None]       # [h, nb, b, m]
            blk_of_key = torch.empty(n, dtype=torch.long, device=x.device)
            blk_of_key[perm[0]] = torch.arange(n, device=x.device) // b         # block id of every key (head 0 permutation)
            same = (torch.arange(nb, device=x.device)[:, None] == blk_of_key[idx][None, :])   # sampled key already counted in the block
            s_smp = s_smp.masked_fill(same[None, :, None, :], float("-inf"))
            w = torch.log(torch.tensor(rest / self.m, device=x.device))          # each sample stands for rest/m keys
            mx = torch.maximum(s_blk.amax(-1, keepdim=True), (s_smp + w).amax(-1, keepdim=True)).detach()
            e1, e2 = torch.exp(s_blk - mx), torch.exp(s_smp + w - mx)
            out = (e1 @ vb + e2 @ vsm[:, None]) / (e1.sum(-1, keepdim=True) + e2.sum(-1, keepdim=True))
        else:
            out = torch.softmax(s_blk, -1) @ vb
        out = out.reshape(self.h, nb * b, self.dh)[:, :n]
        inv = torch.empty_like(perm); inv.scatter_(1, perm, torch.arange(n, device=x.device).expand(self.h, -1))
        out = torch.gather(out, 1, inv[..., None].expand(-1, -1, self.dh))   # undo the permutation
        return self.o(out.transpose(0, 1).reshape(n, -1))

    def forward(self, x):                                                  # x [1, n, d]
        y = x[0] + self.attend(self.n1(x[0]))
        return (y + self.ff(self.n2(y)))[None]
