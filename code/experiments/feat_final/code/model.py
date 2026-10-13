"""Kiến trúc mô hình chốt, từng khối một.

    dấu vân tay FinCast+PCA16 [n, 4, 16]
      ① LayerNorm                                        -> [n, 4, 16]
      ② Bi-LSTM (128 mỗi chiều)                          -> [n, 4, 256]   đọc 4 đoạn theo thời gian
      ③ Pool: self-attention giữa 4 đoạn + gộp có trọng số -> [n, 256]     1 vector / chiến lược
      ④ ISAB (Set Transformer, ICML 2019), 64 điểm tóm tắt -> [n, 256]     so với cả năm, qua bản tóm tắt
      ⑤ HyperAttention (Han et al., ICLR 2024) có cổng    -> [n, 256]     các chiến lược nhìn trực tiếp nhau
      ⑥ ghép 5 số (t-stat, lãi TB, số tháng, Sharpe in-sample, log độ dao động; đã chuẩn hoá) -> [n, 261]
      ⑦ SwiGLU (Shazeer 2020): SiLU(W1 z) ⊙ (W2 z) -> LayerNorm -> Dropout -> Linear 1 -> [n]   đáp án dự báo

n = các chiến lược của CÙNG một năm (lô 4.096 khi học, cả năm khi dự báo).
"""
import torch, torch.nn as nn


class Pool(nn.Module):
    """③ [n, T, d] -> [n, d]. Mỗi đoạn nhìn các đoạn còn lại (multi-head self-attention, cộng dư),
    rồi gộp: điểm_t = v . tanh(W x_t), softmax qua T đoạn, lấy tổng có trọng số."""

    def __init__(self, d, heads=4):
        super().__init__()
        self.proj, self.v = nn.Linear(d, d), nn.Linear(d, 1, bias=False)
        self.mha = nn.MultiheadAttention(d, heads, batch_first=True)

    def forward(self, x, return_weights=False):
        x = x + self.mha(x, x, x, need_weights=False)[0]
        a = torch.softmax(self.v(torch.tanh(self.proj(x))).squeeze(-1), 1)
        out = (a.unsqueeze(-1) * x).sum(1)
        return (out, a) if return_weights else out


class ISAB(nn.Module):
    """④ Induced Set Attention Block (Lee et al., ICML 2019): H = Attn(I, X) (64 điểm tóm tắt đọc cả tập), ra Attn(X, H)
    (mỗi chiến lược đọc bản tóm tắt). Chi phí O(n*m); thông tin đi qua 64 điểm nên khó học thuộc nhiễu."""

    def __init__(self, d, heads=4, m=64):
        super().__init__()
        self.ind = nn.Parameter(torch.randn(1, m, d) * d ** -0.5)
        self.a1 = nn.MultiheadAttention(d, heads, batch_first=True); self.a2 = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1, self.n2, self.n3, self.n4 = (nn.LayerNorm(d) for _ in range(4))
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):                                                      # x [1, n, d]
        h = self.n1(self.ind + self.a1(self.ind, x, x, need_weights=False)[0])  # [1, m, d]
        y = self.n2(x + self.a2(x, h, h, need_weights=False)[0])
        return self.n3(y + self.ff(self.n4(y)))


class HyperAttention(nn.Module):
    """⑤ HyperAttention (Han et al., ICLR 2024) trên tập các chiến lược: sortLSH xếp các chiến lược giống nhau cạnh nhau, attention
    tính chính xác trong từng khối `block`; phần còn lại ước lượng bằng `m` khoá lấy mẫu đều (nhân (n - block)/m). Chi phí O(n(block + m))."""

    def __init__(self, d, heads=4, block=512, m=256, bits=8):
        super().__init__()
        self.h, self.dh, self.block, self.m = heads, d // heads, block, m
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.register_buffer("planes", torch.randn(heads, self.dh, bits))
        self.register_buffer("pow2", 2 ** torch.arange(bits))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def attend(self, x):                                                      # x [n, d] -> [n, d]
        n = x.shape[0]
        q, k, v = self.qkv(x).view(n, 3, self.h, self.dh).unbind(1)
        q, k, v = (t.transpose(0, 1) * (1.0 if i else self.dh ** -0.5) for i, t in enumerate((q, k, v)))   # [h, n, dh]
        b = min(self.block, n); nb = -(-n // b); pad = nb * b - n
        code = ((torch.einsum("hnd,hdr->hnr", k, self.planes) > 0).long() * self.pow2).sum(-1)   # mã LSH (SimHash)
        perm = torch.argsort(code, dim=1)                                                       # sortLSH
        g = lambda t: torch.gather(t, 1, perm[..., None].expand(-1, -1, t.shape[-1]))
        qs, ks, vs = g(q), g(k), g(v)
        if pad:
            qs, ks, vs = (torch.cat([t, t[:, :pad]], 1) for t in (qs, ks, vs))
        qb, kb, vb = (t.view(self.h, nb, b, self.dh) for t in (qs, ks, vs))
        s_blk = qb @ kb.transpose(-1, -2)                                     # điểm attention chính xác trong khối
        if pad:
            valid = torch.arange(nb * b, device=x.device).view(nb, b) < n
            s_blk = s_blk.masked_fill(~valid[None, :, None, :], float("-inf"))
        rest = n - b
        if self.m > 0 and rest > 0:                                           # phần ngoài khối: lấy mẫu m khoá
            idx = torch.randint(0, n, (self.m,), device=x.device)
            ksm, vsm = k[:, idx], v[:, idx]
            s_smp = qs.view(self.h, nb, b, self.dh) @ ksm.transpose(-1, -2)[:, None]
            blk_of_key = torch.empty(n, dtype=torch.long, device=x.device)
            blk_of_key[perm[0]] = torch.arange(n, device=x.device) // b
            same = (torch.arange(nb, device=x.device)[:, None] == blk_of_key[idx][None, :])
            s_smp = s_smp.masked_fill(same[None, :, None, :], float("-inf"))
            w = torch.log(torch.tensor(rest / self.m, device=x.device))
            mx = torch.maximum(s_blk.amax(-1, keepdim=True), (s_smp + w).amax(-1, keepdim=True)).detach()
            e1, e2 = torch.exp(s_blk - mx), torch.exp(s_smp + w - mx)
            out = (e1 @ vb + e2 @ vsm[:, None]) / (e1.sum(-1, keepdim=True) + e2.sum(-1, keepdim=True))
        else:
            out = torch.softmax(s_blk, -1) @ vb
        out = out.reshape(self.h, nb * b, self.dh)[:, :n]
        inv = torch.empty_like(perm); inv.scatter_(1, perm, torch.arange(n, device=x.device).expand(self.h, -1))
        out = torch.gather(out, 1, inv[..., None].expand(-1, -1, self.dh))
        return self.o(out.transpose(0, 1).reshape(n, -1))

    def forward(self, x):                                                     # x [1, n, d]
        y = x[0] + self.attend(self.n1(x[0]))
        return (y + self.ff(self.n2(y)))[None]


class Gated(nn.Module):
    """Cổng khởi tạo 0 (ReZero, Bachlechner et al., UAI 2021): ra = x + a (khối(x) - x); a học được, bắt đầu = 0."""

    def __init__(self, blk):
        super().__init__(); self.blk = blk; self.a = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.a * (self.blk(x) - x)


class SwiGLUHead(nn.Module):
    """⑦ Khối đầu ra SwiGLU (Shazeer 2020, "GLU Variants Improve Transformer"): u = SiLU(W1 z) ⊙ (W2 z) (cổng nhân: mạng tự chọn
    kết hợp nào của vector so sánh và 2 số được đi qua), rồi LayerNorm (chuẩn hoá trong từng mẫu) -> Dropout -> Linear 1."""

    def __init__(self, d, h=64, drop=0.3):
        super().__init__()
        self.w1, self.w2 = nn.Linear(d, h), nn.Linear(d, h)
        self.norm, self.drop, self.out = nn.LayerNorm(h), nn.Dropout(drop), nn.Linear(h, 1)

    def forward(self, z):
        return self.out(self.drop(self.norm(nn.functional.silu(self.w1(z)) * self.w2(z))))


class FinalNet(nn.Module):
    def __init__(self, d_in=16, n_extra=5, hid=128, fc=64, drop=0.3, isab_points=64, hyper_block=512, hyper_m=256):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)                                                   # ①
        self.seq = nn.LSTM(d_in, hid, batch_first=True, bidirectional=True)              # ②
        self.pool = Pool(2 * hid)                                                        # ③
        self.isab = ISAB(2 * hid, 4, isab_points)                                        # ④
        self.hyper = Gated(HyperAttention(2 * hid, 4, hyper_block, hyper_m))             # ⑤
        self.head = SwiGLUHead(2 * hid + n_extra, fc, drop)                              # ⑦

    def forward(self, tokens, extra, trace=None):
        """tokens [n, 4, 16], extra [n, 5] -> [n]. trace: dict để ghi shape và đầu ra từng khối."""
        t = (lambda k, v: trace.__setitem__(k, v.detach())) if trace is not None else (lambda k, v: None)
        x = self.norm(tokens);                      t("1_layernorm", x)
        x, _ = self.seq(x);                         t("2_bilstm", x)
        c, a = self.pool(x, return_weights=True);   t("3_pool", c); t("3_pool_weights", a)
        c = self.isab(c[None])[0];                  t("4_isab", c)
        c = self.hyper(c[None])[0];                 t("5_hyper", c)
        z = torch.cat([c, extra], 1);               t("6_concat", z)
        o = self.head(z).squeeze(-1);               t("7_fc", o)
        return o
