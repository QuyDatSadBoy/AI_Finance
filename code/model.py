"""Kiến trúc mô hình chốt, từng khối một.

    dấu vân tay FinCast+PCA16 [n, 4, 16]
      ① LayerNorm                                        -> [n, 4, 16]
      ② Bi-LSTM (128 mỗi chiều)                          -> [n, 4, 256]   đọc 4 đoạn theo thời gian
      ③ Pool: self-attention giữa 4 đoạn + gộp có trọng số -> [n, 256]     1 vector / chiến lược
      ④ ISAB (Set Transformer, ICML 2019), 64 điểm tóm tắt -> [n, 256]     so với cả năm, qua bản tóm tắt
      ⑤ ghép 2 số (t-stat, lãi TB đã chuẩn hoá)          -> [n, 258]
      ⑥ FC: Linear 64 -> ReLU -> BatchNorm -> Dropout 0,3 -> Linear 1 -> [n]   đáp án dự báo (như baseline)

n = các chiến lược của CÙNG một năm. Tên thuộc tính (norm, seq, pool, xs, head) khớp checkpoint `checkpoints/final_s<seed>/`.
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


class FinalNet(nn.Module):
    def __init__(self, d_in=16, n_extra=2, hid=128, fc=64, drop=0.3, isab_points=64):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)                                                   # ①
        self.seq = nn.LSTM(d_in, hid, batch_first=True, bidirectional=True)              # ②
        self.pool = Pool(2 * hid)                                                        # ③
        self.xs = ISAB(2 * hid, 4, isab_points)                                          # ④
        self.head = nn.Sequential(nn.Linear(2 * hid + n_extra, fc), nn.ReLU(), nn.BatchNorm1d(fc), nn.Dropout(drop), nn.Linear(fc, 1))   # ⑥

    def forward(self, tokens, extra, trace=None):
        """tokens [n, 4, 16], extra [n, 2] -> [n]. trace: dict để ghi shape và đầu ra từng khối."""
        t = (lambda k, v: trace.__setitem__(k, v.detach())) if trace is not None else (lambda k, v: None)
        x = self.norm(tokens);                      t("1_layernorm", x)
        x, _ = self.seq(x);                         t("2_bilstm", x)
        c, a = self.pool(x, return_weights=True);   t("3_pool", c); t("3_pool_weights", a)
        c = self.xs(c[None])[0];                    t("4_isab", c)
        z = torch.cat([c, extra], 1);               t("5_concat", z)
        o = self.head(z).squeeze(-1);               t("6_fc", o)
        return o
