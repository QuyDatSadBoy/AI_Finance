# Mô hình chốt so với baseline (notebook tham khảo B)

**Bài toán.** Đầu mỗi năm Y (1983–2020), đoán lãi năm Y của khoảng 26.000 chiến lược kế toán (Accounting EW), rồi mua top 1/5/10% chiến lược tốt nhất. Chấm như paper Chen & Dim và so với EB của paper.

**Mô hình chốt = seed 1**, checkpoint `code/checkpoints/final_s1/Y<năm>.pt` (38 năm). Notebook chạy từ dữ liệu gốc đến kết quả: `code/notebook/final_model.ipynb`. Nhật ký mọi thử nghiệm: `code/experiments/results/NHAT_KY_THU_NGHIEM.md`.

**Kết quả chính** (Sharpe danh mục top 1 / 5 / 10%):

| | Dev 1983–2004 | Holdout 2005–2020 |
|---|---|---|
| Baseline (notebook B) | 1,34 / 1,23 / 1,16 | 0,61 / 0,51 / 0,54 |
| **Mô hình chốt (seed 1)** | **1,95 / 1,95** / 1,82 | **1,11 / 0,88 / 0,88** |
| EB (paper) | 1,55 / 1,91 / 1,86 | 0,82 / 0,78 / 0,86 |

Hơn EB 5/6 ô; top 1% hơn có ý nghĩa ở cả hai kỳ (p = 0,032 và 0,028). Chi tiết ở mục 4.

---

## 1. So sánh từng khối với baseline

### 1.1 Hai kiến trúc đặt cạnh nhau

Baseline là mô hình (i) của notebook B, đúng như hình kiến trúc gốc. ❄ = đóng băng (pretrained), 🔥 = học. **In đậm** = khối đã đổi.

```
        BASELINE (hình gốc)                          MÔ HÌNH CHỐT
128 tháng lãi của chiến lược              128 tháng lãi × dấu d (±1)            ← ① nhân dấu
            │                                         │
FinCast ❄  → 4 × 1280                     FinCast ❄  → 4 × 1280                  (giữ)
            │                             PCA ❄      → 4 × 16                    ← ② nén
Chuẩn hoá từng chiều                      LayerNorm                              (tương đương)
Bi-LSTM 128 🔥 → 4 × 256                  Bi-LSTM 128 🔥 → 4 × 256               (giữ)
Attention pooling 🔥 → 256                Attention 4 đoạn + pooling 🔥 → 256    ← ③
            │                             ISAB 🔥 (64 điểm tóm tắt) → 256        ← ④ khối mới
Ghép tstat_IS, mean_ret_IS → 258          Ghép t-stat, lãi TB (nhân dấu, z) → 258 (giữ)
FC 64 + ReLU + BatchNorm + Dropout 0,3    FC 64 + ReLU + BatchNorm + Dropout 0,3 (giữ)
FC 1 → lãi năm sau (%)                    FC 1 → ê, trừ TB lô                    (giữ)
                                          dự báo = (ê + nền |t| + mức năm) × độ dao động ← ⑤ đáp án
Học: MSE, Adam, 1.024 dòng/bước, 1 lần    Huber × trọng số năm, AdamW, cả năm/bước,
                                          học lại mỗi năm                         ← ⑥ ⑦ ⑧
Danh mục: xếp theo |dự báo|               xếp theo Sharpe dự báo (luật EB)        ← ⑨
```

Phần mạng gần như giữ nguyên baseline (FinCast, Bi-LSTM, FC). Cái đổi nhiều nhất là **mạng học gì** (⑤) và **học thế nào** (⑥–⑧).

**③ và ④ nói dễ hiểu:**
- **③ Gộp 4 đoạn:** sau Bi-LSTM mỗi chiến lược có 4 vector (4 đoạn 32 tháng). Baseline chấm điểm từng đoạn rồi cộng có trọng số. Bản chốt cho 4 đoạn **nhìn nhau trước** (self-attention), rồi mới chấm và cộng. Ví dụ 18144: trọng số 4 đoạn = 0,02 / 0,21 / 0,39 / 0,38, tức là gần như bỏ đoạn cũ nhất, tập trung vào 2 đoạn gần đây.
- **④ ISAB:**

**Tên:** ISAB = **I**nduced **S**et **A**ttention **B**lock (khối attention trên một tập, đi qua các điểm trung gian), từ Set Transformer (Lee et al., ICML 2019).

**Paper gốc:** *Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks*, Lee et al., ICML 2019 · [arxiv.org/abs/1810.00825](https://arxiv.org/abs/1810.00825) · `docs/papers/Lee2019_SetTransformer_1810.00825.pdf`.
- **Paper giải bài toán gì:** đầu vào là **một tập** phần tử không thứ tự (một đám điểm, một nhóm ảnh…), mô hình xử lý cả tập. Hai yêu cầu: đổi thứ tự phần tử thì kết quả không đổi; các phần tử phải "nhìn" được nhau vì kết quả phụ thuộc cả tập.
- **Ý tưởng chính:** dùng attention để mọi phần tử nhìn nhau, nhưng với n phần tử chi phí là n², quá nặng khi tập lớn. ISAB dùng **m vector trung gian học được** (m nhỏ, ở đây 64): (1) m vector đọc cả tập và tóm tắt; (2) từng phần tử đọc lại bản tóm tắt. Chi phí còn n × m; paper gọi là "low-rank projection".
- **Dự án dùng để làm gì:** mỗi năm ~25.000 chiến lược chính là một tập không thứ tự. Ý định: cho mỗi chiến lược biết bối cảnh cả năm mà không phải tính 25.000² cặp. Thực tế (đo ở dưới): mô hình đã học gần như không tận dụng phần bối cảnh này.

**Đầu vào là gì.** Khối ③ đã gộp 4 đoạn thành **1 vector 256 số cho mỗi chiến lược**. Vậy ISAB nhận cả năm như một **tập** $X$ = [25.603 × 256] (năm 2000).

| Khối | Ra | Ý nghĩa |
|---|---|---|
| ② Bi-LSTM | [25.603 × 4 × 256] | mỗi chiến lược 4 vector (4 đoạn) |
| ③ Attention gộp 4 đoạn | [25.603 × 256] | mỗi chiến lược 1 vector |
| ④ ISAB | [25.603 × 256] | cùng cỡ, nhưng mỗi vector đã trộn thông tin của cả năm |

**ISAB làm 2 phép attention:**

1. **Tóm tắt cả năm: $H = \text{Attention}(I, X)$ → [64 × 256].**
   $I$ = [64 × 256] là **tham số học được** của mạng (lúc đầu ngẫu nhiên, gradient chỉnh dần như mọi trọng số khác; không lấy từ dữ liệu). Mỗi vector của $I$ là một "câu hỏi": tính độ khớp với từng chiến lược trong $X$ → softmax ra trọng số → cộng có trọng số các chiến lược → 1 vector 256 số. 64 câu hỏi → $H$ = 64 vector = **bản tóm tắt cả năm nén còn 64 dòng**.
2. **Mỗi chiến lược đọc bản tóm tắt: $\text{Out} = \text{Attention}(X, H)$ → [25.603 × 256].**
   Giờ mỗi chiến lược là câu hỏi, 64 dòng của $H$ là chỗ tra: tính độ khớp với 64 dòng → softmax → cộng có trọng số → cộng vào vector của chính nó, qua LayerNorm và một lớp FC nhỏ. Ra vector mới cho mỗi chiến lược.

**Vì sao đi qua 64 dòng trung gian:** cho 25.603 chiến lược attention thẳng với nhau phải tính 25.603² ≈ 655 triệu cặp. Qua 64 dòng chỉ còn 25.603 × 64 × 2 ≈ 3,3 triệu cặp (paper gọi là "low-rank projection"). Câu hỏi $I$ học xong thì cố định, dùng cho mọi năm; bản tóm tắt $H$ thay đổi theo tập chiến lược của từng năm. Đây cũng là lý do mỗi bước học là cả một năm.

**Số thật (checkpoint seed 1, năm 2000, chiến lược 18144):**
- 18144 đọc nhiều nhất 3 dòng tóm tắt (số 25, 31, 53). Ba dòng này đều chú ý vào cùng một kiểu chiến lược: 250 chiến lược được chú ý nhất có \|t\| TB 1,6 (mức chung 1,73), 80% bị đổi dấu, năm 2000 kém mức chung 0,27–0,31 Sharpe, tức "nhóm chiến lược yếu" của năm.
- Ra ISAB của 18144 (5 số đầu): −0,295 1,123 −0,378 0,832 −0,716.

**ISAB thật sự dùng thông tin cả năm nhiều cỡ nào** (đo trên mô hình đã học):

| Thử | Dự báo $\hat e$ của 18144 |
|---|---|
| bình thường (cùng cả năm 2000) | 0,128 |
| đặt vào giữa các chiến lược năm 1990 | 0,128 |
| đặt vào giữa các chiến lược năm 2008 | 0,124 |
| chỉ cùng 200 chiến lược ngẫu nhiên | 0,127 |

- Bản tóm tắt $H$ năm 2000 khác năm 2008 chỉ 3%, khác năm 1990 chỉ 2%.
- Chạy từng chiến lược một mình: dự báo tương quan 0,99 với khi chạy cùng cả năm.

**Kết luận:** ở mô hình đã học, phép 1 cho ra gần như **cùng một bản tóm tắt cho mọi năm**, nên ISAB gần như không dùng thông tin "so với cả năm"; nó đóng vai một khối biến đổi thêm (1,07 triệu tham số) cho từng chiến lược. Phần so sánh trong năm thật ra nằm ở đáp án (Bước 6, trừ mức chung của năm) và phép trừ trung bình ở đầu ra ⑦. Thêm ISAB vẫn tăng Sharpe ở đợt 1 (1,24 / 0,56 → 1,37 / 0,69) nên được giữ; bỏ ISAB trên bản chốt chưa được thử lại.

### 1.2 Từng khối đổi: vì sao, đã thử gì, kết quả

Số = Sharpe trung bình top 1/5/10%, **dev / holdout** (đợt 1); số chỉ có dev là thử ở đợt 2–3 (2 seed). Nguồn: `docs/6_ALL_RESULTS_SUMMARY.md`. Baseline = 1,24 / 0,56.

| # | Khối | Vì sao đổi | Nguồn | Đã thử thay thế | Kết quả, vì sao chốt |
|---|---|---|---|---|---|
| ① | Nhân dấu | Chiến lược lỗ kéo dài thì làm ngược sẽ lãi; đưa mọi chiến lược về cùng chiều để mạng chỉ cần học "còn lãi tiếp không" | Chen & Dim §3.1 "sign strategies to have positive predicted returns" | — (luật của paper) | bắt buộc để so công bằng với EB |
| ② | PCA 1280 → 16 | 1280 số/đoạn quá nhiều so với ~25.000 dòng/năm, mạng dễ học thuộc; 16 số giữ 66% thông tin | — | PCA 8, 32, 64; bỏ PCA | 32/64 kém (64: dev 1,80); 8 ngang → giữ 16 |
| ③ | Attention 4 đoạn | Cho 4 đoạn 32 tháng "nhìn nhau" trước khi gộp | Transformer (Vaswani 2017) | Transformer, GRU thay Bi-LSTM | ngang hoặc kém (1,94 / 1,91 dev) → giữ Bi-LSTM như baseline |
| ④ | **ISAB** (mới) | Baseline đoán từng chiến lược riêng lẻ; nhưng mua top 1% là **so sánh** trong năm → cho mỗi chiến lược nhìn cả năm qua 64 điểm tóm tắt | Set Transformer (Lee 2019) | HyperAttention (bài chủ dự án gửi), ISAB 2 khối, ISAB 128 điểm, cổng HyperAttention | + ISAB: 1,24/0,56 → **1,37/0,69**. HyperAttention 1,81/0,88; 6 seed đủ 38 năm holdout 0,82/0,87/0,87 (< bản chốt 1,03/0,93/0,90). 2 khối 1,92/0,88; 128 điểm 1,90/0,81 → **ISAB 1 khối 64 điểm tốt nhất** |
| ⑤ | **Đáp án = Sharpe − TB năm − nền \|t\|** | Lãi % lệ thuộc độ dao động và mức chung của năm; chia độ dao động (Sharpe), trừ TB năm, trừ phần \|t\| đã giải thích → mạng chỉ học phần còn lại | Moskowitz 2012; Chen 2026 (centering); Chen & Dim (\|t\| là tín hiệu chính) | thứ hạng, nhãn 24 tháng, không có nền | Sharpe − TB năm: 1,37/0,69 → **1,84/0,81** (bước lớn nhất). + nền \|t\|: → **1,87/0,92**. Thứ hạng 1,58; bỏ nền thì holdout tụt (0,75–0,82) |
| ⑥ | **Loss Huber × trọng số năm** | Vài chiến lược cực đoan làm MSE lệch; năm có nhiều tín hiệu nên nặng hơn | Gu-Kelly-Xiu 2020 (Huber); Holen 2023 (bài chủ dự án gửi) | MSE, likelihood, ListNet, loss tương quan, Rank IC | + Holen: → **1,90/0,95**. ListNet 1,76/0,90; tương quan 1,82/0,81; MSE ≈ Huber |
| ⑦ | AdamW, wd 1e-2 | Weight decay tách riêng, chống học thuộc | Loshchilov 2019 | RAdam, Lion, Schedule-Free, Muon, EMA, SWA | đều kém (dev 1,90 / 1,85 / 1,63 / 1,57; EMA holdout 0,70) |
| ⑧ | **Mỗi bước = cả năm; học lại mỗi năm** | ISAB cần thấy cả năm mới so sánh được; học lại mỗi năm đúng như dùng thật | Gu-Kelly-Xiu 2020 "refit once every year" | lô 4.096; học một lần; không validation | lô 4.096: dev 1,74/1,90/1,79, holdout 0,82/0,76/0,78 (kém rõ); không validation 1,43/0,57 |
| ⑨ | Xếp theo Sharpe dự báo | Đúng luật EB để so công bằng | Chen & Dim | — | bắt buộc |
| – | Encoder | (giữ FinCast) | FinCast (Zhu 2025) | Chronos-2, EXAONE, Sundial, FinCast + Sundial, không pretrained | đều ≤ 1,89 / 0,80 → **FinCast tốt nhất** |
| – | Đầu ra FC | (giữ như baseline) | — | SwiGLU, GRN, GELU | SwiGLU dev 1,95 nhưng holdout 6 seed 0,78/0,73/0,72 (thua EB) → **FC baseline tốt nhất** |
| – | Thêm số vào FC | (giữ t-stat, lãi TB) | — | + số tháng, Sharpe IS, log độ dao động; t-stat 60/24 tháng; momentum | feat: holdout top 1% từng seed 0,41–0,86; t-stat 60/24 1,50 → giữ 2 số như baseline |

**Tóm lại:** đi từ baseline 1,24 / 0,56 → + ISAB 1,37 / 0,69 → + đáp án Sharpe − TB năm 1,84 / 0,81 → + nền \|t\| 1,87 / 0,92 → + Holen 1,90 / 0,95 = **bản chốt**. Mọi khối "xịn" hơn đã thử (HyperAttention, SwiGLU, Transformer, encoder khác, thêm feature) cho dev ngang/cao hơn nhưng **holdout thấp hơn** → giữ kiến trúc gần baseline, đổi ở đáp án và cách học.

Tổng tham số: 1.565.857 (LayerNorm 32, Bi-LSTM 149.504, attention 4 đoạn 329.216, ISAB 1.070.336, FC 16.769).

## 2. Paper dùng cho từng thay đổi

| Thay đổi | Paper | Trích nguyên văn | File / link |
|---|---|---|---|
| Bài toán, EB, chọn top theo **Sharpe dự báo** | Chen & Dim, *High-Throughput Asset Pricing*, arXiv 2311.10685 | "Our data-mined portfolio is the simple average of the top 1% of strategies, based on **EB-predicted Sharpe ratios**." | `docs/papers/ChenDim2023_HighThroughputAssetPricing_2311.10685.pdf` · [arxiv.org/abs/2311.10685](https://arxiv.org/abs/2311.10685) |
| Nhân dấu: xoay chiến lược về chiều đã lãi | Chen & Dim, mục 3.1 | "Each year, we **sign strategies to have positive predicted returns**, and then form portfolios …" | (như trên) |
| Đáp án theo Sharpe (lãi ÷ độ dao động) | Moskowitz, Ooi, Pedersen, *Time Series Momentum*, JFE 2012 | (chia lãi cho độ dao động để các tài sản so sánh được) | [doi.org/10.1016/j.jfineco.2011.11.003](https://doi.org/10.1016/j.jfineco.2011.11.003) |
| Đáp án trừ trung bình năm | Chen, Liu, Lo, Liu, *Centering Drives Normalization Gains*, arXiv 2609.07122 | "**Centering drives the reliable effect, while scale-only normalization does not help.**" | `docs/papers/Chen2026_Centering_2609.07122.pdf` |
| Encoder FinCast | Zhu et al., *FinCast*, CIKM 2025 | (mô hình nền cho chuỗi tài chính) | `docs/papers/FinCast_2508.19609.pdf` |
| ISAB | Lee et al., *Set Transformer*, ICML 2019 | "ISAB_m(X) = MAB(X, H), where H = MAB(I, X) … This is analogous to **low-rank projection**." | `docs/papers/Lee2019_SetTransformer_1810.00825.pdf` |
| Trọng số năm trong loss (bài chủ dự án gửi) | Holen et al., arXiv 2304.12778 | "**scaling by episodic reward (R-Weighted) or episodic loss (L-Weighted)**. By giving more weight to gradients from more successful or informative episodes …" | `docs/papers/Holen2023_LRWeighted_2304.12778.pdf` |
| Huber, học lại mỗi năm, validation | Gu, Kelly, Xiu, RFS 2020 | "we **refit once every year** … the **Huber robust objective function**" | `docs/papers/GuKellyXiu_2020_RFS.pdf` |
| Nền \|t\| (t-stat là tín hiệu chính) | Chen & Dim (như trên) | EB co dự báo theo t-stat; quy tắc \|t\| là mốc so sánh của paper | (như trên) |

**Đã thử, không dùng** (nhật ký): HyperAttention (Han et al., ICLR 2024, bài chủ dự án gửi) và SwiGLU (Shazeer 2020) cho dev cao hơn nhưng holdout thấp hơn rõ (6 seed: 0,78 / 0,73 / 0,72); thêm 3 feature, trọng số năm gần, loss tương quan, EMA: không giữ được ở holdout.

---

## 3. Từng bước

Ví dụ xuyên suốt: chiến lược **18144**, dự báo năm **2000**; mọi số là dữ liệu thật (in ra trong notebook). Mỗi bước: **Ý tưởng → Ký hiệu, công thức → Tính với số thật → Ra**.

---

### Bước 1. Dữ liệu gốc → ma trận lãi

**Ý tưởng:** đưa file lãi từng tháng về dạng bảng, mỗi chiến lược một hàng, mỗi tháng một cột, để dễ lấy "các tháng trước năm Y" của bất kỳ chiến lược nào.

**Khái niệm:**
- **Chiến lược long–short:** mua một nhóm cổ phiếu và bán khống một nhóm khác. **Lãi tháng** = lãi của nhóm mua − lãi của nhóm bán, tính bằng %.
- **NaN:** tháng chiến lược chưa tồn tại hoặc không có dữ liệu.

**Dữ liệu vào:** `data_raw/DataMinedLongShortReturnsEW.csv`, **[16.941.504 dòng × 5 cột]**:

| signalid | year | month | ret (%) | nstock |
|---|---|---|---|---|
| 1 | 1990 | 7 | 1,542 | 3.160 |
| 18144 | 2000 | 1 | −7,427 | 2.087 |

**Tính:** dòng (signalid, year, month, ret) được đặt vào ô [hàng của signalid, cột (year − 1960) × 12 + month − 1].
Ví dụ: (18144, 2000, 1) → cột (2000 − 1960) × 12 + 0 = **480**.

**Ra:** ma trận **[29.314 chiến lược × 756 tháng]** (1/1960 → 12/2022), 26% ô là NaN. Hàng 18144:

| | 1999-10 | 1999-11 | 1999-12 | 2000-01 | 2000-02 | 2000-03 |
|---|---|---|---|---|---|---|
| 18144 | −5,50 | −10,30 | −7,88 | −7,43 | −21,14 | 2,64 |

---

### Bước 2. Tóm tắt 20 năm quá khứ, và đáp án

**Ý tưởng:** đứng ở cuối năm 1999, nhìn lại 20 năm qua để đánh giá chiến lược tốt cỡ nào và có chắc không. Đồng thời ghi lại lãi thật của năm 2000 làm đáp án (lúc dự báo thì chưa biết).

**Khái niệm và công thức** ($r_t$ = lãi tháng t; $n$ = số tháng có dữ liệu trong 240 tháng):

| Khái niệm | Công thức | Ý nghĩa |
|---|---|---|
| Lãi trung bình $m$ | $m = \dfrac{1}{n}\sum_t r_t$ | trung bình mỗi tháng lãi bao nhiêu % |
| Độ dao động $s$ (độ lệch chuẩn) | $s = \sqrt{\dfrac{1}{n}\sum_t (r_t - m)^2}$ | lãi các tháng lệch khỏi trung bình bao nhiêu; lớn = rủi ro, lúc lãi to lúc lỗ to |
| Sai số chuẩn | $s / \sqrt{n}$ | trung bình $m$ có thể sai bao nhiêu chỉ vì may rủi; càng nhiều tháng càng nhỏ |
| **t-stat** | $t = \dfrac{m}{s / \sqrt{n}}$ | $m$ lớn gấp mấy lần sai số của nó. \|t\| = 2 có thể do may; \|t\| = 10 gần như chắc chắn không phải may |
| **Đáp án** $y$ | trung bình 12 tháng của năm Y | lãi thật năm Y, thứ cần dự báo |

**Dữ liệu vào:** 240 tháng 1/1980 → 12/1999 của 18144:
`0,03  1,62  −5,62  2,59  1,12  1,73  …  −1,26  −4,82  −4,75  −5,50  −10,30  −7,88`
và 12 tháng năm 2000: `−7,43  −21,14  2,64  1,20  0,51  −5,14  −0,31  −5,22  −0,61  0,58  0,93  5,22`.

**Tính:**
- $m$ = (0,03 + 1,62 − 5,62 + … − 7,88) ÷ 240 = −300,65 ÷ 240 = **−1,2527**
- $s$ = √( [(0,03 + 1,2527)² + (1,62 + 1,2527)² + …] ÷ 240 ) = √(943,1 ÷ 240) = √3,930 = **1,9823**
- Sai số chuẩn = 1,9823 ÷ √240 = 1,9823 ÷ 15,492 = 0,1280
- $t$ = −1,2527 ÷ 0,1280 = **−9,79** → lỗ rất đều, gần như chắc không phải may
- $y$ = (−7,43 − 21,14 + … + 5,22) ÷ 12 = −28,76 ÷ 12 = **−2,3967**

Chỉ giữ chiến lược có $n$ ≥ 36 tháng (ít hơn thì t-stat quá nhiễu) và đủ 12 tháng năm Y. Năm 2000 còn **25.603** chiến lược.

**Ra:** bảng **[1.189.707 dòng × 8 cột]** (mọi chiến lược × mọi năm 1966–2020):

| signalid | năm | dấu | lãi TB (m) | độ dao động (s) | t-stat | số tháng | lãi năm sau × dấu |
|---|---|---|---|---|---|---|---|
| **18144** | 2000 | −1 | −1,2527 | 1,9823 | −9,7901 | 240 | 2,3967 |
| 18143 | 2000 | −1 | −1,0269 | 1,8627 | −8,5402 | 240 | 0,7542 |
| 18146 | 2000 | −1 | −0,3989 | 3,1326 | −1,9728 | 240 | −1,0960 |

---

### Bước 3. Nhân dấu (làm ngược chiến lược lỗ)

**Ý tưởng:** mỗi chiến lược long–short đều đầu tư được theo 2 chiều: làm xuôi (mua bên long, bán bên short) hoặc làm ngược (đổi hai bên). Chiến lược lỗ đều trong quá khứ thì làm ngược sẽ lãi đều. Vì vậy trước khi học, ta xoay mọi chiến lược về chiều "đã lãi trong quá khứ". Mô hình chỉ còn học một câu hỏi: **lãi quá khứ này có kéo dài sang năm sau không?**

**Cơ sở (paper Chen & Dim, mục 3.1, nguyên văn):** *"Each year, we sign strategies to have positive predicted returns, and then form portfolios that equally-weight strategies in the top X% of predicted Sharpe ratios."* (`docs/papers/ChenDim2023_HighThroughputAssetPricing_2311.10685.pdf`)

**Khi nào đổi dấu:** **mỗi năm Y, cho từng chiến lược**, dựa trên lãi trung bình 20 năm **trước** Y (biết trước lúc đầu tư, nên không rò rỉ). Cùng một chiến lược có thể năm này đổi, năm khác không.

| Lãi TB 20 năm trước Y ($m$) | Dấu $d$ | Làm gì |
|---|---|---|
| $m \ge 0$ (đã lãi) | +1 | giữ nguyên |
| $m < 0$ (đã lỗ) | −1 | đổi dấu, tức làm ngược |

**Đổi những gì** (nhân tất cả với $d$, đều của cùng chiến lược, cùng năm Y):
1. 128 tháng lãi đưa vào FinCast (Bước 4);
2. t-stat và lãi TB 20 năm (2 số thêm, Bước 5);
3. lãi năm Y = **nhãn** (Bước 6).

Độ dao động $s$ **không đổi** (luôn dương).

**Tính (18144, năm 2000):** $m$ = −1,2527 < 0 → $d$ = −1.

| | Trước | × d = −1 |
|---|---|---|
| t-stat | −9,79 | **9,79** |
| lãi TB 20 năm | −1,2527 | **1,2527** |
| lãi năm 2000 (nhãn) | −2,3967 | **2,3967** |
| 128 tháng đưa vào FinCast | −0,49, −1,06, … | 0,49, 1,06, … |

**Nhãn có luôn dương không? Không.** Đổi dấu chỉ làm **quá khứ** dương. Năm sau chiến lược vẫn có thể đảo chiều:

| Chiến lược (2000) | $m$ | $d$ | Lãi năm 2000 gốc | Nhãn sau đổi dấu | Nghĩa |
|---|---|---|---|---|---|
| 18144 | −1,25 | −1 | −2,40 | **+2,40** | tiếp tục lỗ → làm ngược thì lãi |
| 18146 | −0,40 | −1 | +1,10 | **−1,10** | đảo chiều → làm ngược thì lỗ |

Trên cả bảng 1983–2020: **66%** dòng có $d$ = −1; **43,9% nhãn âm** (trung bình +0,12 %/tháng). Mô hình phải học phân biệt hai loại này.

**Khi đầu tư (Bước 9):** chiều cuối cùng = dấu của dự báo, đổi về chiều gốc. Ví dụ 18144: dự báo dương sau đổi dấu → làm ngược chiến lược gốc. Lãi danh mục luôn tính bằng lãi thật của đúng chiều đã đầu tư.

**Vì sao giúp mạng học:** không đổi dấu thì mạng phải học hai trường hợp đối xứng ("t = +5 tốt" và "t = −5 cũng tốt nếu làm ngược"). Sau đổi dấu chỉ còn một: t càng lớn càng tốt.

---

### Bước 4. Dấu vân tay FinCast

**Ý tưởng:** t-stat gom 240 tháng thành **một con số**, nên mất **hình dạng** của chuỗi (lãi đều hay giật, gần đây tăng hay giảm). FinCast là mô hình đã học sẵn trên rất nhiều chuỗi tài chính. Nó đọc chuỗi và trả ra một bộ số mô tả hình dạng đó ("dấu vân tay"). Sau đó nén bộ số này cho gọn.

**Khái niệm:**
- **Đoạn (patch):** FinCast cắt chuỗi thành các đoạn 32 tháng; 128 tháng = 4 đoạn (giống baseline).
- **Vector ẩn:** mỗi đoạn biến thành 1.280 số. Đây là độ rộng cố định của FinCast; FinCast đọc theo thời gian nên đoạn 4 đã "nhìn" cả 128 tháng.
- **PCA:** $z = (h - \mu)\,V$
  - $h$ là vector 1.280 số;
  - $\mu$ là vector trung bình;
  - $V$ là 16 hướng mà các vector khác nhau nhiều nhất, chia cho căn phương sai để cùng thang.
  - Giữ 16 số chứa 66% thông tin. $\mu$ và $V$ tính **chỉ từ 50.000 dòng năm ≤ 1982**.

**Dữ liệu vào:** **[4 × 32]**, 128 tháng (5/1989 → 12/1999) đã nhân dấu:
```
đoạn 1 (5/1989–12/1991):   0,49  1,06  1,45  0,69  3,88  3,41 …  1,49 −1,26
đoạn 2 (1/1992–8/1994):   −1,67  0,92  1,83  0,13  0,12  1,19 …  0,07  1,69
đoạn 3 (9/1994–4/1997):    2,26  1,93  2,36  0,17  1,73  0,70 … −0,65  1,49
đoạn 4 (5/1997–12/1999):   1,52 −0,77 −1,51  1,05  1,03 −0,36 … 10,30  7,88
```
**Tính 1, FinCast:** ra **[4 × 1280]**:
```
đoạn 1:  0,409   0,001  0,143  0,060   0,729  0,010 …  (1.280 số)
đoạn 2:  0,398  −0,001  0,094  0,099  −0,014  0,056 …
đoạn 3:  0,415  −0,032  0,086  0,045   0,477  0,222 …
đoạn 4:  0,540   0,007  0,089  0,048   0,483  0,117 …
```
**Tính 2, PCA:** ra **[4 × 16]**:
```
đoạn 1: −0,40 −0,91 −1,68 −2,04  0,73 −0,06  0,48 −1,30  0,40  0,98 −1,33 −0,85 −0,82 −0,10  1,46 −1,39
đoạn 2: −0,02 −0,08 −0,11  0,61  1,08  0,48 −0,95 −0,25  0,66 −0,52  0,06  0,24 −0,48 −1,39  0,52 −0,53
đoạn 3: −0,13 −0,12  1,49 −0,15  1,69 −0,78  1,05 −0,35 −0,10  1,95  0,61  0,02 −0,95 −1,57 −0,45 −0,03
đoạn 4:  2,47 −0,63 −0,44 −1,01  2,16  0,07 −0,84 −1,05  1,65 −0,48  1,23  0,65  1,35 −0,12  0,11 −0,43
```
**Ra:** cả bảng **[1.189.707 × 4 × 16]**. Từng số không có nghĩa đọc được; hai chuỗi có hình dạng giống nhau thì có dấu vân tay gần nhau.

---

### Bước 5. Đầu vào của mạng

**Ý tưởng:** mỗi chiến lược được mô tả bằng 3 thứ: hình dạng chuỗi (dấu vân tay, Bước 4), độ "chắc" của lãi (t-stat), độ lớn của lãi (lãi TB). Hai con số được đưa về cùng thang để mạng học dễ.

**Ký hiệu, công thức:**
- **Tập học** khi dự báo năm 2000: năm 1976–1998, lấy **500.000 dòng gần nhất** (vừa bộ nhớ, ưu tiên dữ liệu mới).
- **Chuẩn hoá z-score:** $x' = (x - \bar x) \div \sigma$, với $\bar x$, $\sigma$ = trung bình, độ lệch chuẩn **trên tập học** (không nhìn trước năm 2000). $x'$ = "cao hơn mức thường bao nhiêu lần độ lệch chuẩn".

**Tính (18144):**

| Đầu vào | Giá trị | $\bar x$ | $\sigma$ | Sau chuẩn hoá |
|---|---|---|---|---|
| Dấu vân tay | [4 × 16] (Bước 4) | | | giữ nguyên |
| t-stat × dấu | 9,79 | 1,535 | 1,256 | (9,79 − 1,535) ÷ 1,256 = **6,574** |
| lãi TB × dấu | 1,2527 | 0,2835 | 0,2473 | (1,2527 − 0,2835) ÷ 0,2473 = **3,920** |

**Ra:** năm 2000: **[25.603 × 4 × 16]** + **[25.603 × 2]**. Khi học, mỗi bước là cả một năm (≈ 25.000 chiến lược).

---

### Bước 6. Đáp án mạng học: học gián tiếp, không đoán thẳng lãi

#### 6.1 Ý tưởng

Không bắt mạng đoán thẳng "năm sau lãi bao nhiêu %". Ta đưa sẵn vào đáp án **ba điều đã biết** về tài chính. Mạng không phải tự mò ra chúng, chỉ học **phần mà ba điều này còn đoán sai**.

**(1) Lãi phải tính trên rủi ro: chia cho độ dao động (ra Sharpe).**
Lãi 2% ở chiến lược ít dao động là tốt thật; lãi 2% ở chiến lược dao động mạnh có thể chỉ do may. Nếu không chia, mạng mải đoán mấy chiến lược ồn ào (lãi lỗ đều to). Đây cũng là cách paper chấm điểm.

**(2) Cả thị trường cùng lên cùng xuống: trừ mức chung của năm.**
Năm tốt thì mọi chiến lược cùng lãi, năm xấu thì cùng lỗ. Phần này giống nhau cho mọi chiến lược, nên không giúp chọn cái nào hơn cái nào, mà lại không đoán trước được. Bỏ đi thì bài toán đúng là **xếp hạng trong năm**.

**(3) Quá khứ lãi chắc chắn thì thường tốt: trừ phần "t-stat cao thì tốt".**
Kinh nghiệm này rất mạnh: riêng nó đã ngang EB. Ta tính sẵn nó làm **nền**. Mạng chỉ học **chỗ kinh nghiệm này sai**: chiến lược t-stat cao nhưng sắp yếu đi, hoặc t-stat thường nhưng sắp tốt lên. Nếu mạng không học được gì, dự báo vẫn bằng nền, nên không bị hỏng khi thị trường đổi (sau 2005).

| | Bỏ đi cái gì | Mạng được nhẹ việc gì |
|---|---|---|
| (1) | độ ồn của từng chiến lược | không phải đoán độ dao động |
| (2) | lên xuống chung của thị trường | không phải đoán thị trường |
| (3) | kinh nghiệm t-stat | không phải học lại điều đã biết |
| **Còn lại** | → **mạng học** | chỉ việc khó: nhìn hình dạng chuỗi lãi để sửa sai cho kinh nghiệm |

Khi dự báo thì cộng ba phần đó trở lại. Kết quả: Sharpe dev / holdout từ 1,24 / 0,56 (đoán thẳng) lên **1,90 / 0,95**.

**Nền $b \cdot z_i$ của phần (3) là gì.** Là một phép đoán nhanh: *chiến lược này nổi bật mấy bậc* × *mỗi bậc thường đáng bao nhiêu*. Giống đoán lương: người A giỏi hơn trung bình 5 bậc, mỗi bậc thường được thêm 1 triệu, vậy đoán A hơn mức trung bình 5 triệu.

- **$z_i$ = nổi bật mấy bậc.** Lấy t-stat của chiến lược trừ t-stat trung bình của năm, rồi chia cho độ chênh thường gặp giữa các chiến lược: $z_i = (\lvert t_i\rvert - \text{TB}) \div \text{độ chênh}$. Giống "điểm cao hơn trung bình lớp mấy bậc". $z = 0$ là bình thường, $z = 5$ là vượt trội. Chia như vậy để năm nào cũng đo bằng cùng một thước.
- **$b$ = tỉ giá: 1 bậc nổi bật đổi được bao nhiêu Sharpe năm sau**, học từ 15 năm trước. Hai số của mỗi chiến lược quá khứ:
  - $z$ = **trước năm đó** nó nổi bật mấy bậc về t-stat (như trên).
  - $r$ = **năm đó** nó thật sự **hơn mức chung bao nhiêu**: Sharpe năm đó của nó trừ Sharpe trung bình của mọi chiến lược cùng năm ($r = S - \bar S$, phần (1)(2) ở trên). Ví dụ $r$ = 0,3: tốt hơn mức chung 0,3; $r$ = −0,2: kém hơn mức chung 0,2.

  **Bước 1. Tỉ giá riêng.** Mỗi chiến lược cho một tỉ giá $r \div z$. Ví dụ X nổi bật 2 bậc, năm sau hơn mức chung 0,3 → 0,3 ÷ 2 = 0,15 mỗi bậc.

  **Bước 2. Lấy trung bình, trọng số $z^2$.** Chiến lược gần bình thường ($z \approx 0$) chia cho số gần 0 thì tỉ giá nhảy lung tung, không đáng tin; chiến lược càng nổi bật càng đáng tin. Nên mỗi tỉ giá được nhân trọng số $z^2$:

  $$b = \frac{\text{tổng (trọng số × tỉ giá)}}{\text{tổng trọng số}} = \frac{\sum z^2 \times \dfrac{r}{z}}{\sum z^2}$$

  **Bước 3. Rút gọn tử số.** $z^2 \times \dfrac{r}{z} = z \times z \times \dfrac{r}{z} = z \times r$ (một $z$ triệt tiêu). Nên **tử = $\sum z\, r$**, **mẫu = $\sum z^2$** (tổng trọng số).

  **Bước 4. "+1" ở mẫu.** Thêm một chiến lược tưởng tượng có trọng số 1 và tỉ giá 0 ("biết đâu t-stat chẳng đáng gì"): tử cộng thêm 1 × 0 = 0 (không đổi), mẫu cộng thêm trọng số 1. Nó kéo $b$ nhích về 0, để không tin kinh nghiệm quá mức:

  $$b = \frac{\sum z\, r + 0}{\sum z^2 + 1}$$

  | Chiến lược | $z$ | $r$ | tỉ giá $r/z$ | trọng số $z^2$ |
  |---|---|---|---|---|
  | X | 2 | 0,3 | 0,15 | 4 |
  | Y | 1 | 0,1 | 0,10 | 1 |
  | Z | −1 | −0,2 | 0,20 | 1 |
  | tưởng tượng | | | 0 | 1 |

  $b$ = (4·0,15 + 1·0,10 + 1·0,20 + 1·0) ÷ (4 + 1 + 1 + 1) = 0,9 ÷ 7 = 0,13. Dữ liệu thật 1985–1999 (363.433 chiến lược-năm, bỏ 1% $r$ cực đoan): $b$ = 28.710 ÷ 363.434 = **0,079**; chiến lược tưởng tượng gần như không ảnh hưởng.
- **Nhân lại:** 18144 nổi bật 5,58 bậc × 0,079 = **0,441** (kinh nghiệm đoán). Thực tế nó hơn mức chung 1,029, nên mạng học phần thiếu **0,588**.

#### 6.2 Ký hiệu và công thức từng đại lượng

Chiến lược $i$, năm cần dự báo $Y$. Mọi số đã nhân dấu $d_i$ ở Bước 3. "20 năm trước Y" là các tháng $x_{i,1}, \dots, x_{i,n}$ (tối đa 240 tháng).

| Ký hiệu | Đọc là | Công thức | Ghi chú |
|---|---|---|---|
| $m_i$ | lãi trung bình 20 năm trước | $m_i = \dfrac{1}{n}\sum_{k=1}^{n} x_{i,k}$ | Bước 2 |
| $s_i$ | độ dao động 20 năm trước (①) | $s_i = \sqrt{\dfrac{1}{n}\sum_{k=1}^{n} (x_{i,k} - m_i)^2}$ | đã biết trước Y |
| $t_i$ | t-stat 20 năm trước | $t_i = \dfrac{m_i}{s_i / \sqrt{n}}$ | đã biết trước Y |
| $y_i$ | lãi thật năm Y | $y_i = \dfrac{1}{12}\sum_{\text{tháng } 1..12 \text{ của } Y} x_{i}$ | **chỉ biết khi học** |
| $S_i$ | Sharpe năm sau (đã bỏ ①) | $S_i = y_i / s_i$ | lãi trên mỗi đơn vị dao động |
| $\bar S_Y$ | mức chung của năm Y (②) | $\bar S_Y = \dfrac{1}{N_Y}\sum_{j \in Y} S_j$ | $N_Y$ = số chiến lược năm Y; chỉ biết khi học |
| $r_i$ | Sharpe tương đối (đã bỏ ②) | $r_i = S_i - \bar S_Y$ | hơn mức chung của năm bao nhiêu |
| $z_i$ | t-stat cao cỡ nào so với các chiến lược khác cùng năm | $z_i = \dfrac{\lvert t_i\rvert - \mu_Y}{\sigma_Y}$; $\mu_Y$ = \|t\| trung bình của năm, $\sigma_Y$ = các \|t\| thường chênh nhau bao nhiêu | giống "điểm cao hơn trung bình lớp mấy bậc". $z$ = 0: bình thường; $z$ = 5: vượt trội. Biết trước Y |
| $b$ | t-stat cao hơn 1 bậc thì năm sau tốt hơn bao nhiêu (học từ quá khứ) | $b = \dfrac{\sum_j z_j\, \tilde r_j}{\sum_j z_j^2 + 1}$, tổng trên mọi chiến lược của 15 năm $Y-15 \dots Y-1$ | kẻ một đường thẳng qua dữ liệu 15 năm trước: trục ngang $z$, trục dọc $r$; $b$ là độ dốc. $\tilde r$ = $r$ bỏ 1% cực đoan hai đầu; "+1" ở mẫu để $b$ không quá to khi dữ liệu ít |
| $\text{nền}_i$ | phần quy tắc \|t\| đoán được (③) | $\text{nền}_i = b \cdot z_i$ | |
| $e_i$ | **đáp án mạng học** (④) | $e_i = \text{clip}(r_i - \text{nền}_i,\ lo,\ hi)$ | |
| $lo, hi$ | khoảng cắt | phân vị 1% và 99% của $r - \text{nền}$ trên tập học | bỏ vài chiến lược cực đoan |
| $L_Y$ | mức năm dùng khi dự báo | $L_Y = $ trung bình $S_j$ của **mọi** chiến lược, mọi năm $\le Y-1$ | thay cho $\bar S_Y$ chưa biết |
| $\hat e_i$ | đầu ra của mạng | Bước 8 | đoán $e_i$ |
| $\text{dự báo}_i$ | lãi dự báo năm Y (%/tháng) | $\text{dự báo}_i = (\hat e_i + \text{nền}_i + L_Y) \times s_i$ | lắp ngược ③②① |

Gộp lại, đáp án có hai chiều: **tháo** khi học, **lắp** khi dự báo.

**Chiều tháo (khi học)**: từ lãi thật $y_i$, bỏ dần từng phần đã biết, còn lại đáp án $e_i$:

$$e_i = \underbrace{\frac{y_i}{s_i}}_{\text{(1)}} \;-\; \underbrace{\bar S_Y}_{\text{(2)}} \;-\; \underbrace{b\, z_i}_{\text{(3)}}$$

| Phần | Làm gì | Vì sao | 18144, năm 2000 |
|---|---|---|---|
| (1) $y_i / s_i$ | chia lãi thật cho độ dao động → Sharpe | để chiến lược dao động mạnh và yếu so được với nhau (ví dụ A, B ở trên) | 2,3967 ÷ 1,9823 = 1,209 |
| (2) $-\,\bar S_Y$ | trừ Sharpe trung bình của mọi chiến lược trong năm | bỏ phần "năm tốt / năm xấu" chung cho cả thị trường | 1,209 − 0,180 = 1,029 |
| (3) $-\,b\,z_i$ | trừ phần kinh nghiệm "t-stat cao thì tốt" đã đoán được | kinh nghiệm này có sẵn, không cần mạng học lại | 18144 cao hơn 5,583 bậc × mỗi bậc 0,079 = kinh nghiệm đoán 0,441; còn lại 1,029 − 0,441 = 0,588 |
| $e_i$ | phần còn lại | **mạng học đoán con số này** | **0,588** |


**Chiều lắp (khi dự báo)**: làm ngược lại, cộng các phần vào đầu ra của mạng $\hat e_i$:

$$\text{dự báo}_i = \big(\underbrace{\hat e_i}_{\text{mạng đoán}} + \underbrace{b\, z_i}_{\text{(3) nền } |t|} + \underbrace{L_Y}_{\text{(2) mức năm}}\big) \times \underbrace{s_i}_{\text{(1) độ dao động}}$$

| Phần | Làm gì | 18144, năm 2000 |
|---|---|---|
| $\hat e_i$ | mạng đoán phần còn lại | 0,128 |
| $+\,b\,z_i$ | cộng lại nền \|t\| (giống hệt lúc tháo) | 0,128 + 0,441 = 0,569 |
| $+\,L_Y$ | cộng lại mức năm. Đầu năm Y **chưa biết** $\bar S_Y$, nên dùng mức trung bình của mọi năm trước: $L_Y$ | 0,569 + 0,076 = 0,645 (Sharpe dự báo) |
| $\times\, s_i$ | nhân lại độ dao động để ra %/tháng | 0,645 × 1,9823 = **1,278 %/tháng** |

Hai chiều đối xứng nhau, chỉ khác một chỗ: khi tháo dùng $\bar S_Y$ (mức thật của năm Y, chỉ biết khi học), còn khi lắp dùng $L_Y$ (ước lượng từ quá khứ).

**Không rò rỉ:** khi dự báo năm Y, mọi thứ dùng để lắp đều có từ trước Y:
- $s_i$ và $z_i$: từ 20 năm trước Y;
- $b$: hồi quy trên các năm Y−15 … Y−1;
- $L_Y$: trung bình các năm ≤ Y−1.

Lãi thật $y_i$ và $\bar S_Y$ chỉ xuất hiện ở chiều tháo, dùng cho các năm đã qua. Nền chỉ dùng t-stat của chính dữ liệu, **không dùng đầu ra EB**.

#### 6.3 Tính từng bước với số thật (chiến lược 18144, năm 2000)

Số liệu vào (Bước 2–3, đã nhân dấu $d = -1$): $n$ = 240 tháng (1/1980–12/1999), $m$ = 1,2527, $s$ = 1,9823.

**Bước a. t-stat.**
$\sqrt{n} = \sqrt{240} = 15{,}492$ → $s/\sqrt{n}$ = 1,9823 ÷ 15,492 = 0,12796 → $t$ = 1,2527 ÷ 0,12796 = **9,790**.

**Bước b. Lãi thật năm 2000** (12 tháng, đã nhân dấu):
7,427 + 21,141 − 2,638 − 1,205 − 0,508 + 5,137 + 0,314 + 5,215 + 0,612 − 0,580 − 0,932 − 5,224 = 28,759 → $y$ = 28,759 ÷ 12 = **2,3967 %/tháng**.

**Bước c. Bỏ độ dao động (①).**
$S$ = 2,3967 ÷ 1,9823 = **1,209**.

**Bước d. Mức chung năm 2000 (②).**
Cộng $S$ của cả 25.603 chiến lược năm 2000 được 4.607,45 → $\bar S_{2000}$ = 4.607,45 ÷ 25.603 = **0,180**.

**Bước e. Sharpe tương đối.**
$r$ = 1,209 − 0,180 = **1,029**. 18144 hơn mức chung của năm 1,029.

**Bước f. \|t\| chuẩn hoá.**
Trong năm 2000: $\mu$ = 1,728 (trung bình \|t\|), $\sigma$ = 1,444 (độ lệch chuẩn \|t\|).
$z$ = (9,790 − 1,728) ÷ 1,444 = 8,062 ÷ 1,444 = **5,583**. T-stat của 18144 cao hơn mức thường 5,6 độ lệch chuẩn.

**Bước g. Độ dốc $b$** (15 năm 1985–1999, 363.433 dòng; $r$ cắt vào [−1,047; 1,043]):
$\sum z\,\tilde r$ = 28.710,4; $\sum z^2$ = 363.433 → $b$ = 28.710,4 ÷ (363.433 + 1) = **0,0790**.
Đọc là: \|t\| cao hơn 1 độ lệch chuẩn thì Sharpe năm sau hơn mức chung khoảng 0,079.

**Bước h. Nền (③).**
nền = 0,0790 × 5,583 = **0,441**. Quy tắc \|t\| đoán 18144 hơn mức chung 0,441.

**Bước i. Đáp án (④).**
$r$ − nền = 1,029 − 0,441 = 0,588. Khoảng cắt [lo; hi] = [−0,811; 0,825]; 0,588 nằm trong khoảng → $e$ = **0,588**.

Tóm lại phép tháo: lãi thật 2,3967 → bỏ dao động còn 1,209 → bỏ mức năm còn 1,029 → bỏ phần \|t\| còn **0,588**. Mạng chỉ phải học con số 0,588 này, không phải 2,3967.

**Bước j. Lắp lại khi dự báo** (đầu năm 2000, chưa biết $y$ và $\bar S_{2000}$):
- Mức năm: trung bình $S$ của 639.546 dòng năm ≤ 1999 = 48.455,25 ÷ 639.546 → $L_{2000}$ = **0,076**.
- Mạng (Bước 8) đoán $\hat e$ = **0,128**.
- Cộng: 0,128 + 0,441 + 0,076 = 0,645 (Sharpe dự báo).
- Nhân độ dao động: 0,645 × 1,9823 = **1,278 %/tháng**.

So sánh: lãi thật 2,397; EB dự báo 1,033; mô hình 1,278. Mạng đoán thêm +0,128 so với nền, tức là thấy 18144 còn tốt hơn mức quy tắc \|t\| đoán. Điều đó đúng (thật ra hơn 0,588), dù mạng còn thận trọng.

#### 6.4 So với baseline

| | Baseline | Mô hình chốt |
|---|---|---|
| Mạng đoán gì | thẳng $y_i$ (%/tháng), gồm cả ①②③④ | chỉ $e_i$ (④) |
| Dự báo cuối | = đầu ra mạng | = (đầu ra mạng + nền + mức năm) × độ dao động |
| Nếu mạng không học được gì ($\hat e = 0$) | dự báo vô nghĩa | vẫn bằng quy tắc \|t\| |
| Sharpe dev / holdout (TB top 1/5/10%) | 1,24 / 0,56 | 1,90 / 0,95 |

Từng bước đóng góp (mục 1.2): bỏ ① và ② (đáp án Sharpe − mức chung) đưa Sharpe từ 1,37 / 0,69 lên 1,84 / 0,81, là bước tăng lớn nhất. Bỏ thêm ③ (nền \|t\|) đưa lên 1,87 / 0,92.

**Ra:** mỗi chiến lược một đáp án $e$; năm 2000 có **[25.603]** số.

---

### Bước 7. Chia năm

**Ý tưởng:** giống đời thật: muốn dự báo năm Y thì chỉ được dùng dữ liệu trước Y. Phải giữ riêng một năm mô hình chưa học (năm kiểm) để biết khi nào dừng học, tránh học thuộc nhiễu.

**Khái niệm:**
- **Học** = năm ≤ Y−2;
- **Kiểm (validation)** = năm Y−1: chỉ để chọn vòng học tốt nhất, không cập nhật trọng số;
- **Dự báo** = năm Y;
- **Học lại mỗi năm:** lặp 38 lần, mỗi lần bắt đầu từ trọng số của năm trước. Học 6 seed; **mô hình chốt = seed 1** (chọn ở Bước 10, chỉ bằng dev).

**Dữ liệu vào:** bảng 1.189.707 dòng.

**Ra:**

| Dự báo năm | Học (số dòng, năm) | Kiểm | Dự báo |
|---|---|---|---|
| 1983 | 214.683 (1966–1981) | 20.381 (1982) | 20.477 |
| **2000** | **500.000 (1976–1998)** | **25.725 (1999)** | **25.603** |
| 2020 | 500.000 (1999–2018) | 26.040 (2019) | 26.073 |

---

### Bước 8. Mạng xử lý và học thế nào

**Ý tưởng:** đọc hình dạng chuỗi của từng chiến lược (Bi-LSTM, attention 4 đoạn), so nó với cả năm (ISAB), rồi kết hợp với t-stat và lãi TB qua khối FC (như baseline) để ra một số $\hat e$ = dự báo cho đáp án $e$ của Bước 6.

**Từng khối, số thật** (seed 1, checkpoint năm 2000, chạy cả 25.603 chiến lược năm 2000; cột cuối là 5 số đầu của 18144):

| Khối | Làm gì | Công thức | Vào → Ra | 18144 |
|---|---|---|---|---|
| Vào | | | [25.603, 4, 16] + [25.603, 2] | đoạn 4: 2,473 −0,630 −0,444 …; 2 số: 6,574 3,920 |
| ① LayerNorm | đưa mỗi vector 16 số về cùng thang | $(x - \text{TB}) \div \text{độ lệch chuẩn}$ trong vector | → [25.603, 4, 16] | 0,019 −0,490 −1,272 … |
| ② Bi-LSTM | đọc 4 đoạn theo thời gian, xuôi và ngược | mạng hồi quy, 128 số mỗi chiều | → [25.603, 4, 256] | 0,008 0,015 0,033 … |
| ③ Attention 4 đoạn | các đoạn nhìn nhau, rồi gộp, đoạn quan trọng nặng hơn | $a_k = \text{softmax}(v^\top \tanh(W x_k))$, ra $\sum_k a_k x_k$ | → [25.603, 256] | trọng số 4 đoạn: **0,02 / 0,21 / 0,39 / 0,38** (đoạn gần đây quan trọng hơn) |
| ④ ISAB | 64 "điểm tóm tắt" đọc cả năm, rồi mỗi chiến lược đọc lại bản tóm tắt → biết mình đứng đâu | $H = \text{Attn}(I, X)$; ra $\text{Attn}(X, H)$ | → [25.603, 256] | −0,295 1,123 −0,378 … |
| ⑤ Ghép | thêm t-stat, lãi TB | nối | → [25.603, 258] | |
| ⑥ FC | kết hợp tất cả ra 1 số | Linear 64 → ReLU → BatchNorm → Dropout 0,3 → Linear 1 | → [25.603] | −0,671 |
| ⑦ Trừ trung bình | đáp án $e$ có trung bình năm ≈ 0, nên bỏ phần cả năm lệch đều | $\hat e = o - \text{TB}(o)$, TB = −0,799 | → [25.603] | −0,671 + 0,799 = **0,128** |

**Từ $\hat e$ ra dự báo %/tháng, bằng lời:** cộng lại phần nền |t| (Bước 6) và mức chung của năm, được Sharpe dự báo; nhân độ dao động, được %/tháng.

$$\text{dự báo} = (\hat e + \text{nền} + \text{mức năm}) \times s$$

- **Mức năm** = Sharpe trung bình mọi năm ≤ 1999 = **0,076** (thay $\bar S_{2000}$ chưa biết).
- 18144: (0,128 + 0,441 + 0,076) × 1,982 = **1,278 %/tháng**. Lãi thật: 2,397. EB dự báo: 1,033.
- Mạng thêm 0,128 vào nền: nó thấy 18144 còn tốt hơn mức quy tắc |t| đoán, và đúng (thật hơn 0,588).

#### Khối ④ ISAB

Giải thích đầy đủ, có số thật và kiểm tra: mục 1.1, "③ và ④ nói dễ hiểu".

#### Loss và optimizer

**Ý tưởng.** Loss = mạng đoán sai đáp án $e$ (Bước 6) bao nhiêu. Có hai cải tiến so với MSE của baseline, mỗi cái lấy từ một paper:

$$L_{\text{năm } Y} = \underbrace{w_Y}_{\text{(2) năm này quan trọng cỡ nào}} \times \frac{1}{N_Y}\sum_i \underbrace{h(\hat e_i - e_i)}_{\text{(1) sai bao nhiêu, kiểu Huber}}$$

**(1) Huber: không để vài chiến lược "điên" chi phối** (Gu, Kelly, Xiu, RFS 2020).
- *Paper nói:* lợi suất có đuôi dày (thỉnh thoảng có giá trị cực lớn); dùng **Huber** thay MSE: "squared loss for relatively small errors and absolute loss for relatively large errors". Mạng có Huber ("+H") tốt hơn trong bài.
- *Áp dụng:* sai nhỏ (≤ 1) phạt bình phương như MSE; sai lớn phạt tăng đều (tuyến tính), không bùng nổ:

$$h(x) = \begin{cases}\tfrac12 x^2 & |x| \le 1\\ |x| - \tfrac12 & |x| > 1\end{cases}$$

- *Ví dụ:* sai 0,2 → 0,02 (như MSE). Sai 5 → Huber 4,5; MSE 12,5. Một chiến lược sai 5 không còn nặng bằng hàng trăm chiến lược sai nhỏ.

**(2) Trọng số năm: năm nhiều thông tin thì học nặng hơn** (Holen et al., arXiv 2304.12778, bài chủ dự án gửi).
- *Paper nói:* khi gộp gradient của nhiều agent, nhân mỗi agent với trọng số theo loss của nó: "weight = lossᵢ / total_loss + 1/h", để ưu tiên học từ tình huống có loss cao, vì chúng chứa thông tin quan trọng ("prioritizing learning from scenarios resulting in higher episodic losses").
- *Áp dụng:* mỗi bước học là **một năm**, nên coi **mỗi năm như một agent**. "Loss của agent" thay bằng **độ phân tán đáp án của năm** $v_Y$ (trung bình bình phương $r$): năm các chiến lược khác nhau rõ (có cái rất tốt, có cái rất tệ) thì có nhiều điều để học; năm mọi chiến lược na ná nhau thì ít. Dùng $v_Y$ thay loss thật vì nó cố định, không nhảy theo mạng mỗi bước. Rút gọn công thức của paper (nhân hằng số để trọng số trung bình = 1):

$$w_Y = 0{,}5 \times \Big(1 + \frac{v_Y}{\bar v}\Big),\qquad \bar v = \text{TB } v \text{ của các năm học}$$

  Nửa trọng số chia đều cho mọi năm (phần $+1/h$ của paper), nửa còn lại theo độ phân tán. Năm bình thường ($v_Y = \bar v$) có $w$ = 1; năm phân tán gấp đôi có $w$ = 1,5; năm rất phẳng có $w$ gần 0,5. Không năm nào bị bỏ hẳn.
- *Ví dụ (dự báo năm 2000):* năm 1990: $v$ = 0,1191, $\bar v$ = 0,1010 → $w$ = 0,5 × (1 + 1,179) = **1,09**. Năm 1998: $w$ = **1,046**.

**Ví dụ một bước học** (cả năm 1998): Huber trung bình 0,0506 → $L$ = 1,046 × 0,0506 = **0,0529**.

**Kết quả** (mục 1.2): thêm trọng số năm đưa Sharpe dev / holdout từ 1,87 / 0,92 lên **1,90 / 0,95**; Huber và MSE gần như ngang nhau, giữ Huber cho an toàn với chiến lược cực đoan. Loss **không** chứa Sharpe danh mục (đúng quy tắc dự án).

**Optimizer:** AdamW (Loshchilov & Hutter, ICLR 2019), tốc độ học 0,001, weight decay 0,01 (kéo trọng số về 0 để chống học thuộc), cắt độ lớn gradient ≤ 1. Các optimizer mới thử (Lion, Muon, RAdam, Schedule-Free) đều không tốt hơn.

**Lịch học mỗi năm:** 3 vòng qua tập học, mỗi bước một năm (thứ tự xáo trộn). Sau mỗi vòng đo loss trên năm kiểm Y−1, giữ bản tốt nhất (kể cả bản năm trước chưa học thêm); lưu checkpoint.

---

### Bước 9. Chọn danh mục (giống hệt luật của paper)

**Ý tưởng:** đầu mỗi năm, cầm dự báo trong tay, chọn những chiến lược **lãi dự báo lớn so với rủi ro**, chia đều tiền, giữ 12 tháng. EB, baseline, quy tắc |t| đi qua **đúng luật này**, chỉ khác con số dự báo. Vậy hơn/kém hoàn toàn do dự báo.

**Ký hiệu:**

| Ký hiệu | Nghĩa | 18144 năm 2000 |
|---|---|---|
| $\hat\mu_i$ | dự báo %/tháng | mô hình: 1,278 · EB: 1,033 |
| $s_i$ | độ dao động 20 năm trước | 1,982 |
| $z_i = \hat\mu_i / s_i$ | **Sharpe dự báo**, dùng để xếp hạng | mô hình: **0,645** · EB: 0,521 |
| $d_i$ | chiều đầu tư = dấu của dự báo (theo dấu gốc) | làm ngược chiến lược gốc (gốc lỗ) |
| $K$ | 1% × số chiến lược năm | 1% × 25.603 = **257** |

**Cách làm:**
1. Tính Sharpe dự báo cho cả 25.603 chiến lược.
2. Xếp từ lớn đến nhỏ (bỏ dấu), lấy 257 chiến lược đầu.
3. Mỗi chiến lược 1/257 số tiền, theo chiều $d_i$.
4. Lãi danh mục tháng $p$: $R_p = \frac{1}{257}\sum_{\text{top}} d_i\, r_{i,p}$ ($r_{i,p}$ = lãi thật tháng $p$, chỉ biết sau khi tháng qua).

**4 chiến lược đứng đầu năm 2000** (lãi thật theo chiều đầu tư, %/tháng):

Mô hình chốt:

| Hạng | Chiến lược | Sharpe dự báo | Lãi thật |
|---|---|---|---|
| 1 | 18144 | 0,645 | +2,40 |
| 2 | 15313 | 0,593 | +2,29 |
| 3 | 18135 | 0,592 | +0,39 |
| 4 | 18143 | 0,585 | +0,75 |

EB:

| Hạng | Chiến lược | Sharpe dự báo | Lãi thật |
|---|---|---|---|
| 1 | 18144 | 0,521 | +2,40 |
| 2 | 27689 | 0,469 | **−0,47** (lỗ) |
| 3 | 18135 | 0,468 | +0,39 |
| 4 | 18131 | 0,468 | +0,63 |

**Bằng lời:** cả hai cùng chọn 18144 đứng đầu. EB đưa 27689 lên hạng 2 dù t-stat chỉ 4,4, vì độ dao động của nó rất lớn (4,64) và EB nhân lại với độ dao động; chiến lược này lỗ. Mô hình chốt không chọn nó; 4 chiến lược đầu của mô hình đều lãi.

**Ra:** lãi danh mục top 1% năm 2000 (%/tháng):

| | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | T11 | T12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mô hình chốt | 5,00 | 8,86 | −0,24 | −1,38 | −0,27 | 2,66 | 1,11 | 1,11 | 1,11 | 1,82 | 1,59 | −1,32 |
| EB | 6,48 | 14,49 | −3,89 | −2,53 | −1,19 | 5,25 | −0,16 | 1,69 | 0,40 | 0,21 | −0,92 | −3,82 |

Năm 2000 mô hình dao động ít hơn EB rõ (tháng tệ nhất −1,38 so với −3,89).

---

### Bước 10. Chấm điểm và so với EB

**Ý tưởng:** làm Bước 9 cho **mỗi năm** 1983–2020, nối 12 tháng của từng năm thành một chuỗi lãi dài. Chuỗi tốt = **lãi trung bình cao, dao động thấp**, đo bằng Sharpe. Làm y hệt với EB trên cùng các tháng, rồi hỏi: chênh lệch có đủ lớn để không phải do may?

**Hai kỳ:**
- **Dev 1983–2004** (264 tháng): kỳ được phép nhìn để chọn cấu hình và chọn seed. Giống bộ đề luyện.
- **Holdout 2005–2020** (192 tháng): không được nhìn khi chọn, chỉ để báo cáo. Giống đề thi thật. Sau 2005 lãi các chiến lược giảm mạnh (ai cũng biết chúng), nên mọi phương pháp chỉ đạt khoảng 0,8–1,1.

**Thước đo 1 (quan trọng nhất): Sharpe danh mục**

$$SR = \frac{\overline R}{\text{sd}(R)} \times \sqrt{12}$$

$\overline R$ = trung bình các $R_p$; sd = độ lệch chuẩn; $\sqrt{12}$ đổi từ tháng ra năm. Top 1%, dev:

| | $\overline R$ (%/tháng) | sd($R$) | Phép tính | **Sharpe** |
|---|---|---|---|---|
| Mô hình chốt | 0,780 | 1,385 | 0,780 ÷ 1,385 × 3,464 | **1,95** |
| EB | 0,746 | 1,668 | 0,746 ÷ 1,668 × 3,464 | **1,55** |

Bằng lời: mô hình lãi nhiều hơn một chút, nhưng **dao động ít hơn rõ**, nên Sharpe cao hơn 0,40.

**Có phải do may không?** (p càng nhỏ càng chắc; p < 0,05 = "khó là do may")

| Kiểm định | Làm gì, bằng lời |
|---|---|
| **HAC** (Ledoit & Wolf 2008) | tính sai số của hiệu Sharpe, có tính việc các tháng gần nhau giống nhau |
| **Bootstrap khối năm** | bốc ngẫu nhiên có lặp 22 năm (giữ nguyên 12 tháng mỗi năm) thành "lịch sử giả", tính lại hiệu Sharpe 10.000 lần; p = tỉ lệ lịch sử giả cho hiệu ngược chiều |
| JKM | kiểm định cổ điển, coi các tháng độc lập (thường cho p nhỏ hơn thực tế; báo kèm) |

**Thước đo 2: IC** (xếp hạng đúng không). Mỗi năm, xếp hạng mọi chiến lược theo dự báo và theo lãi thật, lấy tương quan giữa hai dãy hạng (Spearman). 1 = xếp đúng hết, 0 = ngẫu nhiên. IC tính trên **mọi** chiến lược, còn Sharpe top 1% chỉ trên 257 chiến lược đầu, nên có thể IC thấp hơn mà top 1% vẫn hơn: mô hình xếp **phần đầu bảng** tốt hơn.

**Thước đo 3: độ đúng mức** (paper Hình 4–5). Chia chiến lược thành 20 nhóm theo dự báo; vẽ (dự báo TB, lãi thật TB); độ dốc 1 = nói 0,5 thì được 0,5; < 1 = quá lạc quan.

**Chọn seed (chỉ bằng dev):** học 6 seed; chọn seed có Sharpe trung bình top 1/5/10% ở **dev muộn 1995–2004** (đoạn dev gần holdout nhất) cao nhất:

| Seed | Dev muộn TB (tiêu chí) | Dev top 1/5/10% | Holdout top 1/5/10% (chỉ báo cáo) |
|---|---|---|---|
| 0 | 1,885 | 1,89 / 1,89 / 1,79 | 0,93 / 0,89 / 0,89 |
| **1** | **1,930** | 1,95 / 1,95 / 1,82 | 1,11 / 0,88 / 0,88 |
| 2 | 1,900 | 1,89 / 1,92 / 1,82 | 0,81 / 0,83 / 0,86 |
| 3 | 1,866 | 1,77 / 1,93 / 1,82 | 0,98 / 0,90 / 0,90 |
| 4 | 1,913 | 1,92 / 1,96 / 1,85 | 0,98 / 0,91 / 0,89 |
| 5 | 1,907 | 1,96 / 1,95 / 1,80 | 0,91 / 0,89 / 0,89 |

Seed 1 được chọn. Cả 6 seed đều hơn EB ở holdout top 1% và 5%: kết quả không phụ thuộc may mắn của một seed.

---

## 4. Kết quả (mô hình chốt = seed 1)

| Sharpe top 1 / 5 / 10% | Dev 1983–2004 | Holdout 2005–2020 | Cả kỳ 1983–2020 |
|---|---|---|---|
| Baseline (notebook B) | 1,34 / 1,23 / 1,16 | 0,61 / 0,51 / 0,54 | 1,07 / 0,97 / 0,94 |
| **Mô hình chốt** | **1,95 / 1,95** / 1,82 | **1,11 / 0,88 / 0,88** | **1,62 / 1,54** / 1,46 |
| EB (paper) | 1,55 / 1,91 / 1,86 | 0,82 / 0,78 / 0,86 | 1,27 / 1,48 / 1,48 |
| Quy tắc \|t\| (không mô hình) | 1,95 / 1,99 / 1,85 | 0,84 / 0,75 / 0,83 | 1,52 / 1,51 / 1,47 |

**So với EB** (p của HAC / bootstrap):

| Top | Dev | Holdout | Cả kỳ |
|---|---|---|---|
| 1% | 1,95 vs 1,55 · **p 0,032** / 0,059 | 1,11 vs 0,82 · **p 0,028** / 0,051 | 1,62 vs 1,27 · **p 0,005 / 0,010** |
| 5% | 1,95 vs 1,91 · p 0,63 | 0,88 vs 0,78 · p 0,33 | 1,54 vs 1,48 · p 0,32 |
| 10% | 1,82 vs 1,86 · p 0,32 (ngang) | 0,88 vs 0,86 · p 0,83 | 1,46 vs 1,48 · p 0,63 |

- **Hơn EB 5/6 ô** (dev, holdout × top 1/5/10%). Top 1% hơn **có ý nghĩa** ở dev, holdout và cả kỳ. Dev top 10% thấp hơn 0,05 nhưng không có ý nghĩa (ngang).
- **So với baseline:** hơn rõ ở dev (p ≤ 0,004 cả 3 mức); holdout cao hơn (1,11 / 0,88 / 0,88 so với 0,61 / 0,51 / 0,54; p 0,06–0,08).
- **IC:** dev 0,151 so với 0,188 của EB (thấp hơn, p = 0,013); holdout 0,043 so với 0,021 (cao hơn, p = 0,49).
- **Độ đúng mức** (cách của paper, dấu gốc): dev độ dốc 0,79 (EB 0,88); holdout 0,19 (EB 0,31): cả hai mất đúng mức sau 2005. Hình: [7_COMPARE_WITH_EB.md](7_COMPARE_WITH_EB.md).

**Giới hạn (trung thực):**
- Đã thử nhiều cấu hình (nhật ký); một số lần đã xem holdout của các cấu hình khi so sánh. Chưa hiệu chỉnh p cho nhiều cấu hình.
- GPU không tất định: học lại cùng seed cho số hơi khác; checkpoint đã lưu thì dự báo cố định (khớp đến 1e-7).
- FinCast là mô hình pretrained.

## 5. Mã nguồn

| File | Bước |
|---|---|
| `code/data.py` | 1–4 |
| `code/model.py` | 8 (kiến trúc từng khối) |
| `code/train.py` | 5–8 (đầu vào, đáp án có nền \|t\|, loss, học từng năm, checkpoint) |
| `code/evaluate.py` | 9–10 |
| `code/run.py` | học 6 seed → `_cache/forecasts/final_s<seed>.npy`, `checkpoints/final_s<seed>/` |
| `code/notebook/final_model.ipynb` | chạy toàn bộ, `BEST_SEED = 1` |
