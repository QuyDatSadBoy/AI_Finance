# Mục 2 trước, mục 1 sau — chỉ sửa các chỗ SAI so với mô hình và bài hiện tại

Dữ liệu lấy từ bản mới nhất trên Overleaf. Danh sách này **chỉ gồm lỗi** (sai sự thật, sai nguồn trích dẫn, mô tả mô hình cũ, hứa điều bài không làm). Các cải thiện văn phong khác đã bỏ. Mọi chỗ không nhắc đến thì giữ nguyên.

**Cách làm:** mở Overleaf, Ctrl+F dán đoạn **CŨ**, thay đúng đoạn đó bằng đoạn **MỚI**. Không chọn rộng, không comment cả vùng. Mỗi đoạn CŨ đã được kiểm tra xuất hiện đúng một lần trong file hiện tại.

## Bảng tóm tắt (15 lỗi: 6 ở mục 2, 9 ở mục 1)

| Mã | Vị trí | Lỗi |
|---|---|---|
| 2-1 | Mục 2 · 2.3, đoạn FinCast | Mô tả dữ liệu tiền huấn luyện của FinCast chưa đúng với bài gốc: FinCast được huấn luyện trên h |
| 2-2 | Mục 2 · 2.3, câu cuối mục | Câu bị lặp nguyên văn hai lần |
| 2-3 | Mục 2 · 2.5, đoạn 2 | Khoá trích dẫn `Chen` không có trong ` |
| 2-4 | Mục 2 · 2.6, đoạn "To address this gap…" | Mô hình hiện tại có PCA nhưng đoạn này không nhắc PCA (mô tả cũ: FinCast, Bi-LSTM, attention po |
| 2-5 | Mục 2 · 2.6, cùng đoạn | Mô hình hiện tại có ISAB (attention giữa các tín hiệu cùng năm) nhưng đoạn này không nhắc |
| 2-6 | Mục 2 · 2.6, câu cuối | Câu này nói bài so với ba mô hình (i, ii, iii) và đánh giá "interpretability" |
| 1-1 | Mục 1 · 1, đoạn 2 (This flexibility…) | Thiếu dấu chấm sau trích dẫn, hai câu dính nhau |
| 1-2 | Mục 1 · 1, đoạn 2 | Nhận định "nhiều anomaly suy giảm hoặc biến mất ngoài mẫu" đang trích Avramov (bài về ML và ràn |
| 1-3 | Mục 1 · 1, đoạn 2, câu cuối | `chen2024deep` là bài *Deep learning in asset pricing*, không phải bài "high-throughput testing |
| 1-4 | Mục 1 · 1, đoạn 3 (FinCast) | Cùng lỗi như 2 |
| 1-5 | Mục 1 · 1, đoạn 6 (To address this gap…) | Mô tả mô hình cũ: không có PCA và ISAB, trong khi mô hình hiện tại có cả hai |
| 1-6 | Mục 1 · 1, đoạn 6, câu cuối | Hứa trọng số attention cho "giải thích kinh tế" |
| 1-7 | Mục 1 · 1, đoạn câu hỏi nghiên cứu (Motivated by this gap…) | Câu hỏi hỏi về "economic interpretability", điều bài không đo |
| 1-8 | Mục 1 · 1, đoạn đóng góp, câu thứ hai | Hai chỗ sai: (1) "outperforms the benchmark… top 1% và top 5%" chỉ đúng ở top 1% (top 5% không  |
| 1-9 | Mục 1 · 1, đoạn lộ trình (The remainder…) | Mô tả cấu trúc bài cũ: mục 3 là "data and methodology", mục 4 là kết quả, mục 5 là "economic in |

---

# Mục 2 — Literature Review

## 2-1 — Mục 2.3, đoạn FinCast

**Sai ở đâu:** Mô tả dữ liệu tiền huấn luyện của FinCast chưa đúng với bài gốc: FinCast được huấn luyện trên hơn 20 tỷ điểm thời gian thuộc cả lĩnh vực tài chính (crypto, forex, futures, cổ phiếu) lẫn phi tài chính, chứ không chỉ cổ phiếu, ngoại hối, hợp đồng tương lai. Cũng khớp với mục 3.2.

**❌ ĐOẠN CŨ (tìm):**

```latex
pretrained on more than 20 billion observations spanning equities, currencies, and futures
```

**✅ ĐOẠN MỚI (thay bằng):**

```latex
pretrained on more than 20 billion time points across financial and non-financial domains
```

*Tiếng Việt (đoạn mới):* …được tiền huấn luyện trên hơn 20 tỷ điểm thời gian thuộc cả lĩnh vực tài chính và phi tài chính

## 2-2 — Mục 2.3, câu cuối mục

**Sai ở đâu:** Câu bị lặp nguyên văn hai lần.

**❌ ĐOẠN CŨ (tìm):**

```latex
remain disconnected from one another; the two sources of information remain disconnected from one another.
```

**✅ ĐOẠN MỚI (thay bằng):**

```latex
remain disconnected from one another.
```

*Tiếng Việt (đoạn mới):* …hai nguồn thông tin vẫn tách rời nhau. (bỏ phần lặp)

## 2-3 — Mục 2.5, đoạn 2

**Sai ở đâu:** Khoá trích dẫn `Chen` không có trong `.bib`, nên báo undefined. Khoá đúng là bài Chen & Dim.

**❌ ĐOẠN CŨ (tìm):**

```latex
within a unified empirical pipeline \cite{Chen}.
```

**✅ ĐOẠN MỚI (thay bằng):**

```latex
within a unified empirical pipeline \cite{chen2025highthroughputassetpricing}.
```

*Tiếng Việt (đoạn mới):* (sửa khoá trích dẫn)

## 2-4 — Mục 2.6, đoạn "To address this gap…"

**Sai ở đâu:** Mô hình hiện tại có PCA nhưng đoạn này không nhắc PCA (mô tả cũ: FinCast, Bi-LSTM, attention pooling).

**❌ ĐOẠN CŨ (tìm):**

```latex
is first encoded by a frozen, pretrained FinCast encoder; a trainable bidirectional LSTM
```

**✅ ĐOẠN MỚI (thay bằng):**

```latex
is first encoded by a frozen, pretrained FinCast encoder whose output is compressed by whitened principal component analysis (PCA); a trainable bidirectional LSTM
```

*Tiếng Việt (đoạn mới):* …được mã hoá trước bởi FinCast tiền huấn luyện đóng băng, đầu ra được nén bằng PCA trắng hoá; một Bi-LSTM…

## 2-5 — Mục 2.6, cùng đoạn

**Sai ở đâu:** Mô hình hiện tại có ISAB (attention giữa các tín hiệu cùng năm) nhưng đoạn này không nhắc.

**❌ ĐOẠN CŨ (tìm):**

```latex
emphasizes the historical periods most informative about the signal's future performance; and this context vector is concatenated
```

**✅ ĐOẠN MỚI (thay bằng):**

```latex
emphasizes the historical periods most informative about the signal's future performance; an induced set attention block (ISAB) then lets the signals of the same year interact; and the resulting vector is concatenated
```

*Tiếng Việt (đoạn mới):* …; một khối induced set attention (ISAB) sau đó cho các tín hiệu cùng năm tương tác với nhau; và vector kết quả được ghép…

## 2-6 — Mục 2.6, câu cuối

**Sai ở đâu:** Câu này nói bài so với ba mô hình (i, ii, iii) và đánh giá "interpretability". Thực tế bài so với EB mining và naive mining, cộng ablation, và không đo interpretability.

**❌ ĐOẠN CŨ (tìm):**

```latex
The framework is evaluated against (i) a benchmark that extrapolates future performance directly from static in-sample diagnostics, (ii) a model using the pretrained FinCast representation without attention-pooling, and (iii) an attention-based model trained without pretrained temporal representations, to isolate the contribution of each architectural component to forecasting accuracy and interpretability.
```

**✅ ĐOẠN MỚI (thay bằng):**

```latex
We compare the framework with empirical Bayes mining and naive mining using the portfolio tests of \cite{chen2025highthroughputassetpricing}, and we use an ablation study to isolate the contribution of its components.
```

*Tiếng Việt (đoạn mới):* Ta so sánh khung này với empirical Bayes và khai thác đơn giản bằng các phép thử danh mục của Chen & Dim, và dùng nghiên cứu ablation để tách đóng góp của từng thành phần.

---

# Mục 1 — Introduction

## 1-1 — Mục 1, đoạn 2 (This flexibility…)

**Sai ở đâu:** Thiếu dấu chấm sau trích dẫn, hai câu dính nhau.

**❌ ĐOẠN CŨ (tìm):**

```latex
\cite{feng2020taming, giglio2022factor}  Indeed
```

**✅ ĐOẠN MỚI (thay bằng):**

```latex
\cite{feng2020taming, giglio2022factor}. Indeed
```

*Tiếng Việt (đoạn mới):* (thêm dấu chấm)

## 1-2 — Mục 1, đoạn 2

**Sai ở đâu:** Nhận định "nhiều anomaly suy giảm hoặc biến mất ngoài mẫu" đang trích Avramov (bài về ML và ràng buộc kinh tế). Nguồn đúng là McLean & Pontiff.

**❌ ĐOẠN CŨ (tìm):**

```latex
\cite{avramov2023machine, chu2020causal}
```

**✅ ĐOẠN MỚI (thay bằng):**

```latex
\cite{mclean2016does, chu2020causal}
```

*Tiếng Việt (đoạn mới):* (đổi nguồn trích dẫn)

## 1-3 — Mục 1, đoạn 2, câu cuối

**Sai ở đâu:** `chen2024deep` là bài *Deep learning in asset pricing*, không phải bài "high-throughput testing". Nguồn đúng là Chen & Dim.

**❌ ĐOẠN CŨ (tìm):**

```latex
as a high-throughput testing problem \cite{chen2024deep}
```

**✅ ĐOẠN MỚI (thay bằng):**

```latex
as a high-throughput testing problem \cite{chen2025highthroughputassetpricing}
```

*Tiếng Việt (đoạn mới):* (đổi nguồn trích dẫn)

## 1-4 — Mục 1, đoạn 3 (FinCast)

**Sai ở đâu:** Cùng lỗi như 2.3: 20 tỷ điểm thời gian thuộc cả lĩnh vực tài chính và phi tài chính, không chỉ "financial observations".

**❌ ĐOẠN CŨ (tìm):**

```latex
pretrained on over 20 billion financial observations across diverse asset classes including equities, cryptocurrencies, foreign exchange, and futures
```

**✅ ĐOẠN MỚI (thay bằng):**

```latex
pretrained on over 20 billion time points across financial and non-financial domains, including cryptocurrencies, foreign exchange, futures and stocks,
```

*Tiếng Việt (đoạn mới):* …được tiền huấn luyện trên hơn 20 tỷ điểm thời gian thuộc cả lĩnh vực tài chính và phi tài chính, gồm tiền mã hoá, ngoại hối, hợp đồng tương lai và cổ phiếu,

## 1-5 — Mục 1, đoạn 6 (To address this gap…)

**Sai ở đâu:** Mô tả mô hình cũ: không có PCA và ISAB, trong khi mô hình hiện tại có cả hai.

**❌ ĐOẠN CŨ (tìm):**

```latex
rather than treating all periods as equally relevant. This learned context vector
```

**✅ ĐOẠN MỚI (thay bằng):**

```latex
rather than treating all periods as equally relevant. The FinCast representation is compressed by whitened principal component analysis (PCA) before the LSTM, and an induced set attention block (ISAB) \cite{lee2019set} lets the signals of the same year interact before the prediction head. This learned context vector
```

*Tiếng Việt (đoạn mới):* Biểu diễn FinCast được nén bằng PCA trắng hoá trước khi vào LSTM, và một khối induced set attention (ISAB) cho các tín hiệu cùng năm tương tác với nhau trước đầu dự đoán.

## 1-6 — Mục 1, đoạn 6, câu cuối

**Sai ở đâu:** Hứa trọng số attention cho "giải thích kinh tế". Bài không phân tích điều này.

**❌ ĐOẠN CŨ (tìm):**

```latex
, using an attention mechanism whose learned weights offer an economically interpretable account of which periods of a signal's history — and which combination of learned temporal features versus conventional statistics — drive that forecast.
```

**✅ ĐOẠN MỚI (thay bằng):**

```latex
, re-estimating the model every year using only earlier data.
```

*Tiếng Việt (đoạn mới):* …trong thời gian tới, bằng cách học lại mô hình mỗi năm chỉ với dữ liệu trước đó.

## 1-7 — Mục 1, đoạn câu hỏi nghiên cứu (Motivated by this gap…)

**Sai ở đâu:** Câu hỏi hỏi về "economic interpretability", điều bài không đo. Mục 4 và 5 trả lời câu hỏi chọn chiến lược so với EB và naive mining.

**❌ ĐOẠN CŨ (tìm):**

```latex
Motivated by this gap, the present study addresses the following research question: how does an adaptive attention-pooling mechanism, when applied to pretrained financial foundation-model representations and fused with conventional in-sample diagnostics, affect both the out-of-sample predictive accuracy and the economic interpretability of forecasts of long-short signals' future returns?
```

**✅ ĐOẠN MỚI (thay bằng):**

```latex
Motivated by this gap, the present study addresses the following research question: does the return history of a strategy, represented by a pretrained foundation model and combined with conventional in-sample diagnostics, improve the out-of-sample selection of long-short signals relative to empirical Bayes and naive mining?
```

*Tiếng Việt (đoạn mới):* Câu hỏi nghiên cứu: lịch sử lợi suất của một chiến lược, được biểu diễn bởi mô hình nền tiền huấn luyện và kết hợp với các chẩn đoán trong mẫu thông thường, có cải thiện việc chọn tín hiệu long–short ngoài mẫu so với empirical Bayes và khai thác đơn giản không?

## 1-8 — Mục 1, đoạn đóng góp, câu thứ hai

**Sai ở đâu:** Hai chỗ sai: (1) "outperforms the benchmark… top 1% và top 5%" chỉ đúng ở top 1% (top 5% không khác biệt có ý nghĩa với EB); (2) ablation "quantify… ISAB" không đúng, vì ablation không phân biệt được ISAB với dao động giữa các lần chạy.

**❌ ĐOẠN CŨ (tìm):**

```latex
demonstrating that the proposed model outperforms the benchmark, achieving Sharpe ratios of up to 1.95 for portfolios formed from the top 1\% and top 5\% of signals ranked by predicted returns. Ablation analyses further quantify the contribution of each component, including PCA compression and ISAB.
```

**✅ ĐOẠN MỚI (thay bằng):**

```latex
demonstrating that the proposed model outperforms empirical Bayes mining in the top 1\% portfolio, with Sharpe ratios of 1.95 against 1.55 in the development period (1983--2004) and 1.11 against 0.82 in the holdout period (2005--2020). Ablation analyses show that the pretrained encoder, the PCA compression and the label construction contribute to the performance, while we cannot distinguish the contribution of ISAB from run-to-run variation.
```

*Tiếng Việt (đoạn mới):* …cho thấy mô hình đề xuất vượt empirical Bayes ở danh mục top 1%, với Sharpe 1,95 so với 1,55 ở giai đoạn phát triển (1983–2004) và 1,11 so với 0,82 ở holdout (2005–2020). Ablation cho thấy bộ mã hoá pretrained, nén PCA và cách xây nhãn đóng góp, còn đóng góp của ISAB ta không phân biệt được với dao động giữa các lần chạy.

## 1-9 — Mục 1, đoạn lộ trình (The remainder…)

**Sai ở đâu:** Mô tả cấu trúc bài cũ: mục 3 là "data and methodology", mục 4 là kết quả, mục 5 là "economic interpretation of attention weights". Cấu trúc hiện tại khác.

**❌ ĐOẠN CŨ (tìm):**

```latex
Section 3 describes the data and the proposed attention-augmented FinCast–Bi-LSTM methodology. Section 4 presents the empirical results, including out-of-sample forecasting performance and portfolio-level evaluations. Section 5 discusses the economic interpretation of the learned attention weights, robustness checks, and limitations.
```

**✅ ĐOẠN MỚI (thay bằng):**

```latex
Section 3 formulates the problem and describes the proposed model. Section 4 presents the data, the experimental design, the forecast accuracy and the ablation study. Section 5 reports the portfolio results, the robustness to noise strategies, the temporal analysis and the limitations.
```

*Tiếng Việt (đoạn mới):* Mục 3 phát biểu bài toán và mô tả mô hình. Mục 4 trình bày dữ liệu, thiết kế thí nghiệm, độ chính xác dự báo và ablation. Mục 5 báo kết quả danh mục, độ bền trước chiến lược nhiễu, phân tích theo thời gian và giới hạn.

---

## Ghi chú

- **BibTeX:** các khoá dùng ở đây (`mclean2016does`, `chen2025highthroughputassetpricing`, `lee2019set`) đã có trong `.bib` trên Overleaf.
- **Các cải thiện không bắt buộc** (bỏ chữ "novel", thêm "to our knowledge", thêm Chronos và TimesFM, thêm câu về empirical Bayes ở mục 2, lý do chia giai đoạn 2004) đã bỏ khỏi danh sách theo yêu cầu. Cần thì mình đưa lại.
- **Sau khi sửa,** tìm trong mục 1 và 2 các chữ `interpretab`, `cite{Chen}`, `chen2024deep`: không được còn (trừ phần đã comment).
- Sửa xong, mình tải lại file từ Overleaf để đối chiếu từng chỗ.
