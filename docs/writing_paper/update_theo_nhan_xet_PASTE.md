# Hướng dẫn sửa bài theo nhận xét của thầy (bản dễ làm theo)

Bản Overleaf đã đối chiếu: tải ngày 05/11/2026.

## Cách làm theo

Có 2 kiểu sửa, mỗi chỗ đều ghi rõ là kiểu nào:

- 🔁 **THAY:** trên Overleaf bấm Ctrl+F, dán đoạn 🔴 **GỐC** để tìm, bôi đen đúng đoạn đó, rồi dán đoạn 🟢 **MỚI** đè lên.
- ➕ **THÊM:** Ctrl+F tìm đoạn 🔴 **GỐC** nhưng **không xoá nó**. Click ngay sau đoạn đó, nhấn Enter 2 lần (tạo dòng trống), rồi dán đoạn 🟢 **MỚI**.

Thêm vài điều cần biết:

- Các sửa xếp **từ trên xuống dưới** theo file, kèm số dòng trên Overleaf. Số dòng sẽ lệch dần khi bạn sửa, nên cứ tìm bằng Ctrl+F.
- Mình đã kiểm tra: mọi đoạn GỐC đều xuất hiện **đúng 1 lần** trong bài. Ctrl+F ra 0 kết quả nghĩa là đã sửa rồi, hoặc lúc copy bị sót ký tự.
- Với các câu thêm vào giữa đoạn, MỚI đã chứa sẵn GỐC ở đầu, nên chỉ việc dán đè lên.

---

## Tiến độ

| Việc | Trạng thái |
|---|---|
| `.bib`: thêm `}` sau bài Bowles (lỗi Chen & Dim hiện "?") | ✅ Đã xong |
| `.bib`: xoá 3 khoá trùng (gu2020empirical, vaswani2017attention, wu2020connecting) | ✅ Đã xong |
| `.bib`: sửa tác giả Holen, thêm số WP cho Kelly, đổi `// Dat` thành `% Dat` | ✅ Đã xong |
| `.bib`: sửa chen2023does, guijarro2025deep, thêm 8 bài mới | ⬜ Còn lại: **Phần 1** dưới đây |
| `.tex`: 49 chỗ sửa | ⬜ Chưa làm: **Phần 2** dưới đây |

---

# PHẦN 1 — File `sn-bibliography.bib` (4 việc)

### B1 · 🔁 THAY · bài `chen2023does` (khoảng dòng 520)

🔴 **GỐC** (xoá cả 6 dòng):
```bibtex
@techreport{chen2023does,
  title={Does peer-reviewed theory help predict the cross-section of stock returns?},
  author={Chen, Andrew Y and Lopez-Lira, Alejandro and Zimmermann, Tom},
  year={2023},
  institution={Working Paper}
}
```
🟢 **MỚI:**
```bibtex
@article{chen2023does,
  title   = {Does Peer-Reviewed Research Help Predict Stock Returns?},
  author  = {Chen, Andrew Y. and Lopez-Lira, Alejandro and Zimmermann, Tom},
  journal = {arXiv preprint arXiv:2212.10317},
  year    = {2022}
}
```
💬 **Vì sao:** đây là tên của bản nháp cũ. Bản hiện tại trên arXiv đã đổi tên.

### B2 · 🔁 THAY · bài `guijarro2025deep` (khoảng dòng 580)

🔴 **GỐC** (xoá cả 7 dòng):
```bibtex
@article{guijarro2025deep,
  title={Deep learning statistical arbitrage},
  author={Guijarro-Ordonez, Jorge and Pelger, Markus and Zanotti, Greg},
  journal={Management Science},
  year={2025},
  publisher={INFORMS}
}
```
🟢 **MỚI:**
```bibtex
@article{guijarro2025deep,
  title   = {Deep Learning Statistical Arbitrage},
  author  = {Guijarro-Ordonez, Jorge and Pelger, Markus and Zanotti, Greg},
  journal = {Management Science},
  volume  = {72},
  number  = {9},
  pages   = {7502--7549},
  year    = {2026},
  doi     = {10.1287/mnsc.2022.03132}
}
```
💬 **Vì sao:** bài đã được in chính thức (theo CrossRef), nên cần có tập, số và trang.

### B3 · ➕ THÊM · 8 bài mới, dán vào **cuối file** `.bib`

Các sửa 3, 21, 22, 23 và 27 ở Phần 2 sẽ trích những bài này. Phải dán ở đây trước, nếu không PDF sẽ hiện "?".

🟢 **MỚI:**
```bibtex

% ---- Them theo nhan xet cua thay
@article{campbell2000asset,
  title   = {Asset Pricing at the Millennium},
  author  = {Campbell, John Y.},
  journal = {The Journal of Finance},
  volume  = {55},
  number  = {4},
  pages   = {1515--1567},
  year    = {2000},
  doi     = {10.1111/0022-1082.00260}
}

@article{yae2023robust,
  title   = {Robust Monitoring Machine: A Machine Learning Solution for Out-of-Sample {$R^2$}-Hacking in Return Predictability Monitoring},
  author  = {Yae, James and Luo, Yang},
  journal = {Financial Innovation},
  volume  = {9},
  number  = {1},
  pages   = {94},
  year    = {2023},
  doi     = {10.1186/s40854-023-00497-z}
}

@article{zhou2025cross,
  title   = {Cross-Sectional Anomalies and Conditional Asset Pricing Models Based on Investor Sentiment: Evidence from the {Chinese} Stock Market},
  author  = {Zhou, Zhong-Qiang and Wu, Jiajia and Huang, Ping and Xiong, Xiong},
  journal = {Financial Innovation},
  volume  = {11},
  number  = {1},
  pages   = {103},
  year    = {2025},
  doi     = {10.1186/s40854-025-00774-z}
}

@article{wang2024stock,
  title   = {Stock Return Prediction with Multiple Measures Using Neural Network Models},
  author  = {Wang, Cong},
  journal = {Financial Innovation},
  volume  = {10},
  number  = {1},
  pages   = {72},
  year    = {2024},
  doi     = {10.1186/s40854-023-00608-w}
}

@article{kang2025how,
  title   = {How Well Do Machine Learning Models in Finance Work?},
  author  = {Kang, Yeonchan and Ryu, Doojin and Webb, Robert I.},
  journal = {Financial Innovation},
  volume  = {11},
  number  = {1},
  pages   = {128},
  year    = {2025},
  doi     = {10.1186/s40854-025-00870-0}
}

@article{htun2024forecasting,
  title   = {Forecasting Relative Returns for {S\&P} 500 Stocks Using Machine Learning},
  author  = {Htun, Htet Htet and Biehl, Michael and Petkov, Nicolai},
  journal = {Financial Innovation},
  volume  = {10},
  number  = {1},
  pages   = {118},
  year    = {2024},
  doi     = {10.1186/s40854-024-00644-0}
}

@article{htun2023survey,
  title   = {Survey of Feature Selection and Extraction Techniques for Stock Market Prediction},
  author  = {Htun, Htet Htet and Biehl, Michael and Petkov, Nicolai},
  journal = {Financial Innovation},
  volume  = {9},
  number  = {1},
  pages   = {26},
  year    = {2023},
  doi     = {10.1186/s40854-022-00441-7}
}

@article{gunduz2021efficient,
  title   = {An Efficient Stock Market Prediction Model Using Hybrid Feature Reduction Method Based on Variational Autoencoders and Recursive Feature Elimination},
  author  = {Gunduz, Hakan},
  journal = {Financial Innovation},
  volume  = {7},
  number  = {1},
  pages   = {28},
  year    = {2021},
  doi     = {10.1186/s40854-021-00243-3}
}
```
💬 **Vì sao:** thầy yêu cầu trích bài của tạp chí đích. 7 bài cuối đều thuộc *Financial Innovation*, còn bài Campbell dùng cho câu "asset pricing quan trọng". Tất cả đã đối chiếu với CrossRef.

### B4 · (tuỳ chọn, làm sau cùng) dọn rác

Việc này **không bắt buộc**: các bài không được trích sẽ không in ra.
- Xoá từ dòng 1 (`%% Journal article`) đến ngay trước `@article{sharpe1964capital,`. Đây là 13 bài mẫu của template.
- Xoá entry `@article{oy1tz1952portfolio, ... }` (tên bài hỏng "PORTFOLIO) SELECTION").

**Recompile.** Nếu PDF không còn "?" ở các trích dẫn thì sang Phần 2.

---

# PHẦN 2 — File `sn-article.tex` (49 chỗ, từ trên xuống)

## Tiêu đề và Abstract

### Sửa 1 · 🔁 THAY · Tiêu đề · dòng 95
🔴 **GỐC:**
```latex
\title[Article Title]{A novel attention network for assets pricing}
```
🟢 **MỚI:**
```latex
\title[Article Title]{An attention network with pretrained representations for selecting long-short strategies in asset pricing}
```
💬 **Vì sao:** "assets pricing" sai ngữ pháp, và "novel" là từ quảng cáo nên tránh.

### Sửa 2 · 🔁 THAY · Abstract · dòng 124
🔴 **GỐC:**
```latex
Furthermore, the proposed model outperforms the benchmark methods, achieving Sharpe ratios of up to 1.95 among the top 1\% and top 5\% of strategies.
```
🟢 **MỚI:**
```latex
Furthermore, the top 1\% portfolio of the proposed model has a higher Sharpe ratio than that of empirical Bayes mining in both the development period (1.95 against 1.55) and the holdout period (1.11 against 0.82).
```
💬 **Vì sao:** ở top 5% mô hình không hơn có ý nghĩa thống kê (ở dev, naive mining còn đạt 1,99, cao hơn 1,95). Chỉ top 1% là thắng thật.

---

## Mục 1 — Introduction

### Sửa 3 · 🔁 THAY · Intro đoạn 1, câu đầu · dòng 148
🔴 **GỐC:**
```latex
Asset pricing theory explains how financial markets translate
```
🟢 **MỚI** (thêm 3 câu ở đầu đoạn, giữ câu cũ ở cuối):
```latex
Asset pricing is central to finance. Expected returns determine the cost of capital that firms face, the allocation of savings across assets, and the benchmarks against which investment strategies and portfolio managers are evaluated \cite{campbell2000asset}. Knowing which sources of return predictability persist is therefore relevant both for researchers who test asset pricing theories and for investors who allocate capital to trading strategies. Asset pricing theory explains how financial markets translate
```
💬 **Vì sao:** thầy yêu cầu nêu vai trò của asset pricing trong tài chính.
📖 **Nghĩa:** Định giá tài sản là trung tâm của tài chính. Lợi suất kỳ vọng quyết định chi phí vốn của doanh nghiệp, cách phân bổ tiết kiệm giữa các tài sản, và chuẩn để đánh giá chiến lược đầu tư và người quản lý danh mục. Vì vậy, biết nguồn dự báo lợi suất nào còn bền vững là điều quan trọng với cả nhà nghiên cứu lẫn nhà đầu tư.

### Sửa 4 · 🔁 THAY · Intro đoạn 2 · dòng 150
🔴 **GỐC:**
```latex
significantly elevates the risk of detecting spurious, sample-specific patterns that fail to generalize out-of-sample
```
🟢 **MỚI:**
```latex
increases the risk of detecting spurious, sample-specific patterns that may not generalize out of sample
```
💬 **Vì sao:** thầy dặn tránh "fail" và các từ mạnh.

### Sửa 5 · 🔁 THAY · Intro đoạn 2 · dòng 150
🔴 **GỐC:**
```latex
\cite{avramov2023machine, chu2020causal}
```
🟢 **MỚI:**
```latex
\cite{mclean2016does, chu2020causal}
```
💬 **Vì sao:** câu này nói anomaly suy giảm ngoài mẫu. Bài nói điều đó là McLean & Pontiff, còn Avramov nói về ML với ràng buộc kinh tế.

### Sửa 6 · 🔁 THAY · Intro đoạn 2 · dòng 150
🔴 **GỐC:**
```latex
as a high-throughput testing problem \cite{chen2024deep}
```
🟢 **MỚI:**
```latex
as a high-throughput testing problem \cite{chen2025highthroughputassetpricing}
```
💬 **Vì sao:** người đặt ra bài toán "high-throughput" là Chen & Dim. `chen2024deep` là bài tổng quan về deep learning.

### Sửa 7 · 🔁 THAY · Intro đoạn 3 (FinCast) · dòng 152
🔴 **GỐC:**
```latex
Among these advances, FinCast \cite{zhu2025fincast} marks a major methodological milestone as the first foundation model tailored specifically to financial time series.
```
🟢 **MỚI:**
```latex
Among these advances, FinCast \cite{zhu2025fincast} is presented as the first foundation model designed specifically for financial time series.
```
💬 **Vì sao:** "major milestone" là từ quảng cáo. Ta chỉ thuật lại điều chính tác giả FinCast nói.

### Sửa 8 · 🔁 THAY · Intro đoạn 3 · dòng 152
🔴 **GỐC:**
```latex
pretrained on over 20 billion financial observations across diverse asset classes including equities, cryptocurrencies, foreign exchange, and futures
```
🟢 **MỚI:**
```latex
pretrained on over 20 billion time points across financial and non-financial domains, including cryptocurrencies, foreign exchange, futures and stocks,
```
💬 **Vì sao:** theo bài FinCast, dữ liệu huấn luyện gồm cả lĩnh vực phi tài chính, không chỉ tài chính.

### Sửa 9 · 🔁 THAY · Intro đoạn 3 · dòng 152
🔴 **GỐC:**
```latex
directly mitigating the temporal non-stationarity and structural breaks that typically confound conventional machine learning models
```
🟢 **MỚI:**
```latex
which is intended to mitigate the temporal non-stationarity and structural breaks that complicate conventional machine learning models
```
💬 **Vì sao:** "directly" và "confound" là khẳng định mạnh.

### Sửa 10 · 🔁 THAY · Intro đoạn 3 · dòng 152
🔴 **GỐC:**
```latex
It inherently lacks a targeted mechanism to assess whether a signal's historical return pattern conveys genuine predictive power about its future performance, nor does it offer a principled framework to integrate
```
🟢 **MỚI:**
```latex
It is therefore not designed to assess whether a signal's historical return pattern carries information about its future performance, and it does not by itself provide a way to integrate
```
💬 **Vì sao:** "inherently lacks" và "principled" là từ mạnh. Câu mới nêu hạn chế một cách nhẹ nhàng.

### Sửa 11 · 🔁 THAY · Intro đoạn 4 · dòng 154
🔴 **GỐC:**
```latex
summary numbers that compress an entire historical path into two figures and discard the sequential and regime-dependent information that path contains.
```
🟢 **MỚI:**
```latex
summary numbers that compress an entire historical path into two figures and therefore do not retain the sequential and regime-dependent information that the path may contain.
```
💬 **Vì sao:** "discard" là từ mạnh.

### Sửa 12 · 🔁 THAY · Intro đoạn 5 · dòng 156
🔴 **GỐC:**
```latex
such as FinCast generalize well across temporal non-stationarity but provide no mechanism for adaptively weighting
```
🟢 **MỚI:**
```latex
such as FinCast generalize well across temporal non-stationarity but do not by themselves provide a mechanism for adaptively weighting
```
💬 **Vì sao:** làm nhẹ "provide no mechanism".

### Sửa 13 · 🔁 THAY · Intro đoạn 5 · dòng 156
🔴 **GỐC:**
```latex
and no existing framework combines them with a learned, attention-weighted temporal representation
```
🟢 **MỚI:**
```latex
and, to our knowledge, no existing framework combines them with a learned temporal representation
```
💬 **Vì sao:** "No existing" là khẳng định tuyệt đối, nên thêm "to our knowledge".

### Sửa 14 · 🔁 THAY · Intro đoạn 5 · dòng 156
🔴 **GỐC:**
```latex
No prior study has examined whether fusing a pretrained
```
🟢 **MỚI:**
```latex
To our knowledge, no prior study has examined whether fusing a pretrained
```
💬 **Vì sao:** giống sửa 13.

### Sửa 15 · ➕ THÊM · đoạn mới "phương pháp mới cần làm được gì" · sau dòng 156
🔴 **GỐC** (câu cuối đoạn 5, **giữ nguyên**, click ngay sau dấu `\\`):
```latex
with conventional in-sample diagnostics improves out-of-sample forecasts of a signal's future return relative to either source of information alone.\\
```
🟢 **MỚI** (Enter 2 lần rồi dán):
```latex
These limitations suggest what a forecasting method for long-short strategies needs to provide, and each requirement motivates one part of our design. First, it should read the full return sequence rather than only its summary statistics, because two strategies with the same t-stat can earn their returns in very different ways; a pretrained encoder provides such a reading without fitting a large sequence model to noisy returns. Second, it should keep the number of trainable inputs small relative to the roughly 26,000 strategies observed each year, because monthly long-short returns are noisy and a high-dimensional input invites overfitting; this motivates compressing the pretrained representation before any trainable block. Third, it should learn which periods of a strategy's history matter for the forecasting task, which a generic pretrained representation does not do by itself; this motivates a trainable sequence reader with attention pooling. Fourth, it should compare strategies with each other, because a strategy is selected only if it looks attractive relative to the others of the same year; this motivates cross-sectional interaction and a training label defined relative to the year. Finally, it should build on, rather than replace, the t-stat, which already carries much of the information at the top of the ranking \cite{chen2025highthroughputassetpricing}.\\
```
💬 **Vì sao:** đây là ý chính thầy yêu cầu: vì sao cần phương pháp mới. Đoạn này nêu 5 yêu cầu, mỗi yêu cầu dẫn tới một khối của mô hình.
📖 **Nghĩa:** Một phương pháp tốt cần: (1) đọc cả chuỗi lợi suất chứ không chỉ t-stat, vì hai chiến lược cùng t-stat có thể kiếm lời theo cách rất khác, và bộ mã hoá pretrained làm được việc này mà không phải học mô hình lớn trên dữ liệu nhiễu; (2) giữ ít đầu vào so với khoảng 26.000 chiến lược mỗi năm để tránh overfitting, nên phải nén; (3) học giai đoạn nào trong lịch sử là quan trọng, nên cần Bi-LSTM kèm attention pooling; (4) so sánh các chiến lược với nhau, vì chọn chiến lược là chọn tương đối trong năm, nên cần ISAB và nhãn tính theo năm; (5) dựa trên t-stat chứ không thay thế nó.

### Sửa 16 · 🔁 THAY CẢ ĐOẠN · đoạn "To address this gap" · dòng 158–161

Đoạn này nằm trên **4 dòng** của Overleaf, kể cả 1 dòng bắt đầu bằng `%`. Bôi đen từ chữ `To address this gap, we propose` (dòng 158) đến hết `drive that forecast.\\` (dòng 161).

🔴 **GỐC** (toàn bộ 4 dòng):
```latex
To address this gap, we propose an attention-augmented FinCast framework that forecasts the out-of-sample mean return of long-short signals by combining pretrained temporal representations with a learned, adaptive summary of each signal's own history. For every candidate signal, the historical monthly long-short return series is first passed through a frozen, pretrained FinCast encoder to obtain a generalized temporal representation of the sequence. A trainable bidirectional LSTM further processes this representation to capture sequential dependencies specific to the return-forecasting task, and an attention-pooling layer then aggregates the resulting hidden states into a single context vector, adaptively weighting the historical periods that are most informative about the signal's future behavior 
rather than treating all periods as equally relevant. The FinCast representation is compressed by whitened principal component analysis (PCA) before the LSTM, and an induced set attention block (ISAB) \cite{lee2019set} lets the signals of the same year interact before the prediction head. This learned context vector
% rather than treating all periods as equally relevant. This learned context vector
is concatenated with the signal's conventional in-sample diagnostics — its in-sample t-statistic and mean return — before being passed through a fully connected prediction head to estimate the signal's expected out-of-sample mean return. In this way, the framework can be interpreted as a signal-of-signals model: rather than pricing individual assets directly, it learns to forecast which previously documented long-short signals are likely to remain informative going forward, using an attention mechanism whose learned weights offer an economically interpretable account of which periods of a signal's history — and which combination of learned temporal features versus conventional statistics — drive that forecast.\\
```
🟢 **MỚI:**
```latex
To address these requirements, we propose an attention-augmented FinCast framework in which each block addresses one of them and passes to the next block a representation suited to its role. A frozen, pretrained FinCast encoder reads the 128 months of returns before the forecasting year and produces four patch representations, one for each 32-month segment. Because these representations are high-dimensional relative to the number of strategies, a whitened principal component analysis (PCA), estimated only on data before the first forecasting year, compresses each patch to 16 components on a common scale. A trainable bidirectional LSTM then reads the four compressed patches in temporal order, and attention pooling weights them into one context vector; this is the task-specific step that the frozen encoder lacks. An induced set attention block (ISAB) \cite{lee2019set} then lets the context vectors of all strategies in the same year interact through a small number of learned summaries, at a cost that is linear in the number of strategies. Finally, the context vector is concatenated with the in-sample t-stat and mean return, and a small fully connected head forecasts the part of next-year performance that the t-stat does not already explain. The trainable part of the model is kept small (about 1.6 million parameters) on top of the frozen encoder, and the model is re-estimated every year using only earlier data. In this way, the framework can be interpreted as a signal-of-signals model: rather than pricing individual assets directly, it learns to forecast which long-short signals are likely to remain profitable.\\
```
💬 **Vì sao:** thầy muốn đoạn "To address" sâu hơn, không chỉ quảng bá. Đoạn cũ chỉ liệt kê các khối, và còn hứa "economically interpretable attention weights" mà bài không phân tích. Đoạn mới nói khối nào giải quyết yêu cầu nào ở sửa 15, và khối đó chuyển gì cho khối sau.
📖 **Nghĩa:** FinCast (đóng băng) đọc 128 tháng lợi suất và tạo 4 patch, mỗi patch 32 tháng. Vì các patch có quá nhiều chiều so với số chiến lược, PCA nén mỗi patch còn 16 thành phần; PCA chỉ ước lượng trên dữ liệu trước năm dự báo đầu tiên. Bi-LSTM đọc 4 patch theo thứ tự thời gian, rồi attention pooling gộp thành 1 vector; đây là bước gắn với nhiệm vụ mà FinCast không có. ISAB cho các chiến lược cùng năm tương tác, với chi phí tuyến tính. Cuối cùng, vector được ghép với t-stat và lãi trung bình để dự báo phần mà t-stat chưa giải thích. Phần cần học nhỏ (khoảng 1,6 triệu tham số), và mô hình được học lại mỗi năm.

### Sửa 17 · 🔁 THAY CẢ CÂU · câu hỏi nghiên cứu · dòng 163
🔴 **GỐC** (cả câu, đến hết dấu `?\\`):
```latex
Motivated by this gap, the present study addresses the following research question: how does an adaptive attention-pooling mechanism, when applied to pretrained financial foundation-model representations and fused with conventional in-sample diagnostics, affect both the out-of-sample predictive accuracy and the economic interpretability of forecasts of long-short signals' future returns?\\
```
🟢 **MỚI:**
```latex
Motivated by this gap, the present study addresses the following research question: does the return history of a strategy, represented by a pretrained foundation model and combined with conventional in-sample diagnostics, improve the out-of-sample selection of long-short signals relative to empirical Bayes and naive mining?\\
```
💬 **Vì sao:** câu cũ hỏi về "economic interpretability", điều bài không đo.

### Sửa 18 · 🔁 THAY · đoạn đóng góp · dòng 167
🔴 **GỐC:**
```latex
we propose a novel attention-augmented FinCast–Bi-LSTM framework that adapts
```
🟢 **MỚI:**
```latex
we propose an attention-augmented FinCast–Bi-LSTM framework that adapts
```
💬 **Vì sao:** bỏ chữ "novel".

### Sửa 19 · 🔁 THAY · đoạn đóng góp · dòng 167
🔴 **GỐC:**
```latex
demonstrating that the proposed model outperforms the benchmark, achieving Sharpe ratios of up to 1.95 for portfolios formed from the top 1\% and top 5\% of signals ranked by predicted returns. Ablation analyses further quantify the contribution of each component, including PCA compression and ISAB.
```
🟢 **MỚI:**
```latex
demonstrating that the proposed model outperforms empirical Bayes mining in the top 1\% portfolio, with Sharpe ratios of 1.95 against 1.55 in the development period (1983--2004) and 1.11 against 0.82 in the holdout period (2005--2020). Ablation analyses show that the pretrained encoder, the PCA compression and the label construction contribute to the performance, while the contribution of ISAB cannot be distinguished from run-to-run variation.
```
💬 **Vì sao:** câu cũ sai ở top 5%, giống sửa 2. Câu cũ cũng nói ablation đo được đóng góp của ISAB, trái với bảng ablation ở mục 4.3.

### Sửa 20 · 🔁 THAY · đoạn lộ trình bài · dòng 169
🔴 **GỐC:**
```latex
Section 3 describes the data and the proposed attention-augmented FinCast–Bi-LSTM methodology. Section 4 presents the empirical results, including out-of-sample forecasting performance and portfolio-level evaluations. Section 5 discusses the economic interpretation of the learned attention weights, robustness checks, and limitations.
```
🟢 **MỚI:**
```latex
Section 3 formulates the problem and describes the proposed architecture and the rationale for each of its blocks. Section 4 presents the data, the experimental design, the forecast accuracy and the ablation study. Section 5 reports the portfolio results, the robustness to noise strategies and the temporal analysis.
```
💬 **Vì sao:** lộ trình cũ mô tả sai nội dung các mục 3, 4, 5 hiện tại.

---

## Mục 2 — Literature Review

### Sửa 21 · 🔁 THAY · mục 2.1 · dòng 176
🔴 **GỐC:**
```latex
simply survived a data-mining search \cite{chen2023does} .
```
🟢 **MỚI** (bỏ dấu cách thừa trước dấu chấm, thêm 2 câu):
```latex
simply survived a data-mining search \cite{chen2023does}. Even out-of-sample evaluation can be compromised when model choices draw on patterns that appear only in the test sample, which motivates evaluation designs that keep the test period unseen \cite{yae2023robust}. Evidence from the Chinese market further shows that the pricing of cross-sectional anomalies depends on conditioning information such as investor sentiment \cite{zhou2025cross}.
```
💬 **Vì sao:** thêm 2 bài *Financial Innovation*. Câu đầu nối thẳng với cách bài giữ kín giai đoạn holdout 2005–2020.
📖 **Nghĩa:** Ngay cả đánh giá ngoài mẫu cũng có thể sai lệch nếu chọn mô hình dựa vào tập kiểm tra. Ở Trung Quốc, việc định giá anomaly phụ thuộc vào tâm lý nhà đầu tư.

### Sửa 22 · 🔁 THAY · mục 2.2 · dòng 179
🔴 **GỐC:**
```latex
substantially outperform traditional linear benchmarks in out-of-sample return prediction.
```
🟢 **MỚI:**
```latex
substantially outperform traditional linear benchmarks in out-of-sample return prediction. The gains depend on the setting: neural networks predict several measures of stock returns consistently from firm characteristics and improve further when macroeconomic and sentiment variables are added \cite{wang2024stock}, whereas moderately complex tree-based models perform best in the data-constrained Korean market \cite{kang2025how}.
```
💬 **Vì sao:** thêm 2 bài *Financial Innovation*. Ý "lợi ích phụ thuộc bối cảnh, ít dữ liệu thì mô hình vừa phải tốt hơn" là lý do bài giữ phần cần học nhỏ.

### Sửa 23 · 🔁 THAY · mục 2.2 · dòng 181
🔴 **GỐC:**
```latex
rather than learning which periods of that history are most informative for the prediction task at hand.\\
```
🟢 **MỚI:**
```latex
rather than learning which periods of that history are most informative for the prediction task at hand. Related work also frames prediction relative to a benchmark, for example forecasting whether S\&P 500 stocks will outperform their index with random forests, support vector machines and LSTM networks \cite{htun2024forecasting}, which is closer to the selection problem studied here.\\
```
💬 **Vì sao:** thêm 1 bài *Financial Innovation* về dự báo lợi suất **tương đối**, gần với bài toán chọn chiến lược của ta.

### Sửa 24 · 🔁 THAY · mục 2.3 · dòng 187
🔴 **GỐC:**
```latex
pretrained on more than 20 billion observations spanning equities, currencies, and futures
```
🟢 **MỚI:**
```latex
pretrained on more than 20 billion time points spanning financial and non-financial domains
```
💬 **Vì sao:** giống sửa 8.

### Sửa 25 · 🔁 THAY · mục 2.3 · dòng 189
🔴 **GỐC:**
```latex
remain disconnected from one another; the two sources of information remain disconnected from one another.
```
🟢 **MỚI:**
```latex
remain disconnected from one another.
```
💬 **Vì sao:** câu bị lặp ý.

### Sửa 26 · 🔁 THAY · mục 2.3 · dòng 189
🔴 **GỐC:**
```latex
Second, they offer no principled mechanism for combining
```
🟢 **MỚI:**
```latex
Second, they do not by themselves provide a mechanism for combining
```
💬 **Vì sao:** "no principled" là từ mạnh.

### Sửa 27 · 🔁 THAY · mục 2.4 · dòng 192
🔴 **GỐC:**
```latex
reducing the input dimension before the attention-based predictor improves generalization and lowers computational cost.
```
🟢 **MỚI:**
```latex
reducing the input dimension before the attention-based predictor lowers computational cost without sacrificing generalization accuracy. Feature selection and extraction are widely used for the same purpose in stock market prediction \cite{htun2023survey}, for example to reduce the inputs of LSTM and gradient boosting models with variational autoencoders \cite{gunduz2021efficient}.
```
💬 **Vì sao:** bài Guo chỉ nói giảm chiều làm giảm chi phí tính toán **mà không mất** độ chính xác, chứ không nói nó "improves generalization". Sửa này cũng thêm 2 bài *Financial Innovation* làm cơ sở cho bước nén PCA.

### Sửa 28 · 🔁 THAY · mục 2.5 · dòng 218
🔴 **GỐC:**
```latex
and has not incorporated learned, attention-weighted representations of a signal's full historical return path of the kind made possible by pretrained time-series foundation models.
```
🟢 **MỚI:**
```latex
and, to our knowledge, has not used learned representations of a signal's full historical return path, such as those made possible by pretrained time-series foundation models.
```
💬 **Vì sao:** thêm "to our knowledge" cho nhẹ.

### Sửa 29 · 🔁 THAY · mục 2.6 · dòng 222
🔴 **GỐC:**
```latex
through large-scale pretraining, but provide no mechanism for adaptively weighting
```
🟢 **MỚI:**
```latex
through large-scale pretraining, but do not by themselves provide a mechanism for adaptively weighting
```
💬 **Vì sao:** giống sửa 12.

### Sửa 30 · 🔁 THAY · mục 2.6 · dòng 222
🔴 **GỐC:**
```latex
No existing framework adapts a pretrained foundation model's temporal representation
```
🟢 **MỚI:**
```latex
To our knowledge, no existing framework adapts a pretrained foundation model's temporal representation
```
💬 **Vì sao:** giống sửa 13.

### Sửa 31 · 🔁 THAY · mục 2.6 · dòng 222
🔴 **GỐC:**
```latex
dimensionality reduction before attention has been shown to improve generalization in noisy financial data \citep{guo2026hybrid}
```
🟢 **MỚI:**
```latex
dimensionality reduction before attention has been shown to lower computational cost without sacrificing generalization accuracy in noisy financial data \citep{guo2026hybrid}
```
💬 **Vì sao:** giống sửa 27, cùng một lỗi về bài Guo.

### Sửa 32 · 🔁 THAY · mục 2.6 · dòng 222
🔴 **GỐC:**
```latex
No prior study has examined whether fusing such a representation
```
🟢 **MỚI:**
```latex
To our knowledge, no prior study has examined whether fusing such a representation
```
💬 **Vì sao:** giống sửa 14.

### Sửa 33 · 🔁 THAY · mục 2.6 · dòng 228
🔴 **GỐC:**
```latex
estimates its expected out-of-sample mean return.We compare
```
🟢 **MỚI:**
```latex
estimates its expected out-of-sample mean return. We compare
```
💬 **Vì sao:** thiếu dấu cách sau dấu chấm.

---

## Mục 3 — Kiến trúc (giải thích vì sao các khối khớp nhau)

### Sửa 34 · ➕ THÊM · đoạn và bảng "lý do thiết kế" · sau dòng 401 (cuối 3.2.1 Overall architecture)
🔴 **GỐC** (câu cuối của 3.2.1, **giữ nguyên**, click ngay sau dấu chấm):
```latex
or on other labels of the underlying assets.
```
🟢 **MỚI** (Enter 2 lần rồi dán):
```latex
The blocks are arranged so that each one resolves a difficulty left by the previous one and passes on a representation that the next block can use (Table \ref{tab:design}). The pretrained encoder supplies a rich description of the return sequence without fitting a large model to noisy returns, but its output is high-dimensional and task-agnostic. PCA reduces this output to a size that the roughly 26,000 strategies of a year can support, and whitening places the components on a common scale for the trainable blocks. The compressed patches remain ordered in time, so a bidirectional LSTM can read them as a short sequence, and attention pooling turns them into one vector by learning which segments matter for the forecasting task, which is the task-specific step that the frozen encoder does not provide. Only after each strategy is summarized in this compact vector does ISAB compare strategies within the year, which keeps the cross-sectional step inexpensive. The head then combines the learned vector with the t-stat and mean return, and the label of Equation \eqref{eq:label} asks the network to forecast only what the t-stat does not already explain, so the learned representation and the classical diagnostics play complementary rather than competing roles. Finally, the Huber loss and the year weights adapt the training objective to the heavy tails and the uneven informativeness of the yearly cross-sections.

\begin{table}[h]
\caption{Design rationale of the proposed architecture. Each block addresses a difficulty of the forecasting problem and passes its output to the next block. The last column refers to the ablation of Section \ref{subsec4_3}; the comparison with the baseline model in Section \ref{subsec5_1} measures the joint effect of PCA, ISAB, sign orientation and the label.}\label{tab:design}%
\small
\begin{tabular}{@{}p{2.6cm}p{3.9cm}p{2.3cm}p{3.6cm}@{}}
\toprule
Block & Difficulty addressed & Output passed on & Evidence \\
\midrule
FinCast (pretrained, frozen) & The t-stat discards the shape of the return path; a sequence model trained from scratch on noisy returns may overfit & $4 \times 1280$ patch representations & Random initialization lowers the holdout Sharpe ratio (0.66 against 0.84) \\
Whitened PCA (fixed) & 5,120 inputs per strategy are many relative to about 26,000 strategies per year & $4 \times 16$ components on a common scale & 8 components lower the holdout Sharpe ratio ($-0.17$); 64 components lower the development Sharpe ratio ($-0.20$) \\
Bi-LSTM and attention pooling & The frozen representation is task-agnostic; segments differ in relevance & 256-dimensional context vector & Not removed separately; also used by the baseline model \\
ISAB (64 inducing points) & Selection is relative to the other strategies of the year; full attention costs $O(N_Y^2)$ & Context vector placed in the cross-section & Not distinguishable from run-to-run variation \\
Head with t-stat, mean return and the label & The t-stat already carries much of the information & Forecast of the part not explained by the t-stat & Label: 1.37 to 1.84 (development) and 0.69 to 0.81 (holdout) \\
Huber loss and year weights & Heavy-tailed returns and uneven years & Training objective & Not distinguishable from run-to-run variation \\
\botrule
\end{tabular}
\end{table}
```
💬 **Vì sao:** đây là phần thầy nhấn mạnh nhất: vì sao dùng từng khối và các khối nối với nhau thế nào. Bảng ghi **trung thực** cả những khối mà ablation chưa phân biệt được.
📖 **Nghĩa:** Mỗi khối giải quyết khó khăn mà khối trước để lại:
- FinCast mô tả chuỗi rất kỹ, nhưng đầu ra quá nhiều chiều và không gắn với nhiệm vụ.
- PCA nén xuống cỡ vừa với khoảng 26.000 chiến lược mỗi năm.
- Các patch vẫn giữ thứ tự thời gian, nên Bi-LSTM đọc được như một chuỗi; attention pooling gộp chúng bằng cách học đoạn nào quan trọng.
- Chỉ khi mỗi chiến lược đã thành một vector gọn, ISAB mới so sánh các chiến lược, nên bước này rẻ.
- Đầu dự đoán ghép vector với t-stat, và nhãn chỉ bắt mạng học phần t-stat chưa giải thích, nên hai nguồn thông tin bổ sung cho nhau.
- Huber và trọng số năm xử lý đuôi dày và các năm không đều nhau.

### Sửa 35 · 🔁 THAY · cuối 3.2.2 FinCast · dòng 405
🔴 **GỐC:**
```latex
so no parameter of the encoder is fitted to our sample.
```
🟢 **MỚI:**
```latex
so no parameter of the encoder is fitted to our sample. The resulting representation is rich but has 5,120 numbers per strategy and is not tailored to our task, which the next two blocks are designed to address: PCA reduces its size, and the trainable sequence block adapts it to the forecasting objective.
```
💬 **Vì sao:** câu nối sang khối sau, nói vì sao cần PCA và Bi-LSTM.

### Sửa 36 · 🔁 THAY · cuối 3.2.3 PCA · dòng 413
🔴 **GỐC:**
```latex
intended to reduce the complexity of the model and the risk of overfitting.
```
🟢 **MỚI:**
```latex
intended to reduce the complexity of the model and the risk of overfitting. Because the compression is applied to each patch separately, the temporal order of the four patches is preserved, so the next block can still read them as a sequence. The ablation in Section \ref{subsec4_3} supports this choice: 8 components lower the holdout Sharpe ratio, and 64 components lower the development Sharpe ratio.
```
💬 **Vì sao:** giải thích vì sao nén riêng từng patch (để giữ thứ tự thời gian cho Bi-LSTM), và dẫn bằng chứng từ ablation.

### Sửa 37 · 🔁 THAY · sau công thức attention pooling (3.2.4) · dòng 416
🔴 **GỐC:**
```latex
and pool them into a 256-dimensional context vector $c_i = \sum_{j=1}^{4} a_{i,j}\, \tilde{h}_{i,j}$.
```
🟢 **MỚI:**
```latex
and pool them into a 256-dimensional context vector $c_i = \sum_{j=1}^{4} a_{i,j}\, \tilde{h}_{i,j}$. A recurrent reader suits this input because it is short and ordered (four patches, from the oldest to the most recent), and attention pooling is the step that learns, for the forecasting task, which segments of the history matter; this is precisely what the frozen encoder does not provide.
```
💬 **Vì sao:** giải thích vì sao chọn Bi-LSTM (đầu vào ngắn, có thứ tự) và vì sao cần attention pooling.

### Sửa 38 · 🔁 THAY · cuối phần ISAB · dòng 424
🔴 **GỐC:**
```latex
The output $c'_i$ keeps the dimension 256.
```
🟢 **MỚI:**
```latex
The output $c'_i$ keeps the dimension 256. Placing ISAB after the per-strategy summary, rather than on the raw patches, means that the cross-sectional step operates on one compact vector per strategy, which keeps its cost linear in the number of strategies and lets it compare strategies on the features that the previous blocks have already selected.
```
💬 **Vì sao:** giải thích vì sao ISAB đứng sau bước tóm tắt mà không đứng trước.

### Sửa 39 · 🔁 THAY · cuối phần đầu dự đoán · dòng 432
🔴 **GỐC:**
```latex
and the final forecast follows from $\hat{y}_i = \hat{\sigma}_i(\hat{e}_i + b\, z_i + L_Y)$.
```
🟢 **MỚI:**
```latex
and the final forecast follows from $\hat{y}_i = \hat{\sigma}_i(\hat{e}_i + b\, z_i + L_Y)$. This construction lets the learned representation and the classical diagnostics complement each other: the t-stat enters both as an input and through the baseline $b\, z_i$, so the network is designed to model the residual information in the return history rather than to relearn what the t-stat already captures.
```
💬 **Vì sao:** giải thích vì sao ghép t-stat với nhãn lại khớp nhau: mạng chỉ học phần còn lại, không học lại điều t-stat đã biết.

---

## Mục 4 — Thử nghiệm

### Sửa 40 · 🔁 THAY · đầu 4.3 Ablation · dòng 493
🔴 **GỐC:**
```latex
We replace one component of the final configuration at a time and train each configuration three times, because retraining is not deterministic.
```
🟢 **MỚI:**
```latex
Because the main contribution of this study is the architecture, the ablation follows its chain of blocks, from the input representation to the training objective. We replace one component of the final configuration at a time and train each configuration three times, because retraining is not deterministic.
```
💬 **Vì sao:** thầy muốn phần thử nghiệm nhấn mạnh kiến trúc, vì kiến trúc là đóng góp chính.

### Sửa 41 · ➕ THÊM · đoạn tổng hợp cuối 4.3 · sau dòng 522
🔴 **GỐC** (câu cuối 4.3, **giữ nguyên**, click ngay sau dấu chấm):
```latex
since the summaries of its inducing points differ by only 2--3\% between years.
```
🟢 **MỚI** (Enter 2 lần rồi dán):
```latex
\textit{Taken together.} The ablation supports the design at the two ends of the chain. At the input end, the pretrained encoder and a compression size matched to the amount of data improve the holdout performance; at the output end, the label that compares strategies within a year gives the largest improvement in our experiments. The components in between, ISAB, the robust loss and the year weights, cannot be separated from run-to-run variation when removed one at a time, so their contribution, if any, may only appear jointly. The comparison with the baseline model in Section \ref{subsec5_1}, which shares the FinCast encoder and attention pooling but lacks the compression, the cross-strategy block, the sign orientation and the label, measures the joint effect of these design choices.
```
💬 **Vì sao:** đoạn này đọc lại kết quả ablation theo chuỗi kiến trúc, nói thật chỗ nào chưa phân biệt được, và chỉ sang so sánh với baseline.
📖 **Nghĩa:** Ablation ủng hộ thiết kế ở hai đầu của chuỗi. Ở đầu vào, FinCast pretrained và cỡ nén vừa phải làm tăng kết quả holdout. Ở đầu ra, nhãn tính theo năm cho cải thiện lớn nhất. Các khối ở giữa (ISAB, Huber, trọng số năm) chưa tách được khỏi dao động giữa các lần chạy. So với baseline (cùng FinCast và attention nhưng thiếu các khối kia), ta đo được hiệu ứng chung của cả thiết kế.

---

## Mục 5 — Kết quả

### Sửa 42 · 🔁 THAY · 5.1.1 · dòng 664
🔴 **GỐC:**
```latex
so the improvement of the proposed model comes from the design of Section \ref{subsec3} and not from the FinCast representations alone
```
🟢 **MỚI:**
```latex
so the improvement of the proposed model comes from the design of Section \ref{subsec3} and not from the FinCast representations alone: the baseline model uses the same encoder and attention pooling, but it does not compress the representation, does not let strategies interact, does not orient the signs and is trained on raw mean returns rather than on the label of Equation \eqref{eq:label}
```
💬 **Vì sao:** nói rõ baseline thiếu những khối nào, để người đọc thấy phần hơn đến từ kiến trúc.

### Sửa 43 · 🔁 THAY · 5.1.3 · dòng 709
🔴 **GỐC:**
```latex
This suggests that the return history carries information about future performance that the summary statistics used by EB mining do not.
```
🟢 **MỚI:**
```latex
This suggests that the return history carries information about future performance that the summary statistics used by EB mining do not. These are the strategies on which the ranking of the proposed architecture, which uses the return history in addition to the summary statistics, departs from a ranking by summary statistics alone, and their realized returns support these departures.
```
💬 **Vì sao:** nối kết quả "chỗ hai phương pháp chọn khác nhau" với kiến trúc (mô hình dùng thêm lịch sử lợi suất).

---

## Mục 6 — Conclusion (không thuộc nhận xét của thầy, nhưng đang **viết sai**)

### Sửa 44 · 🔁 THAY · dòng 909
🔴 **GỐC:**
```latex
while the Induced Set Attention Block (ISAB) aggregates relevant information into a compact context vector.
```
🟢 **MỚI:**
```latex
while attention pooling aggregates the temporal segments into a compact context vector and the Induced Set Attention Block (ISAB) lets the strategies of the same year interact.
```
💬 **Vì sao:** gộp thành vector là việc của attention pooling. ISAB làm việc khác: cho các chiến lược tương tác với nhau.

### Sửa 45 · 🔁 THAY · dòng 909
🔴 **GỐC:**
```latex
leveraging their complementary strengths for stock return prediction.
```
🟢 **MỚI:**
```latex
leveraging their complementary strengths to forecast the returns of long-short strategies.
```
💬 **Vì sao:** bài dự báo lợi suất của chiến lược, không phải của cổ phiếu.

### Sửa 46 · 🔁 THAY · dòng 911
🔴 **GỐC:**
```latex
It outperforms benchmark methods for the top 1\% and top 5\% of strategies across both periods and the full sample, as well as for the top 10\% during the holdout period, demonstrating its potential for out-of-sample return prediction.
```
🟢 **MỚI:**
```latex
Its top 1\% portfolio has a higher Sharpe ratio than that of empirical Bayes mining in the development, holdout and full periods, with statistically significant differences, and it has the highest holdout Sharpe ratio of the compared methods at all three portfolio sizes.
```
💬 **Vì sao:** giống sửa 2: top 5% không thắng có ý nghĩa thống kê.

### Sửa 47 · 🔁 THAY · dòng 913
🔴 **GỐC:**
```latex
by capturing nonlinear relationships among market factors and adapting to changing market dynamics.
```
🟢 **MỚI:**
```latex
by reading the return histories of strategies and being re-estimated every year as market conditions change.
```
💬 **Vì sao:** bài không mô hình hoá "market factors".

### Sửa 48 · 🔁 THAY · dòng 918
🔴 **GỐC:**
```latex
\section{Practical implications}
```
🟢 **MỚI:**
```latex
\subsection{Key findings and contributions}
```
💬 **Vì sao:** hiện có 3 `\section` liền nhau ở cuối bài. Mục này chứa kết quả chính chứ không phải ý nghĩa thực tiễn, nên chuyển nó thành mục con của Conclusion.

### Sửa 49 · 🔁 THAY · dòng 925
🔴 **GỐC:**
```latex
\section{Limitations and directions for future research}
```
🟢 **MỚI:**
```latex
\subsection{Limitations and future research}
```
💬 **Vì sao:** giống sửa 48.

---

# Kiểm tra sau khi sửa xong

1. Recompile. Trong PDF không còn chỗ nào hiện "?" (trích dẫn hoặc số bảng).
2. Bảng "Design rationale" ở mục 3 nằm gọn trong trang. Các bảng sau tự lùi số, LaTeX làm việc này.
3. Ctrl+F lần lượt các chữ: `novel`, `fail to`, `interpretab`, `stock return prediction`, `chen2024deep`. Không được còn chữ nào ở các chỗ đã sửa.
4. Báo mình, mình sẽ tải bản Overleaf về và đối chiếu từng chỗ.

**Bạn tự kiểm (mình không đọc được toàn văn vì bài thu phí):** ở mục 2.4, các chi tiết về `guo2026hybrid` ("three Chinese stocks", "outperforms CNN, LSTM, TCN, Informer, PatchTST") và về `ma2023attention` ("72 firm characteristics and 8 macroeconomic variables", "SHAP"). Nếu bạn đã đọc toàn văn và đúng như vậy thì giữ nguyên.
