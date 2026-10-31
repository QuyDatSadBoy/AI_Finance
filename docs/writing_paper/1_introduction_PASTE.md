# Mục 1 Introduction — đề xuất cải tiến và cách paste từng bước

Bạn paste **từng cải tiến một**, không thay cả mục. Mỗi cải tiến ghi rõ: đoạn nào (nhận biết bằng vài chữ đầu), làm gì (giữ, sửa một câu, thay, xoá, thêm), nội dung tiếng Anh để paste và bản tiếng Việt để đọc. Sau mỗi bước bạn có thể recompile.

## Nhận xét về Introduction hiện tại

Căn cứ: file hướng dẫn *How to write research paper 3* (bố cục 4 bước của Introduction), mục Introduction của Chen & Dim, `WRITING_GUIDE.md`, và mục 2 đến 6 đã viết.

**Bố cục chuẩn trong file hướng dẫn:** (1) bối cảnh và câu hỏi nghiên cứu, (2) điểm mạnh và yếu của các phương pháp hiện có, (3) kết quả của phương pháp so với đối chứng, (4) đóng góp chính. File còn nhấn: Introduction nêu câu hỏi, mục tiêu, giả thuyết; Literature Review đánh giá chi tiết. Phản biện hay nhận xét "trùng giữa Introduction và Related Work".

**Chen & Dim viết thế nào:** mở bằng định nghĩa và vấn đề (data mining bias), nêu giải pháp, nêu kết quả chính kèm con số ngay trong Introduction (5,7% mỗi năm so với 5,9%), nêu cả điểm lưu ý (lợi suất tập trung trước 2004), rồi mới đến insight và cách dùng.

| Vấn đề của bản hiện tại | Hệ quả |
|---|---|
| Đoạn 3, 4, 5 là một mục tổng quan tài liệu thu nhỏ (FinCast, AIPM, t-stat) | **Trùng với mục 2.4 và 2.5**, đúng lỗi phản biện hay nêu |
| Mô tả mô hình cũ (FinCast, Bi-LSTM, attention pooling) | Không có PCA, ISAB, nhãn Sharpe, Huber; sai với mục 3 |
| Hứa "interpretable attention weights" và "state-of-the-art ML models" | Bài không làm; vi phạm guide (không SOTA, không hứa điều chưa có) |
| Còn `[PLACEHOLDER]` ở phần kết quả | Chưa có bước 3 của bố cục: kết quả so với đối chứng |
| Không nhắc EB, khai thác dữ liệu, look-ahead bias | Thiếu đối thủ trực tiếp và động cơ chính |
| Từ hype: "marks a major methodological milestone", "first", "novel", "directly mitigating" | Vi phạm guide |
| `\cite{chen2024deep}` gắn với "high-throughput testing" | **Sai nguồn**: `chen2024deep` là *Deep learning in asset pricing*; nguồn đúng là `chen2025highthroughputassetpricing` |
| Thiếu dấu chấm sau `\cite{feng2020taming, giglio2022factor}` | Lỗi nhỏ |
| Lộ trình bài sai (Section 3 "data", Section 5 "interpretation of attention weights") | Không khớp mục 3 đến 6 |

## Bố cục đề xuất (giữ phần tốt, thay phần lệch)

| Đoạn | Việc làm | Vai trò theo file hướng dẫn |
|---|---|---|
| Đ1 Bối cảnh định giá tài sản (cũ) | **Giữ nguyên** | Bối cảnh |
| Đ2 Rủi ro dự báo giả (cũ) | **Giữ, sửa 2 chỗ** (cải tiến 1, 2) | Bối cảnh và vấn đề |
| Đ3 Điểm mạnh và yếu: EB và mô hình nền (mới, thay cũ Đ3) | **Thay** (cải tiến 3) | Bước 2 |
| Đ4 Câu hỏi và điều kiện cần (mới, thay cũ Đ4 và Đ5) | **Thay** hai đoạn bằng một (cải tiến 4) | Bước 2 và mục tiêu |
| Đ5 Cách tiếp cận (thay cũ Đ6) | **Thay** (cải tiến 5) | Phương pháp |
| Đ6 Câu hỏi nghiên cứu (thay cũ Đ7) | **Thay** (cải tiến 6) | Câu hỏi |
| Đ7 Kết quả chính (mới) | **Thêm** (cải tiến 7) | Bước 3 |
| Đ8 Đóng góp (thay cũ Đ8) | **Thay** (cải tiến 8) | Bước 4 |
| Đ9 Lộ trình (thay cũ Đ9) | **Thay** (cải tiến 9) | Lộ trình |

Độ dài sau khi sửa khoảng 900 từ (bản cũ khoảng 1.150 từ). Không có BibTeX mới: mọi khoá (`efron1973stein`, `chen2022open`, `lee2019set`, …) đã có.

---

## Cải tiến 1 — Đoạn 2: thêm dấu chấm bị thiếu và thay nguồn trích dẫn sai

**Tìm** (trong đoạn bắt đầu "This flexibility, however, comes with notable empirical trade-offs"):

```latex
\cite{feng2020taming, giglio2022factor}  Indeed, recent empirical evidence
```

**Thay bằng:**

```latex
\cite{feng2020taming, giglio2022factor}. Indeed, recent empirical evidence
```

**Tìm tiếp trong cùng đoạn:**

```latex
rather than genuine structural signals \cite{avramov2023machine, chu2020causal}.
```

**Thay bằng** (nguồn đúng cho nhận định "tín hiệu suy giảm ngoài mẫu" là McLean & Pontiff; `avramov2023machine` là về ML và ràng buộc kinh tế):

```latex
rather than genuine structural signals \cite{mclean2016does, chu2020causal}.
```

*Tiếng Việt:* sửa dấu chấm bị thiếu; đổi nguồn trích dẫn của nhận định "nhiều anomaly suy giảm hoặc biến mất ngoài mẫu" từ Avramov sang McLean & Pontiff.

---

## Cải tiến 2 — Đoạn 2: thay câu cuối để dẫn tới khai thác dữ liệu chặt chẽ (EB)

**Xoá câu cuối của đoạn 2:**

```latex
Disentangling robust, persistent signals from unreplicable noise has thus emerged as a central challenge in empirical finance, prompting recent efforts to formalize the systematic, large-scale evaluation of predictor candidate space as a high-throughput testing problem \cite{chen2024deep}.\\
```

**Thay bằng** (giữ câu mở, bỏ nguồn sai, thêm hướng EB):

```latex
Disentangling robust, persistent signals from unreplicable noise has thus emerged as a central challenge in empirical finance. One response is to mine data rigorously rather than less. \citet{chen2025highthroughputassetpricing} apply empirical Bayes \cite{efron1973stein} to about 136,000 long-short strategies built by systematically searching accounting ratios, past returns and tickers, and show that the top 1\% of strategies, selected using only information available in real time, earns out-of-sample returns comparable to those of published strategies. Because the selection contains no look-ahead bias, this approach provides a transparent benchmark for any method that aims to forecast which strategies will perform.\\
```

*Tiếng Việt:* Tách tín hiệu bền khỏi nhiễu không tái lập là thách thức trung tâm của tài chính thực nghiệm. Một cách đáp lại là khai thác dữ liệu chặt chẽ chứ không phải ít đi. Chen & Dim áp dụng empirical Bayes cho khoảng 136.000 chiến lược long–short tạo bằng tìm kiếm có hệ thống trên tỷ số kế toán, lợi suất quá khứ và mã cổ phiếu, và cho thấy top 1% chiến lược, chọn chỉ bằng thông tin có sẵn theo thời gian thực, cho lợi suất ngoài mẫu tương đương các chiến lược đã công bố. Vì việc chọn không có look-ahead bias, cách này là chuẩn so sánh minh bạch cho mọi phương pháp muốn dự báo chiến lược nào sẽ hiệu quả.

---

## Cải tiến 3 — Đoạn 3 (FinCast): thay bằng đoạn trung lập, nêu điểm mạnh và yếu của EB và mô hình nền

**Xoá cả đoạn bắt đầu** "A parallel and more recent strand of literature develops large-scale, pretrained time-series foundation models…" (đoạn 3 cũ, kết bằng "…evaluate signal quality.\\"). **Thay bằng:**

```latex
Empirical Bayes ranks strategies with summary statistics of their return histories, namely the t-stat, and therefore does not explicitly model how returns are distributed over time, although two strategies with the same t-stat can differ in this respect. At the same time, large-scale pretrained time-series foundation models have become available for financial data. FinCast \cite{zhu2025fincast}, a decoder-only Transformer pretrained on more than 20 billion time points, represents a return sequence without task-specific fine-tuning. Such representations are task-agnostic, however, and it is not obvious whether they help to forecast which strategies will perform, nor how to combine them with the summary statistics that researchers conventionally use to judge a signal \cite{harvey2016and}.\\
```

*Tiếng Việt:* Empirical Bayes xếp hạng chiến lược bằng các thống kê tóm tắt của lịch sử lợi suất, cụ thể là t-stat, nên không mô hình hoá tường minh việc lợi suất phân bố theo thời gian, dù hai chiến lược cùng t-stat có thể khác nhau ở điểm này. Đồng thời, các mô hình nền chuỗi thời gian tiền huấn luyện quy mô lớn đã có cho dữ liệu tài chính. FinCast, một Transformer decoder-only tiền huấn luyện trên hơn 20 tỷ điểm thời gian, biểu diễn một chuỗi lợi suất mà không cần tinh chỉnh riêng. Tuy vậy các biểu diễn này không gắn với nhiệm vụ, và chưa rõ chúng có giúp dự báo chiến lược nào sẽ hiệu quả không, cũng như cách kết hợp chúng với các thống kê tóm tắt mà nhà nghiên cứu thường dùng để đánh giá tín hiệu.

*Lý do:* bản cũ viết "first foundation model", "major methodological milestone", "directly mitigating" (hype, và trùng mục 2.4); chi tiết FinCast đã ở 2.4 và 3.2.

---

## Cải tiến 4 — Đoạn 4 và 5 cũ (attention, AIPM, "two limitations", "No prior study"): thay cả hai bằng một đoạn

**Xoá hai đoạn liên tiếp:** đoạn bắt đầu "Attention mechanisms offer a natural way to address this shortcoming…" (kết bằng "…discard the sequential and regime-dependent information that path contains.\\") và đoạn bắt đầu "Two complementary limitations therefore remain unresolved." (kết bằng "…relative to either source of information alone.\\"). Nội dung đã ở mục 2.5 và 2.6. **Thay bằng:**

```latex
We therefore ask whether the return history of a strategy carries information about its future performance that its t-stat does not. Answering this question requires three ingredients that have not, to our knowledge, been combined: a representation of the full return sequence, a way to compare each strategy with the other strategies of the same year, and an evaluation against empirical Bayes and naive mining that avoids look-ahead bias.\\
```

*Tiếng Việt:* Vì vậy ta hỏi liệu lịch sử lợi suất của một chiến lược có chứa thông tin về hiệu quả tương lai mà t-stat của nó không có hay không. Trả lời câu hỏi này cần ba thành phần mà theo hiểu biết của chúng tôi chưa được kết hợp: biểu diễn của toàn bộ chuỗi lợi suất, cách so sánh từng chiến lược với các chiến lược khác cùng năm, và một đánh giá so với empirical Bayes và khai thác đơn giản mà không có look-ahead bias.

*Lý do:* file hướng dẫn nêu lỗi "trùng giữa Introduction và Related Work"; bỏ "No prior study…" (quá tuyệt đối), bỏ "interpretability".

---

## Cải tiến 5 — Đoạn cách tiếp cận (cũ "To address this gap, we propose an attention-augmented FinCast framework…"): thay bằng mô tả mô hình hiện tại

**Xoá cả đoạn** bắt đầu "To address this gap, we propose an attention-augmented FinCast framework…" (kết bằng "…drive that forecast.\\"). **Thay bằng:**

```latex
We propose a forecasting model that is re-estimated every year using only earlier data. For each strategy, a frozen pretrained FinCast encoder represents the 128 months of returns before the forecasting year, and principal component analysis compresses this representation. A bidirectional LSTM and attention pooling summarize it, an induced set attention block \cite{lee2019set} lets the strategies of the same year interact, and the result is combined with the t-stat and mean return before a fully connected head forecasts the mean return of the year. We train the model with a label that expresses future performance in Sharpe ratio units relative to the year mean and to the part explained by the absolute t-stat, using a weighted Huber loss. We study the equal-weighted accounting strategies of \citet{chen2025highthroughputassetpricing} from 1983 to 2020, and compare the model with empirical Bayes mining and naive mining using the portfolio tests of that study.\\
```

*Tiếng Việt:* Ta đề xuất một mô hình dự báo được học lại mỗi năm chỉ bằng dữ liệu trước đó. Với mỗi chiến lược, bộ mã hoá FinCast tiền huấn luyện đóng băng biểu diễn 128 tháng lợi suất trước năm dự báo, và PCA nén biểu diễn này. Bi-LSTM và attention pooling tóm tắt nó, khối induced set attention cho các chiến lược cùng năm tương tác, và kết quả được ghép với t-stat và lãi trung bình trước khi một đầu fully connected dự báo lãi trung bình của năm. Mô hình học với nhãn biểu diễn hiệu quả tương lai theo đơn vị Sharpe, so với trung bình năm và phần |t-stat| đã giải thích, dùng Huber loss có trọng số. Ta nghiên cứu các chiến lược kế toán chia đều của Chen & Dim từ 1983 đến 2020, và so mô hình với empirical Bayes và khai thác đơn giản bằng các phép thử danh mục của họ.

---

## Cải tiến 6 — Đoạn câu hỏi nghiên cứu (cũ "Motivated by this gap…"): thay

**Xoá đoạn** bắt đầu "Motivated by this gap, the present study addresses the following research question…" **Thay bằng:**

```latex
The research question of this study is whether the return history of a strategy, represented by a pretrained foundation model and combined with its t-stat, improves the out-of-sample selection of long-short strategies relative to empirical Bayes and naive mining.\\
```

*Tiếng Việt:* Câu hỏi nghiên cứu là liệu lịch sử lợi suất của một chiến lược, biểu diễn bằng mô hình nền tiền huấn luyện và kết hợp với t-stat, có cải thiện việc chọn chiến lược long–short ngoài mẫu so với empirical Bayes và khai thác đơn giản hay không.

*Lý do:* bản cũ hỏi về "attention pooling" và "economic interpretability"; bài không đo những điều đó.

---

## Cải tiến 7 — Thêm đoạn kết quả chính (bước 3 của bố cục; thay `[PLACEHOLDER]`)

**Thêm một đoạn mới ngay sau đoạn câu hỏi nghiên cứu** (trước đoạn "This study makes three main contributions"):

```latex
The model selects better strategies at the top of the ranking. Its top 1\% portfolio has a Sharpe ratio of 1.95 in the development period (1983--2004), 1.11 in the holdout period (2005--2020) and 1.62 over the full sample, compared with 1.55, 0.82 and 1.27 for empirical Bayes mining, and each difference is statistically significant. The strategies that only the model selects earn a Sharpe ratio of 1.63, against 0.85 for those that only empirical Bayes selects. The model almost never selects ticker strategies, which carry no predictability. Its advantage does not extend to the top 5\% and 10\% portfolios, where the methods are statistically indistinguishable, and empirical Bayes remains more accurate in the magnitude of its forecasts. All results use only information available in real time and are before transaction costs.\\
```

*Tiếng Việt:* Mô hình chọn được chiến lược tốt hơn ở đầu bảng xếp hạng. Danh mục top 1% của mô hình có Sharpe 1,95 ở giai đoạn phát triển (1983–2004), 1,11 ở holdout (2005–2020) và 1,62 cả mẫu, so với 1,55, 0,82 và 1,27 của empirical Bayes, và mỗi chênh lệch đều có ý nghĩa thống kê. Các chiến lược chỉ mô hình chọn đạt Sharpe 1,63, so với 0,85 của các chiến lược chỉ empirical Bayes chọn. Mô hình hầu như không chọn ticker, vốn không có khả năng dự báo. Lợi thế không mở rộng sang danh mục top 5% và 10%, nơi các phương pháp không phân biệt được về thống kê, và empirical Bayes vẫn chính xác hơn về độ lớn dự báo. Mọi kết quả chỉ dùng thông tin có sẵn theo thời gian thực và trước chi phí giao dịch.

*Lý do:* bố cục trong file hướng dẫn và Chen & Dim đều đưa kết quả chính kèm số vào Introduction. Số khớp mục 4 và 5; câu cuối nêu giới hạn như Chen & Dim ("with an important caveat").

---

## Cải tiến 8 — Đoạn đóng góp (cũ "This study makes three main contributions"): thay

**Xoá đoạn** bắt đầu "This study makes three main contributions." (có `[PLACEHOLDER]`, "novel", "state-of-the-art", "interpretable attention weights"). **Thay bằng:**

```latex
This study makes three contributions. First, we propose a label that expresses future performance in Sharpe ratio units and removes the year mean and the part explained by the absolute t-stat, so that the model learns what the t-stat does not already explain. Second, we integrate pretrained temporal representations, cross-strategy interaction and a robust loss into a forecasting model that is re-estimated every year without look-ahead bias. Third, we evaluate the model against empirical Bayes mining and naive mining with out-of-sample portfolio tests, and report through an ablation study which components contribute and which we cannot distinguish from run-to-run variation.\\
```

*Tiếng Việt:* Nghiên cứu có ba đóng góp. Một, ta đề xuất nhãn biểu diễn hiệu quả tương lai theo đơn vị Sharpe, loại trung bình năm và phần |t-stat| đã giải thích, để mô hình học điều mà t-stat chưa giải thích. Hai, ta tích hợp biểu diễn thời gian tiền huấn luyện, tương tác giữa các chiến lược và loss bền vững thành một mô hình dự báo học lại mỗi năm mà không có look-ahead bias. Ba, ta đánh giá mô hình so với empirical Bayes và khai thác đơn giản bằng các phép thử danh mục ngoài mẫu, và báo qua nghiên cứu ablation thành phần nào đóng góp, thành phần nào ta không phân biệt được với dao động giữa các lần chạy.

*Lý do:* khớp mục 6.2; "propose" chỉ dùng cho nhãn (thứ mới), "integrate" cho thành phần kế thừa (đúng guide).

---

## Cải tiến 9 — Đoạn lộ trình (cũ "The remainder of this paper is organized…"): thay

**Xoá đoạn** bắt đầu "The remainder of this paper is organized as follows." **Thay bằng:**

```latex
The remainder of this paper is organized as follows. Section \ref{sec2} reviews the literature on return predictability and data mining, the decay of signals, machine learning in asset pricing, financial time-series models and attention. Section \ref{sec3} formulates the problem and describes the proposed model. Section \ref{sec4} presents the data, the experimental design, the forecast accuracy and the ablation study. Section \ref{sec5} reports the portfolio results, the robustness to noise strategies, the temporal analysis and the limitations. Section \ref{sec6} concludes.
```

*Tiếng Việt:* Phần còn lại của bài được tổ chức như sau. Mục 2 điểm lại tài liệu về khả năng dự báo lợi suất và khai thác dữ liệu, sự suy giảm của tín hiệu, học máy trong định giá tài sản, mô hình chuỗi thời gian tài chính và attention. Mục 3 phát biểu bài toán và mô tả mô hình đề xuất. Mục 4 trình bày dữ liệu, thiết kế thí nghiệm, độ chính xác dự báo và nghiên cứu ablation. Mục 5 báo kết quả danh mục, độ bền trước chiến lược nhiễu, phân tích theo thời gian và giới hạn. Mục 6 kết luận.

*Lưu ý:* bản của bạn có hai mục `\section{Practical implications}` và `\section{Limitations and directions…}` tách riêng ở cuối; nếu bạn giữ cấu trúc đó thì câu về mục 5 và 6 cần điều chỉnh cho khớp.

---

## Thứ tự làm và kiểm tra

1. Làm cải tiến **1 và 2** (sửa đoạn 2), recompile.
2. Làm **3, 4, 5, 6** (thay các đoạn giữa), recompile.
3. Làm **7, 8, 9** (thêm kết quả, đóng góp, lộ trình), recompile.
4. Kiểm tra lại: tìm trong file các chữ `PLACEHOLDER`, `state-of-the-art`, `novel`, `interpretab`; không được còn chữ nào ở mục 1.
5. Mọi `\ref{secN}` ở cải tiến 9 dùng nhãn `sec2` đến `sec6` có sẵn trong file.

Sau khi bạn làm xong, mình tải lại file từ Overleaf để đối chiếu từng đoạn nếu bạn muốn.
