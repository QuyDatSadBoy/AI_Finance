# Mục 6 Conclusion — nội dung để paste vào Overleaf

Mục 6 **tóm lại, không lặp mục 5**: chỉ 3 con số chính, không bảng, không số liệu chi tiết. Giới hạn đã liệt kê đủ ở 5.4.3, nên 6.3 chỉ gói thành một câu rồi chuyển sang hướng tiếp theo. Đây là chỗ duy nhất tổng kết mô hình "cao nhất trong bốn phương pháp được so sánh" (5.4 không lặp lại).

## Cách dùng

1. Trong `sn-article.tex`, xoá `\section{Conclusion}\label{sec6}` cùng đoạn mẫu bên dưới nó (đến ngay trước `\section{Algorithms, Program codes and Listings}`), rồi paste khối LaTeX bên dưới.
2. Không có BibTeX mới.
3. **Introduction hiện cũ:** đoạn đóng góp và đoạn "The remainder of this paper" trong `\section{Introduction}` còn mô tả cấu trúc và kết quả cũ (có `[PLACEHOLDER]`, "state-of-the-art", "diễn giải attention weights", "Section 5 discusses…"). Mục 6 này **không** tuyên bố những điều đó. Khi viết lại Introduction, hãy sửa cho khớp (mình làm được nếu bạn yêu cầu).

## Cấu trúc của mục 6

| Mục | Tên | Nội dung | Không lặp với |
|---|---|---|---|
| 6.1 | Research summary | bài toán và mô hình đề xuất, một đoạn | 3.1, 3.2 (chỉ nhắc ý) |
| 6.2 | Key findings | kết quả chính (3 số), đóng góp, ý nghĩa | mục 5 (không bảng, không chi tiết) |
| 6.3 | Limitations and future work | giới hạn một câu, bốn hướng tiếp theo | 5.4.3 |

---

## Bản tiếng Anh (paste vào LaTeX)

```latex
\section{Conclusion}\label{sec6}

\subsection{Research summary}
This study investigated whether the return history of a long-short strategy, and not only its summary statistics, helps to select strategies that will earn returns out of sample. For each year from 1983 to 2020, we forecast the mean return of about 26,000 equal-weighted accounting strategies using only earlier data, the quantity that empirical Bayes (EB) predicts. The proposed model employs a pretrained FinCast encoder to represent the 128-month return sequence, compresses it with PCA, summarizes it with a Bi-LSTM and attention pooling, lets the strategies of the same year interact through an ISAB, and combines the result with the t-stat and mean return. It is trained on a label that removes the year mean and the part explained by the absolute t-stat, with a weighted Huber loss, and it is re-estimated every year.

\subsection{Key findings}
Evaluated with the portfolio tests of \cite{chen2025highthroughputassetpricing}, the proposed model selects a top 1\% portfolio with a higher Sharpe ratio than EB mining in the development period (1.95 against 1.55), the holdout period (1.11 against 0.82) and the full period (1.62 against 1.27), and each difference is statistically significant. Among the four methods compared on the same strategies and years, it has the highest top 1\% Sharpe ratio in the holdout and full periods (tied with naive mining in the development period) and the highest holdout Sharpe ratio at all three portfolio sizes. Its losses in bad years are smaller than those of EB mining. The model also recognizes ticker strategies, which carry no predictability, as noise at the top of its ranking (0.19\% of its top 1\% portfolio, against 43.5\% of the pool), although it shrinks them less than EB mining beyond the top 1\%. The ablation indicates that the pretrained encoder, the PCA compression and the label construction contribute to this result.

This study makes three contributions. First, we propose a label that expresses future performance in Sharpe ratio units and removes the year mean and the part explained by the absolute t-stat, so that the model learns what the t-stat does not already explain. Second, we integrate pretrained temporal representations, cross-strategy interaction and a robust loss into a forecasting model that is re-estimated every year without look-ahead bias. Third, we evaluate the model against EB mining and naive mining with the portfolio tests of \cite{chen2025highthroughputassetpricing}, and report which components contribute and which do not.

The results have two implications. The return history carries information beyond the t-stat at the top of the ranking, and the advantage remains after 2004, when predictability is weaker. Ranking strategies and estimating their expected returns are different tasks, since EB mining remains more accurate in the magnitude of its forecasts, so the two approaches may complement each other rather than substitute for each other.

\subsection{Limitations and future work}
The evidence is limited to one family of strategies in one market, before transaction costs, and from a model selected on the development period (Section \ref{subsec5_4}). Future work may explore four directions. First, evaluating all six families of strategies and other markets would test whether the advantage generalizes. Second, adding transaction costs and value-weighted portfolios would show whether it survives trading frictions. Third, learning a shrinkage of the model's own forecasts may improve calibration without using EB as an input. Fourth, repeating the ablation with more runs, and examining the learned attention weights, may clarify the roles of ISAB and the other components that we could not distinguish from run-to-run variation.
```

---

## Bản tiếng Việt (dịch sát bản tiếng Anh, để bạn đọc, không paste)

### 6 Kết luận

#### 6.1 Research summary (Tóm tắt nghiên cứu)

Nghiên cứu này xem xét liệu lịch sử lợi suất của một chiến lược long–short, chứ không chỉ các thống kê tóm tắt, có giúp chọn các chiến lược sẽ sinh lời ngoài mẫu hay không. Với mỗi năm từ 1983 đến 2020, ta dự báo lợi suất trung bình của khoảng 26.000 chiến lược kế toán chia đều chỉ bằng dữ liệu trước đó, đúng đại lượng mà empirical Bayes (EB) dự báo. Mô hình đề xuất dùng bộ mã hoá FinCast pretrained để biểu diễn chuỗi lợi suất 128 tháng, nén bằng PCA, tóm tắt bằng Bi-LSTM và attention pooling, cho các chiến lược cùng năm tương tác qua ISAB, rồi kết hợp với t-stat và lãi trung bình. Mô hình học trên nhãn đã loại trung bình năm và phần |t-stat| đã giải thích, với Huber loss có trọng số, và học lại mỗi năm.

#### 6.2 Key findings (Phát hiện chính)

Đánh giá bằng các phép thử danh mục của Chen & Dim, mô hình đề xuất chọn danh mục top 1% có Sharpe cao hơn EB ở dev (1,95 so với 1,55), holdout (1,11 so với 0,82) và cả kỳ (1,62 so với 1,27), mỗi chênh lệch đều có ý nghĩa thống kê. Trong bốn phương pháp so sánh trên cùng chiến lược và năm, mô hình có Sharpe top 1% cao nhất ở holdout và cả kỳ (ngang naive mining ở dev) và Sharpe holdout cao nhất ở cả ba cỡ danh mục. Thua lỗ ở các năm xấu nhỏ hơn EB. Mô hình cũng nhận ra ticker, vốn không có khả năng dự báo, là nhiễu ở đầu bảng xếp hạng (0,19% top 1%, so với 43,5% của tập gộp), dù co ticker yếu hơn EB ngoài top 1%. Ablation cho thấy bộ mã hoá pretrained, nén PCA và cách xây nhãn đóng góp vào kết quả này.

Nghiên cứu có ba đóng góp. Một, ta đề xuất nhãn biểu diễn hiệu quả tương lai theo đơn vị Sharpe, loại trung bình năm và phần |t-stat| đã giải thích, để mô hình học điều mà t-stat chưa giải thích. Hai, ta tích hợp biểu diễn thời gian pretrained, tương tác giữa các chiến lược và loss bền vững thành một mô hình dự báo học lại mỗi năm mà không có look-ahead bias. Ba, ta đánh giá mô hình so với EB và naive mining bằng các phép thử danh mục của Chen & Dim, và báo thành phần nào đóng góp, thành phần nào không.

Kết quả có hai hàm ý. Lịch sử lợi suất chứa thông tin ngoài t-stat ở đầu bảng xếp hạng, và lợi thế vẫn còn sau 2004, khi khả năng dự báo yếu hơn. Xếp hạng chiến lược và ước lượng lãi kỳ vọng là hai việc khác nhau, vì EB vẫn chính xác hơn về độ lớn dự báo, nên hai cách tiếp cận có thể bổ sung cho nhau thay vì thay thế nhau.

#### 6.3 Limitations and future work (Giới hạn và hướng tiếp theo)

Bằng chứng giới hạn ở một họ chiến lược trong một thị trường, trước chi phí giao dịch, và từ một mô hình chọn trên giai đoạn dev (mục 5.4). Các hướng tiếp theo: một, đánh giá cả 6 họ chiến lược và thị trường khác để kiểm tra lợi thế có tổng quát hoá không. Hai, thêm chi phí giao dịch và danh mục theo vốn hoá để xem lợi thế có qua được ma sát giao dịch không. Ba, học một phép co (shrinkage) cho chính dự báo của mô hình có thể cải thiện độ đúng mức mà không dùng EB làm đầu vào. Bốn, lặp lại ablation với nhiều lần chạy hơn và xem các trọng số attention đã học có thể làm rõ vai trò của ISAB và các thành phần khác mà ta chưa phân biệt được với dao động giữa các lần chạy.

---

## Cần bạn biết trước khi paste

1. **Số duy nhất lặp lại từ mục 5** là Sharpe top 1% (dev, holdout, cả kỳ) của mô hình và EB. Đó là kết quả chính nên nhắc lại là đúng chỗ.
2. **"Đóng góp"** viết thận trọng: chỉ "propose" cho nhãn (thứ mới); biểu diễn pretrained, tương tác, loss bền vững là "integrate". Không viết "SOTA", không viết ISAB hay attention là đóng góp có chứng minh (ablation không phân biệt được).
3. **Hướng tiếp theo thứ ba** (co dự báo) ghi rõ "không dùng EB làm đầu vào", đúng nguyên tắc B của dự án.
4. Phần Introduction hiện cũ (có `[PLACEHOLDER]`, "state-of-the-art", diễn giải attention weights) **không khớp** mục 3–6; cần viết lại khi đến lượt.
5. Chưa biên dịch LaTeX.
