# Mục 2 Literature Review — sửa từng ý trên bản gốc của bạn

Bạn đã paste lại bản gốc của mục 2. Dưới đây là **13 sửa đổi nhỏ**, mỗi sửa đổi nói rõ: **tìm** đoạn nào (copy y nguyên từ file của bạn), **làm gì** (thay, chèn, xoá) và nội dung tiếng Anh để paste, kèm bản tiếng Việt để đọc. Mọi chỗ không nhắc đến thì **giữ nguyên**. Không đổi thứ tự các mục con. Mình đã kiểm tra mọi chuỗi "Tìm" xuất hiện đúng một lần trong file hiện tại trên Overleaf.

## Vì sao sửa (tóm tắt)

| Vấn đề trong bản gốc | Sửa ở |
|---|---|
| Không có lời dẫn nêu cấu trúc mục | 1 |
| Câu khẳng định "linear form là hạn chế chính" chưa có bằng chứng | 2 |
| Thiếu hướng khai thác dữ liệu / empirical Bayes, đối thủ trực tiếp của bài | 3 |
| Chưa nói rõ nhóm ML nhắm đến lợi suất tài sản, không phải chọn chiến lược | 4 |
| Thiếu mô hình nền đa dụng (Chronos, TimesFM) để đặt FinCast vào bối cảnh | 5 |
| Câu bị lặp ("disconnected from one another" hai lần) | 6 |
| Tiêu đề 2.4 và hai câu khẳng định "bài khác AIPM vì chỉ attention theo thời gian" **không còn đúng** (mô hình có ISAB, attention giữa các chiến lược) | 7, 8 |
| Câu hứa "interpretability goals" (bài không làm) | 8 |
| Thiếu Set Transformer (ISAB ở mục 3.2) | 9 |
| `\cite{Chen}` không có trong `.bib` | 10 |
| Câu "attention-weighted representations" (quá hẹp so với mô hình hiện tại) | 11 |
| Thiếu giải thích suy giảm sau 2004 (lý do chia dev / holdout) | 12 |
| 2.6 hứa kiểm tra "interpretability", ba mô hình đối chứng sai, "No prior study" | 13 |

**BibTeX:** sáu mục mới (`efron1973stein`, `chen2022open`, `chordia2014have`, `peng2005learning`, `ansari2024chronos`, `das2024decoder`) đã có trong `sn-bibliography.bib` trên Overleaf (mình đã kiểm tra). Nếu bạn paste lại `.bib` gốc thì thêm lại từ phần BibTeX cuối file này.

---

## Sửa 1 — Thêm lời dẫn ngay sau tiêu đề mục 2

**Tìm:**

```latex
\section{Literature Review}\label{sec2}
```

**Thay bằng** (giữ dòng cũ, thêm đoạn dẫn bên dưới):

```latex
\section{Literature Review}\label{sec2}

We review the literature from the broad question of return predictability to the specific modeling tools that we build on, and we end with the gap that this study addresses.
```

*Tiếng Việt:* Ta điểm lại tài liệu từ câu hỏi rộng về khả năng dự báo lợi suất đến các công cụ mô hình cụ thể mà ta dựa vào, và kết thúc ở khoảng trống mà nghiên cứu này giải quyết.

---

## Sửa 2 — Mục 2.1: bỏ câu khẳng định chưa có bằng chứng, nối thẳng sang câu hỏi

**Tìm** (trong đoạn "This proliferation of predictors…", đoạn thứ hai của mục 2.1), từ chữ `Taken together, this literature suggests` đến hết `These concerns also raise a question`:

```latex
Taken together, this literature suggests that the explanatory power of linear factor models is constrained less by the number of candidate predictors available than by the rigidity of the linear functional form used to combine them — a limitation that becomes more acute as the number of candidate factors continues to expand and as return-generating processes become more complex. These concerns also raise a question
```

**Thay bằng:**

```latex
Taken together, these concerns raise a question
```

*Tiếng Việt:* Bỏ nhận định "sức giải thích của mô hình nhân tố tuyến tính bị giới hạn bởi dạng hàm hơn là số biến" vì chưa có nguồn; câu nối trực tiếp sang "câu hỏi đi trước mô hình hoá từng nhân tố".

---

## Sửa 3 — Mục 2.1: thêm một đoạn về khai thác dữ liệu chặt chẽ và empirical Bayes

**Tìm** (cuối mục 2.1, kết thúc đoạn):

```latex
how reliably can its own future performance be forecast at all?\\
```

**Thay bằng** (giữ nguyên dòng trên, thêm một đoạn mới ngay sau nó, cách một dòng trống):

```latex
how reliably can its own future performance be forecast at all?\\

A recent response is to mine data rigorously rather than less. \citet{chen2025highthroughputassetpricing} apply empirical Bayes \cite{efron1973stein} to about 136,000 long-short strategies built by systematically searching accounting ratios, past returns and tickers, following the construction of \citet{yan2017fundamental}. They show that a portfolio of the top 1\% of strategies, selected with information available in real time, earns out-of-sample returns comparable to those of published strategies \cite{chen2022open}, and that naively ranking strategies by their in-sample Sharpe ratio performs similarly. Empirical Bayes forecasts the performance of each strategy from summary statistics of its return history, namely the t-stat, and therefore does not explicitly model how the returns are distributed over time.\\
```

*Tiếng Việt:* Một cách đáp lại gần đây là khai thác dữ liệu chặt chẽ chứ không phải ít đi. Chen & Dim áp dụng empirical Bayes (Efron & Morris) cho khoảng 136.000 chiến lược long–short tạo bằng tìm kiếm có hệ thống trên tỷ số kế toán, lợi suất quá khứ và mã cổ phiếu, theo cách xây dựng của Yan & Zheng. Họ cho thấy danh mục top 1%, chọn bằng thông tin có sẵn theo thời gian thực, cho lợi suất ngoài mẫu tương đương các chiến lược đã công bố (Chen & Zimmermann), và việc xếp hạng đơn giản theo Sharpe trong mẫu cũng cho kết quả tương tự. Empirical Bayes dự báo hiệu quả từng chiến lược từ các thống kê tóm tắt của lịch sử lợi suất, cụ thể là t-stat, nên không mô hình hoá tường minh việc lợi suất phân bố theo thời gian.

---

## Sửa 4 — Mục 2.2: thêm một câu cuối đoạn "A more recent line of work…"

**Tìm** (cuối đoạn thứ hai của mục 2.2):

```latex
rather than learning which periods of that history are most informative for the prediction task at hand.\\
```

**Thay bằng:**

```latex
rather than learning which periods of that history are most informative for the prediction task at hand. They are also designed to forecast the returns of assets or portfolios, rather than to choose among thousands of strategies whose performance is only observed through noisy return histories.\\
```

*Tiếng Việt:* Chúng cũng được thiết kế để dự báo lợi suất của tài sản hoặc danh mục, chứ không phải để chọn giữa hàng nghìn chiến lược mà hiệu quả chỉ quan sát được qua các lịch sử lợi suất nhiễu.

---

## Sửa 5 — Mục 2.3: thêm mô hình nền đa dụng trước câu giới thiệu FinCast

**Tìm:**

```latex
\citep{zhu2025fincast} introduce FinCast
```

**Thay bằng** (thêm một câu, và đổi `\citep` thành `\citet` vì đây là chủ ngữ của câu):

```latex
General-purpose models such as Chronos \cite{ansari2024chronos} and TimesFM \cite{das2024decoder} are pretrained on large collections of public time series and forecast new series in a zero-shot manner. \citet{zhu2025fincast} introduce FinCast
```

*Tiếng Việt:* Các mô hình đa dụng như Chronos và TimesFM được tiền huấn luyện trên tập lớn chuỗi thời gian công khai và dự báo chuỗi mới theo kiểu zero-shot. Zhu và cộng sự giới thiệu FinCast…

---

## Sửa 6 — Mục 2.3: bỏ câu bị lặp ở cuối mục

**Tìm:**

```latex
remain disconnected from one another; the two sources of information remain disconnected from one another.
```

**Thay bằng:**

```latex
remain disconnected from one another.
```

*Tiếng Việt:* Bỏ phần lặp "hai nguồn thông tin vẫn tách rời nhau".

---

## Sửa 7 — Mục 2.4: đổi tiêu đề mục con

**Tìm:**

```latex
\subsection{Attention Mechanisms in the Stochastic Discount Factor}
```

**Thay bằng:**

```latex
\subsection{Attention Mechanisms in Asset Pricing and Set-Based Learning}
```

*Tiếng Việt:* "Cơ chế attention trong định giá tài sản và học trên tập" (vì mục giờ có thêm Set Transformer ở sửa 9).

---

## Sửa 8 — Mục 2.4: sửa hai chỗ không còn đúng (khác biệt với AIPM và "interpretability")

**8a. Tìm** (trong đoạn "AIPM's attention layers, however…"), từ `These architectural differences manifest along two distinct axes` đến hết `\cite{vaswani2017attention}.` (câu ngay trước "A smaller strand of finance applications…"):

```latex
These architectural differences manifest along two distinct axes: the origin of the underlying representations, and the dimension along which attention operates. On the latter axis, the attention mechanism explored in the present study fundamentally differs from AIPM. Rather than reweighting across assets within a given time period—as captured by AIPM's dynamic, time-varying attention matrix over the cross-section of assets—our framework reweights across temporal periods within a single signal's own historical return trajectory, aligning more closely with the sequence-pooling paradigms of the foundational Transformer architecture \cite{vaswani2017attention}.
```

**Thay bằng:**

```latex
These architectural differences concern the origin of the underlying representations and the dimension along which attention operates. AIPM reweights across assets within a given period, whereas attention pooling reweights across the temporal periods of a single series, in line with the sequence-pooling paradigm of the Transformer \cite{vaswani2017attention}. The model of this study uses both dimensions: attention pooling over time within each strategy, and a set-based attention block across the strategies of the same year.
```

*Tiếng Việt:* Những khác biệt kiến trúc này liên quan đến nguồn gốc của biểu diễn và chiều mà attention hoạt động. AIPM đặt lại trọng số giữa các tài sản trong một kỳ, còn attention pooling đặt lại trọng số giữa các giai đoạn thời gian của một chuỗi, theo mô hình gộp chuỗi của Transformer. Mô hình của bài dùng cả hai chiều: attention pooling theo thời gian trong từng chiến lược, và khối attention trên tập giữa các chiến lược cùng năm.

**8b. Tìm**, từ `Our specific contribution` đến hết `directly motivates the interpretability goals pursued in this study.\\`:

```latex
Our specific contribution—temporal attention pooling over a single signal's history atop a pretrained foundation model backbone remains, to our knowledge, unexplored in this precise form. Nonetheless, AIPM remains highly relevant: it demonstrates that attention weights, when optimized within an asset-pricing framework, yield transparent and economically meaningful interpretations—a finding that directly motivates the interpretability goals pursued in this study.\\
```

**Thay bằng:**

```latex
To our knowledge, combining temporal attention pooling over a single strategy's history with a pretrained foundation model backbone has not been explored in this form. AIPM nonetheless remains relevant: it shows that attention optimized within an asset-pricing framework can be economically meaningful, which supports the use of attention across the cross-section of strategies.\\
```

*Tiếng Việt:* Theo hiểu biết của chúng tôi, việc kết hợp attention pooling theo thời gian trên lịch sử của một chiến lược với nền mô hình tiền huấn luyện chưa được khảo sát dưới dạng này. Dù vậy AIPM vẫn liên quan: nó cho thấy attention được tối ưu trong khung định giá tài sản có thể có ý nghĩa kinh tế, điều này ủng hộ việc dùng attention trên mặt cắt ngang các chiến lược.

---

## Sửa 9 — Mục 2.4: thêm đoạn về Set Transformer (cơ sở của ISAB ở mục 3.2)

**Tìm** (đúng chuỗi vừa thay ở sửa 8b, nghĩa là cuối mục 2.4):

```latex
which supports the use of attention across the cross-section of strategies.\\
```

**Thay bằng** (giữ dòng trên, thêm đoạn mới ngay sau, cách một dòng trống, trước dòng `\subsection{Factor and Anomaly Decay…}`):

```latex
which supports the use of attention across the cross-section of strategies.\\

Selecting strategies is likewise a cross-sectional problem, because a strategy is chosen only if it looks attractive relative to the other strategies of the same year. Full self-attention over tens of thousands of strategies is costly, and the Set Transformer \cite{lee2019set} addresses this with induced set attention blocks, which route the interaction between set elements through a small number of learned inducing points. To our knowledge, such set-based attention has not been applied to the cross-section of long-short strategies.\\
```

*Tiếng Việt:* Chọn chiến lược cũng là bài toán mặt cắt ngang, vì một chiến lược chỉ được chọn nếu trông hấp dẫn so với các chiến lược khác cùng năm. Self-attention đầy đủ trên hàng chục nghìn chiến lược rất tốn kém, và Set Transformer giải quyết bằng các khối induced set attention, đưa tương tác giữa các phần tử của tập qua một số ít điểm cảm ứng học được. Theo hiểu biết của chúng tôi, attention trên tập như vậy chưa được áp dụng cho mặt cắt ngang các chiến lược long–short.

---

## Sửa 10 — Mục 2.5: sửa khoá trích dẫn sai

**Tìm:**

```latex
\cite{Chen}
```

**Thay bằng:**

```latex
\cite{chen2025highthroughputassetpricing}
```

*Tiếng Việt:* Khoá `Chen` không có trong `.bib` nên sẽ báo undefined; khoá đúng là `chen2025highthroughputassetpricing`.

---

## Sửa 11 — Mục 2.5: sửa câu cuối đoạn thứ hai cho khớp mô hình hiện tại

**Tìm:**

```latex
and has not incorporated learned, attention-weighted representations of a signal's full historical return path of the kind made possible by pretrained time-series foundation models.
```

**Thay bằng:**

```latex
and, to our knowledge, has not used learned representations of a signal's full historical return path, such as those made possible by pretrained time-series foundation models.
```

*Tiếng Việt:* Và theo hiểu biết của chúng tôi chưa dùng các biểu diễn học được của toàn bộ đường lợi suất lịch sử của một tín hiệu, như các biểu diễn mà mô hình nền chuỗi thời gian tiền huấn luyện cho phép.

---

## Sửa 12 — Mục 2.5: thêm đoạn giải thích suy giảm sau 2004

**Tìm** (cuối đoạn đầu của mục 2.5):

```latex
reinforcing the multiple-testing concerns.\\
```

**Thay bằng** (giữ dòng trên, thêm đoạn mới ngay sau nó, trước đoạn "Because the number of candidate signals…"):

```latex
reinforcing the multiple-testing concerns.\\

The decline is not uniform over time. \citet{chen2025highthroughputassetpricing} find that the returns of data-mined strategies are concentrated in accounting strategies and in the period before 2004. They interpret this pattern as consistent with limited attention and the slow incorporation of information into prices \cite{peng2005learning}, and they associate the later decline with the rise of information technology \cite{chordia2014have}. This structural break motivates evaluating a forecasting method separately before and after 2004, as we do in Section \ref{sec4}.\\
```

*Tiếng Việt:* Sự suy giảm không đồng đều theo thời gian. Chen & Dim thấy lợi suất của các chiến lược khai thác dữ liệu tập trung ở chiến lược kế toán và ở giai đoạn trước 2004. Họ diễn giải mẫu hình này phù hợp với sự chú ý hạn chế và việc thông tin được phản ánh chậm vào giá (Peng), và gắn sự suy giảm sau đó với sự phát triển của công nghệ thông tin (Chordia và cộng sự). Điểm gãy cấu trúc này là lý do ta đánh giá phương pháp dự báo riêng cho trước và sau 2004, như ở mục 4.

---

## Sửa 13 — Mục 2.6: thay cả mục con "Research Gap"

Mục con cuối đã lệch nhiều nhất, nên thay cả mục (một lần). **Xoá** từ dòng `\subsection{Research Gap: Toward an Attention-Augmented FinCast Framework for Signal Persistence Forecasting}` đến hết mục 2 (dòng cuối kết thúc bằng "…to forecasting accuracy and interpretability."), rồi **paste:**

```latex
\subsection{Research Gap and Contribution}

The literature reviewed above leaves three gaps. First, rigorous data mining and naive mining rank strategies with summary statistics of their return histories \cite{chen2025highthroughputassetpricing}, whereas pretrained time-series foundation models can represent the full return sequence, but have not, to our knowledge, been used to forecast the out-of-sample performance of long-short strategies. Second, choosing the best strategies depends on how a strategy compares with its peers in the same year, yet the existing temporal and cross-sectional attention models have not been combined for this purpose. Third, evidence on predictability must be obtained without look-ahead bias and compared with empirical Bayes and naive mining through out-of-sample portfolio tests, which requires a training procedure that forecasts every year using only earlier data.\\

To address these gaps, this study forecasts the out-of-sample mean return of equal-weighted accounting strategies for each year from 1983 to 2020. The proposed model represents each strategy's return history with a frozen pretrained FinCast encoder compressed by principal component analysis, summarizes it with a Bi-LSTM and attention pooling, lets the strategies of the same year interact through an induced set attention block, and combines the result with the t-stat and mean return. It is trained with a label that expresses future performance relative to the year and to the t-stat, and it is re-estimated every year. We compare it with empirical Bayes mining and naive mining using the portfolio tests of \cite{chen2025highthroughputassetpricing}, and we report which components contribute to its performance.
```

*Tiếng Việt:* **2.6 Khoảng trống nghiên cứu và đóng góp.** Tài liệu ở trên để lại ba khoảng trống. Thứ nhất, khai thác dữ liệu chặt chẽ và khai thác đơn giản xếp hạng chiến lược bằng các thống kê tóm tắt của lịch sử lợi suất (Chen & Dim), trong khi các mô hình nền chuỗi thời gian tiền huấn luyện có thể biểu diễn toàn bộ chuỗi lợi suất, nhưng theo hiểu biết của chúng tôi chưa được dùng để dự báo hiệu quả ngoài mẫu của các chiến lược long–short. Thứ hai, chọn chiến lược tốt nhất phụ thuộc vào việc chiến lược so với các chiến lược cùng năm ra sao, nhưng các mô hình attention theo thời gian và theo mặt cắt ngang hiện có chưa được kết hợp cho mục đích này. Thứ ba, bằng chứng về khả năng dự báo phải đạt được mà không có look-ahead bias và được so với empirical Bayes và khai thác đơn giản qua các phép thử danh mục ngoài mẫu, đòi hỏi một quy trình huấn luyện dự báo mỗi năm chỉ bằng dữ liệu trước đó.

Để giải quyết các khoảng trống này, nghiên cứu dự báo lợi suất trung bình ngoài mẫu của các chiến lược kế toán chia đều cho từng năm từ 1983 đến 2020. Mô hình đề xuất biểu diễn lịch sử lợi suất của mỗi chiến lược bằng bộ mã hoá FinCast tiền huấn luyện đóng băng được nén bằng PCA, tóm tắt bằng Bi-LSTM và attention pooling, cho các chiến lược cùng năm tương tác qua khối induced set attention, và kết hợp kết quả với t-stat và lãi trung bình. Mô hình được huấn luyện với nhãn biểu diễn hiệu quả tương lai so với năm và so với t-stat, và được học lại mỗi năm. Ta so sánh với empirical Bayes và khai thác đơn giản bằng các phép thử danh mục của Chen & Dim, và báo thành phần nào đóng góp vào hiệu quả của mô hình.

---

## BibTeX (chỉ thêm nếu `.bib` của bạn chưa có)

Đã đối chiếu CrossRef (bốn mục đầu) và arXiv (hai mục cuối, đăng dạng preprint).

```bibtex
@article{efron1973stein,
  title   = {Stein's Estimation Rule and Its Competitors---An Empirical {Bayes} Approach},
  author  = {Efron, Bradley and Morris, Carl},
  journal = {Journal of the American Statistical Association},
  volume  = {68},
  number  = {341},
  pages   = {117--130},
  year    = {1973}
}

@article{chen2022open,
  title   = {Open Source Cross-Sectional Asset Pricing},
  author  = {Chen, Andrew Y. and Zimmermann, Tom},
  journal = {Critical Finance Review},
  volume  = {11},
  number  = {2},
  pages   = {207--264},
  year    = {2022}
}

@article{chordia2014have,
  title   = {Have Capital Market Anomalies Attenuated in the Recent Era of High Liquidity and Trading Activity?},
  author  = {Chordia, Tarun and Subrahmanyam, Avanidhar and Tong, Qing},
  journal = {Journal of Accounting and Economics},
  volume  = {58},
  number  = {1},
  pages   = {41--58},
  year    = {2014}
}

@article{peng2005learning,
  title   = {Learning with Information Capacity Constraints},
  author  = {Peng, Lin},
  journal = {Journal of Financial and Quantitative Analysis},
  volume  = {40},
  number  = {2},
  pages   = {307--329},
  year    = {2005}
}

@article{ansari2024chronos,
  title   = {Chronos: Learning the Language of Time Series},
  author  = {Ansari, Abdul Fatir and Stella, Lorenzo and Turkmen, Caner and Zhang, Xiyuan and Mercado, Pedro and Shen, Huibin and Shchur, Oleksandr and Rangapuram, Syama Sundar and Arango, Sebastian Pineda and Kapoor, Shubham and Zschiegner, Jasper and Maddix, Danielle C. and Wang, Hao and Mahoney, Michael W. and Torkkola, Kari and Wilson, Andrew Gordon and Bohlke-Schneider, Michael and Wang, Yuyang},
  journal = {arXiv preprint arXiv:2403.07815},
  year    = {2024}
}

@article{das2024decoder,
  title   = {A Decoder-Only Foundation Model for Time-Series Forecasting},
  author  = {Das, Abhimanyu and Kong, Weihao and Sen, Rajat and Zhou, Yichen},
  journal = {arXiv preprint arXiv:2310.10688},
  year    = {2024}
}
```

---

## Thứ tự làm và kiểm tra

1. Làm **sửa 1 đến 6** (các sửa ngắn, an toàn), recompile.
2. Làm **sửa 7 đến 9** (mục 2.4), recompile.
3. Làm **sửa 10 đến 12** (mục 2.5), recompile.
4. Làm **sửa 13** (mục 2.6), recompile.
5. Trước khi làm, **đừng comment cả vùng**: chỉ chọn đúng đoạn cần thay, để không làm mất dòng `\section{Methodology}` và khối hình kiến trúc ngay sau mục 2.
6. Sửa xong, tìm trong mục 2 các chữ `interpretab`, `Chen}`, `No prior study`: không được còn.

Sau khi bạn làm xong, mình tải lại file từ Overleaf để đối chiếu từng sửa đổi.
