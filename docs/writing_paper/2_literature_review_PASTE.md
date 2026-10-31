# Mục 2 Literature Review — nội dung để paste vào Overleaf

Viết lại trên nền bản của bạn: **giữ nguyên các đoạn đã tốt** (CAPM và factor zoo, ML trong asset pricing, FinCast, AIPM, suy giảm anomaly), **sửa** các chỗ không còn đúng với mô hình hiện tại, **thêm** hướng khai thác dữ liệu / EB và Set Transformer. Bố cục đi từ rộng đến hẹp, mỗi mục con dẫn sang mục sau, và kết ở khoảng trống nghiên cứu khớp với ba đóng góp ở mục 6.2.

## Cách dùng

1. Trong `overleaf_project/sn-article.tex`, xoá từ `\section{Literature Review}\label{sec2}` đến ngay trước `\section{Methodology}\label{sec3}`, rồi paste khối LaTeX bên dưới.
2. Thêm 6 mục BibTeX (phần dưới) vào cuối `overleaf_project/sn-bibliography.bib`.
3. Sửa dòng `\cite{Chen}` cũ: khối mới đã bỏ nó (khoá đúng là `chen2025highthroughputassetpricing`).
4. Sau khi paste, recompile và gửi mình lỗi nếu có.

## Giữ, sửa, thêm so với bản cũ

| Mục mới | Nguồn từ bản cũ | Thay đổi |
|---|---|---|
| 2.1 Return predictability, the factor zoo and data mining | 2.1 cũ | **Giữ** hai đoạn đầu, rút câu về "linear form" (chuyển sang 2.3). **Thêm** đoạn khai thác dữ liệu có hệ thống và EB (đối thủ trực tiếp) |
| 2.2 Decay and persistence of signals | 2.5 cũ | **Chuyển lên** trước, **giữ** McLean–Pontiff, Bowles, Chen–Velikov. **Thêm** giải thích suy giảm sau 2004 (Chordia, Peng) để dẫn tới cách chia dev / holdout. Bỏ `\cite{Chen}` lỗi |
| 2.3 Machine learning in asset pricing | 2.2 cũ | **Giữ** hai đoạn, rút gọn câu cuối, thêm câu nối với bài toán chọn chiến lược |
| 2.4 Sequence and foundation models for financial time series | 2.3 cũ | **Giữ** LSTM và FinCast, **rút** GNN còn một câu, **thêm** Chronos và TimesFM, **sửa** câu lặp "disconnected from one another" |
| 2.5 Cross-sectional attention and set-based learning | 2.4 cũ | **Viết lại**: bỏ hai công thức AIPM, **bỏ** khẳng định "khác AIPM vì chỉ attention theo thời gian" (không còn đúng vì mô hình có ISAB). **Thêm** Set Transformer |
| 2.6 Research gap and contribution | 2.6 cũ | **Viết lại**: ba khoảng trống khớp ba đóng góp; bỏ "No prior study…", bỏ hứa "interpretability" và ba mô hình đối chứng (i, ii, iii) không còn đúng; không nêu kết quả |

Độ dài: khoảng 1.500 từ (bản cũ khoảng 1.900).

---

## Bản tiếng Anh (paste vào LaTeX)

```latex
\section{Literature Review}\label{sec2}

We review the literature from the broad question to the specific method. Section \ref{subsec2_1} covers return predictability and data mining, Section \ref{subsec2_2} the decay of predictability, Sections \ref{subsec2_3} to \ref{subsec2_5} the modeling tools that we build on, and Section \ref{subsec2_6} the gap that this study addresses.

\subsection{Return predictability, the factor zoo and data mining}\label{subsec2_1}

Modern asset pricing theory originates from the Capital Asset Pricing Model \cite{sharpe1964capital} and the Arbitrage Pricing Theory \cite{ross1978current}, later extended to a dynamic, multi-period setting by \cite{merton1973intertemporal}. These models represent expected returns as a linear function of exposure to a small number of systematic risk factors. Over subsequent decades, empirical research documented an ever-expanding array of return predictors, spanning global firm-level characteristics \cite{jensen2023there}, time-varying macroeconomic and market-wide state variables \cite{farmer2023pockets}, and hundreds of candidate risk factors collectively formalized as the factor zoo \cite{feng2020taming}.

This proliferation of predictors has raised serious methodological concerns. Standard hypothesis-testing procedures do not account for the multiple-testing problem inherent in searching over hundreds of candidate factors, which inflates the likelihood of spurious findings \cite{harvey2016and, chen2025t}. Related work questions whether published anomalies replicate reliably out of sample across international markets \cite{jensen2023there}, and whether peer-reviewed economic theory helps identify factors with genuine predictive power, as opposed to factors that simply survived a data-mining search \cite{chen2023does}.

A recent response is to mine data rigorously rather than less. \citet{chen2025highthroughputassetpricing} apply empirical Bayes \cite{efron1973stein} to about 136,000 long-short strategies built by systematically searching accounting ratios, past returns and tickers, following the construction of \citet{yan2017fundamental}. They show that a portfolio of the top 1\% of strategies, selected with information available in real time, earns out-of-sample returns comparable to those of published strategies \cite{chen2022open}, and that naively ranking strategies by their in-sample Sharpe ratio performs similarly. Empirical Bayes forecasts the performance of each strategy from summary statistics of its return history, namely the t-stat, and therefore does not explicitly model how the returns are distributed over time. This makes the forecasting of a strategy's own future performance a question that is logically prior to modeling how any individual factor prices the cross-section.

\subsection{Decay and persistence of signals}\label{subsec2_2}

A closely related literature asks whether a documented signal will continue to earn abnormal returns. \citet{mclean2016does} show that the average return of a documented anomaly declines substantially after its discovery and further after its publication, consistent with a combination of statistical overfitting during the discovery process and gradual arbitrage by investors who learn about the anomaly. \citet{bowles2024anomaly} show that anomaly returns are concentrated in the period immediately following the public release of the underlying information and decay rapidly thereafter, so that even the timing of a signal's profitability is unstable out of sample. \citet{chen2023zeroing} further show that, once publication bias and realistic trading costs are accounted for, the expected returns that remain for most anomalies are considerably smaller than their original in-sample estimates. Taken together, this evidence implies that a signal's in-sample statistics, however large the associated t-stat, are on their own an unreliable guide to its future performance \cite{harvey2016and}.

The decline is not uniform over time. \citet{chen2025highthroughputassetpricing} find that the returns of data-mined strategies are concentrated in accounting strategies and in the period before 2004. They interpret this pattern as consistent with limited attention and the slow incorporation of information into prices \cite{peng2005learning, hirshleifer2003limited, barber2008all}, and they associate the later decline with the rise of information technology \cite{chordia2014have}. This structural break motivates evaluating a forecasting method separately before and after 2004, as we do in Section \ref{sec4}.

\subsection{Machine learning in asset pricing}\label{subsec2_3}

To address the limitations of linear specifications, subsequent research applies machine learning (ML) and deep learning (DL) methods to the cross-section of expected returns. \citet{gu2020empirical} show that non-linear ML models, including regularized regressions, tree ensembles and neural networks, substantially outperform traditional linear benchmarks in out-of-sample return prediction. Related approaches compress large characteristic sets into lower-dimensional latent factors using autoencoders \cite{gu2021autoencoder, chen2024deep} and kernel-based regularization \cite{kozak2020shrinking}, while nonparametric techniques allow characteristic-return relationships to take flexible forms without imposing rigid functional structures \cite{freyberger2020dissecting, avramov2023machine}.

A more recent line of work examines why increasing model complexity does not necessarily lead to overfitting in return prediction, a phenomenon termed the ``virtue of complexity'': highly parameterized models can generalize well out of sample even when the number of parameters greatly exceeds the number of observations \cite{kelly2024virtue}. This insight has motivated specialized architectures, including panel regression trees \cite{cong2025growing}, multilayer perceptrons applied to large-cap equities \cite{lai2025multilayer} and deep statistical arbitrage models \cite{guijarro2025deep}. Most of these architectures process each input independently through static, feedforward transformations, and they typically summarize an input history with a single fixed-form operation, such as a terminal hidden state or an unweighted average. They are also designed to forecast returns of assets or portfolios, rather than to choose among thousands of strategies whose performance is only observed through noisy return histories.

\subsection{Sequence and foundation models for financial time series}\label{subsec2_4}

Because asset prices are sequential, recurrent networks and Long Short-Term Memory (LSTM) networks were for some time the standard approach to financial sequence modeling \cite{yu2020stock}, although they are known to struggle with long-range dependencies. Graph-based models extend this view across assets, representing markets as graphs to capture comovement and spillovers \cite{wu2020connecting, chen2018incorporating}.

A more recent line of research develops large-scale, pretrained time-series foundation models. General-purpose models such as Chronos \cite{ansari2024chronos} and TimesFM \cite{das2024decoder} are pretrained on large collections of public time series and forecast new series in a zero-shot manner. \citet{zhu2025fincast} introduce FinCast, a one-billion-parameter decoder-only Transformer with a sparse Mixture-of-Experts architecture, pretrained on more than 20 billion time points and designed for financial time series, using learnable frequency embeddings and a joint Point-Quantile loss. FinCast achieves strong zero-shot forecasting performance across financial domains and sampling frequencies without domain-specific fine-tuning.

These models are pretrained to minimize a generic forecasting error, so the representation that they produce is task-agnostic and is not optimized to identify which parts of a series are diagnostic of its future performance. They also offer no mechanism for combining the learned representation with the static diagnostics that empirical researchers conventionally use to judge a signal, such as the in-sample t-stat and mean return \cite{harvey2016and, chen2025t}, so the two sources of information remain disconnected.

\subsection{Cross-sectional attention and set-based learning}\label{subsec2_5}

Attention mechanisms offer a way to weight information adaptively. Rather than summarizing a sequence with its last hidden state or an unweighted average, attention-based pooling assigns data-dependent weights to different time steps \cite{vaswani2017attention}. In finance, \citet{wood2021trading} combine attention with an LSTM backbone to weight past time steps of a trading signal, and \citet{ma2023attention} show that transformer attention in an asset-pricing context can yield economically interpretable weightings.

Attention has also been applied across assets. \citet{kelly2025artificial} propose the Artificial Intelligence Pricing Model (AIPM), which places a Transformer within the stochastic discount factor so that portfolio weights depend on the characteristics of the entire asset universe rather than on each asset's own characteristics alone, consistent with limited-attention theories of information processing \cite{hirshleifer2003limited, barber2008all}. AIPM is estimated end to end within a cross-sectional pricing objective on firm characteristics, and it does not use a pretrained temporal backbone.

Selecting strategies is likewise a cross-sectional problem, because a strategy is chosen only if it looks attractive relative to the other strategies of the same year. Full self-attention over tens of thousands of strategies is costly, and the Set Transformer \cite{lee2019set} addresses this with induced set attention blocks, which route the interaction between set elements through a small number of learned inducing points. To our knowledge, such set-based attention has not been applied to the cross-section of long-short strategies.

\subsection{Research gap and contribution}\label{subsec2_6}

The literature reviewed above leaves three gaps. First, rigorous data mining and naive mining rank strategies with summary statistics of their return histories \cite{chen2025highthroughputassetpricing}, whereas pretrained time-series foundation models can represent the full return sequence, but have not, to our knowledge, been used to forecast the out-of-sample performance of long-short strategies. Second, choosing the best strategies depends on how a strategy compares with its peers in the same year, yet the existing temporal and cross-sectional attention models have not been combined for this purpose. Third, evidence on predictability must be obtained without look-ahead bias and compared with empirical Bayes and naive mining through out-of-sample portfolio tests, which requires a training procedure that forecasts every year using only earlier data.

To address these gaps, this study forecasts the out-of-sample mean return of equal-weighted accounting strategies for each year from 1983 to 2020. The proposed model represents each strategy's return history with a frozen pretrained FinCast encoder compressed by principal component analysis, summarizes it with a Bi-LSTM and attention pooling, lets the strategies of the same year interact through an induced set attention block, and combines the result with the t-stat and mean return. It is trained with a label that expresses future performance relative to the year and to the t-stat, and it is re-estimated every year. We compare it with empirical Bayes mining and naive mining using the portfolio tests of \cite{chen2025highthroughputassetpricing}, and we report which components contribute to its performance.
```

---

## BibTeX cần thêm vào `overleaf_project/sn-bibliography.bib`

Đã đối chiếu qua CrossRef (4 mục đầu) và arXiv (2 mục cuối, đăng dạng preprint).

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

## Bản tiếng Việt (dịch sát bản tiếng Anh, để bạn đọc, không paste)

### 2 Tổng quan tài liệu

Ta điểm lại tài liệu từ câu hỏi rộng đến phương pháp cụ thể. Mục 2.1 nói về khả năng dự báo lợi suất và khai thác dữ liệu, mục 2.2 về sự suy giảm của khả năng dự báo, các mục 2.3 đến 2.5 về các công cụ mô hình mà ta dựa vào, và mục 2.6 về khoảng trống mà nghiên cứu này giải quyết.

#### 2.1 Return predictability, the factor zoo and data mining (Khả năng dự báo lợi suất, "vườn thú nhân tố" và khai thác dữ liệu)

Lý thuyết định giá tài sản hiện đại bắt nguồn từ Capital Asset Pricing Model (Sharpe) và Arbitrage Pricing Theory (Ross), sau đó được Merton mở rộng sang bối cảnh động nhiều kỳ. Các mô hình này biểu diễn lợi suất kỳ vọng là hàm tuyến tính của mức phơi nhiễm với một số ít nhân tố rủi ro hệ thống. Qua nhiều thập kỷ, nghiên cứu thực nghiệm ghi nhận ngày càng nhiều biến dự báo lợi suất: các đặc trưng cấp doanh nghiệp toàn cầu (Jensen và cộng sự), các biến trạng thái vĩ mô và thị trường thay đổi theo thời gian (Farmer và cộng sự), và hàng trăm nhân tố rủi ro ứng viên được gọi chung là factor zoo (Feng và cộng sự).

Sự bùng nổ các biến dự báo gây ra những lo ngại phương pháp nghiêm trọng. Các quy trình kiểm định giả thuyết chuẩn không tính đến vấn đề kiểm định nhiều lần (multiple testing) vốn có khi tìm trong hàng trăm nhân tố ứng viên, làm tăng khả năng có phát hiện giả (Harvey và cộng sự; Chen). Các nghiên cứu liên quan đặt câu hỏi liệu các anomaly đã công bố có tái lập đáng tin ngoài mẫu ở các thị trường quốc tế không, và liệu lý thuyết kinh tế đã qua bình duyệt có giúp nhận ra các nhân tố có khả năng dự báo thật, thay vì chỉ là các nhân tố vừa sống sót qua một cuộc tìm kiếm dữ liệu (Chen, Lopez-Lira, Zimmermann).

Một cách đáp lại gần đây là khai thác dữ liệu một cách chặt chẽ chứ không phải ít đi. Chen & Dim áp dụng empirical Bayes (Efron & Morris) cho khoảng 136.000 chiến lược long–short được tạo bằng cách tìm kiếm có hệ thống trên các tỷ số kế toán, lợi suất quá khứ và mã cổ phiếu, theo cách xây dựng của Yan & Zheng. Họ cho thấy danh mục top 1% chiến lược, chọn bằng thông tin có sẵn theo thời gian thực, cho lợi suất ngoài mẫu tương đương các chiến lược đã công bố (Chen & Zimmermann), và việc xếp hạng đơn giản theo Sharpe trong mẫu cũng cho kết quả tương tự. Empirical Bayes dự báo hiệu quả của từng chiến lược từ các thống kê tóm tắt của lịch sử lợi suất, cụ thể là t-stat, nên không mô hình hoá tường minh việc lợi suất phân bố theo thời gian như thế nào. Điều này làm cho việc dự báo hiệu quả tương lai của chính một chiến lược là câu hỏi đi trước, về mặt logic, việc mô hình hoá từng nhân tố định giá mặt cắt ngang ra sao.

#### 2.2 Decay and persistence of signals (Sự suy giảm và độ bền của tín hiệu)

Một dòng tài liệu liên quan chặt chẽ hỏi liệu một tín hiệu đã được ghi nhận có tiếp tục sinh lợi bất thường hay không. McLean & Pontiff cho thấy lợi suất trung bình của một anomaly đã ghi nhận giảm mạnh sau khi được phát hiện và giảm thêm sau khi công bố, phù hợp với sự kết hợp của overfitting thống kê trong quá trình phát hiện và việc nhà đầu tư dần kinh doanh chênh lệch giá khi biết đến anomaly. Bowles và cộng sự cho thấy lợi suất anomaly tập trung ngay sau khi thông tin được công bố và suy giảm nhanh sau đó, nên ngay cả thời điểm sinh lợi của tín hiệu cũng không ổn định ngoài mẫu. Chen & Velikov chỉ ra thêm rằng khi tính đến thiên lệch công bố và chi phí giao dịch thực tế, lợi suất kỳ vọng còn lại của đa số anomaly nhỏ hơn đáng kể so với ước lượng trong mẫu ban đầu. Gộp lại, bằng chứng này hàm ý các thống kê trong mẫu của một tín hiệu, dù t-stat lớn đến đâu, tự chúng là chỉ dẫn không đáng tin cho hiệu quả tương lai (Harvey và cộng sự).

Sự suy giảm không đồng đều theo thời gian. Chen & Dim thấy lợi suất của các chiến lược khai thác dữ liệu tập trung ở chiến lược kế toán và ở giai đoạn trước 2004. Họ diễn giải mẫu hình này phù hợp với sự chú ý hạn chế và việc thông tin được phản ánh chậm vào giá (Peng; Hirshleifer & Teoh; Barber & Odean), và gắn sự suy giảm sau đó với sự phát triển của công nghệ thông tin (Chordia và cộng sự). Điểm gãy cấu trúc này là lý do ta đánh giá phương pháp dự báo riêng cho trước và sau 2004, như ở mục 4.

#### 2.3 Machine learning in asset pricing (Học máy trong định giá tài sản)

Để giải quyết hạn chế của các mô hình tuyến tính, các nghiên cứu sau đó áp dụng học máy (ML) và học sâu (DL) cho mặt cắt ngang của lợi suất kỳ vọng. Gu và cộng sự cho thấy các mô hình ML phi tuyến, gồm hồi quy có chính quy hoá, rừng cây và mạng nơ-ron, vượt đáng kể các chuẩn tuyến tính trong dự báo lợi suất ngoài mẫu. Các cách tiếp cận liên quan nén tập đặc trưng lớn thành các nhân tố tiềm ẩn chiều thấp bằng autoencoder (Gu và cộng sự; Chen và cộng sự) và chính quy hoá dựa trên kernel (Kozak và cộng sự), trong khi các kỹ thuật phi tham số cho phép quan hệ giữa đặc trưng và lợi suất có dạng linh hoạt mà không áp cấu trúc hàm cứng nhắc (Freyberger và cộng sự; Avramov và cộng sự).

Một dòng gần đây xem xét vì sao tăng độ phức tạp mô hình không nhất thiết dẫn đến overfitting khi dự báo lợi suất, hiện tượng gọi là "virtue of complexity": các mô hình nhiều tham số có thể tổng quát hoá tốt ngoài mẫu dù số tham số vượt xa số quan sát (Kelly và cộng sự). Điều này thúc đẩy các kiến trúc chuyên biệt như cây hồi quy dạng panel (Cong và cộng sự), perceptron nhiều lớp áp dụng cho cổ phiếu vốn hoá lớn (Lai và cộng sự) và các mô hình kinh doanh chênh lệch giá thống kê sâu (Guijarro và cộng sự). Phần lớn các kiến trúc này xử lý từng đầu vào độc lập qua các phép biến đổi tĩnh, và thường tóm tắt lịch sử đầu vào bằng một phép toán dạng cố định, như trạng thái ẩn cuối hoặc trung bình không trọng số. Chúng cũng được thiết kế để dự báo lợi suất của tài sản hoặc danh mục, chứ không phải để chọn giữa hàng nghìn chiến lược mà hiệu quả chỉ quan sát được qua các lịch sử lợi suất nhiễu.

#### 2.4 Sequence and foundation models for financial time series (Mô hình chuỗi và mô hình nền cho chuỗi thời gian tài chính)

Vì giá tài sản là chuỗi tuần tự, mạng hồi quy và mạng Long Short-Term Memory (LSTM) từng là cách tiếp cận chuẩn cho mô hình hoá chuỗi tài chính (Yu & Yan), dù được biết là khó với phụ thuộc xa. Các mô hình dựa trên đồ thị mở rộng góc nhìn này sang nhiều tài sản, biểu diễn thị trường bằng đồ thị để nắm bắt đồng biến động và lan truyền (Wu và cộng sự; Chen và cộng sự).

Một dòng nghiên cứu gần đây phát triển các mô hình nền chuỗi thời gian tiền huấn luyện quy mô lớn. Các mô hình đa dụng như Chronos (Ansari và cộng sự) và TimesFM (Das và cộng sự) được tiền huấn luyện trên tập lớn chuỗi thời gian công khai và dự báo chuỗi mới theo kiểu zero-shot. Zhu và cộng sự giới thiệu FinCast, một Transformer decoder-only một tỷ tham số với kiến trúc Mixture-of-Experts thưa, được tiền huấn luyện trên hơn 20 tỷ điểm thời gian và thiết kế cho chuỗi thời gian tài chính, dùng embedding tần suất học được và hàm mất mát Point-Quantile kết hợp. FinCast đạt hiệu quả dự báo zero-shot mạnh trên nhiều lĩnh vực tài chính và tần suất lấy mẫu mà không cần tinh chỉnh riêng.

Các mô hình này được tiền huấn luyện để cực tiểu sai số dự báo chung, nên biểu diễn chúng tạo ra là biểu diễn không gắn với nhiệm vụ và không được tối ưu để nhận ra phần nào của chuỗi có giá trị chẩn đoán cho hiệu quả tương lai. Chúng cũng không có cơ chế kết hợp biểu diễn đã học với các thống kê tĩnh mà nhà nghiên cứu thực nghiệm thường dùng để đánh giá một tín hiệu, như t-stat và lãi trung bình trong mẫu (Harvey và cộng sự; Chen), nên hai nguồn thông tin vẫn tách rời nhau.

#### 2.5 Cross-sectional attention and set-based learning (Attention mặt cắt ngang và học trên tập)

Cơ chế attention là một cách trọng số hoá thông tin thích nghi. Thay vì tóm tắt một chuỗi bằng trạng thái ẩn cuối hoặc trung bình không trọng số, attention pooling gán trọng số phụ thuộc dữ liệu cho các bước thời gian khác nhau (Vaswani và cộng sự). Trong tài chính, Wood và cộng sự kết hợp attention với nền LSTM để đặt trọng số cho các bước thời gian quá khứ của một tín hiệu giao dịch, và Ma và cộng sự cho thấy attention kiểu transformer trong bối cảnh định giá tài sản có thể cho trọng số diễn giải được về mặt kinh tế.

Attention cũng đã được áp dụng giữa các tài sản. Kelly và cộng sự đề xuất Artificial Intelligence Pricing Model (AIPM), đặt một Transformer trong stochastic discount factor để trọng số danh mục phụ thuộc vào đặc trưng của toàn bộ vũ trụ tài sản chứ không chỉ đặc trưng của từng tài sản, phù hợp với các lý thuyết chú ý hạn chế trong xử lý thông tin (Hirshleifer & Teoh; Barber & Odean). AIPM được ước lượng đầu-cuối trong một mục tiêu định giá mặt cắt ngang trên đặc trưng doanh nghiệp, và không dùng nền thời gian tiền huấn luyện.

Chọn chiến lược cũng là bài toán mặt cắt ngang, vì một chiến lược chỉ được chọn nếu trông hấp dẫn so với các chiến lược khác cùng năm. Self-attention đầy đủ trên hàng chục nghìn chiến lược rất tốn kém, và Set Transformer (Lee và cộng sự) giải quyết bằng các khối induced set attention, đưa tương tác giữa các phần tử của tập qua một số ít điểm cảm ứng học được. Theo hiểu biết của chúng tôi, attention trên tập như vậy chưa được áp dụng cho mặt cắt ngang các chiến lược long–short.

#### 2.6 Research gap and contribution (Khoảng trống nghiên cứu và đóng góp)

Tài liệu ở trên để lại ba khoảng trống. Thứ nhất, khai thác dữ liệu chặt chẽ và khai thác đơn giản xếp hạng chiến lược bằng các thống kê tóm tắt của lịch sử lợi suất (Chen & Dim), trong khi các mô hình nền chuỗi thời gian tiền huấn luyện có thể biểu diễn toàn bộ chuỗi lợi suất, nhưng theo hiểu biết của chúng tôi chưa được dùng để dự báo hiệu quả ngoài mẫu của các chiến lược long–short. Thứ hai, chọn chiến lược tốt nhất phụ thuộc vào việc chiến lược so với các chiến lược cùng năm ra sao, nhưng các mô hình attention theo thời gian và theo mặt cắt ngang hiện có chưa được kết hợp cho mục đích này. Thứ ba, bằng chứng về khả năng dự báo phải đạt được mà không có look-ahead bias và được so với empirical Bayes và khai thác đơn giản qua các phép thử danh mục ngoài mẫu, đòi hỏi một quy trình huấn luyện dự báo mỗi năm chỉ bằng dữ liệu trước đó.

Để giải quyết các khoảng trống này, nghiên cứu dự báo lợi suất trung bình ngoài mẫu của các chiến lược kế toán chia đều cho từng năm từ 1983 đến 2020. Mô hình đề xuất biểu diễn lịch sử lợi suất của mỗi chiến lược bằng bộ mã hoá FinCast tiền huấn luyện đóng băng được nén bằng PCA, tóm tắt bằng Bi-LSTM và attention pooling, cho các chiến lược cùng năm tương tác qua khối induced set attention, và kết hợp kết quả với t-stat và lãi trung bình. Mô hình được huấn luyện với nhãn biểu diễn hiệu quả tương lai so với năm và so với t-stat, và được học lại mỗi năm. Ta so sánh với empirical Bayes và khai thác đơn giản bằng các phép thử danh mục của Chen & Dim, và báo thành phần nào đóng góp vào hiệu quả của mô hình.

---

## Cần bạn biết trước khi paste

1. **Mục 2 không nêu kết quả** (đúng guide): kết quả chỉ ở 4, 5, 6. Mục 2.6 chỉ nêu khoảng trống và hướng giải quyết, khớp ba đóng góp ở 6.2.
2. **Đã bỏ hai thứ không còn đúng với bài:** (a) khẳng định "attention của bài chỉ theo thời gian nên khác AIPM" (mô hình có ISAB, attention giữa các chiến lược), (b) lời hứa phân tích "interpretability" và ba mô hình đối chứng (i, ii, iii) (bài so với EB, naive mining, baseline).
3. **Cách nói giữ thận trọng:** "to our knowledge" chỉ dùng cho hai nhận định về chưa thấy công trình; không dùng "No prior study". Nhận xét về EB trung lập (chỉ nêu EB dựa vào t-stat).
4. **Tham chiếu chéo mới** dùng `\ref{subsec2_1}` đến `\ref{subsec2_6}`, `\ref{sec4}`: cần paste đủ mục 4 để không báo undefined (đã có trên Overleaf).
5. **Sáu mục BibTeX mới** đã đối chiếu (CrossRef: Efron–Morris, Chen–Zimmermann, Chordia và cộng sự, Peng; arXiv: Chronos, TimesFM đăng dạng preprint). Các khoá còn lại đã có sẵn trong `.bib`.
6. **Các khoá trong `.bib` nhưng không dùng ở mục 2 mới:** `bailey2014deflated`, `fukasawa2026predicting`, `zong2025stock`, `zhu2025mci` (dự báo giá cổ phiếu bằng cross-attention). Nếu muốn nhắc dòng cross-attention thì có thể thêm một câu ở 2.5; hiện tôi không đưa vào để mục ngắn gọn và đúng trọng tâm.
7. Chưa biên dịch LaTeX.
