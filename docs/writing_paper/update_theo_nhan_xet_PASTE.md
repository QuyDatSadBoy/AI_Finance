# Cập nhật bài theo nhận xét của thầy — sửa tay từng ý

**Tạp chí đích:** *Financial Innovation* (Springer, ISSN 2199-4730).
**Bản đã đọc:** `sn-article.tex` và `sn-bibliography.bib` tải từ Overleaf ngày 03/11/2026, bao gồm các phần bạn đã tự viết ở mục 3, 4, 5, 6.

## Cách dùng

- Mỗi sửa đổi gồm: **vị trí**, **❌ CŨ** (copy y nguyên, Ctrl+F trên Overleaf), **✅ MỚI** (thay đúng đoạn đó), **giải thích** vì sao sửa, và bản tiếng Việt của đoạn mới.
- Sửa đổi kiểu **chèn**: đoạn CŨ là câu đứng ngay trước chỗ chèn; đoạn MỚI giữ nguyên câu đó và thêm phần mới phía sau.
- Mình đã kiểm tra bằng script: mọi đoạn CŨ trong file này xuất hiện **đúng một lần** trong bản Overleaf hiện tại.
- Không comment cả vùng, không chọn rộng. Chỉ thay đúng đoạn được nêu.

## Nhận xét của thầy và chỗ đáp ứng

| Nhận xét | Đáp ứng ở |
|---|---|
| Intro: tiếp cận "tại sao cần phương pháp mới", nêu hạn chế bằng từ nhẹ (tránh "fail", từ mạnh) | I-2 đến I-5 |
| Mỗi lần đưa ra cái gì phải giải thích tại sao | I-6 (đoạn "yêu cầu" mới), I-7, toàn bộ mục 3 |
| Thêm tầm quan trọng của asset pricing trong tài chính | I-1 |
| Đoạn "To address" sâu hơn, không chỉ quảng bá, nói rõ vì sao cần | I-7 |
| Mục 2 trích vài bài trong tạp chí | II-1 đến II-4 (7 bài *Financial Innovation*) |
| Mục 3: giải thích kiến trúc theo sự hợp lý, các bước liên kết, vì sao kết hợp hiệu quả | III-1 (đoạn và bảng mới), III-2 đến III-6 |
| Thử nghiệm: nhấn mạnh các khối khớp nhau, vì đóng góp là kiến trúc | IV-1, IV-2, V-1, V-2 |
| Soát hết reference, thêm bài trong tạp chí | Phần A, B, C |
| (Thêm) tiêu đề sai ngữ pháp và dùng "novel" | Phần 0 |

> **Lưu ý trung thực (quan trọng):** ablation cho thấy pretrained, PCA và nhãn có đóng góp; ISAB, Huber, trọng số năm **không phân biệt được** với dao động giữa các lần chạy. Vì vậy các đoạn mới nhấn mạnh kiến trúc bằng **lý do thiết kế** ("is designed to", "so that") và **bằng chứng có thật**: ablation, chênh lệch với baseline cùng FinCast (1,95 so với 1,34), và phân tích "nơi hai phương pháp bất đồng". Không có câu nào nói ISAB làm tăng hiệu quả. Nếu viết vậy, phản biện so với Bảng ablation sẽ thấy mâu thuẫn ngay.

---

# A. Lỗi tài liệu tham khảo cần sửa ngay

## A-1 (NGHIÊM TRỌNG) — `.bib` thiếu dấu `}`, làm mất bài Chen & Dim

Mục `bowles2024anomaly` không có dấu `}` đóng. BibTeX sẽ đọc lẫn sang mục ngay sau là `chen2025highthroughputassetpricing`, nên **bài Chen & Dim, bài được trích nhiều nhất (khoảng 15 lần), sẽ hiện "?" trong PDF**, và mục Bowles có thể hỏng theo.

**Trong `sn-bibliography.bib`, ❌ CŨ:**

```bibtex
  year={2024},
  publisher={Wiley Online Library}

@misc{chen2025highthroughputassetpricing,
```

**✅ MỚI** (thêm `}` ở dòng trống):

```bibtex
  year={2024},
  publisher={Wiley Online Library}
}

@misc{chen2025highthroughputassetpricing,
```

## A-2 — Khoá bị trùng

`gu2020empirical`, `vaswani2017attention`, `wu2020connecting` mỗi khoá có **hai** mục trong `.bib`. BibTeX báo lỗi "repeated entry" và chỉ dùng mục đầu. Xoá **mục thứ hai** của mỗi khoá (số dòng theo bản `.bib` hiện tại): `gu2020empirical` ở **dòng 380**, `vaswani2017attention` ở **dòng 450**, `wu2020connecting` ở **dòng 617**. Giữ các mục ở dòng 237, 303, 512.

## A-3 — Sai hoặc cũ thông tin (đã đối chiếu CrossRef, arXiv, trang nhà xuất bản)

| Khoá | Vấn đề | Sửa thành |
|---|---|---|
| `holen2024loss` | **Sai thứ tự tác giả**. arXiv 2304.12778 ghi Holen, Andersen, Knausgård, Goodwin | `author = {Holen, Martin and Andersen, Per-Arne and Knausg{\aa}rd, Kristian Muri and Goodwin, Morten}` |
| `chen2023does` | Tên cũ, loại `techreport` "Working Paper". Bản hiện tại: *Does Peer-Reviewed Research Help Predict Stock Returns?*, arXiv 2212.10317 (2022) | xem BibTeX ở phần C |
| `guijarro2025deep` | Thiếu tập và trang; bài đã in: *Management Science* 72(9):7502–7549 (2026), DOI 10.1287/mnsc.2022.03132 | xem BibTeX ở phần C |
| `kelly2025artificial` | Thiếu số working paper | thêm `number = {w33351}`, `doi = {10.3386/w33351}` |
| dòng `// Dat` | BibTeX không coi `//` là chú thích (vô hại, nhưng sai cú pháp) | đổi thành `% Dat` |
| `bib1`–`bib13`, `oy1tz1952portfolio`, `velickovic2017graph`, `kim2019hats`, `bailey2014deflated`, `diebold*`, `fama1993common`, `fama2016dissecting`, `graham2001theory`, `banz1981relationship`, `zong2025stock`, `zhu2025mci`, `fukasawa2026predicting`, `efron1973stein`, `chen2022open`, `chordia2014have`, `peng2005learning`, `ansari2024chronos`, `das2024decoder` | **Không được trích** trong bài nên không in ra. `bib1`–`bib13` là mẫu của template (vật lý, y học); `oy1tz1952portfolio` có tiêu đề hỏng "PORTFOLIO) SELECTION" | Nên xoá `bib1`–`bib13` và `oy1tz1952portfolio` cho sạch; các mục khác giữ nếu định trích |

## A-4 — Trích dẫn đúng bài nhưng **dùng sai nội dung** (nằm trong văn bản)

| Vị trí | Vấn đề | Sửa ở |
|---|---|---|
| Intro, đoạn 2 | `chen2024deep` (*Deep learning in asset pricing*) được gắn cho câu "high-throughput testing" | I-3 |
| Intro, đoạn 2 | `avramov2023machine` (ML và ràng buộc kinh tế) được gắn cho "anomaly suy giảm ngoài mẫu"; nguồn đúng là McLean & Pontiff | I-3 |
| Mục 2.4, bài `guo2026hybrid` | Bản tóm tắt nói giảm thuộc tính "giảm độ phức tạp tính toán **mà không làm giảm** độ chính xác tổng quát hoá", không phải "**improves** generalization" | II-6 |
| Mục 2.4, chi tiết `guo2026hybrid` ("three Chinese stocks", "outperforms CNN, LSTM, TCN, Informer, PatchTST") và `ma2023attention` ("72 firm characteristics and 8 macroeconomic variables", "SHAP") | Mình **không kiểm được** qua bản tóm tắt công khai (bài thu phí). Nếu bạn đã đọc toàn văn thì giữ; nếu chưa, nên đối chiếu lại | tự kiểm |

---

# B. Kết quả soát toàn bộ tài liệu tham khảo

**Cách soát:** Google Scholar không cho truy vấn tự động, nên mình đối chiếu từng mục được trích với **CrossRef** (cơ sở DOI mà Google Scholar cũng dùng), **arXiv** và **trang nhà xuất bản**. Mỗi mục được so tên bài, tác giả đầu, năm, tạp chí, tập, số, trang.

**Kết quả cho 50 khoá được trích trong bài:**

| Trạng thái | Khoá |
|---|---|
| ✅ Khớp hoàn toàn | sharpe1964capital, ross1978current, merton1973intertemporal, fama1992cross, fama2015five, green2017characteristics, gu2020empirical, chen2024deep, feng2020taming, mclean2016does, harvey2016and, giglio2022factor, avramov2023machine, chu2020causal, hirshleifer2003limited, barber2008all, jensen2023there, farmer2023pockets, wu2020connecting, chen2023zeroing, chen2025t, gu2021autoencoder, freyberger2020dissecting, kozak2020shrinking, kelly2024virtue, cong2025growing, yu2020stock, chen2018incorporating, ma2023attention, bowles2024anomaly (nội dung đúng, chỉ thiếu `}`), yan2017fundamental, zhu2025fincast (CIKM 2025, tr. 4539–4549), guo2026hybrid (*Applied Intelligence* 56(2), bài 52), huber1964robust, hotelling1933analysis, hochreiter1997long, schuster1997bidirectional, ledoit2008robust, lee2019set (ICML, PMLR 97), ioffe2015batch (ICML, PMLR 37), srivastava2014dropout (JMLR 15), loshchilov2019decoupled (ICLR 2019), vaswani2017attention (NeurIPS 30), ba2016layer (arXiv 1607.06450), lai2025multilayer (arXiv 2505.01921), wood2021trading (arXiv 2112.08534), chen2025highthroughputassetpricing (arXiv 2311.10685) |
| ⚠️ Cần sửa | holen2024loss (thứ tự tác giả), chen2023does (tên cũ), guijarro2025deep (đã in, thiếu tập/trang), kelly2025artificial (thiếu số WP), bowles2024anomaly (thiếu `}`) |
| ⚠️ Dùng sai nội dung | chen2024deep và avramov2023machine ở Intro; guo2026hybrid ở 2.4 |

Năm của vài bài (harvey2016and, barber2008all, kelly2024virtue, fama2016dissecting) lệch một năm giữa bản online và bản in. Mình giữ **năm in** như trong `.bib`, đúng cách Google Scholar ghi.

---

# C. BibTeX: sửa và thêm (dán vào cuối `sn-bibliography.bib`, xoá mục cũ cùng khoá)

Các bài *Financial Innovation* đều được đối chiếu với CrossRef, và nội dung câu trích được viết theo bản tóm tắt của bài.

```bibtex
% ---- sửa
@article{chen2023does,
  title   = {Does Peer-Reviewed Research Help Predict Stock Returns?},
  author  = {Chen, Andrew Y. and Lopez-Lira, Alejandro and Zimmermann, Tom},
  journal = {arXiv preprint arXiv:2212.10317},
  year    = {2022}
}

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

% ---- thêm: tầm quan trọng của asset pricing (I-1)
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

% ---- thêm: các bài Financial Innovation (mục 2)
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

Sửa `holen2024loss` tại chỗ: chỉ thay dòng `author` như bảng A-3.

# 0. Tiêu đề bài

**Giải thích:** "assets pricing" sai ngữ pháp (đúng là "asset pricing"), và "novel" là từ guide khuyên tránh. Tên mới nêu đúng ba ý của bài: attention, biểu diễn tiền huấn luyện, chọn chiến lược long–short.

**❌ CŨ:**

```latex
\title[Article Title]{A novel attention network for assets pricing}
```

**✅ MỚI:**

```latex
\title[Article Title]{An attention network with pretrained representations for selecting long-short strategies in asset pricing}
```

---

# I. Mục 1 — Introduction

## I-1 — Thêm tầm quan trọng của asset pricing (đầu đoạn 1)

**Giải thích:** thầy yêu cầu nêu asset pricing quan trọng thế nào trong tài chính. Hai câu mở đầu nêu ba vai trò cụ thể (chi phí vốn, phân bổ vốn, chuẩn đánh giá chiến lược) rồi nối sang lý do bài này quan tâm.

**❌ CŨ:**

```latex
Asset pricing theory explains how financial markets translate
```

**✅ MỚI:**

```latex
Asset pricing is central to finance. Expected returns determine the cost of capital that firms face, the allocation of savings across assets, and the benchmarks against which investment strategies and portfolio managers are evaluated \cite{campbell2000asset}. Knowing which sources of return predictability persist is therefore relevant both for researchers who test asset pricing theories and for investors who allocate capital to trading strategies. Asset pricing theory explains how financial markets translate
```

*Tiếng Việt:* Định giá tài sản là trung tâm của tài chính. Lợi suất kỳ vọng quyết định chi phí vốn của doanh nghiệp, cách phân bổ tiết kiệm giữa các tài sản, và chuẩn để đánh giá chiến lược đầu tư và người quản lý danh mục. Vì vậy biết nguồn dự báo lợi suất nào còn bền vững là điều quan trọng cho cả nhà nghiên cứu kiểm định lý thuyết lẫn nhà đầu tư phân bổ vốn vào các chiến lược. (Sau đó giữ nguyên câu "Asset pricing theory explains…".)

## I-2 — Đoạn 2: dùng từ nhẹ hơn

**Giải thích:** thầy dặn tránh "fail" và từ mạnh. "Significantly elevates" và "fail to generalize" được đổi thành cách nói có chừng mực.

**❌ CŨ:**

```latex
significantly elevates the risk of detecting spurious, sample-specific patterns that fail to generalize out-of-sample
```

**✅ MỚI:**

```latex
increases the risk of detecting spurious, sample-specific patterns that may not generalize out of sample
```

*Tiếng Việt:* …làm tăng nguy cơ phát hiện các mẫu hình giả, chỉ đúng trong mẫu, có thể không tổng quát hoá ra ngoài mẫu.

## I-3 — Đoạn 2: sửa hai nguồn trích dẫn sai

**❌ CŨ (1):**

```latex
\cite{avramov2023machine, chu2020causal}
```

**✅ MỚI (1):**

```latex
\cite{mclean2016does, chu2020causal}
```

**❌ CŨ (2):**

```latex
as a high-throughput testing problem \cite{chen2024deep}
```

**✅ MỚI (2):**

```latex
as a high-throughput testing problem \cite{chen2025highthroughputassetpricing}
```

**Giải thích:** xem bảng A-4. Phản biện tài chính sẽ nhận ra ngay hai nguồn này không nói điều được gán cho chúng.

## I-4 — Đoạn 3 (FinCast): sửa thông tin và bỏ từ mạnh

**Giải thích:** (1) "major methodological milestone" là từ quảng cáo; ta chỉ thuật lại điều tác giả FinCast tự nhận. (2) Theo bài FinCast (mục 3.7.1), dữ liệu tiền huấn luyện là 20 tỷ điểm thời gian **thuộc cả lĩnh vực tài chính lẫn phi tài chính**, khớp mục 3.2 của bài. (3) "Directly mitigating", "inherently lacks", "no principled framework" là từ mạnh, được đổi thành cách nói trung lập.

**❌ CŨ (1):**

```latex
Among these advances, FinCast \cite{zhu2025fincast} marks a major methodological milestone as the first foundation model tailored specifically to financial time series.
```

**✅ MỚI (1):**

```latex
Among these advances, FinCast \cite{zhu2025fincast} is presented as the first foundation model designed specifically for financial time series.
```

**❌ CŨ (2):**

```latex
pretrained on over 20 billion financial observations across diverse asset classes including equities, cryptocurrencies, foreign exchange, and futures
```

**✅ MỚI (2):**

```latex
pretrained on over 20 billion time points across financial and non-financial domains, including cryptocurrencies, foreign exchange, futures and stocks,
```

**❌ CŨ (3):**

```latex
directly mitigating the temporal non-stationarity and structural breaks that typically confound conventional machine learning models
```

**✅ MỚI (3):**

```latex
which is intended to mitigate the temporal non-stationarity and structural breaks that complicate conventional machine learning models
```

**❌ CŨ (4):**

```latex
It inherently lacks a targeted mechanism to assess whether a signal's historical return pattern conveys genuine predictive power about its future performance, nor does it offer a principled framework to integrate
```

**✅ MỚI (4):**

```latex
It is therefore not designed to assess whether a signal's historical return pattern carries information about its future performance, and it does not by itself provide a way to integrate
```

*Tiếng Việt:* (1) FinCast được giới thiệu là foundation model đầu tiên thiết kế riêng cho chuỗi thời gian tài chính. (2) …được tiền huấn luyện trên hơn 20 tỷ điểm thời gian thuộc cả lĩnh vực tài chính và phi tài chính… (3) …nhằm giảm bớt tính phi dừng và các điểm gãy cấu trúc… (4) Vì vậy nó không được thiết kế để đánh giá liệu mẫu hình lợi suất quá khứ của một tín hiệu có chứa thông tin về hiệu quả tương lai không, và tự nó không cung cấp cách kết hợp…

## I-5 — Đoạn 4 và 5: từ nhẹ hơn

**❌ CŨ (1):**

```latex
summary numbers that compress an entire historical path into two figures and discard the sequential and regime-dependent information that path contains.
```

**✅ MỚI (1):**

```latex
summary numbers that compress an entire historical path into two figures and therefore do not retain the sequential and regime-dependent information that the path may contain.
```

**❌ CŨ (2):**

```latex
No prior study has examined whether fusing a pretrained
```

**✅ MỚI (2):**

```latex
To our knowledge, no prior study has examined whether fusing a pretrained
```

**Giải thích:** "discard" và "No prior study" là khẳng định tuyệt đối. Thầy dặn nêu hạn chế nhẹ nhàng.

## I-6 — Thêm đoạn "vì sao cần phương pháp mới" (chèn sau đoạn 5)

**Giải thích:** đây là ý chính thầy yêu cầu: chỉ ra phương pháp mới **cần** làm được gì, và mỗi yêu cầu dẫn tới một khối của kiến trúc. Nhờ vậy khi đọc đoạn "To address" (I-7), người đọc đã hiểu vì sao có từng khối.

**❌ CŨ** (câu cuối đoạn 5, giữ nguyên):

```latex
with conventional in-sample diagnostics improves out-of-sample forecasts of a signal's future return relative to either source of information alone.\\
```

**✅ MỚI** (giữ câu trên, thêm đoạn mới ngay sau, cách một dòng trống):

```latex
with conventional in-sample diagnostics improves out-of-sample forecasts of a signal's future return relative to either source of information alone.\\

These limitations suggest what a forecasting method for long-short strategies needs to provide, and each requirement motivates one part of our design. First, it should read the full return sequence rather than only its summary statistics, because two strategies with the same t-stat can earn their returns in very different ways; a pretrained encoder provides such a reading without fitting a large sequence model to noisy returns. Second, it should keep the number of trainable inputs small relative to the roughly 26,000 strategies observed each year, because monthly long-short returns are noisy and a high-dimensional input invites overfitting; this motivates compressing the pretrained representation before any trainable block. Third, it should learn which periods of a strategy's history matter for the forecasting task, which a generic pretrained representation does not do by itself; this motivates a trainable sequence reader with attention pooling. Fourth, it should compare strategies with each other, because a strategy is selected only if it looks attractive relative to the others of the same year; this motivates cross-sectional interaction and a training label defined relative to the year. Finally, it should build on, rather than replace, the t-stat, which already carries much of the information at the top of the ranking \cite{chen2025highthroughputassetpricing}.\\
```

*Tiếng Việt:* Những hạn chế này cho thấy một phương pháp dự báo cho chiến lược long–short cần làm được gì, và mỗi yêu cầu dẫn tới một phần trong thiết kế của ta. Một, nó nên đọc toàn bộ chuỗi lợi suất chứ không chỉ thống kê tóm tắt, vì hai chiến lược cùng t-stat có thể kiếm lợi suất theo cách rất khác nhau; bộ mã hoá tiền huấn luyện đọc được chuỗi mà không phải khớp một mô hình chuỗi lớn trên lợi suất nhiễu. Hai, số đầu vào cần học phải nhỏ so với khoảng 26.000 chiến lược mỗi năm, vì lợi suất tháng nhiễu và đầu vào nhiều chiều dễ gây overfitting; điều này dẫn tới việc nén biểu diễn trước mọi khối huấn luyện. Ba, nó cần học giai đoạn nào trong lịch sử quan trọng cho nhiệm vụ dự báo, điều mà biểu diễn tiền huấn luyện chung không tự làm; điều này dẫn tới khối đọc chuỗi có attention pooling. Bốn, nó cần so sánh các chiến lược với nhau, vì một chiến lược chỉ được chọn nếu hấp dẫn hơn các chiến lược khác cùng năm; điều này dẫn tới tương tác giữa các chiến lược và nhãn định nghĩa so với năm. Cuối cùng, nó nên dựa trên t-stat chứ không thay thế t-stat, vì t-stat đã chứa phần lớn thông tin ở đầu bảng xếp hạng.

## I-7 — Đoạn "To address this gap…": viết sâu hơn, mỗi khối có lý do

**Giải thích:** đoạn cũ mô tả các khối như danh sách, chưa nói vì sao, và hứa "economically interpretable attention weights" (bài không phân tích). Đoạn mới nối **mỗi khối với một yêu cầu** ở I-6, nói khối đó nhận gì và chuyển gì cho khối sau, và nêu lý do kích thước mô hình. Đoạn này bị tách thành nhiều dòng trong file, nên thay **cả đoạn**:

**❌ CŨ:** xoá từ `To address this gap, we propose an attention-augmented FinCast framework` (đầu đoạn) đến hết `drive that forecast.\\` (cuối đoạn, sau "…versus conventional statistics —").

**✅ MỚI:**

```latex
To address these requirements, we propose an attention-augmented FinCast framework in which each block addresses one of them and passes to the next block a representation suited to its role. A frozen, pretrained FinCast encoder reads the 128 months of returns before the forecasting year and produces four patch representations, one for each 32-month segment. Because these representations are high-dimensional relative to the number of strategies, a whitened principal component analysis (PCA), estimated only on data before the first forecasting year, compresses each patch to 16 components on a common scale. A trainable bidirectional LSTM then reads the four compressed patches in temporal order, and attention pooling weights them into one context vector; this is the task-specific step that the frozen encoder lacks. An induced set attention block (ISAB) \cite{lee2019set} then lets the context vectors of all strategies in the same year interact through a small number of learned summaries, at a cost that is linear in the number of strategies. Finally, the context vector is concatenated with the in-sample t-stat and mean return, and a small fully connected head forecasts the part of next-year performance that the t-stat does not already explain. The trainable part of the model is kept small (about 1.6 million parameters) on top of the frozen encoder, and the model is re-estimated every year using only earlier data. In this way, the framework can be interpreted as a signal-of-signals model: rather than pricing individual assets directly, it learns to forecast which long-short signals are likely to remain profitable.\\
```

*Tiếng Việt:* Để đáp ứng các yêu cầu này, ta đề xuất khung attention-augmented FinCast, trong đó mỗi khối giải quyết một yêu cầu và chuyển cho khối sau một biểu diễn phù hợp với vai trò của nó. Bộ mã hoá FinCast đóng băng đọc 128 tháng lợi suất trước năm dự báo và tạo bốn biểu diễn patch, mỗi patch cho một đoạn 32 tháng. Vì các biểu diễn này nhiều chiều so với số chiến lược, PCA trắng hoá, ước lượng chỉ bằng dữ liệu trước năm dự báo đầu tiên, nén mỗi patch còn 16 thành phần trên cùng thang đo. Bi-LSTM đọc bốn patch đã nén theo thứ tự thời gian, và attention pooling đặt trọng số để gộp chúng thành một vector ngữ cảnh; đây là bước gắn với nhiệm vụ mà bộ mã hoá đóng băng không có. Khối ISAB sau đó cho vector của mọi chiến lược cùng năm tương tác qua một số ít bản tóm tắt học được, với chi phí tuyến tính theo số chiến lược. Cuối cùng vector được ghép với t-stat và lãi trung bình trong mẫu, và một đầu fully connected nhỏ dự báo phần hiệu quả năm sau mà t-stat chưa giải thích. Phần huấn luyện được giữ nhỏ (khoảng 1,6 triệu tham số) trên nền bộ mã hoá đóng băng, và mô hình được học lại mỗi năm chỉ với dữ liệu trước đó. Theo cách này, khung có thể hiểu là một mô hình "tín hiệu của tín hiệu": thay vì định giá từng tài sản, nó học dự báo tín hiệu long–short nào có khả năng còn sinh lời.

## I-8 — Câu hỏi nghiên cứu

**Giải thích:** câu cũ hỏi về "economic interpretability", điều bài không đo.

**❌ CŨ:** cả câu bắt đầu bằng

```latex
Motivated by this gap, the present study addresses the following research question
```

(đến hết dấu `?` của câu).

**✅ MỚI:**

```latex
Motivated by this gap, the present study addresses the following research question: does the return history of a strategy, represented by a pretrained foundation model and combined with conventional in-sample diagnostics, improve the out-of-sample selection of long-short signals relative to empirical Bayes and naive mining?
```

*Tiếng Việt:* Câu hỏi nghiên cứu: lịch sử lợi suất của một chiến lược, được biểu diễn bởi mô hình nền tiền huấn luyện và kết hợp với các chẩn đoán trong mẫu thông thường, có cải thiện việc chọn tín hiệu long–short ngoài mẫu so với empirical Bayes và khai thác đơn giản không?

## I-9 — Đoạn đóng góp: bỏ "novel", sửa hai khẳng định không khớp kết quả

**❌ CŨ (1):**

```latex
we propose a novel attention-augmented FinCast–Bi-LSTM framework that adapts
```

**✅ MỚI (1):**

```latex
we propose an attention-augmented FinCast–Bi-LSTM framework that adapts
```

**❌ CŨ (2):**

```latex
demonstrating that the proposed model outperforms the benchmark, achieving Sharpe ratios of up to 1.95 for portfolios formed from the top 1\% and top 5\% of signals ranked by predicted returns. Ablation analyses further quantify the contribution of each component, including PCA compression and ISAB.
```

**✅ MỚI (2):**

```latex
demonstrating that the proposed model outperforms empirical Bayes mining in the top 1\% portfolio, with Sharpe ratios of 1.95 against 1.55 in the development period (1983--2004) and 1.11 against 0.82 in the holdout period (2005--2020). Ablation analyses show that the pretrained encoder, the PCA compression and the label construction contribute to the performance, while the contribution of ISAB cannot be distinguished from run-to-run variation.
```

**Giải thích:** "outperforms… top 5%" chưa đúng (ở top 5% không khác biệt có ý nghĩa, và naive mining cao hơn ở dev). "Ablation quantify… ISAB" mâu thuẫn với mục 4.3. Phản biện sẽ so ngay với bảng.

## I-10 — Đoạn lộ trình bài

**❌ CŨ:**

```latex
Section 3 describes the data and the proposed attention-augmented FinCast–Bi-LSTM methodology. Section 4 presents the empirical results, including out-of-sample forecasting performance and portfolio-level evaluations. Section 5 discusses the economic interpretation of the learned attention weights, robustness checks, and limitations.
```

**✅ MỚI:**

```latex
Section 3 formulates the problem and describes the proposed architecture and the rationale for each of its blocks. Section 4 presents the data, the experimental design, the forecast accuracy and the ablation study. Section 5 reports the portfolio results, the robustness to noise strategies and the temporal analysis.
```

---

# II. Mục 2 — Literature Review

## II-1 — Mục 2.1: thêm hai bài *Financial Innovation* (out-of-sample và anomaly)

**❌ CŨ:**

```latex
simply survived a data-mining search \cite{chen2023does} .
```

**✅ MỚI:**

```latex
simply survived a data-mining search \cite{chen2023does}. Even out-of-sample evaluation can be compromised when model choices draw on patterns that appear only in the test sample, which motivates evaluation designs that keep the test period unseen \cite{yae2023robust}. Evidence from the Chinese market further shows that the pricing of cross-sectional anomalies depends on conditioning information such as investor sentiment \cite{zhou2025cross}.
```

*Tiếng Việt:* …Ngay cả đánh giá ngoài mẫu cũng có thể bị sai lệch khi lựa chọn mô hình dựa vào các mẫu hình chỉ có trong tập kiểm tra, điều này đòi hỏi thiết kế đánh giá giữ kín giai đoạn kiểm tra (Yae & Luo). Bằng chứng ở thị trường Trung Quốc cho thấy việc định giá các anomaly mặt cắt ngang phụ thuộc vào thông tin điều kiện như tâm lý nhà đầu tư (Zhou và cộng sự).

**Giải thích:** câu đầu nối thẳng với cách bài giữ holdout 2005–2020 không nhìn thấy khi chọn mô hình. Bỏ luôn dấu cách thừa trước dấu chấm.

## II-2 — Mục 2.2: thêm hai bài *Financial Innovation* về ML dự báo lợi suất

**❌ CŨ:**

```latex
substantially outperform traditional linear benchmarks in out-of-sample return prediction.
```

**✅ MỚI:**

```latex
substantially outperform traditional linear benchmarks in out-of-sample return prediction. The gains depend on the setting: neural networks predict several measures of stock returns consistently from firm characteristics and improve further when macroeconomic and sentiment variables are added \cite{wang2024stock}, whereas moderately complex tree-based models perform best in the data-constrained Korean market \cite{kang2025how}.
```

*Tiếng Việt:* …Mức cải thiện phụ thuộc bối cảnh: mạng nơ-ron dự báo ổn định nhiều thước đo lợi suất từ đặc trưng doanh nghiệp và tốt hơn nữa khi thêm biến vĩ mô và tâm lý (Wang), còn ở thị trường Hàn Quốc ít dữ liệu thì mô hình cây độ phức tạp vừa phải cho kết quả tốt nhất (Kang và cộng sự).

**Giải thích:** câu "gains depend on the setting" dẫn tới lý do bài giữ phần huấn luyện nhỏ (I-6, yêu cầu 2).

## II-3 — Mục 2.2: thêm bài dự báo lợi suất **tương đối** (gần với bài toán chọn chiến lược)

**❌ CŨ:**

```latex
rather than learning which periods of that history are most informative for the prediction task at hand.\\
```

**✅ MỚI:**

```latex
rather than learning which periods of that history are most informative for the prediction task at hand. Related work also frames prediction relative to a benchmark, for example forecasting whether S\&P 500 stocks will outperform their index with random forests, support vector machines and LSTM networks \cite{htun2024forecasting}, which is closer to the selection problem studied here.\\
```

*Tiếng Việt:* …Một số nghiên cứu đặt bài toán so với một chuẩn, ví dụ dự báo cổ phiếu S&P 500 có vượt chỉ số hay không bằng random forest, SVM và LSTM (Htun và cộng sự), gần với bài toán chọn chiến lược ở đây.

## II-4 — Mục 2.4: sửa khẳng định về Guo và thêm hai bài về giảm chiều

**Giải thích:** bản tóm tắt của Guo và cộng sự nói việc giảm thuộc tính giảm độ phức tạp tính toán **mà không làm mất** độ chính xác tổng quát hoá, không nói "improves generalization". Hai bài *Financial Innovation* bổ sung cơ sở cho bước nén PCA của ta.

**❌ CŨ:**

```latex
reducing the input dimension before the attention-based predictor improves generalization and lowers computational cost.
```

**✅ MỚI:**

```latex
reducing the input dimension before the attention-based predictor lowers computational cost without sacrificing generalization accuracy. Feature selection and extraction are widely used for the same purpose in stock market prediction \cite{htun2023survey}, for example to reduce the inputs of LSTM and gradient boosting models with variational autoencoders \cite{gunduz2021efficient}.
```

*Tiếng Việt:* …giảm chiều đầu vào trước bộ dự báo dùng attention giúp giảm chi phí tính toán mà không làm mất độ chính xác tổng quát hoá. Chọn và trích xuất đặc trưng được dùng rộng rãi cho cùng mục đích trong dự báo thị trường chứng khoán (Htun và cộng sự), ví dụ giảm đầu vào của LSTM và gradient boosting bằng variational autoencoder (Gunduz).

## II-5 — Mục 2.3: sửa thông tin FinCast và câu bị lặp

**❌ CŨ (1):**

```latex
pretrained on more than 20 billion observations spanning equities, currencies, and futures
```

**✅ MỚI (1):**

```latex
pretrained on more than 20 billion time points spanning financial and non-financial domains
```

**❌ CŨ (2):**

```latex
remain disconnected from one another; the two sources of information remain disconnected from one another.
```

**✅ MỚI (2):**

```latex
remain disconnected from one another.
```

## II-6 — Mục 2.5 và 2.6: từ nhẹ hơn và lỗi dính chữ

**❌ CŨ (1):**

```latex
and has not incorporated learned, attention-weighted representations of a signal's full historical return path of the kind made possible by pretrained time-series foundation models.
```

**✅ MỚI (1):**

```latex
and, to our knowledge, has not used learned representations of a signal's full historical return path, such as those made possible by pretrained time-series foundation models.
```

**❌ CŨ (2):**

```latex
No existing framework adapts a pretrained foundation model's temporal representation
```

**✅ MỚI (2):**

```latex
To our knowledge, no existing framework adapts a pretrained foundation model's temporal representation
```

**❌ CŨ (3):**

```latex
No prior study has examined whether fusing such a representation
```

**✅ MỚI (3):**

```latex
To our knowledge, no prior study has examined whether fusing such a representation
```

**❌ CŨ (4):**

```latex
estimates its expected out-of-sample mean return.We compare
```

**✅ MỚI (4):**

```latex
estimates its expected out-of-sample mean return. We compare
```

## II-7 — Mục 2.3: từ mạnh "no principled mechanism"

**❌ CŨ:**

```latex
Second, they offer no principled mechanism for combining
```

**✅ MỚI:**

```latex
Second, they do not by themselves provide a mechanism for combining
```

*Tiếng Việt:* Hai, tự chúng không cung cấp cơ chế kết hợp…

## II-8 — Mục 2.6: cùng lỗi khẳng định quá mức về Guo và cộng sự

**Giải thích:** câu này lặp lại khẳng định "giảm chiều cải thiện tổng quát hoá" (xem II-4); bản tóm tắt chỉ nói giảm chi phí tính toán mà không mất độ chính xác.

**❌ CŨ:**

```latex
dimensionality reduction before attention has been shown to improve generalization in noisy financial data \citep{guo2026hybrid}
```

**✅ MỚI:**

```latex
dimensionality reduction before attention has been shown to lower computational cost without sacrificing generalization accuracy in noisy financial data \citep{guo2026hybrid}
```

*Tiếng Việt:* …giảm chiều trước attention đã được chứng minh giúp giảm chi phí tính toán mà không làm mất độ chính xác tổng quát hoá trên dữ liệu tài chính nhiễu.

---

# III. Mục 3 — Kiến trúc: vì sao các khối hợp lý và khớp nhau

## III-1 — Thêm đoạn và bảng "lý do thiết kế" vào cuối 3.2.1 Overall architecture

**Giải thích:** đây là phần thầy nhấn mạnh nhất. Đoạn mới đi theo **chuỗi**: mỗi khối giải quyết khó khăn do khối trước để lại, và đầu ra giữ dạng mà khối sau dùng được. Bảng tóm tắt mỗi khối giải quyết vấn đề gì, chuyển gì cho khối sau, và **bằng chứng** nằm ở đâu. Cột bằng chứng ghi trung thực cả chỗ ablation không phân biệt được.

**❌ CŨ** (câu cuối của 3.2.1, giữ nguyên):

```latex
or on other labels of the underlying assets.
```

**✅ MỚI** (giữ câu trên, thêm đoạn và bảng ngay sau, cách một dòng trống):

```latex
or on other labels of the underlying assets.

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

*Tiếng Việt (đoạn):* Các khối được sắp sao cho mỗi khối giải quyết một khó khăn do khối trước để lại và chuyển tiếp một biểu diễn mà khối sau dùng được (Bảng thiết kế). Bộ mã hoá tiền huấn luyện cho một mô tả phong phú về chuỗi lợi suất mà không phải khớp mô hình lớn trên lợi suất nhiễu, nhưng đầu ra nhiều chiều và không gắn với nhiệm vụ. PCA giảm đầu ra xuống cỡ mà khoảng 26.000 chiến lược mỗi năm đủ để học, và trắng hoá đưa các thành phần về cùng thang đo cho các khối huấn luyện. Các patch đã nén vẫn giữ thứ tự thời gian, nên Bi-LSTM đọc chúng như một chuỗi ngắn, và attention pooling gộp thành một vector bằng cách học đoạn nào quan trọng cho nhiệm vụ dự báo, đây là bước gắn với nhiệm vụ mà bộ mã hoá đóng băng không có. Chỉ sau khi mỗi chiến lược đã được tóm tắt thành vector gọn, ISAB mới so sánh các chiến lược trong năm, nên bước mặt cắt ngang này rẻ. Đầu dự đoán ghép vector đã học với t-stat và lãi trung bình, và nhãn yêu cầu mạng chỉ dự báo phần t-stat chưa giải thích, nên biểu diễn học được và chẩn đoán cổ điển bổ sung chứ không cạnh tranh nhau. Cuối cùng, Huber loss và trọng số năm điều chỉnh mục tiêu huấn luyện cho đuôi dày và mức thông tin không đều giữa các năm.

**Lưu ý:** bảng làm bảng của mục 4 và 5 lùi số thứ tự một bậc (LaTeX tự đánh số lại).

## III-2 — Cuối 3.2.2 FinCast: nối sang khối sau

**❌ CŨ:**

```latex
so no parameter of the encoder is fitted to our sample.
```

**✅ MỚI:**

```latex
so no parameter of the encoder is fitted to our sample. The resulting representation is rich but has 5,120 numbers per strategy and is not tailored to our task, which the next two blocks are designed to address: PCA reduces its size, and the trainable sequence block adapts it to the forecasting objective.
```

*Tiếng Việt:* …Biểu diễn thu được giàu thông tin nhưng có 5.120 số mỗi chiến lược và không gắn với nhiệm vụ của ta; hai khối tiếp theo được thiết kế để giải quyết điều này: PCA giảm kích thước, và khối chuỗi huấn luyện được thì thích nghi nó với mục tiêu dự báo.

## III-3 — Cuối 3.2.3 PCA: vì sao nén theo từng patch, và bằng chứng

**❌ CŨ:**

```latex
intended to reduce the complexity of the model and the risk of overfitting.
```

**✅ MỚI:**

```latex
intended to reduce the complexity of the model and the risk of overfitting. Because the compression is applied to each patch separately, the temporal order of the four patches is preserved, so the next block can still read them as a sequence. The ablation in Section \ref{subsec4_3} supports this choice: 8 components lower the holdout Sharpe ratio, and 64 components lower the development Sharpe ratio.
```

*Tiếng Việt:* …Vì nén riêng từng patch nên thứ tự thời gian của bốn patch được giữ, và khối sau vẫn đọc chúng như một chuỗi. Ablation ở mục 4.3 ủng hộ lựa chọn này: 8 thành phần làm giảm Sharpe holdout, 64 thành phần làm giảm Sharpe giai đoạn phát triển.

## III-4 — Sau công thức attention pooling (3.2.4): vì sao Bi-LSTM và attention

**❌ CŨ:**

```latex
and pool them into a 256-dimensional context vector $c_i = \sum_{j=1}^{4} a_{i,j}\, \tilde{h}_{i,j}$.
```

**✅ MỚI:**

```latex
and pool them into a 256-dimensional context vector $c_i = \sum_{j=1}^{4} a_{i,j}\, \tilde{h}_{i,j}$. A recurrent reader suits this input because it is short and ordered (four patches, from the oldest to the most recent), and attention pooling is the step that learns, for the forecasting task, which segments of the history matter; this is precisely what the frozen encoder does not provide.
```

*Tiếng Việt:* …Bộ đọc hồi quy phù hợp với đầu vào này vì nó ngắn và có thứ tự (bốn patch, từ cũ nhất đến gần nhất), và attention pooling là bước học, cho nhiệm vụ dự báo, đoạn nào của lịch sử là quan trọng; đó chính là điều bộ mã hoá đóng băng không cung cấp.

## III-5 — Cuối phần ISAB: vì sao đặt ISAB sau bước tóm tắt

**❌ CŨ:**

```latex
The output $c'_i$ keeps the dimension 256.
```

**✅ MỚI:**

```latex
The output $c'_i$ keeps the dimension 256. Placing ISAB after the per-strategy summary, rather than on the raw patches, means that the cross-sectional step operates on one compact vector per strategy, which keeps its cost linear in the number of strategies and lets it compare strategies on the features that the previous blocks have already selected.
```

*Tiếng Việt:* …Đặt ISAB sau bước tóm tắt từng chiến lược, thay vì trên các patch thô, giúp bước mặt cắt ngang làm việc trên một vector gọn cho mỗi chiến lược, nên chi phí tuyến tính theo số chiến lược và việc so sánh dựa trên các đặc trưng mà các khối trước đã chọn lọc.

## III-6 — Cuối phần đầu dự đoán: vì sao ghép t-stat và nhãn khớp với nhau

**❌ CŨ:**

```latex
and the final forecast follows from $\hat{y}_i = \hat{\sigma}_i(\hat{e}_i + b\, z_i + L_Y)$.
```

**✅ MỚI:**

```latex
and the final forecast follows from $\hat{y}_i = \hat{\sigma}_i(\hat{e}_i + b\, z_i + L_Y)$. This construction lets the learned representation and the classical diagnostics complement each other: the t-stat enters both as an input and through the baseline $b\, z_i$, so the network is designed to model the residual information in the return history rather than to relearn what the t-stat already captures.
```

*Tiếng Việt:* …Cách xây dựng này cho phép biểu diễn học được và các chẩn đoán cổ điển bổ sung nhau: t-stat vừa là đầu vào vừa đi qua nền $b\,z_i$, nên mạng được thiết kế để mô hình hoá phần thông tin còn lại trong lịch sử lợi suất thay vì học lại điều t-stat đã nắm.

---

# IV. Mục 4 — Thử nghiệm: nhấn mạnh kiến trúc

## IV-1 — Mở đầu 4.3 Ablation: ablation đi theo chuỗi kiến trúc

**❌ CŨ:**

```latex
We replace one component of the final configuration at a time and train each configuration three times, because retraining is not deterministic.
```

**✅ MỚI:**

```latex
Because the main contribution of this study is the architecture, the ablation follows its chain of blocks, from the input representation to the training objective. We replace one component of the final configuration at a time and train each configuration three times, because retraining is not deterministic.
```

*Tiếng Việt:* Vì đóng góp chính của nghiên cứu là kiến trúc, ablation đi theo chuỗi các khối, từ biểu diễn đầu vào đến mục tiêu huấn luyện. (Sau đó giữ nguyên câu cũ.)

## IV-2 — Thêm đoạn tổng hợp cuối 4.3

**Giải thích:** đọc kết quả ablation theo chuỗi kiến trúc, nói rõ cả chỗ chưa phân biệt được, và chỉ sang so sánh với baseline (đo hiệu ứng chung của thiết kế).

**❌ CŨ** (câu cuối 4.3, giữ nguyên):

```latex
since the summaries of its inducing points differ by only 2--3\% between years.
```

**✅ MỚI:**

```latex
since the summaries of its inducing points differ by only 2--3\% between years.

\textit{Taken together.} The ablation supports the design at the two ends of the chain. At the input end, the pretrained encoder and a compression size matched to the amount of data improve the holdout performance; at the output end, the label that compares strategies within a year gives the largest improvement in our experiments. The components in between, ISAB, the robust loss and the year weights, cannot be separated from run-to-run variation when removed one at a time, so their contribution, if any, may only appear jointly. The comparison with the baseline model in Section \ref{subsec5_1}, which shares the FinCast encoder and attention pooling but lacks the compression, the cross-strategy block, the sign orientation and the label, measures the joint effect of these design choices.
```

*Tiếng Việt:* **Tổng hợp.** Ablation ủng hộ thiết kế ở hai đầu của chuỗi. Ở đầu vào, bộ mã hoá tiền huấn luyện và kích thước nén vừa với lượng dữ liệu cải thiện hiệu quả holdout; ở đầu ra, nhãn so sánh các chiến lược trong cùng năm cho cải thiện lớn nhất. Các khối ở giữa (ISAB, loss bền vững, trọng số năm) không tách được khỏi dao động giữa các lần chạy khi bỏ từng khối, nên đóng góp của chúng, nếu có, có thể chỉ xuất hiện khi kết hợp. So sánh với mô hình baseline ở mục 5.1, vốn dùng chung FinCast và attention pooling nhưng không có nén, khối giữa các chiến lược, đảo dấu và nhãn, đo hiệu ứng chung của các lựa chọn thiết kế này.

---

# V. Mục 5 — Kết quả: gắn kết quả với kiến trúc

## V-1 — 5.1.1: nói rõ baseline thiếu những khối nào

**❌ CŨ:**

```latex
so the improvement of the proposed model comes from the design of Section \ref{subsec3} and not from the FinCast representations alone
```

**✅ MỚI:**

```latex
so the improvement of the proposed model comes from the design of Section \ref{subsec3} and not from the FinCast representations alone: the baseline model uses the same encoder and attention pooling, but it does not compress the representation, does not let strategies interact, does not orient the signs and is trained on raw mean returns rather than on the label of Equation \eqref{eq:label}
```

*Tiếng Việt:* …nên phần cải thiện đến từ thiết kế ở mục 3.2 chứ không chỉ từ biểu diễn FinCast: baseline dùng cùng bộ mã hoá và attention pooling, nhưng không nén biểu diễn, không cho các chiến lược tương tác, không đảo dấu và học trên lãi trung bình thô thay vì nhãn ở mục 3.1.

## V-2 — 5.1.3: nối phân tích "nơi hai phương pháp bất đồng" với kiến trúc

**❌ CŨ:**

```latex
This suggests that the return history carries information about future performance that the summary statistics used by EB mining do not.
```

**✅ MỚI:**

```latex
This suggests that the return history carries information about future performance that the summary statistics used by EB mining do not. These are the strategies on which the ranking of the proposed architecture, which uses the return history in addition to the summary statistics, departs from a ranking by summary statistics alone, and their realized returns support these departures.
```

*Tiếng Việt:* …Đây là những chiến lược mà cách xếp hạng của kiến trúc đề xuất, vốn dùng thêm lịch sử lợi suất ngoài thống kê tóm tắt, khác với cách xếp hạng chỉ bằng thống kê tóm tắt, và lợi suất thực tế ủng hộ những khác biệt này.

---

# VI. Ngoài yêu cầu nhưng **sai**: mục 6 và Abstract (nên sửa trước khi nộp)

| Mã | ❌ CŨ | ✅ MỚI | Vì sao |
|---|---|---|---|
| VI-1 | `while the Induced Set Attention Block (ISAB) aggregates relevant information into a compact context vector.` | `while attention pooling aggregates the temporal segments into a compact context vector and the Induced Set Attention Block (ISAB) lets the strategies of the same year interact.` | Gộp thành vector ngữ cảnh là việc của attention pooling; ISAB làm tương tác giữa các chiến lược |
| VI-2 | `leveraging their complementary strengths for stock return prediction.` | `leveraging their complementary strengths to forecast the returns of long-short strategies.` | Bài dự báo lợi suất chiến lược, không phải cổ phiếu |
| VI-3 | `It outperforms benchmark methods for the top 1\% and top 5\% of strategies across both periods and the full sample, as well as for the top 10\% during the holdout period, demonstrating its potential for out-of-sample return prediction.` | `Its top 1\% portfolio has a higher Sharpe ratio than that of empirical Bayes mining in the development, holdout and full periods, with statistically significant differences, and it has the highest holdout Sharpe ratio of the compared methods at all three portfolio sizes.` | Ở top 5% dev, naive mining cao hơn (1,99 so với 1,95) và mọi chênh lệch ở top 5% không có ý nghĩa thống kê |
| VI-4 | `by capturing nonlinear relationships among market factors and adapting to changing market dynamics.` | `by reading the return histories of strategies and being re-estimated every year as market conditions change.` | Bài không mô hình hoá "market factors" |
| VI-5 (Abstract) | `Furthermore, the proposed model outperforms the benchmark methods, achieving Sharpe ratios of up to 1.95 among the top 1\% and top 5\% of strategies.` | `Furthermore, the top 1\% portfolio of the proposed model has a higher Sharpe ratio than that of empirical Bayes mining in both the development period (1.95 against 1.55) and the holdout period (1.11 against 0.82).` | Như VI-3 |


**Cấu trúc mục 6 (nên xem lại):** hiện có ba `\section` liền nhau: `Conclusion`, `Practical implications`, `Limitations and directions for future research`. Mục `Practical implications` thực chất chứa kết quả chính và đóng góp, không phải ý nghĩa thực tiễn. Cách gọn nhất: đổi hai dòng `\section{Practical implications}` và `\section{Limitations and directions for future research}` thành `\subsection{Key findings and contributions}` và `\subsection{Limitations and future research}`, để cả phần nằm trong mục 6 Conclusion. Mục 5.4 *Discussion and limitations* đang bị comment, nên phần giới hạn chỉ còn ở mục 6; như vậy là đủ, không cần mở lại.

---

# Thứ tự làm và kiểm tra

1. **Làm A-1 trước tiên** (thêm `}` trong `.bib`), rồi A-2, A-3, rồi dán BibTeX ở phần C. Recompile và kiểm tra không còn "?" ở các trích dẫn Chen & Dim.
2. Làm mục I (Intro), recompile.
3. Làm mục II (Literature Review), recompile.
4. Làm mục III (Kiến trúc, có bảng mới), recompile; kiểm tra bảng hiển thị gọn trong trang.
5. Làm mục IV, V, VI, recompile.
6. Tìm trong toàn bài các chữ: `fail`, `novel`, `interpretab`, `chen2024deep` (trong Intro), `stock return prediction` (trong Kết luận). Không được còn chữ nào ở các chỗ đã sửa.

Mình chưa sửa gì trong `overleaf_project/` hay trên Overleaf. Bạn paste xong thì báo, mình tải lại để đối chiếu từng chỗ.
