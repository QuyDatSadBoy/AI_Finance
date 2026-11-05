# Kiểm tra cuối bản Overleaf (tải 05/11/2026) — việc còn lại trước khi nộp

Cách làm như file `update_theo_nhan_xet_PASTE.md`: 🔁 **THAY** = Ctrl+F đoạn 🔴 GỐC, bôi đen, dán 🟢 MỚI đè lên. Mọi đoạn GỐC dưới đây đã kiểm tra xuất hiện **đúng 1 lần** trong phần chữ đang chạy (không tính dòng bắt đầu bằng `%`).

**Đã kiểm tra và ổn:** 45 trong 49 chỗ sửa trước đã vào; `.bib` không còn khoá trùng, mọi `\cite` đều có entry, mọi `\ref` đều có nhãn; 58 tài liệu trích dẫn đều có thật; **mọi số ở Bảng 1–3 (Sharpe, p so với EB và naive mining) đã tính lại từ dự báo đã lưu và khớp từng chữ số.**

---

# A. LỖI THẬT (làm trước)

## A-1 · 🔁 Câu bị mất ở cuối 4.3 "Other components" (lỗi do paste sửa 41)

Khi paste sửa 41, câu cuối của 4.3 bị xoá, nên câu về ISAB bị cụt ("...of each strategy, **\textit{Taken together.}**"). Phải trả lại câu và xuống dòng. (Sửa 41 là kiểu ➕ THÊM: **giữ nguyên** câu GỐC, chỉ thêm đoạn mới sau nó.)

🔴 **GỐC:**
```latex
acts mainly as an additional transformation of each strategy, \textit{Taken together.}
```
🟢 **MỚI** (có dòng trống trước `\textit{Taken together.}`):
```latex
acts mainly as an additional transformation of each strategy, since the summaries of its inducing points differ by only 2--3\% between years.

\textit{Taken together.}
```

## A-2 · 🔁 Câu bị lặp trong mục 2.3 (sửa 25 chưa vào)

🔴 **GỐC:**
```latex
the two sources of information remain disconnected from one another; the two sources of information remain disconnected from one another.
```
🟢 **MỚI:**
```latex
the two sources of information remain disconnected from one another.
```

## A-3 · 🔁 Abstract trích dẫn tài liệu (Springer cấm)

Hướng dẫn của *Financial Innovation*: "do not cite references in the abstract". Abstract đang có `\cite`.

🔴 **GỐC:**
```latex
Building on the high-throughput asset pricing framework of Chen and Dim \cite{chen2025highthroughputassetpricing} , this study
```
🟢 **MỚI:**
```latex
Building on the high-throughput asset pricing framework of Chen and Dim (2025), this study
```

## A-4 · 🔁 Abstract nói điều bài không kiểm tra

Bài chỉ huấn luyện và đánh giá họ **kế toán EW**, và dùng Ticker làm phép kiểm tra nhiễu. Câu "predictability is concentrated in equal-weighted accounting-based strategies … their in-sample success reflects luck" là phát hiện của Chen & Dim, không phải của bài này.

🔴 **GỐC:**
```latex
Consistent with the empirical Bayes benchmark, we find that predictability is concentrated in equal-weighted accounting-based strategies, whereas ticker-based strategies show no reliable out-of-sample performance, indicating that their in-sample success reflects luck.
```
🟢 **MỚI:**
```latex
Focusing on equal-weighted accounting strategies, we find that the model almost never selects ticker-based strategies, a placebo family with no economic content, which indicates that it does not mistake noise for signal.
```
📖 Tập trung vào họ kế toán chia đều, mô hình gần như không chọn chiến lược theo ticker (họ đối chứng không có ý nghĩa kinh tế), nghĩa là nó không nhầm nhiễu với tín hiệu.

## A-5 · Phần mẫu (template) còn nguyên, **bắt buộc thay trước khi nộp**

| Chỗ | Đang là | Cần |
|---|---|---|
| Tác giả (dòng 106, 108) | `iauthor@gmail.com`, `iiauthor@gmail.com` | email thật của tác giả |
| Địa chỉ cơ quan (dòng 114–118) | `Department, Organization, Street, City, State, Country` | tên khoa, trường, địa chỉ thật |
| Cuối bài (dòng 1191–1256) | Acknowledgements mẫu, Declarations mẫu, khối "Editorial Policies for Nature/BMC", Appendix mẫu | **xoá hết** và thay bằng khối A-6 |

## A-6 · Declarations theo yêu cầu của *Financial Innovation*

Tạp chí yêu cầu đúng 5 mục dưới tiêu đề `Declarations`: Availability of data and materials, Competing interests, Funding, Authors' contributions, Acknowledgements (mục nào không có thì ghi "Not applicable"). Cần có thêm **List of abbreviations** và, ở hệ thống nộp bài, **Short Bio** của mọi tác giả.

**Xoá** từ `\bmhead{Acknowledgements}` (dòng ~1191) đến hết `\end{appendices}` (dòng ~1256), rồi dán khối dưới. Chỗ nào có `[...]` thì chỉ bạn điền được.

```latex
\section*{List of abbreviations}
BN: batch normalization; Bi-LSTM: bidirectional long short-term memory; EB: empirical Bayes; HAC: heteroskedasticity and autocorrelation consistent; IC: information coefficient; ISAB: induced set attention block; LN: layer normalization; MoE: mixture of experts; PCA: principal component analysis.

\section*{Declarations}

\bmhead{Availability of data and materials}
The long-short strategy returns analysed in this study are the public data of Chen and Dim, available at \url{https://sites.google.com/site/chenandrewy/Data}. The code, the trained weights of the final model and the forecasts supporting the conclusions of this article are available in the [NAME] repository, [PERSISTENT WEB LINK OR DOI].

\bmhead{Competing interests}
The authors declare that they have no competing interests.

\bmhead{Funding}
[Funding sources, or ``This research received no specific funding.'']

\bmhead{Authors' contributions}
[Use the authors' initials, for example: NDH designed the study, QDT implemented the model and ran the experiments, TNPL supervised the work. All authors read and approved the final manuscript.]

\bmhead{Acknowledgements}
Not applicable.
```

⚠️ Repo GitHub hiện là **private**. Câu "available in the repository" chỉ đúng khi repo public hoặc đã lưu lên Zenodo (có DOI).
⚠️ Chính sách của Springer Nature: dùng công cụ AI (LLM) ngoài mức sửa câu chữ thì phải **ghi rõ trong phần Methods**. Bạn đọc chính sách của tạp chí và khai báo đúng thực tế.

---

# B. Lỗi câu chữ nhỏ (sửa nhanh)

| # | 🔴 GỐC | 🟢 MỚI | Vì sao |
|---|---|---|---|
| B-1 | `long-short portfolio \cite{harvey2016and} summary numbers` | `long-short portfolio \cite{harvey2016and}, which are summary numbers` | câu thiếu dấu nối, đọc cụt |
| B-2 | `however large the associated t-statistic are, on their own,` | `however large the associated t-statistic, are, on their own,` | thiếu dấu phẩy đóng cụm xen |
| B-3 | `in noisy financial data \citep{guo2026hybrid}` + xuống dòng + `. Existing` | `in noisy financial data \citep{guo2026hybrid}. Existing` (nối thành 1 dòng) | dòng mới làm dấu chấm đứng một mình |
| B-4 | `\cite{chen2025highthroughputassetpricing} yet this literature` | `\cite{chen2025highthroughputassetpricing}, yet this literature` | thiếu dấu phẩy |
| B-5 | `factor zoo\cite{feng2020taming}` | `factor zoo \cite{feng2020taming}` | thiếu dấu cách |
| B-6 | `markets\cite{jensen2023there}` | `markets \cite{jensen2023there}` | thiếu dấu cách |
| B-7 | `\citep{zhu2025fincast} introduce FinCast` | `\citet{zhu2025fincast} introduce FinCast` | `\citep` đặt làm chủ ngữ phải là `\citet` |

---

# C. Nhất quán với kết quả (đã đối chiếu với dữ liệu)

## C-1 · 🔁 Số chiến lược mỗi năm: 25.600 hay 26.000

Chữ "roughly 26,000" dùng ở mục 1, 3; mục 4.1 viết "about 25,600". Số trung bình thật là 25.600.

🔴 **GỐC:**
```latex
which leaves about 25,600 strategies per year
```
🟢 **MỚI:**
```latex
which leaves about 26,000 strategies per year (25,600 on average)
```

## C-2 · 🔁 Mục 5.2.2: "cao nhất" mà thực ra gần ngang naive mining ở dev

Pool, dev top 1%: mô hình 2,02, naive 2,01.

🔴 **GỐC:**
```latex
The proposed model has the highest top 1\% Sharpe ratio on the pool in both periods (2.02 and 1.05).
```
🟢 **MỚI:**
```latex
The proposed model has the highest top 1\% Sharpe ratio on the pool in both periods (2.02 and 1.05), although in the development period it is only 0.01 above naive mining (2.01).
```

## C-3 · 🔁 Mục 4.3 PCA: "both fall" vượt quá bằng chứng

PCA 64 thành phần: dev giảm 0,20 (rõ), holdout giảm 0,10 (nhỏ hơn dao động 0,13). Chính bài đặt tiêu chuẩn "chỉ coi là có ý nghĩa khi vượt dao động".

🔴 **GỐC:**
```latex
and with 64 components both fall.
```
🟢 **MỚI:**
```latex
and with 64 components the development one falls ($-0.20$), while the holdout one falls ($-0.10$) by less than the run-to-run variation.
```

## C-4 · Kết luận (mục 6) nói quá so với kết quả

| 🔴 GỐC | 🟢 MỚI | Vì sao |
|---|---|---|
| `Together, these components enable the model to learn complex patterns from historical strategy performance and improve out-of-sample return prediction.` | `Together, these components enable the model to learn patterns in the historical performance of strategies and, as the results show, to select strategies with a higher out-of-sample Sharpe ratio at the top of the ranking.` | EB dự báo **độ lớn** lợi suất chính xác hơn (calibration 0,88 so với 0,79), nên không thể nói mô hình "improve return prediction" nói chung |
| `the proposed method demonstrates the potential of deep learning for return prediction by` | `the proposed method demonstrates the potential of deep learning for selecting long-short strategies by` | bài là chọn chiến lược, không phải dự báo lợi suất cổ phiếu |

## C-5 · 🔁 Mục Limitations: thiếu chi phí giao dịch

Chọn **kết quả không bắt đầu bằng `%`** (cụm này cũng nằm trong một dòng đã comment).

🔴 **GỐC:**
```latex
could provide stronger evidence of their respective contributions.
```
🟢 **MỚI:**
```latex
could provide stronger evidence of their respective contributions. In addition, returns are measured before transaction costs, so whether the advantage survives trading frictions remains to be tested.
```

---

# D. Phần thầy yêu cầu mà bản hiện tại chưa đáp ứng đủ

**Đoạn "To address this gap" (dòng ~162) chưa sửa** (sửa 16). Thầy yêu cầu đoạn này sâu hơn, không chỉ quảng bá. Đoạn đang có:
- vẫn mở đầu "To address this gap" ngay sau đoạn mới liệt kê 5 yêu cầu (sửa 15);
- vẫn hứa "an attention mechanism whose learned weights offer an **economically interpretable** account", điều bài không phân tích và mâu thuẫn với câu hỏi nghiên cứu mới.

**Cách tốt nhất:** làm đúng sửa 16 trong `update_theo_nhan_xet_PASTE.md` (thay cả đoạn, mỗi khối gắn với một yêu cầu).
**Cách tối thiểu** (giữ phần lớn đoạn cũ, bỏ mâu thuẫn):

🔁 **(a)** 🔴 GỐC:
```latex
To address this gap, we propose an attention-augmented FinCast framework that forecasts
```
🟢 MỚI:
```latex
To address these requirements, we propose an attention-augmented FinCast framework that forecasts
```

🔁 **(b)** 🔴 GỐC:
```latex
likely to remain informative going forward, using an attention mechanism whose learned weights offer an economically interpretable account of which periods of a signal's history — and which combination of learned temporal features versus conventional statistics — drive that forecast.\\
```
🟢 MỚI:
```latex
likely to remain informative going forward.\\
```

---

# E. Việc cần làm khi nộp (không sửa bằng Ctrl+F)

1. **Biên dịch lại trên Overleaf** và xem PDF: không còn `?`; **Bảng "Design rationale"** (4 cột `p{}` tổng ~12,4 cm) nằm gọn trong trang; hình `final_model.png` và `cumret_top1.png` hiển thị. PDF `sn-article.pdf` trong bản tải về là bản cũ (12/2024), mình không thể dùng để kiểm tra giao diện.
2. Tạp chí yêu cầu **giãn dòng đôi và đánh số dòng**. Dòng `\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}` hiện chưa có hai tuỳ chọn đó; thêm `referee` (giãn dòng đôi) và `lineno` vào danh sách tuỳ chọn khi nộp.
3. **Short Bio** của mọi tác giả (nhập ở hệ thống nộp bài, theo mẫu title page của tạp chí).
4. Các chi tiết bạn tự viết ở mục 2.4 về `guo2026hybrid` ("three Chinese stocks", các mô hình so sánh) và `ma2023attention` ("72 firm characteristics", "SHAP"): mình không đọc được toàn văn (thu phí). Bạn đối chiếu với bài gốc.
5. Mục 6 hiện có 3 `\section` liền nhau (sửa 48, 49 bạn chưa làm): **chấp nhận được**, chỉ là cấu trúc hơi rời.
