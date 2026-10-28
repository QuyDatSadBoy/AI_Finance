# Hướng dẫn viết paper (đọc trước khi viết bất kỳ mục nào)

Áp dụng cho mọi agent và người viết bài trong `docs/writing_paper/`. Bài đang viết: `sn-article.tex` (mẫu Springer Nature). Người dùng chép nội dung vào Overleaf, nên **không sửa `sn-article.tex`** trừ khi được yêu cầu. Mỗi mục viết ra một file `<số mục>_<tên>_PASTE.md`.

## 0. Quy tắc bắt buộc

1. **Đọc paper gốc trước khi viết:** `ChenDim2023_HighThroughputAssetPricing_2311.10685.pdf` (cùng thư mục; bản đầy đủ ở `docs/papers/`). Paper này đã được chấp nhận, nên cách họ dùng từ, lập luận và hiểu về tài chính là chuẩn để học. Đọc kỹ phần mình sắp viết (Introduction, mục 2 Data and Methods, mục 3 Performance) rồi mới viết. Không viết từ trí nhớ.
2. **Học cách viết, không chép.** Học thuật ngữ, cấu trúc lập luận, giọng văn. Không sao chép câu.
3. **Bám bản nháp của người dùng** trong `sn-article.tex`: giữ câu đã tốt, chỉ sửa chỗ sai, thiếu chiều sâu hoặc không đúng mô hình chốt. Không viết lại hoàn toàn khi chưa được yêu cầu.
4. **Khớp mô hình chốt:** số liệu, ký hiệu, kiến trúc lấy từ mã nguồn (`code/model.py`, `train.py`, `data.py`) và `docs/5_FINAL_MODEL_VS_BASELINE.md`. Không bịa số. Không đưa kết quả chưa có.
5. **Mỗi file PASTE.md gồm:** (a) hướng dẫn chép vào đâu, (b) khối LaTeX tiếng Anh để paste, (c) bản tiếng Việt dịch sát để người dùng đọc hiểu, (d) BibTeX cần thêm, (e) danh sách cần kiểm tra. Trả lời người dùng bằng tiếng Việt.
6. **Trích dẫn phải xác minh** (MCP `arxiv`, `paper-search`, hoặc PDF trong `docs/papers/`). Chỉ dùng key có trong `.bib` của bài. Nếu thêm nguồn mới, đưa BibTeX vào file PASTE.md và nói rõ chỗ nào chưa xác minh.

## 1. Giọng văn: tự tin về phương pháp, chính xác về đóng góp, thận trọng về kết luận

Phong cách học thuật, khách quan, chính xác, khiêm tốn, dựa trên bằng chứng (NeurIPS, ICML, ICLR, IEEE).

### Đại từ và chủ ngữ

| Dùng | Khi nào |
|---|---|
| **We** | Điều nhóm nghiên cứu làm, đề xuất, chứng minh. Đại từ chính, không né bằng câu bị động liên tục |
| The proposed model / framework | Mô tả kiến trúc, cơ chế |
| This study / This work | Mục tiêu, phạm vi, kết quả |
| The results / Our findings | Phân tích kết quả thực nghiệm |

### Nên dùng và hạn chế

| Nên dùng | Hạn chế |
|---|---|
| We propose / We investigate / We aim to | We invent a revolutionary… / We prove that our model is the best… |
| We hypothesize that… | It is obvious that… |
| The results suggest / indicate… | The results undoubtedly prove… |
| This may improve… | This always improves… |
| A possible explanation is… | The only explanation is… |

### Nguyên tắc cho paper này

1. **Phân biệt đóng góp mới với thành phần kế thừa.** FinCast, PCA, ISAB, Bi-LSTM, Huber, AdamW là thành phần có sẵn: viết *We employ / adopt / integrate / apply*. Chỉ viết *We propose / introduce* cho thiết kế thật sự mới (ví dụ nhãn "Sharpe − mức năm − nền |t|", cách ghép các khối, quy trình walk-forward cho bài toán này).
2. **Không khẳng định quá mức.** Không viết "PCA removes noise". Viết "PCA is employed to reduce dimensionality and potentially mitigate redundant variation" hoặc "with the aim of…".
3. **Không nhận xét tiêu cực về nghiên cứu trước.** Không viết "EB fails to…". Viết "EB relies on summary statistics and therefore does not explicitly model the full temporal dynamics of return sequences."
4. **Methodology dùng ngôn ngữ thiết kế, Results mới dùng ngôn ngữ kết quả.**
   - Methodology: *is designed to, aims to, enables the model to, is intended to, may*.
   - Results: *achieves, improves, outperforms* chỉ khi có số liệu kèm theo.
5. **Giả thuyết khác kết quả.** Lý do thiết kế chỉ được viết là động cơ thiết kế hoặc giả thuyết. Một lý do chỉ được viết thành kết quả khi có thí nghiệm bỏ khối đó cho chênh lệch vượt dao động giữa các lần học lại (xem `4_experiments_PASTE.md`, mục 4.3: hiện có FinCast pretrained, PCA và cách xây nhãn; ISAB, Huber, trọng số năm, nền |t| chưa chứng minh được).
6. **Thuật ngữ nhất quán** (xem bảng ở mục 3). Chọn một từ và giữ xuyên suốt.
7. **Thừa nhận giới hạn một cách khoa học:** điều kiện áp dụng, hạn chế, kết luận chưa kiểm chứng. Không tự hạ thấp đóng góp.
8. **Tránh từ quảng cáo:** groundbreaking, revolutionary, extremely powerful, perfect, obviously, state-of-the-art (khi chưa có cơ sở).

### Văn phong theo từng phần

| Phần | Mẫu |
|---|---|
| Introduction | We investigate… / We aim to address… |
| Related Work | Prior studies have explored… / However, limited attention has been given to… |
| Problem Formulation | Let … denote … / We formulate the task as… |
| Proposed Model | We adopt… / We integrate… / The module computes… |
| Experiments | We evaluate… / We compare our approach against… |
| Results & Discussion | The results indicate… / One possible explanation is… |
| Limitations | Our approach has several limitations… |
| Conclusion | This study investigated… / Future work may explore… |

### Ví dụ

- Không nên: *Our novel model completely eliminates noise and significantly outperforms all existing methods.*
- Nên: *We employ PCA to reduce the dimensionality of the extracted representations, with the aim of mitigating redundancy and improving generalization.*

## 2. Học cách hiểu và viết về tài chính từ Chen & Dim

Đây là phần quan trọng nhất: bài của ta được đọc bởi người làm tài chính thực nghiệm, nên phải nói đúng ngôn ngữ và đúng cách nghĩ của họ.

### Cách họ nghĩ

1. **Khai thác dữ liệu có thể làm cho an toàn.** Họ gọi là *rigorous data mining*, *atheoretical* (không dùng lý thuyết kinh tế để chọn mô hình), *high-throughput asset pricing*. Họ thừa nhận thiên lệch rồi mới sửa: chọn chiến lược có $r_i$ lớn cũng chọn luôn sai số lấy mẫu $\varepsilon_i$ lớn, nên $E(r_i \mid r_i = \bar r)$ cao hơn hiệu suất thật $\mu_i$. Họ tách $r_i = \mu_i + \varepsilon_i$ ("actual performance" và "sampling error or luck").
2. **Look-ahead bias là vấn đề trung tâm.** Chiến lược đã công bố được chọn khi tác giả biết các mẫu hình đã xảy ra trong mẫu; chiến lược khai thác dữ liệu có thể xây chỉ bằng thông tin có sẵn *in real time*.
3. **Đánh giá bằng danh mục ngoài mẫu đơn giản:** mỗi năm đảo dấu chiến lược theo dự báo, xếp theo *predicted Sharpe ratios*, lập danh mục chia đều top 1%, 5%, 10%. Báo lợi suất %/năm, t-stat, Sharpe; so với *published strategies*; vẽ giá trị 1 USD đầu tư qua thời gian.
4. **Predictability suy giảm.** Lợi suất tập trung trước 2004, giải thích bằng sự phát triển của công nghệ thông tin (Chordia, Subrahmanyam, Tong 2014) hơn là việc nhà đầu tư học từ paper (McLean & Pontiff 2016). Họ nêu giới hạn của chính mình ("with an important caveat") và nói rõ EB dùng cửa sổ trượt 20 năm nên không tính đến điểm gãy cấu trúc.
5. **Giải thích kinh tế phía sau số liệu:** chiến lược kế toán chia đều chiếm ~91% top 1%; dự báo tập trung ở cổ phiếu nhỏ, trước 2004, phù hợp với *limited attention* và *slow diffusion of information* (Peng 2005). Chiến lược dựa trên ticker có t-stat gần phân phối chuẩn nên không có dự báo được.
6. **Giải thích bằng trực giác trước, công thức sau.** "In essence, EB measures the distance between the empirical t-stat distribution and the standard normal null."
7. **Chuẩn so sánh trung thực:** luôn đặt cạnh một mốc đơn giản (*naive mining*, xếp theo t-stat) và một mốc học thuật (*published strategies*).

### Cách họ viết

- **Mở bằng câu hỏi rồi trả lời:** "Can data mining generate out-of-sample returns? To answer this question we construct simple out-of-sample portfolio tests."
- **Nêu vấn đề ("This practice is dangerous because…") trước, giải pháp sau.**
- **Mỗi lựa chọn kèm lý do ngắn:** "Mixture normals are parsimonious, easy to understand, and yet allow for skewness and fat tails."
- **Câu ngắn, chủ động, số cụ thể:** "It earns out-of-sample returns of 5.7% per year over the 1983-2020 sample, compared to…"
- **Từ nối:** *Thus, Indeed, In essence, In contrast, Unlike, Taken together, Moreover*.
- **Dẫn bảng/hình bằng câu ngắn:** "Table 2 shows the result." "Figure 1 takes a closer look by plotting…"
- **Nêu giới hạn ngay trong văn bản:** "The caveat is that returns are concentrated in the pre-2004 sample."
- **Nhấn mạnh ý nghĩa thống kê bằng từ có chừng mực:** "within 2 standard errors", "extremely statistically significant" chỉ khi có số.

## 3. Bảng thuật ngữ cố định cho bài này

| Dùng | Ý nghĩa | Tránh trộn với |
|---|---|---|
| strategy / signal | một luật long–short duy nhất (`signalid`); dùng như nhau, đã khai báo ở 3.1 | portfolio, factor (trừ khi nói đúng nghĩa) |
| long-short return | lợi suất tháng của chiến lược | profit, gain |
| in-sample (IS) / out-of-sample (OOS) | cửa sổ đầu vào / năm dự báo | training / test (dành cho nói về mô hình) |
| forecasting year $Y$ | năm cần dự báo, 1983–2020 | test year |
| mean (monthly) return | lợi suất trung bình tháng của năm $Y$ | average profit |
| t-stat, Sharpe ratio, predicted Sharpe ratio | như paper gốc | t-value |
| top X% portfolio | danh mục chia đều top X% chiến lược theo Sharpe dự báo | top-k |
| look-ahead bias, real time | như paper gốc | data leakage (chỉ dùng khi nói về mô hình) |
| empirical Bayes (EB), naive mining | các chuẩn so sánh | baseline (dành cho mô hình baseline của ta) |
| equal-weighted accounting strategies | họ chiến lược đang học | Accounting EW (viết tắt chỉ dùng trong bảng) |
| pretrained / frozen encoder | FinCast | fine-tuned (ta không tinh chỉnh) |
| cross-strategy interaction | ISAB: các chiến lược cùng năm tương tác | attention giữa các mẫu |
| walk-forward, re-estimated every year | học lại mỗi năm | rolling (EB của paper dùng cửa sổ trượt 20 năm) |

## 4. Kiểm tra trước khi giao một mục

- [ ] Đã đọc phần liên quan của paper gốc và dùng đúng từ vựng ở mục 2–3.
- [ ] Ký hiệu, số liệu khớp với mã nguồn và `docs/5_FINAL_MODEL_VS_BASELINE.md`.
- [ ] Thành phần kế thừa dùng *employ/adopt/integrate*; không dùng *propose* cho thứ không mới.
- [ ] Methodology chỉ dùng ngôn ngữ thiết kế (*is designed to, aims to, may*), không tuyên bố hiệu quả.
- [ ] Nhận xét về EB trung lập (chỉ nêu điều EB dùng và không dùng).
- [ ] Mọi `\cite` có trong `.bib` hoặc có BibTeX đã xác minh kèm theo.
- [ ] Có bản tiếng Việt dịch sát và danh sách cần kiểm tra.
- [ ] Không có từ quảng cáo, không có kết luận mà dữ liệu chưa cho phép.

## 5. Tình trạng

| Mục | File | Trạng thái |
|---|---|---|
| 3.1 Problem formulation | `3.1_problem_formulation_PASTE.md` | đã viết, người dùng sẽ paste |
| 3.2 Proposed model | `3.2_proposed_model_PASTE.md` | đã viết, người dùng sẽ paste |
| 4 Experiments (4.1 Data, 4.2 Analysis and results, 4.3 Ablation) | `4_experiments_PASTE.md` | đã viết; cách làm, chạy lại EB đối chứng, ablation; không có so sánh mô hình với EB |
| 5 Results (5.1 so sánh tổng thể, 5.2 theo họ, 5.3 theo thời gian, 5.4 thảo luận) | `5_results_PASTE.md` | đã viết; mô hình đề xuất luôn đứng cuối bảng/đoạn |
| 6 Conclusion | `6_conclusion_PASTE.md` | đã viết; chỉ tóm tắt, không lặp mục 5 |
| Còn lại (Introduction cần viết lại cho khớp mục 3–6, Related Work) | chưa viết | |

Lưu ý: nhãn huấn luyện "Sharpe năm sau − mức năm − nền |t|" và PCA/ISAB/Huber/trọng số năm là các thành phần cần ghi nguồn đúng (xem `docs/5_FINAL_MODEL_VS_BASELINE.md`). Việc FinCast được tiền huấn luyện trên dữ liệu bên ngoài phải luôn được nêu rõ, và chưa có khẳng định "không rò rỉ".

## 6. Lưu ý rút ra khi viết mục thực nghiệm

- Mọi kết quả chính tính từ checkpoint đã lưu (`code/checkpoints/final_s1/`). GPU không tất định nên học lại cùng seed cho số khác (holdout Sharpe 0,71 / 0,87 / 0,94 qua ba lần học lại, so với 0,96 của mô hình lưu). Mọi so sánh giữa các cấu hình phải học lại nhiều lần và chỉ coi chênh lệch là có ý nghĩa khi vượt dao động này.
- Nhấn mạnh điểm mạnh thật (top 1% hơn EB có ý nghĩa thống kê) bằng cách đặt tóm tắt lên đầu mục kết quả, nhưng vẫn nêu giới hạn (IC và đúng mức, naive mining ngang ở dev, p chưa hiệu chỉnh nhiều phép so sánh). Không viết "state-of-the-art".
- Phân vai các mục để không lặp: 3.1 = bài toán, nhãn, loss, walk-forward, luật top X%; 3.2 = kiến trúc và tối ưu; 4 = dữ liệu, chia dev/holdout, cài đặt, đối chứng, chỉ số, chạy lại EB, ablation; 5 = toàn bộ kết quả và thảo luận; 6 = tóm tắt. Tiêu đề mục con không được trùng nhau giữa các mục (ví dụ "Key findings" chỉ ở 6.2). Ý đã có ở mục trước thì trỏ `Section \ref{...}`, không viết lại.
- Khi so sánh, đặt mô hình đề xuất ở dòng/câu cuối và kết thúc bằng điều nó đạt được, nhưng chỉ dùng "cao nhất trong các phương pháp được so sánh", không viết SOTA.
