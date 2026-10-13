# Notebook tham khảo: chi tiết từng bước và kết quả (baseline để so sánh)

Nguồn: hai file trong [Research/reference_notebooks/](../Research/reference_notebooks/). Mình đọc mã và kết quả đã lưu trong notebook, không chạy lại. Số liệu trong file này lấy nguyên văn từ kết quả đã lưu. Kết quả dùng làm baseline để so sánh nằm ở mục 5 của chính file này.

| Notebook | Làm gì | Có huấn luyện mô hình không |
|---|---|---|
| A. `rep_original_R_LSTM_Attention_highthroughput_PointB.ipynb` (24 ô) | Chạy lại phân tích của paper bằng Python (bản chuyển mã của tác giả), vẽ hình, rồi đẩy kết quả lên GitHub | **Không** |
| B. `submit_financial_fincast_high_throughput.ipynb` (40 ô) | Huấn luyện mô hình học sâu để dự đoán lợi suất năm sau của từng chiến lược, rồi so với EB | **Có** (ba kiểu mô hình khác nhau, xem mục 3) |

**Trả lời ngắn: notebook B dự đoán gì và chấm thế nào?** Mô hình dự đoán `pred_ret`, lợi suất trung bình tháng của chiến lược trong năm sau. **MSE giữa dự đoán và lợi suất thật là loss khi huấn luyện** (và được in ra ở tập kiểm định), nhưng **thước đo báo cáo chính không phải MSE**: đó là IC theo năm (tương quan hạng giữa dự đoán và lợi suất thật), chênh lệch giữa nhóm dự đoán cao nhất và thấp nhất, và bảng lợi suất/t-stat/Sharpe của danh mục top 1/5/10% so với EB. Mình không thấy kết quả MSE trên tập kiểm tra được báo cáo ở bước cuối.

---

## 1. Dữ liệu dùng chung

**Chiến lược.** Mỗi chiến lược là một chuỗi lợi suất long-short theo tháng. Có 136.192 chiến lược trong 6 họ. Số chiến lược mỗi họ, in ra từ notebook A:

| Họ | Số chiến lược |
|---|---|
| Accounting EW và Accounting VW | 29.314 mỗi họ |
| Past Return EW và Past Return VW | 19.402 mỗi họ |
| Ticker EW và Ticker VW | 19.380 mỗi họ |

Ba nguồn (kế toán, lợi suất quá khứ, ticker) nhân hai cách tính (EW chia đều, VW theo vốn hóa) ra 6 họ. Tổng 2 × (29.314 + 19.402 + 19.380) = 136.192.

**Ba loại file mà cả hai notebook dùng:**

1. **Lợi suất theo tháng** (nguồn: `data_raw/`). Ví dụ thật, dòng đầu của `DataMinedLongShortReturnsEW.csv`:
   `signalid=1, year=1990, month=7, ret=1,5422, nstock=3160`, nghĩa là chiến lược số 1 lãi 1,54% trong tháng 7/1990 (với 3.160 cổ phiếu).
2. **Thống kê trong mẫu và ngoài mẫu** (`OOS_signal_tstat_OosNyears1.csv.gzip`). Mỗi chiến lược, mỗi năm Y có **hai dòng**:
   - dòng `IS`: 240 tháng (20 năm) trước Y, gồm `mean_ret` và `tstat`;
   - dòng `OOS`: 12 tháng của năm Y, gồm `mean_ret` và `tstat` của năm đó.

   Ví dụ thật (chiến lược 66, năm Y = 2020, file mẫu `Research/output_minh_hoa/`):

   | s_flag | Giai đoạn | nobs | mean_ret (%/tháng) | tstat |
   |---|---|---|---|---|
   | IS | 2000–2019 | 240 | −0,549 | −3,76 |
   | OOS | 2020 | 12 | −1,170 | −1,09 |

3. **Dự báo của EB** (`Nelly_Predict_SignalYear.csv.gzip`): file trên cộng thêm cột `thetahat` (t-stat đã co) và `pred_ret` (dự báo lợi suất tháng của EB). Với chiến lược 66: `thetahat = −2,5`, `pred_ret = −2,5 × (−0,549) / (−3,76) = −0,365` %/tháng. Thực tế năm 2020 là −1,170, tức EB đúng hướng nhưng chưa đủ mạnh.

Ngoài ra, notebook B dùng thêm **embedding FinCast** (tải từ Hugging Face, kho `ngandh/fincast-embeddings`): mỗi (chiến lược, năm) có một mảng `4 × 1280`, tính trước từ lịch sử lợi suất. Họ Accounting EW có 1.033.883 mảng như vậy. Notebook không chứa mã tạo embedding, nên độ dài cửa sổ lịch sử đưa vào FinCast không thấy trong notebook.

**Đại lượng cần dự đoán ở mọi nơi:** `mean_ret_OOS`, lợi suất trung bình tháng của chiến lược trong năm Y.

---

## 2. Notebook A: chạy lại phân tích của paper (không có AI)

Đầu ô 2 ghi "Created on Mon Feb 5 2024, author cdim": đây là mã Python của Chen & Dim. Ô 0–1 tải mã gốc `chenandrewy/high-throughput-ap`.

| Bước | Ô | Việc làm | Đầu ra |
|---|---|---|---|
| 1 | 0–2 | Tải repo, đặt đường dẫn, đọc file thống kê và file dự báo EB, ghép phần IS với phần OOS của cùng chiến lược-năm thành bảng `df_est_oos` (mỗi dòng: `ret_is`, `ret_oos`, `tstat_is`, `pred_ret`, ...) | bảng dữ liệu |
| 2 | 2 | Tóm tắt số chiến lược và lợi suất mỗi họ (phân vị 5%, 50%, 95% của lợi suất trung bình năm) | bảng `summary_stat.csv`, số chiến lược như mục 1 |
| 3 | 2 | So sánh phân phối t-stat thật với phân phối do **mô hình hỗn hợp** (tham số lấy từ file `QML_FamilyYear`) tạo ra, cho các năm 1983, 2004, 2020 | hình histogram, 6 họ |
| 4 | 3–4 | Chia chiến lược mỗi họ-tháng thành **20 nhóm theo lợi suất trong mẫu**; vẽ lợi suất trong mẫu, lợi suất ngoài mẫu và lợi suất dự báo của EB theo nhóm, cho 1983–2004 và 2004–2020 | hình "In-sample / OOS / Predicted" |
| 5 | 5 | Tính ngưỡng t-stat kiểm soát tỉ lệ phát hiện sai (FDR) theo hai phương pháp BY và Storey; chia chiến lược thành 20 nhóm theo t-stat và vẽ lợi suất ngoài mẫu so với ngưỡng | hình BY / Storey |
| 6 | 6–23 | Dọn dẹp: sao chép dữ liệu vào repo `ngandh/asset-pricing-exp`, commit, push, rồi xóa thư mục kết quả cũ. Tên các thư mục bị xóa gợi ý nhóm từng thử các hướng LSTM khác (`output_v2_A_LSTM`, `output_v3_C_Stacking`, `output_v4_E2E_LSTM`, `pointB_lb20_attention_output`) nhưng **không có mã hay kết quả của chúng trong notebook này** | không có kết quả khoa học |

Kết luận về notebook A: **nó tái hiện phần EB và các hình của paper, không dự đoán gì bằng AI**. Tên file có chữ "LSTM_Attention" nhưng trong notebook không có mô hình học nào.

---

## 3. Notebook B: huấn luyện mô hình dự đoán lợi suất và so với EB

### 3.1 Ba kiểu mô hình trong notebook

| Kiểu | Ô | Đầu vào | Chia dữ liệu | Có đáng tin không |
|---|---|---|---|---|
| (i) FinCast + Bi-LSTM + Attention | 10–11, 17–18 | embedding `4 × 1280` và 2 số (`tstat_IS`, `mean_ret_IS`) | một lần chia cố định theo năm | **Có** (đúng thời gian) |
| (ii) Bi-LSTM + Attention trên chuỗi thô, không FinCast | 13, 19 | **20 tháng** lợi suất cuối trước năm Y và 2 số như trên | giống (i), để so công bằng | **Có** |
| (iii) "New version to submit": LSTM trên 240 tháng | 28–39 | 240 tháng lợi suất kèm mặt nạ giá trị thiếu | **K-fold ngẫu nhiên trong cùng một năm** | **Có nguy cơ rò rỉ** (xem 3.6) |

### 3.2 Mô hình (i): FinCast + Bi-LSTM + Attention (họ Accounting EW, ô 10–11)

Các bước theo thứ tự trong mã:

1. **Đọc dữ liệu cho đúng một họ**: `FAMILY = 'DataMinedLongShortReturnsEW'`. Tải embedding (`1033883 × 4 × 1280`) và bảng khóa (`signalid`, `signal_family`, `oos_begin_year`, chỉ số hàng `emb_idx`).
2. **Kiểm tra căn hàng** (ô 2–6): số hàng embedding bằng số hàng khóa, không khóa trùng, và độ tương tự cosine giữa các embedding. Với một cặp riêng lẻ, cosine "cùng chiến lược hai năm liên tiếp" là 0,813 còn "khác chiến lược cùng năm" là 0,986 (tức ngược với kỳ vọng); sau khi trừ vector trung bình chung là 0,651 so với 0,975. Với 100 cặp ngẫu nhiên: cùng chiến lược 0,849 (độ lệch chuẩn 0,077), khác chiến lược 0,679 (0,174), t = 8,86, p ≈ 5e-16. Hai kiểm tra đơn lẻ không ủng hộ, 100 cặp thì ủng hộ rằng embedding gắn đúng với chiến lược.
3. **Ghép nhãn**: từ file thống kê, lấy dòng `IS` (`tstat_IS`, `mean_ret_IS`) và dòng `OOS` (`mean_ret_OOS`), ghép theo (chiến lược, họ, năm) rồi ghép với `emb_idx`. Bỏ dòng thiếu.
4. **Chia theo năm**: huấn luyện < 2011, kiểm định 2011–2013, kiểm tra ≥ 2014.
5. **Chuẩn hóa chỉ bằng thống kê của tập huấn luyện**: trung bình và độ lệch chuẩn từng chiều của embedding (tính từng khối để tiết kiệm RAM); `StandardScaler` cho 2 số bổ sung.
6. **Mô hình** `LSTMAttentionRegressor`:
   ```
   embedding [B,4,1280] -> Bi-LSTM(128) [B,4,256]
        -> attention: điểm = v·tanh(W h), softmax qua 4 vị trí, tổng có trọng số -> [B,256]
        -> ghép 2 số (tstat_IS, mean_ret_IS) -> [B,258]
        -> Linear 258→64, ReLU, BatchNorm, Dropout 0,3 -> Linear 64→1
   ```
7. **Loss và cách học**: `MSELoss` giữa dự đoán và `mean_ret_OOS`; Adam, lr 1e-3, weight decay 1e-4, batch 1024, tối đa 100 epoch.
8. **Chọn checkpoint**: sau mỗi vài epoch tính MSE và **IC** (tương quan hạng Spearman) trên tập kiểm định; giữ mô hình có **IC kiểm định cao nhất**; dừng sớm sau 15 epoch không cải thiện.
9. **Chấm trên tập kiểm tra (≥ 2014)** bằng hàm `portfolio_relevant_report_yearly`:
   - IC tính **riêng từng năm** (không gộp các năm), rồi lấy trung bình và độ lệch chuẩn;
   - chia 10 nhóm theo dự đoán trong từng năm, tính lợi suất thật trung bình mỗi nhóm; "long-short spread" = nhóm cao nhất trừ nhóm thấp nhất, trung bình các năm, kèm t-stat.
10. **Lưu dự đoán** của tập kiểm tra vào file parquet để bước so sánh dùng.

**Kết quả đã lưu (Accounting EW, 8 năm kiểm tra 2014–2021):** IC trung bình theo năm 0,0393 (độ lệch chuẩn 0,0977); IC kiểm định tốt nhất 0,0169; spread long-short 0,0748 với t-stat 0,74. Nghĩa là dự đoán gần như không có khả năng xếp hạng đáng kể.

### 3.3 Mô hình (ii): không FinCast (ô 13)

Giống (i) nhưng thay embedding bằng chuỗi thô:
- Đầu vào: **20 tháng lợi suất cuối cùng trước năm Y** (từ tháng 12 năm Y−1 lùi về), 1 chiều mỗi bước, bỏ chiến lược không đủ 20 tháng.
- Mạng nhỏ hơn: Bi-LSTM hidden 32, attention, ghép 2 số, `Linear → 32, ReLU, Dropout 0,3, Linear → 1`.
- Cùng loss, cách chọn checkpoint và mốc năm để so công bằng.

**Kết quả đã lưu:** Accounting EW: IC 0,0839 (độ lệch chuẩn 0,1308), spread 0,2115, t-stat 1,22. Past Return EW: IC 0,0556 (0,2983), spread 0,6053, t-stat 0,79.

### 3.4 So với EB và Naive (ô 15, 22)

Ghép dự đoán với thông tin của EB rồi tạo bốn danh mục, mỗi danh mục lấy top 1%, 5%, 10%:

| Danh mục | Dấu (mua hay bán) | Xếp hạng bằng | Trong từng |
|---|---|---|---|
| EB Mining | dấu của `pred_ret` của EB | **Sharpe dự báo** `thetahat/√n` | tháng |
| Naive Mining | dấu của t-stat trong mẫu | abs(t-stat) trong mẫu | tháng |
| FinCast Mining | dấu của dự đoán mô hình (i) | **abs(dự đoán)** (độ lớn lợi suất dự đoán) | tháng |
| No-FinCast Mining | dấu của dự đoán mô hình (ii) | abs(dự đoán) | tháng |

Chiến lược nào có dự đoán âm thì được đảo chiều (bán thay vì mua). Danh mục chia đều tiền cho các chiến lược được chọn; lấy lợi suất của danh mục theo tháng, **chỉ trên các năm kiểm tra (≥ 2014)**.

**Cách chấm bảng cuối:** gom lợi suất danh mục theo tháng, tính trung bình, độ lệch chuẩn; sau đó:
- `Mean Return (% ann)` = trung bình tháng × 12;
- `t-stat` = trung bình / độ lệch chuẩn × √(số tháng);
- `Sharpe (ann)` = trung bình / độ lệch chuẩn × √12.

(Theo mã của notebook, cột lợi suất là %/năm và Sharpe đã quy năm.) Kết quả từng bảng nằm ở mục 5.

### 3.5 Các họ khác (ô 16–27)

Ô 16–20 làm lại đúng quy trình (i), (ii) và so sánh cho **Past Return EW**: IC của (i) là −0,0364 (độ lệch chuẩn 0,2644, dừng sớm ở epoch 30), của (ii) là 0,0556. Ô 23–25 bắt đầu làm lại cho Accounting VW (embedding 1.033.882 hàng); nhật ký còn lại cho thấy Val MSE khoảng 0,77–0,82 và Val IC âm (từ −0,05 đến −0,01), và không có kết quả kiểm tra đi kèm.

### 3.6 Mô hình (iii): "new version to submit" (ô 28–39)

Các bước:
1. Đọc lợi suất Accounting EW; lọc `year ≥ 1963`, `nstock ≥ 10`; kiểm tra khoảng trống chuỗi và ngoại lệ (|ret| > 50%).
2. Với mỗi năm Y từ 1983 đến 2019: lấy 240 tháng trước Y (nếu thiếu thì đệm giá trị thiếu ở đầu, cần tối thiểu 36 tháng), kèm một mặt nạ báo ô nào có dữ liệu. Nhãn là `mean_ret` của dòng `OOS`.
3. Chuẩn hóa chuỗi bằng trung bình và độ lệch chuẩn của chính năm đó.
4. Mô hình `ReturnLSTM`: LSTM một chiều hidden 32, đầu vào 2 kênh (giá trị và mặt nạ), lấy trạng thái cuối, một lớp tuyến tính.
5. **Huấn luyện kiểu "out-of-fold"**: chia **ngẫu nhiên** 5 phần các chiến lược **của cùng năm Y**, huấn luyện trên 4 phần, dự đoán phần còn lại, lặp 5 lần để mọi chiến lược đều có một dự đoán. Loss MSE, Adam 1e-3, dừng sớm theo MSE của phần còn lại (tối đa 200 epoch, kiên nhẫn 10).
6. Chấm: **tương quan Pearson giữa dự đoán và lợi suất thật** của các chiến lược trong từng năm; trung bình 37 năm là **0,633** (từ 0,37 đến 0,86). Ô 39 tính, như một mốc, tương quan giữa lợi suất trong mẫu và ngoài mẫu: trung bình **0,160**.

**Vì sao cần cẩn thận:** trong bước 5, mạng được huấn luyện bằng **nhãn của chính năm Y** (lợi suất ngoài mẫu của các chiến lược khác trong cùng năm). Các chiến lược cùng năm có lợi suất tương quan mạnh với nhau (cùng chịu một năm thị trường), nên mạng có thể học "năm Y lên hay xuống" từ nhãn của chiến lược khác, điều mà một dự báo thật vào cuối năm Y−1 không thể biết. Tương quan 0,633 (cao hơn nhiều so với 0,160 của mốc trong mẫu) phù hợp với khả năng đó. Do đó mình **không** xem đây là một dự báo ngoài mẫu hợp lệ và không đưa nó vào baseline. Mình chưa tự chạy lại để chứng minh có rò rỉ; đây là nhận xét từ thiết kế huấn luyện.

---

## 4. Những điểm cần nhớ khi so với dự án này

| Điểm | Notebook B | Dự án này |
|---|---|---|
| Phạm vi | một họ mỗi lần | cả 6 họ gộp, top-k chọn trong toàn bộ |
| Cách chia | một lần: huấn luyện < 2011, kiểm tra ≥ 2014 (8 năm) | walk-forward mỗi năm 1983–2020 |
| Chọn mô hình | IC kiểm định | MSE trên dev 1983–2004; holdout 2005–2020 chỉ báo cáo |
| Xếp hạng chọn top-k | mô hình AI: abs(dự đoán); EB: Sharpe dự báo | cùng luật của paper: Sharpe dự báo |
| Số chiến lược được mô hình chấm | cả họ | 10% có abs(t) cao nhất mỗi họ-năm |
| Seed | một | sẽ báo độ phân tán giữa các seed |

Vì các khác biệt này, số của notebook là mốc tham khảo chứ chưa so thẳng được với kết quả của dự án.

---

## 5. Kết quả đã lưu (baseline để so sánh)

Đây là mốc để so các cải tiến của dự án. Cấu hình gốc (mô hình (i)) theo sơ đồ:

```
Chuỗi lợi suất long-short của 1 chiến lược (năm, tháng, ret)
   ▼
FinCast Encoder (pretrained, đóng băng)            -> Hidden Sequence 4 x 1280
   ▼
Bi-LSTM (hidden = 128)                              -> 4 x 256
   ▼
Attention Pooling                                   -> Context Vector 256 chiều
   ▼
Ghép với Extra Features: tstat_IS, mean_ret_IS      -> 258 chiều
   ▼
Fully Connected (64) + ReLU + BatchNorm + Dropout 0,3
   ▼
Fully Connected (1), tuyến tính                     -> Predicted Mean OOS Return
```

Số liệu lấy nguyên văn từ kết quả đã lưu, trên tập kiểm tra 2014–2021, mỗi lần một họ. Cột lợi suất là %/năm, Sharpe đã quy năm (mục 3.4).

**IC trung bình theo năm trên tập kiểm tra 2014–2021** (độ lệch chuẩn giữa các năm trong ngoặc, 8 năm):

| Họ | FinCast + Bi-LSTM + Attention | Bi-LSTM + Attention trên chuỗi thô (không FinCast) |
|---|---|---|
| Accounting EW | 0,0393 (0,0977) | 0,0839 (0,1308) |
| Past Return EW | −0,0364 (0,2644) | 0,0556 (0,2983) |

**Bảng top-k trong notebook** (số quan sát, `ret_mean`, t-stat, Sharpe `sr`; chỉ trên tập kiểm tra):

*Accounting EW*

| Phương pháp | Top 1% | Top 5% | Top 10% |
|---|---|---|---|
| EB (paper) | 1,53 \| 1,05 \| 0,40 | 1,74 \| 1,48 \| 0,56 | 1,96 \| 1,92 \| 0,73 |
| Naive | 1,81 \| 1,26 \| 0,47 | 1,67 \| 1,44 \| 0,54 | 1,83 \| 1,81 \| 0,68 |
| FinCast + Bi-LSTM + Attention | 0,78 \| 0,80 \| 0,28 | 0,56 \| 0,95 \| 0,34 | 0,61 \| 1,16 \| 0,41 |
| Không FinCast | 2,16 \| 0,99 \| 0,35 | 2,17 \| 1,39 \| 0,49 | 1,71 \| 1,32 \| 0,47 |

*Past Return EW*

| Phương pháp | Top 1% | Top 5% | Top 10% |
|---|---|---|---|
| EB (paper) | −4,69 \| −2,57 \| −0,97 | −3,45 \| −2,42 \| −0,91 | −2,66 \| −2,20 \| −0,83 |
| Naive | −4,71 \| −2,59 \| −0,98 | −3,46 \| −2,57 \| −0,97 | −2,47 \| −2,32 \| −0,88 |
| FinCast + Bi-LSTM + Attention | 2,60 \| 0,76 \| 0,27 | −0,70 \| −0,31 \| −0,11 | −1,96 \| −0,92 \| −0,33 |
| Không FinCast | 6,18 \| 1,27 \| 0,45 | 6,18 \| 1,36 \| 0,48 | 6,24 \| 1,42 \| 0,50 |

Ba điều rút ra, giới hạn trong những gì notebook đã in:
- Trong Accounting EW, FinCast + Bi-LSTM + Attention **thấp hơn** EB ở cả ba mức top-k (Sharpe 0,28/0,34/0,41 so với 0,40/0,56/0,73), và IC (0,039) thấp hơn bản không FinCast (0,084). Tức là trong notebook này FinCast không giúp.
- Trong Past Return EW, EB và Naive cho lợi suất âm trên giai đoạn 2014–2021, còn bản không FinCast dương (Sharpe 0,45–0,50). Giai đoạn này chỉ có 8 năm nên kết luận rất yếu.
- Các số trên là tập kiểm tra ngắn (8 năm), một họ, một seed, không có p-value; không nên coi là bằng chứng chắc chắn cho bên nào.

Ô 25 (Accounting VW) có nhật ký huấn luyện cho thấy Val IC âm (từ −0,05 đến −0,01) và không có kết quả kiểm tra; bản (iii) có nguy cơ rò rỉ (mục 3.6). Cả hai không đưa vào baseline.


**Khi dùng làm mốc:** dự án này đánh giá walk-forward 1983–2020 trên cả 6 họ, nên để so công bằng cần chấm mô hình mới trên đúng cửa sổ 2014–2020 của hai họ Accounting EW và Past Return EW và báo cùng IC, cùng bảng top-k. Kết quả của chính dự án nằm ở [5_FINAL_MODEL_VS_BASELINE.md](5_FINAL_MODEL_VS_BASELINE.md).

---

## 6. Ba thước đo báo cáo, giải thích từng bước

Cả ba đều trả lời cùng một câu hỏi: dự đoán của mô hình có giúp chọn được chiến lược tốt không. Mỗi thước nhìn một góc khác.

### 6.1 IC (tương quan hạng): mô hình xếp thứ tự có đúng không

IC chỉ xét **thứ tự**, bỏ qua độ lớn của dự đoán. Tính cho **một năm** như sau:

1. Mỗi chiến lược có hai số: dự đoán của mô hình và lợi suất thực tế của năm đó.
2. Xếp hạng cột dự đoán từ thấp đến cao (hạng 1 = thấp nhất). Xếp hạng cột thực tế theo cùng cách.
3. Tính `d` = hạng dự đoán − hạng thực tế, cho từng chiến lược.
4. `IC = 1 − 6·Σd² / (n·(n² − 1))`, n là số chiến lược.

Ví dụ 4 chiến lược:

| | Dự đoán | Thực tế | Hạng dự đoán | Hạng thực tế | d |
|---|---|---|---|---|---|
| A | 0,9 | 0,4 | 4 | 3 | 1 |
| B | 0,3 | 0,7 | 3 | 4 | −1 |
| C | 0,0 | −0,3 | 2 | 1 | 1 |
| D | −0,5 | −0,1 | 1 | 2 | −1 |

Σd² = 4, nên IC = 1 − 6·4 / (4·15) = **0,6**. IC = 1 là xếp đúng hoàn toàn, **IC = 0 là không hơn đoán mò**, IC âm là xếp ngược.

**Tính riêng từng năm:** mỗi năm một IC trên chỉ các chiến lược của năm đó, không trộn 8 năm thành một IC. Lý do: trong một năm mọi chiến lược chịu chung một môi trường thị trường; trộn nhiều năm sẽ lẫn hiệu ứng "cả năm lên hoặc xuống" với việc xếp hạng đúng.

**Gộp 8 năm (2014–2021):** có 8 giá trị IC. Lấy **trung bình** (mức xếp đúng trung bình) và **độ lệch chuẩn** (độ dao động giữa các năm). Ví dụ Accounting EW, FinCast: trung bình 0,039, độ lệch chuẩn 0,098. Độ lệch chuẩn lớn gấp mấy lần trung bình nghĩa là có năm dương, có năm âm, nên chưa có bằng chứng mô hình xếp hạng tốt.

### 6.2 Chia 10 nhóm và spread: nhóm dự đoán cao có lãi hơn nhóm dự đoán thấp không

Làm cho **một năm**:

1. Xếp các chiến lược theo dự đoán, cắt thành 10 nhóm bằng nhau (nhóm 1 = dự đoán thấp nhất, nhóm 10 = cao nhất).
2. Với mỗi nhóm, tính lợi suất **thực tế** trung bình.
3. `spread_năm = (nhóm 10) − (nhóm 1)`.

Mô hình tốt thì lợi suất thực tế tăng dần từ nhóm 1 lên nhóm 10, nên spread dương. Gộp 8 năm: lấy trung bình 8 spread, rồi

`t = trung bình(spread) / độ lệch chuẩn(spread) · √8`

t cho biết spread trung bình có đủ lớn so với độ dao động giữa các năm để loại trừ may rủi không. Quy ước thường dùng: **t từ 2 trở lên mới coi là có ý nghĩa**. Baseline Accounting EW, FinCast: spread 0,075, t = 0,74, chưa đạt.

| | IC | Spread |
|---|---|---|
| Nhìn vào | thứ tự của mọi chiến lược | hai đầu: nhóm 10 và nhóm 1 |
| Đơn vị | −1 đến +1 | %/tháng |

### 6.3 Bảng danh mục top 1/5/10% (giống Table 2 của paper)

Gần nhất với việc dùng dự đoán để đầu tư. Làm lần lượt:

1. Mỗi tháng, lấy top k% chiến lược theo luật xếp hạng (mục 3.4).
2. Chiến lược có dự đoán âm thì **đảo chiều** (bán thay vì mua): lợi suất = dấu × lợi suất thực.
3. Chia đều tiền: lợi suất danh mục tháng đó = trung bình các lợi suất đã nhân dấu.
4. Gộp các tháng kiểm tra, tính trung bình `m` và độ lệch chuẩn `s` của lợi suất tháng của danh mục:

| Cột | Công thức | Ví dụ: m = 0,5%, s = 2%, 96 tháng |
|---|---|---|
| Lợi suất (%/năm) | `12 · m` | 6,0% |
| Sharpe (năm) | `m / s · √12` | 0,25 × 3,46 = 0,87 |
| t-stat | `m / s · √(số tháng)` | 0,25 × 9,8 = 2,45 |

Bảng này chỉ nhìn **đầu bảng**, nơi mô hình thực sự chọn, và là thước paper dùng để so EB với Naive.

### 6.4 Khác MSE ở chỗ nào

`MSE = trung bình((dự đoán − thực tế)²)` đo sai số độ lớn trên **mọi** chiến lược, kể cả những chiến lược mô hình không chọn. Mô hình có MSE nhỏ vẫn có thể xếp hạng kém, và mô hình dự đoán độ lớn lệch nhưng xếp đúng thứ tự vẫn cho danh mục tốt. Dự án này báo cáo cả MSE (so với EB, kèm kiểm định Diebold–Mariano) lẫn danh mục.

---

## 7. Số liệu chi tiết của từng mô hình (lấy từ kết quả đã lưu của notebook B)

### 7.1 Kích thước dữ liệu

| | Accounting EW, FinCast (ô 11) | Accounting EW, không FinCast (ô 13) | Past Return EW, FinCast (ô 18) | Past Return EW, không FinCast (ô 19) |
|---|---|---|---|---|
| Số mẫu (chiến lược-năm) | 979.380 | 977.901 | 756.678 | 756.678 |
| Huấn luyện (< 2011) | 692.447 | 691.348 | không in trong phần mình xem | 543.256 |
| Kiểm định (2011–2013) | 79.133 | 79.010 | không in | 58.206 |
| Kiểm tra (≥ 2014) | 207.800 | 207.543 | 155.216 | 155.216 |
| Dừng sớm ở epoch | không in trong phần mình xem | 16 | 30 | 29 |
| IC kiểm định tốt nhất | 0,0169 | 0,1542 | **0,4339** | 0,0191 |
| IC kiểm tra | 0,0393 | 0,0839 | **−0,0364** | 0,0556 |

Ghi chú: ở Accounting EW có 1.033.883 embedding nhưng chỉ 1.005.582 dòng IS của họ này (tỉ lệ 102,8%); dòng "Matched 19,2%" in ra là phần so với **tất cả 6 họ** (5.113.700 dòng), nên con số thấp đó không phải lỗi. Ở Past Return EW có 756.678 embedding trên 776.080 dòng IS (97,5%). Chênh lệch giữa IC kiểm định và IC kiểm tra của Past Return FinCast (0,43 so với −0,04) lớn bất thường: kiểm định chỉ có 3 năm (2011–2013) nên việc chọn checkpoint theo IC trên đó rất bất ổn.

### 7.2 IC theo từng năm kiểm tra

| Năm | Accounting, FinCast | Accounting, không FinCast | Past Return, FinCast | Past Return, không FinCast |
|---|---|---|---|---|
| 2014 | 0,108 | 0,068 | −0,055 | −0,356 |
| 2015 | 0,198 | 0,294 | 0,361 | 0,094 |
| 2016 | 0,120 | 0,238 | 0,050 | 0,363 |
| 2017 | −0,065 | 0,007 | −0,202 | −0,051 |
| 2018 | 0,037 | 0,106 | −0,179 | −0,288 |
| 2019 | 0,015 | 0,043 | 0,220 | 0,083 |
| 2020 | −0,090 | −0,118 | −0,493 | 0,532 |
| 2021 | −0,009 | 0,032 | 0,007 | 0,068 |
| **Trung bình** | **0,039** | **0,084** | **−0,036** | **0,056** |
| Độ lệch chuẩn | 0,098 | 0,131 | 0,264 | 0,298 |

Past Return có IC dao động rất mạnh giữa các năm (từ −0,49 đến +0,53), nên trung bình 8 năm gần như không mang thông tin.

### 7.3 Lợi suất thực tế trung bình theo nhóm dự đoán (%/tháng, nhóm 1 = dự đoán thấp nhất)

| Nhóm | Accounting, FinCast | Accounting, không FinCast | Past Return, FinCast | Past Return, không FinCast |
|---|---|---|---|---|
| 1 | −0,060 | −0,143 | 0,317 | −0,240 |
| 2 | −0,045 | −0,033 | 0,194 | −0,105 |
| 3 | −0,016 | −0,079 | 0,078 | −0,072 |
| 4 | −0,009 | −0,045 | −0,057 | −0,100 |
| 5 | 0,000 | 0,017 | −0,146 | −0,063 |
| 6 | 0,006 | 0,012 | −0,167 | −0,062 |
| 7 | 0,018 | 0,036 | −0,155 | −0,113 |
| 8 | 0,009 | 0,040 | −0,140 | −0,038 |
| 9 | −0,004 | 0,053 | −0,123 | 0,146 |
| 10 | 0,015 | 0,069 | −0,083 | 0,366 |
| Spread (nhóm 10 − nhóm 1) | 0,075 (t = 0,74) | 0,212 (t = 1,22) | **−0,400 (t = −0,88)** | 0,605 (t = 0,79) |

Hình dạng nhóm cho thấy rõ chất lượng: Accounting không FinCast tăng gần đơn điệu từ −0,14 lên +0,07 (xếp hạng có ích, dù t = 1,22 chưa đạt ngưỡng 2). Past Return FinCast thì **giảm** từ +0,32 xuống −0,08: dự đoán ngược chiều thực tế. Mọi t-stat đều dưới 2 nên chưa có kết quả nào có ý nghĩa thống kê.

---

## 8. Luồng shape trong một batch (B = 1024 mẫu)

**Mô hình (i), FinCast + Bi-LSTM + Attention (1.526.657 tham số học được, mình tính từ cấu trúc trong mã):**

| Bước | Phép tính | Shape |
|---|---|---|
| Embedding FinCast đã chuẩn hóa theo trung bình/độ lệch chuẩn của tập huấn luyện | đọc từ file `.npy` | [1024, 4, 1280] |
| Bi-LSTM (1280 → 128, hai chiều) | nối trạng thái hai chiều | [1024, 4, 256] |
| Điểm attention: `v·tanh(W h)`, softmax qua 4 patch | | [1024, 4] |
| Context = tổng trọng số của 4 patch | | [1024, 256] |
| Ghép `tstat_IS`, `mean_ret_IS` (đã chuẩn hóa) | nối cột | [1024, 258] |
| Linear 258 → 64, ReLU, BatchNorm, Dropout 0,3 | | [1024, 64] |
| Linear 64 → 1 | dự đoán `mean_ret_OOS` | [1024] |

**Mô hình (ii), chuỗi thô (15.361 tham số, mình tính từ mã):** chuỗi [1024, 20, 1] → Bi-LSTM(1 → 32) [1024, 20, 64] → attention qua 20 bước → [1024, 64] → ghép 2 số [1024, 66] → Linear 66 → 32, ReLU, Dropout 0,3 → Linear 32 → 1 → [1024].

**Mô hình (iii), LSTM walk-forward (4.641 tham số, mình tính từ mã):** [n, 240, 2] (giá trị và mặt nạ) → LSTM(2 → 32) lấy trạng thái cuối [n, 32] → Linear 32 → 1 → [n].

---

## 9. Bản đồ các ô của hai notebook

**Notebook A (24 ô):**

| Ô | Việc |
|---|---|
| 0–1 | tải mã gốc `chenandrewy/high-throughput-ap` |
| 2 | cấu hình, đọc dữ liệu, các hàm BY/Storey, bảng tóm tắt số chiến lược, hình phân phối t-stat và mô hình hỗn hợp |
| 3–4 | hình lợi suất trong mẫu / ngoài mẫu / dự báo theo 20 nhóm (hai ô giống nhau) |
| 5 | ngưỡng BY và Storey, hình lợi suất ngoài mẫu theo t-stat |
| 6–23 | sao chép dữ liệu, commit, push lên GitHub, xóa thư mục kết quả |

**Notebook B (40 ô):**

| Ô | Việc |
|---|---|
| 0 | đăng nhập Hugging Face |
| 2–8 | kiểm tra embedding: số hàng, khóa trùng, độ tương tự cosine |
| 10–11 | FinCast + Bi-LSTM + Attention, Accounting EW: tải dữ liệu, huấn luyện, chấm, lưu dự đoán |
| 13 | không FinCast, Accounting EW |
| 15 | bảng so sánh EB / Naive / FinCast / không FinCast, Accounting EW |
| 17–18 | FinCast, Past Return EW |
| 19 | không FinCast, Past Return EW |
| 21–22 | bảng so sánh, Past Return EW |
| 24–25 | FinCast, Accounting VW (nhật ký huấn luyện chưa hoàn tất) |
| 28–39 | "new version to submit": kiểm tra dữ liệu (29–31), tải lợi suất (32), hàm huấn luyện (33–34), vòng walk-forward OOF 1983–2019 (35), thử dựng dữ liệu một năm (36), so sánh và tương quan theo năm (37–39) |

---

## 10. Paper gốc (Chen & Dim), từng bước, để đối chiếu

Mô tả theo cách dự án này đã cài đặt lại và kiểm tra khớp với paper (top 1% của EB: 5,79%/năm, Sharpe 1,49 so với 5,70% và 1,46 trong Table 2). Mã: [code/data.py](../code/data.py).

| Bước | Đầu vào | Phép tính | Đầu ra |
|---|---|---|---|
| P1. Lợi suất | 136.192 chuỗi lợi suất tháng (6 họ) | không | ma trận chiến lược × tháng |
| P2. Thống kê trong mẫu | với mỗi năm Y (1983–2020) và chiến lược i: 240 tháng trước Y, cần ít nhất 36 tháng có dữ liệu | `m_i` = trung bình, `s_i` = độ lệch chuẩn (chia cho n), `n_i` = số tháng, `t_i = m_i / (s_i / √n_i)` | một vector `t` cho mỗi họ-năm |
| P3. Ước lượng phân phối tiên nghiệm (theo từng họ-năm) | vector `t` của họ đó | giả định `θ ~ p·N(μa, σa²) + (1−p)·N(μb, σb²)` và `t = θ + N(0,1)`, nên `t` là hỗn hợp hai phân phối chuẩn với phương sai `σ²+1`; tìm 5 tham số làm cực đại `Σ log f(t_i)` (quasi-maximum likelihood) | 5 tham số cho họ-năm |
| P4. Trung bình hậu nghiệm | `t_i` và 5 tham số | `θ̂_i = E[θ \| t_i] = Σ_k w_k(t_i)·( μ_k + σ_k²/(σ_k²+1) · (t_i − μ_k) )`, với `w_k ∝ π_k · N(t_i; μ_k, σ_k²+1)` | `θ̂_i`, t-stat đã co |
| P5. Dự báo lợi suất | `θ̂_i`, `m_i`, `t_i` | `pred_ret_i = θ̂_i · m_i / t_i` (áp hệ số co `θ̂/t` lên lợi suất trong mẫu); Sharpe dự báo tháng = `θ̂_i / √n_i` | `pred_ret` (%/tháng) |
| P6. Chọn chiến lược | `pred_ret`, `θ̂`, `n` của **cả 6 họ gộp**, mỗi năm | dấu `sign_i = sign(pred_ret_i)`; điểm xếp hạng `SRahat_i = sign_i · θ̂_i/√n_i · √12`; lấy top 1%, 5%, 10% | tập chiến lược được chọn kèm chiều mua/bán |
| P7. Danh mục | chiến lược được chọn, lợi suất tháng của năm Y | mỗi tháng: trung bình đều của `sign_i · r_{i,t}` | chuỗi lợi suất tháng của danh mục 1983–2020 |
| P8. Bảng kết quả | chuỗi danh mục | lợi suất = trung bình × 12; Sharpe = trung bình / độ lệch chuẩn × √12; t-stat = trung bình / độ lệch chuẩn × √(số tháng) | Table 2 |

Đối thủ Naive của paper dùng cùng quy trình P6–P8 nhưng với `sign(t_i)` và điểm xếp hạng `abs(t_i)`, không co. Paper: EB top 1% 5,70%/năm, Sharpe 1,46; Naive top 1% 5,72%/năm, Sharpe 1,45.

**Điểm quan trọng:** trong P3–P5 paper **không dùng lợi suất của năm Y** để ước lượng gì cả; EB chỉ nhìn phân phối `t` trong mẫu của họ-năm đó và tự co kéo. Lợi suất năm Y chỉ xuất hiện ở P7–P8 để chấm điểm.

**Paper báo cáo gì theo họ, và gì gộp chung (đã kiểm tra trong PDF, trang 42–43):**
- **Table 2 gộp chung 6 họ.** Cột "Num Strats Combined" cho top 1% là **1.278** chiến lược, top 5% là 6.389, top 10% là 12.777. Paper không có bảng top-k riêng cho từng họ.
- **Table 3 Panel A** cho biết top 1% của EB đến từ họ nào (trung bình các năm): Acct EW **91,0%**, Past Ret EW **8,6%**, Acct VW 0,3%, Past Ret VW 0,1%, Ticker EW 0,0%, Ticker VW 0,0%.
- **Hình 2–5** vẽ theo từng họ: phân phối t-stat (1983, 2004) và đồ thị lợi suất trong mẫu / dự báo EB / ngoài mẫu theo 20 nhóm (1983–2004 và 2004–2020).

Hệ quả: hai họ mà notebook tham khảo chạy xong (Accounting EW, Past Return EW) chiếm 99,6% top 1% của paper, nên chọn đúng hai họ này là hợp lý. Nhưng số của notebook cho từng họ **không so trực tiếp được với Table 2** (vì Table 2 là gộp).

**Ví dụ số xuyên suốt (chiến lược 66, năm 2020, họ Accounting EW, file mẫu):**

| Bước | Giá trị |
|---|---|
| P2: `m` (%/tháng), `t`, `n` | −0,549; −3,76; 240 |
| P4: `θ̂` | −2,5 (trong file mẫu) |
| P5: `pred_ret = −2,5 · (−0,549) / (−3,76)` | **−0,365** %/tháng |
| P6: dấu | `pred_ret` âm nên chiến lược bị đảo chiều (bán) |
| P6: `SRahat = 2,5 / √240 · √12` | 2,5 × 0,0645 × 3,464 = **0,559** |
| Naive: dấu và điểm | dấu `sign(t) = −1`, điểm `abs(t) = 3,76` |
| P7: lợi suất thực tế năm 2020 (%/tháng) | −1,170, sau đảo chiều là **+1,170** nếu chiến lược này nằm trong danh mục |

---

## 11. Notebook B, từng bước: đầu vào, phép tính, đầu ra, shape, loss, cách chấm

Áp dụng cho mô hình (i) FinCast + Bi-LSTM + Attention, họ Accounting EW (ô 10–11). Mô hình (ii) giống hệt từ B3 trở đi, chỉ khác ở B1 và B6 (ghi chú cuối bảng). `N` là số mẫu (979.380), `B` = 1024 là batch.

| Bước | Đầu vào | Phép tính | Đầu ra và shape |
|---|---|---|---|
| B1. Nạp embedding | file `.npy` do FinCast tạo sẵn từ lịch sử lợi suất (cửa sổ chính xác không có trong notebook) và bảng khóa | đọc dạng mmap, không nạp hết RAM | mảng `[1033883, 4, 1280]` (float), bảng khóa `[1033883]` với (`signalid`, `signal_family`, `oos_begin_year`, `emb_idx`) |
| B2. Nạp nhãn và hai số bổ sung | file thống kê `OOS_signal_tstat_OosNyears1` | lấy dòng `IS`: `tstat_IS`, `mean_ret_IS`; lấy dòng `OOS`: `mean_ret_OOS`; ghép theo (chiến lược, họ, năm) với bảng khóa; bỏ dòng thiếu | bảng `merged`: `N = 979380` dòng, mỗi dòng có `emb_idx`, `tstat_IS`, `mean_ret_IS`, `mean_ret_OOS` (nhãn, %/tháng) |
| B3. Chia theo năm | `oos_begin_year` | huấn luyện: năm < 2011; kiểm định: 2011–2013; kiểm tra: ≥ 2014 | 692.447 / 79.133 / 207.800 mẫu |
| B4. Chuẩn hóa (chỉ dùng tập huấn luyện) | embedding và 2 số | embedding: trừ trung bình và chia độ lệch chuẩn **từng trong 1280 chiều** (tính trên mọi vị trí patch của tập huấn luyện); 2 số: `StandardScaler` | `[B, 4, 1280]` và `[B, 2]` mỗi batch |
| B5. Bi-LSTM | `[B, 4, 1280]` | LSTM hai chiều, hidden 128 mỗi chiều: mỗi chiều đọc 4 patch theo thứ tự, nối đầu ra hai chiều | `[B, 4, 256]` |
| B6. Attention pooling | `[B, 4, 256]` | `e_t = v · tanh(W h_t)`, `α = softmax(e)` qua 4 patch, `c = Σ α_t h_t` | trọng số `[B, 4]`, context `[B, 256]` |
| B7. Ghép đặc trưng bổ sung | `c` `[B, 256]` và 2 số `[B, 2]` | nối cột | `[B, 258]` |
| B8. Đầu hồi quy | `[B, 258]` | `Linear 258→64`, ReLU, BatchNorm, Dropout 0,3, `Linear 64→1` | dự đoán `ŷ` `[B]`, đơn vị %/tháng |
| B9. Loss | `ŷ`, nhãn `y` | `L = (1/B) · Σ (ŷ − y)²` (MSE); Adam lr 1e-3, weight decay 1e-4 (cộng `1e-4·w` vào gradient), batch 1024, tối đa 100 epoch | một số vô hướng; cập nhật tham số |
| B10. Kiểm định và dừng sớm | tập kiểm định | sau mỗi vài epoch: tính MSE và `IC_val = Spearman(ŷ, y)` trên toàn tập kiểm định (gộp 3 năm); giữ mô hình có `IC_val` cao nhất; dừng sau 15 epoch không cải thiện | checkpoint tốt nhất (ở đây `IC_val` = 0,0169) |
| B11. Dự đoán tập kiểm tra | tập kiểm tra | chạy mô hình tốt nhất | `pred_ret_fincast` `[207800]`, lưu cùng khóa vào file parquet |
| B12. Chấm bằng IC | `ŷ`, `y`, `oos_begin_year` | với từng năm: `IC_Y = Spearman(ŷ, y)` trong năm đó; rồi trung bình và độ lệch chuẩn qua 8 năm | `IC` trung bình = **0,0393** (độ lệch chuẩn 0,0977) |
| B13. Chấm bằng nhóm | `ŷ`, `y` | trong từng năm chia 10 nhóm theo `ŷ`; lợi suất thực tế trung bình của nhóm; `spread_Y = nhóm 10 − nhóm 1`; trung bình qua năm; `t = mean(spread) / std(spread) · √(số năm)` | spread **0,0748**, t = **0,74** |
| B14. Danh mục top-k (ô 15) | `ŷ` gộp với thống kê của EB | dấu `sign(ŷ)`; điểm `abs(ŷ)` xếp hạng phần trăm **theo từng tháng**; top 1/5/10%; lợi suất tháng = trung bình đều của `sign · r` | chuỗi danh mục theo tháng, chỉ năm ≥ 2014 |
| B15. Bảng | chuỗi danh mục | lợi suất = trung bình × 12; Sharpe = trung bình / độ lệch chuẩn × √12; t = trung bình / độ lệch chuẩn × √(số tháng) | bảng Panel A–D (xem mục 5) |

Với mô hình (ii) (ô 13): B1 thay bằng 20 tháng lợi suất gần nhất trước năm Y, shape `[N, 20]` rồi `[B, 20, 1]` (N = 977.901, vì cần đủ 20 tháng); B5 là LSTM(1→32) hai chiều cho `[B, 20, 64]`; B6 cho `[B, 64]`; B7 cho `[B, 66]`; B8 là `Linear 66→32`, ReLU, Dropout 0,3, `Linear 32→1`. Phần còn lại giống nhau (cùng mốc năm, cùng loss MSE, cùng chọn checkpoint theo IC).

### 11.1 So sánh từng khía cạnh với paper gốc

| Khía cạnh | Paper gốc (EB) | Notebook B |
|---|---|---|
| Phạm vi | 6 họ gộp, chọn top-k từ toàn bộ | một họ mỗi lần |
| Đại lượng dự báo | `pred_ret = θ̂·m/t` (lợi suất trong mẫu bị co), cùng đơn vị %/tháng | `mean_ret_OOS` do mạng dự đoán trực tiếp, %/tháng |
| Phương pháp | công thức đóng, 5 tham số cho mỗi họ-năm | mạng 1,5 triệu tham số (i) hoặc 15 nghìn (ii), học một lần |
| Có dùng lợi suất tương lai để học không | **không**: chỉ nhìn phân phối `t` trong mẫu | **có**: nhãn `mean_ret_OOS` của các năm < 2011 |
| Cập nhật theo thời gian | ước lượng lại mỗi năm 1983–2020 | huấn luyện một lần, dùng cho mọi năm ≥ 2014 |
| Loss / mục tiêu | cực đại likelihood biên của `t` | MSE của lợi suất |
| Cách xếp hạng chọn top-k | `SRahat = sign·θ̂/√n·√12` (Sharpe dự báo, có tính rủi ro) | `abs(ŷ)` (chỉ độ lớn lợi suất dự đoán, không chia cho rủi ro) |
| Đầu vào chuỗi | 240 tháng, qua `m`, `s`, `t` | embedding FinCast (cửa sổ chưa rõ) hoặc 20 tháng; cộng `t`, `m` |
| Thước đo báo cáo | Table 2: lợi suất, t-stat, Sharpe của top 1/5/10% trên 1983–2020 | IC, spread, và bảng giống Table 2 nhưng chỉ 2014–2021, một họ |
| Công thức lợi suất/Sharpe/t của bảng | giống hệt (× 12; ÷ độ lệch chuẩn × √12; × √số tháng) | giống |

### 11.2 Chỗ cần cẩn thận khi đọc kết quả của notebook B

- **Cửa sổ của EB và của mô hình AI có vẻ lệch một năm.** Theo mã ô 22: hàng EB và Naive lọc `oos_begin_year <= YEAR_MAX` (YEAR_MAX = 2020), nên chỉ gồm 2014–2020 (7 năm); còn hàm `build_lstm_best_ret` cho mô hình AI chỉ lọc `>= 2014` và bảng IC của notebook có năm 2021, nên hàng của mô hình AI gồm 2014–2021 (8 năm). Đây là nhận xét từ việc đọc mã, mình chưa chạy lại để xác nhận. Nếu đúng, các hàng trong bảng so sánh không cùng giai đoạn.
- **Luật chọn không cùng loại:** EB xếp hạng theo Sharpe dự báo, mô hình AI theo `abs(ŷ)`. Một phần chênh lệch giữa các hàng của bảng có thể đến từ luật chọn chứ không chỉ từ chất lượng dự báo.
- **Chọn checkpoint theo IC trên 3 năm kiểm định** (2011–2013) rất bất ổn: Past Return FinCast có `IC_val` 0,43 nhưng IC kiểm tra −0,04.
- **MSE không phải thước báo cáo:** notebook chỉ in MSE lúc huấn luyện và kiểm định; dự án này bổ sung MSE trên tập kiểm tra và kiểm định Diebold–Mariano để so với EB.
