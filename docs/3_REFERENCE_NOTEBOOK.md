# Notebook B tham khảo làm gì: tuần tự từng bước, kèm dữ liệu thật

Notebook: `Research/reference_notebooks/submit_financial_fincast_high_throughput.ipynb` (40 ô): huấn luyện mô hình học sâu để dự báo lợi suất năm sau của từng chiến lược, rồi so với EB. (Notebook A, chỉ chạy lại EB của paper, mô tả ở [3_REFERENCE_NOTEBOOK.md](3_REFERENCE_NOTEBOOK.md).)

## Cách đọc các con số trong file này

Mỗi con số có nhãn nguồn:

- **[N]** là số **in nguyên văn trong notebook** (kết quả đã lưu). Đây là số chính thức của notebook.
- **[R]** là số mình **tính lại bằng dữ liệu của dự án** (`data_raw/`, `code/_cache/`) để có một ví dụ cụ thể mà notebook không in ra. Hai chỗ cần lưu ý:
  - Dữ liệu embedding FinCast của notebook nằm trên Hugging Face (`ngandh/fincast-embeddings`, hàng chục GB), **không có trên máy này**. Bước dùng embedding chỉ có số [N] của notebook.
  - Mình chạy lại mô hình (ii) (LSTM trên 20 tháng thô) để có dự báo cụ thể cho từng hàng. Kết quả gần giống chứ không trùng với notebook (xem mục B8).

Ví dụ xuyên suốt: **chiến lược 18144** (họ kế toán EW), năm dự báo **2000** (nằm trong tập huấn luyện) và **2020** (nằm trong tập kiểm tra).

---

Notebook B có **3 kiểu mô hình**:

| Kiểu | Ô | Đầu vào | Cách chia |
|---|---|---|---|
| (i) FinCast + Bi-LSTM + Attention | 10–11, 17–18, 24–25 | embedding 4 × 1280 + 2 số | cố định: train < 2011, val 2011–2013, test ≥ 2014 |
| (ii) Bi-LSTM + Attention trên chuỗi thô | 13, 19 | 20 tháng lợi suất cuối + 2 số | giống (i) |
| (iii) "New version to submit": LSTM 240 tháng | 28–39 | 240 tháng + mặt nạ | K-fold ngẫu nhiên trong cùng một năm |

Mô hình **(i)** là mô hình được dùng làm baseline của dự án.

## B0. Đăng nhập Hugging Face (ô 0)

`userdata.get('HF_TOKEN')` rồi `HfApi(token=...)`, để tải embedding.

## B1. Kiểm tra embedding có khớp đúng chiến lược không (ô 2–8)

Notebook có sẵn các biến `final_embeddings` (mảng embedding) và `final_keys_df` (bảng khóa). Các kiểm tra, và kết quả [N]:

| Ô | Kiểm tra | Kết quả in ra |
|---|---|---|
| 2 | Số hàng embedding = số hàng khóa | ✅ khớp |
| 3 | Khóa `(signal_family, signalid, oos_begin_year)` có trùng không | 0 dòng trùng |
| 4 | Cosine giữa 2 embedding của **cùng chiến lược, 2 năm liên tiếp** (signal 1, 1994 và 1995) so với **chiến lược khác cùng năm** | cùng: **0,8132**; khác: **0,9860** |
| 5 | Như ô 4, sau khi trừ vector trung bình chung (lấy mẫu 2.000 embedding) | cùng: **0,6513**; khác: **0,9747** |
| 6 | 100 cặp: cùng chiến lược liên tiếp so với một embedding ngẫu nhiên | cùng: TB **0,8486** (độ lệch chuẩn 0,0768); khác: TB **0,6789** (0,1744); t = **8,857**, p = 4,684e-16 |
| 7 | Hồi quy logistic đoán `signalid` từ embedding, 20 chiến lược, mỗi chiến lược 1 năm (2000) | accuracy **0,000** (đoán mò ≈ 0,050) |
| 8 | Như ô 7 nhưng 15 chiến lược, mỗi chiến lược nhiều năm | accuracy **0,203** (đoán mò ≈ 0,067, 15 lớp) |

Đọc bằng lời:
- Ô 4–5 (một cặp duy nhất) cho kết quả **ngược** kỳ vọng: cosine của hai embedding khác chiến lược lại cao hơn cùng chiến lược.
- Ô 6 (100 cặp) cho kết quả đúng chiều và có ý nghĩa thống kê.
- Ô 7 gần như không phân biệt được chiến lược; ô 8 (nhiều năm hơn) phân biệt được khoảng 3 lần đoán mò.
- Kết luận của notebook không được ghi lại bằng chữ; mình chỉ có các số trên.

## B1b. Hai file đầu vào cần biết trước: thống kê IS/OOS và lợi suất tháng

(Notebook A và B dùng chung các file này.)

File `OOS_signal_tstat_OosNyears1.csv.gzip` (và bản `Nelly_Predict_SignalYear.csv.gzip` có thêm dự báo của EB): mỗi chiến lược, mỗi năm Y có **hai dòng**:

| Cột | Ý nghĩa |
|---|---|
| `s_flag` = `IS` | 240 tháng (20 năm) **trước** Y. `mean_ret`, `tstat`, `nobs` = 240 |
| `s_flag` = `OOS` | 12 tháng của **năm Y**. `mean_ret`, `tstat`, `nobs` = 12 |
| `thetahat` | t-stat của EB sau khi co |
| `pred_ret` | dự báo lợi suất tháng của EB |

**Dữ liệu thật** (5 dòng đầu của `OOS_signal_tstat_OosNyears1.csv.gzip`, in từ ô 39 của notebook B) [N]:

```
                 signal_family  oos_begin_year  signalid  mean_ret     tstat  nobs s_flag  is_begin_year
0  DataMinedLongShortReturnsVW            1983        66  0.143098  0.825577 240.0     IS           1963
1  DataMinedLongShortReturnsVW            1983        66 -0.103663 -0.149915  12.0    OOS           1963
2  DataMinedLongShortReturnsVW            1983        67 -0.140025 -0.648427 240.0     IS           1963
3  DataMinedLongShortReturnsVW            1983        67 -0.249557 -0.238849  12.0    OOS           1963
4  DataMinedLongShortReturnsVW            1983        68 -0.039744 -0.242839 240.0     IS           1963
```

(Toàn bộ file: **10.227.400 dòng × 8 cột**: `signal_family, oos_begin_year, signalid, mean_ret, tstat, nobs, s_flag, is_begin_year` [N].)

**Ví dụ của mình** [R] với chiến lược 18144, hai năm, theo đúng định dạng của file:

| Năm | s_flag | nobs | mean_ret | tstat | thetahat | pred_ret |
|---|---|---|---|---|---|---|
| 2000 | IS (1980–1999) | 240 | −1,2527 | −9,7901 | −8,0705 | **−1,0327** |
| 2000 | OOS (2000) | 12 | −2,3967 | −1,2559 | | |
| 2020 | IS (2000–2019) | 240 | −0,6669 | −4,0115 | −2,7415 | **−0,4557** |
| 2020 | OOS (2020) | 12 | −0,2526 | | | |

(Giá trị OOS tstat năm 2000 = mean ÷ (độ lệch chuẩn ÷ √12), độ lệch chuẩn chia cho n.)

**Ghép với lợi suất tháng.** Ở bước so sánh (B9) notebook ghép bảng thống kê (mỗi chiến lược-năm 1 dòng) với lợi suất tháng của đúng năm đó, theo `(signal_family, signalid, oos_begin_year)`. Mỗi chiến lược-năm trở thành **12 dòng** (12 tháng của năm Y), mỗi dòng mang theo `ret_is`, `tstat_is`, `pred_ret`, `ret_oos` (lãi **tháng đó**). Bảng này tên là `df_est_oos`.

Ví dụ [R], chiến lược 18144 năm 2020 thành 12 dòng, lãi các tháng:

```
 T1     T2     T3     T4     T5     T6     T7     T8     T9    T10    T11    T12
-0,04  -0,19  -7,55  +0,91  -0,57  +0,87  -0,93  +1,53  +1,64  -0,89  +2,80  -0,62     trung bình −0,2526
```

Con số "trung bình −0,2526" chính là nhãn `mean_ret_OOS` của hàng (18144, 2020) mà notebook sẽ dùng.

## B2. Dữ liệu cho họ Accounting EW (ô 10–11, phần đầu)

### B2a. Tải embedding (ô 10)

```
FAMILY = 'DataMinedLongShortReturnsEW'     # chỉ 1 họ
hidden_arr = np.load(hidden_path, mmap_mode='r')   # không nạp hết vào RAM
hidden_arr shape: (1033883, 4, 1280)
```

Nghĩa là **1.033.883 embedding**, mỗi cái là **4 vector × 1280 số** (4 "patch"). FinCast chia chuỗi thành các mảnh 32 tháng, nên 4 patch ứng với 128 tháng; **cửa sổ chính xác mà notebook đưa vào FinCast không được ghi trong notebook** (mình suy ra 128 tháng từ hình dạng 4 × 32). Bảng khóa `keys` (file parquet `fincast_keys_DataMinedLongShortReturnsEW.parquet`) có các cột `signalid` (chuỗi), `signal_family`, `oos_begin_year` và `emb_idx` (chỉ số hàng của embedding). Hai hàng khóa đầu, in từ ô 4 [N]:

```
signal_family                  signalid  oos_begin_year
DataMinedLongShortReturnsEW    1         1994
DataMinedLongShortReturnsEW    1         1995
```

**Một embedding trông thế nào** (minh họa từ phần trích xuất FinCast riêng của dự án [R], không phải embedding của notebook; cùng kiến trúc FinCast v1, cửa sổ 128 tháng): cho chiến lược 18144 năm 2000, vector patch thứ nhất có 5 số đầu `0,406  −0,001  0,143  0,061  0,732` trong tổng số 1.280 số; tổng cộng 4 × 1.280 = 5.120 số.

### B2b. Ghép nhãn (ô 11, hàm `join_rollsignal_with_sequence`)

1. Đọc `OOS_signal_tstat_OosNyears1.csv.gzip` (10.227.400 dòng), chỉ giữ cột `signalid, signal_family, s_flag, oos_begin_year, tstat, mean_ret`.
2. Tách hai bảng:
   - **`is_df`**: dòng `IS`, đổi tên `tstat → tstat_IS`, `mean_ret → mean_ret_IS`;
   - **`oos_df`**: dòng `OOS`, đổi tên `mean_ret → mean_ret_OOS`.
3. Ghép `is_df` với bảng khóa theo `(signalid, signal_family, oos_begin_year)` để lấy `emb_idx`.
4. Ghép với `oos_df` (inner join).
5. Bỏ dòng thiếu `emb_idx`, `tstat_IS`, `mean_ret_IS`, `mean_ret_OOS`.
6. Kiểm tra `emb_idx` không trùng.

**Số in ra** [N]:

```
[Chẩn đoán] Family trong keys_df: ['DataMinedLongShortReturnsEW']
[Chẩn đoán] Family trong is_df (rollsignal, IS): ['DataMinedLongShortReturnsEW', 'DataMinedLongShortReturnsVW', 'PastReturnSignalsLongShort_ew', 'PastReturnSignalsLongShort_vw', 'TickerSignalsLongShort_ew', 'TickerSignalsLongShort_vw']
[Chẩn đoán] Số dòng is_df (toàn bộ family có trong rollsignal): 1005582
[Chẩn đoán] Số dòng keys_df (đã extract FinCast embedding):    1033883
[Chẩn đoán] Tỷ lệ đã extract / tổng cần có: 102.8%
Matched: 979380/5113700 (19.2%)
Số signal-window: 979380
```

Giải thích mấy con số này:
- `5.113.700` là tổng số dòng IS của **cả 6 họ**, nên "19,2%" thấp là do chia cho tất cả họ chứ không phải thiếu dữ liệu.
- Số embedding (1.033.883) nhiều hơn số dòng IS của họ này (1.005.582). Notebook không giải thích vì sao; mình không biết khóa nào có embedding mà không có dòng IS.
- Còn lại **979.380 hàng** dùng được: mỗi hàng là một cặp (chiến lược, năm) với embedding, 2 số trong mẫu và nhãn.

**Mỗi hàng huấn luyện trông thế nào** (dữ liệu thật của mình [R], hai hàng ví dụ):

| signalid | năm | embedding | tstat_IS | mean_ret_IS (%/tháng) | **nhãn** mean_ret_OOS (%/tháng) |
|---|---|---|---|---|---|
| 18144 | 2000 | mảng 4 × 1280 | −9,7901 | −1,2527 | **−2,3967** |
| 18144 | 2020 | mảng 4 × 1280 | −4,0115 | −0,6669 | **−0,2526** |

Lưu ý nhãn là **lợi suất gốc có dấu** (chưa đảo dấu). Mô hình dự báo cả số âm.

**Phân phối nhãn** trong tập huấn luyện [R]: trung bình −0,138, độ lệch chuẩn 1,153, nhỏ nhất −35,58, lớn nhất +34,27 (%/tháng). Có vài giá trị cực lớn, nên MSE bị chi phối bởi các điểm ngoại lai.

Tái lập số hàng bằng dữ liệu của dự án [R]: **980.653** hàng cho năm 1983–2021 (chênh khoảng 0,13% so với 979.380; chênh lệch nhỏ đến từ chi tiết lọc trong file của tác giả mà mình không có).

## B3. Chia theo năm cố định (ô 11)

```
Split theo năm CỐ ĐỊNH: train < 2011 (n=692447), val [2011, 2014) (n=79133), test >= 2014 (n=207800)
```

| Phần | Năm | Số hàng [N] | Số hàng tái lập [R] |
|---|---|---|---|
| Train | 1983–2010 | 692.447 | 692.718 |
| Validation | 2011–2013 | 79.133 | 79.190 |
| Test | 2014–2021 | 207.800 | 208.745 |

Mỗi năm test có khoảng 26.000 hàng (ví dụ 2014: 26.055 hàng [N], theo bảng IC ở B7). **Chỉ có một lần chia**: không có walk-forward theo năm.

## B4. Chuẩn hóa (ô 11)

Chỉ dùng thống kê **tập train** (không rò rỉ):

1. **Embedding:** tính trung bình và độ lệch chuẩn **từng chiều trong 1280 chiều**, trên mọi patch của mọi hàng train (692.447 × 4 vector). Tính từng khối 50.000 hàng một. Mỗi batch: `(embedding − trung bình) ÷ (độ lệch chuẩn + 1e-8)`.
2. **Hai số bổ sung** (`tstat_IS`, `mean_ret_IS`): `StandardScaler` fit trên train.

**Ví dụ hai số bổ sung** [R, thống kê train của dự án]: trung bình `tstat_IS` = −0,862, độ lệch chuẩn 1,890; trung bình `mean_ret_IS` = −0,134, độ lệch chuẩn 0,377.

| Hàng | tstat_IS | → chuẩn hóa | mean_ret_IS | → chuẩn hóa |
|---|---|---|---|---|
| 18144, 2000 | −9,790 | **−4,72** | −1,2527 | **−2,97** |
| 18144, 2020 | −4,012 | **−1,67** | −0,6669 | **−1,42** |

Nhãn `mean_ret_OOS` **không** được chuẩn hóa.

## B5. Mô hình (i): FinCast + Bi-LSTM + Attention (ô 11)

Mã lớp `LSTMAttentionRegressor`. Luồng dữ liệu trong một batch B = 1024 [suy từ mã]:

| Bước | Phép tính | Hình dạng |
|---|---|---|
| Đầu vào | embedding đã chuẩn hóa | [1024, 4, 1280] |
| Bi-LSTM, ẩn 128 mỗi chiều | LSTM hai chiều, nối hai chiều | [1024, 4, 256] |
| Điểm attention | `attn_v(tanh(attn_proj(h)))` cho từng patch | [1024, 4] |
| Trọng số attention | softmax qua 4 patch | [1024, 4] |
| Context | tổng có trọng số của 4 vector | [1024, 256] |
| Ghép 2 số bổ sung | nối cột | [1024, 258] |
| Linear 258 → 64, ReLU, BatchNorm, Dropout 0,3 | | [1024, 64] |
| Linear 64 → 1 | **dự báo `pred_ret`** | [1024] |

Số tham số học được: **1.526.657** (mình tính từ cấu trúc: Bi-LSTM 1.443.840 + attention 66.048 + đầu hồi quy 16.769).

## B6. Huấn luyện (ô 11)

- **Loss:** `MSELoss` giữa dự báo và `mean_ret_OOS`.
- **Tối ưu:** Adam, lr 1e-3, weight decay 1e-4, batch 1024, tối đa 100 epoch, mỗi epoch xáo trộn thứ tự.
- **Mỗi 5 epoch:** tính trên tập validation: MSE, và **IC** (tương quan hạng Spearman) trên **toàn bộ 79.133 hàng gộp 3 năm**.
- **Chọn checkpoint:** giữ mô hình có **IC validation cao nhất**; dừng sớm nếu 15 epoch không cải thiện.
- Seed 42.

**Nhật ký thật** [N] (họ Accounting EW):

```
Epoch 1/100 (31.8s) | Train MSE (running): 0.846007
Epoch 2/100 (30.5s) | Train MSE (running): 0.685243
Epoch 3/100 (31.5s) | Train MSE (running): 0.645686
Epoch 4/100 (31.4s) | Train MSE (running): 0.607183
Epoch 5/100 (65.4s) | Train MSE: 0.593264 | Val MSE: 0.733109 | Val IC: -0.0110
Epoch 6/100 (30.7s) | Train MSE (running): 0.569273
Epoch 7/100 (30.7s) | Train MSE (running): 0.543463
Epoch 8/100 (30.4s) | Train MSE (running): 0.535817
Epoch 9/100 (30.3s) | Train MSE (running): 0.538407
Epoch 10/100 (33.5s) | Train MSE: 0.525405 | Val MSE: 0.768277 | Val IC: -0.0286
Epoch 11/100 (30.5s) | Train MSE (running): 0.509286
Epoch 12/100 (30.4s) | Train MSE (running): 0.505176
Epoch 13/100 (30.5s) | Train MSE (running): 0.496646
Epoch 14/100 (30.4s) | Train MSE (running): 0.494040
Epoch 15/100 (33.5s) | Train MSE: 0.496767 | Val MSE: 0.758080 | Val IC: 0.0002
Epoch 16/100 (31.1s) | Train MSE (running): 0.490554
Epoch 17/100 (30.7s) | Train MSE (running): 0.481396
Epoch 18/100 (30.5s) | Train MSE (running): 0.478934
Epoch 19/100 (30.3s) | Train MSE (running): 0.482863
Epoch 20/100 (33.3s) | Train MSE: 0.500424 | Val MSE: 0.798899 | Val IC: -0.0110
Epoch 21/100 (30.6s) | Train MSE (running): 0.480174
Epoch 22/100 (30.4s) | Train MSE (running): 0.480630
Epoch 23/100 (30.4s) | Train MSE (running): 0.484702
Epoch 24/100 (30.5s) | Train MSE (running): 0.489227
Epoch 25/100 (33.9s) | Train MSE: 0.477212 | Val MSE: 0.758864 | Val IC: 0.0169
Epoch 26/100 (30.8s) | Train MSE (running): 0.462287
Epoch 27/100 (30.6s) | Train MSE (running): 0.459362
Epoch 28/100 (30.3s) | Train MSE (running): 0.452677
Epoch 29/100 (30.6s) | Train MSE (running): 0.442879
Epoch 30/100 (33.6s) | Train MSE: 0.461752 | Val MSE: 0.789387 | Val IC: -0.0367
Epoch 31/100 (30.6s) | Train MSE (running): 0.468781
Epoch 32/100 (30.4s) | Train MSE (running): 0.454964
Epoch 33/100 (30.5s) | Train MSE (running): 0.462004
Epoch 34/100 (30.6s) | Train MSE (running): 0.495044
Epoch 35/100 (33.8s) | Train MSE: 0.464203 | Val MSE: 0.830741 | Val IC: -0.0353
Epoch 36/100 (30.8s) | Train MSE (running): 0.469253
Epoch 37/100 (30.6s) | Train MSE (running): 0.450687
Epoch 38/100 (30.8s) | Train MSE (running): 0.454103
Epoch 39/100 (30.5s) | Train MSE (running): 0.456363
Epoch 40/100 (33.6s) | Train MSE: 0.438138 | Val MSE: 0.852722 | Val IC: -0.0277
⏹ Early stopping tại epoch 40
✅ Khôi phục best model (val_IC=0.0169)
```

Đọc bằng lời:
- **MSE huấn luyện giảm đều** (0,846 → 0,438) nhưng **MSE validation tăng** (0,733 → 0,853): mô hình học thuộc tập huấn luyện.
- **IC validation** chỉ dao động quanh 0 (từ −0,037 đến +0,017). Mô hình được chọn có IC 0,0169, ở epoch 25.
- Mất khoảng 31 giây mỗi epoch.

## B7. Chấm trên tập test và lưu dự báo (ô 11)

1. Dự báo toàn bộ 207.800 hàng test bằng mô hình tốt nhất.
2. Hàm `portfolio_relevant_report_yearly`:
   - **IC từng năm**: Spearman giữa dự báo và lợi suất thật, tính **riêng cho mỗi năm** (không gộp năm), rồi lấy trung bình và độ lệch chuẩn.
   - **Chia 10 nhóm theo dự báo, trong từng năm** (`pd.qcut`); tính lợi suất thật trung bình của mỗi nhóm; rồi lấy trung bình các năm.
   - **Spread** = nhóm 10 − nhóm 1, tính theo từng năm; t-stat = trung bình ÷ độ lệch chuẩn × √(số năm).
3. Lưu dự báo test vào file parquet (207.800 hàng × 4 cột: họ, signalid, năm, `pred_ret_fincast`).

**Kết quả in ra** [N]:

```
Mean yearly IC: 0.0393 (std=0.0977, n_years=8)
 oos_begin_year     n        IC
           2014 26055  0.108481
           2015 26039  0.197828
           2016 26014  0.119900
           2017 26009 -0.064924
           2018 26003  0.036925
           2019 25908  0.015116
           2020 25921 -0.089826
           2021 25851 -0.009044

Decile spread (trung bình các năm, KHÔNG pooled):
 decile     n  mean_pred  mean_actual_ret
      1 20784  -0.857500        -0.059796
      2 20779  -0.469646        -0.044541
      3 20780  -0.323342        -0.015845
      4 20779  -0.221960        -0.009013
      5 20781  -0.138421         0.000158
      6 20777  -0.060156         0.006203
      7 20779   0.020100         0.018075
      8 20780   0.113738         0.008853
      9 20779   0.243705        -0.003996
     10 20782   0.572945         0.014976

Long-short spread (trung bình theo năm): 0.0748, t-stat: 0.7427
Best validation IC: 0.0169
Mean yearly test IC: 0.0393
Kiểm tra: tồn tại=True, shape=(207800, 4)
```

Đọc bằng lời:
- IC trung bình **0,039**, độ lệch chuẩn giữa các năm **0,098**. Có 3 năm âm trong 8 năm (2017, 2020, 2021). Không đủ để nói mô hình xếp hạng được.
- Nhóm dự báo cao nhất (decile 10) có lãi thật trung bình chỉ **+0,015%**/tháng, và 3 nhóm đầu (decile 1–3) lỗ thật nhẹ (−0,06 đến −0,02). Đường lợi suất thật gần như phẳng, không tăng đều theo nhóm.
- Spread 0,0748 với t = 0,74: chưa đạt ngưỡng 2.

## B8. Mô hình (ii): LSTM + Attention trên 20 tháng thô, không FinCast (ô 13)

Dùng làm đối chứng "không có FinCast". Cùng mốc năm, cùng loss, cùng cách chọn checkpoint.

### B8a. Chuẩn bị dữ liệu (hàm `prepare_lookback_data`)

Với mỗi hàng của bảng ghép (chiến lược, năm Y):
1. Lấy **20 tháng cuối cùng trước năm Y** (tháng 12 năm Y−1 lùi về 20 tháng), từ file lợi suất tháng.
2. Nếu chuỗi **không đủ 20 tháng** thì bỏ hàng này.
3. Đầu vào: chuỗi 20 số + 2 số bổ sung (như B4).

**Số in ra** [N]: `seq_arr shape: (977901, 20)`, chia thành train **691.348**, val **79.010**, test **207.543** (ít hơn mô hình (i) vì phải đủ 20 tháng liền).

**Ví dụ thật** [R], hàng (18144, 2000): 20 tháng từ 5/1998 đến 12/1999:

```
raw (%):
-0.80 -0.26 -0.61 -3.98 -1.40 -1.51 -8.58 -3.20     ← tháng 5–12/1998
-4.18 -2.48 -0.28 +1.23 -2.30 -2.06 -1.26 -4.82 -4.75 -5.50 -10.30 -7.88   ← 12 tháng 1999

chuẩn hóa = (x − trung bình train) ÷ độ lệch chuẩn train, với trung bình −0,066 và độ lệch chuẩn 3,302 [R]:
-0.22 -0.06 -0.17 -1.19 -0.41 -0.44 -2.58 -0.95 -1.25 -0.73 -0.06 +0.39
-0.68 -0.60 -0.36 -1.44 -1.42 -1.65 -3.10 -2.37
```

(Hai hàng đầu của bảng ví dụ nói chuỗi lỗ nặng ở cuối năm 1999; mô hình thấy −10,30 và −7,88 ở hai tháng cuối.)

Hàng (18144, 2020) [R]: chuỗi từ 5/2018 đến 12/2019:

```
raw: -3.38 +3.13 +0.43 -1.01 +0.07 -1.25 +0.25 -0.87 +3.21 -2.38 -2.13 +1.00 +0.58 +0.26 +0.61 +0.67 +1.96 +0.51 +1.95 -2.55
```

### B8b. Mô hình

`LSTMAttentionRaw`: Bi-LSTM (1 → 32 mỗi chiều) → attention qua 20 bước → context 64 → ghép 2 số (66) → Linear 66 → 32, ReLU, Dropout 0,3 → Linear 32 → 1. **15.361 tham số.** Hình dạng: [B, 20, 1] → [B, 20, 64] → [B, 64] → [B, 66] → [B, 32] → [B]. Batch 4096; Adam lr 1e-3, weight decay 1e-4; MSE; **mỗi epoch** tính IC validation; dừng sau 15 epoch không cải thiện.

### B8c. Kết quả của notebook [N]

```
seq_arr shape: (977901, 20)
Split ...: train < 2011 (n=691348), val [2011, 2014) (n=79010), test >= 2014 (n=207543)
⏹ Early stopping tại epoch 16, best_val_ic=0.1542

Mean yearly IC: 0.0839 (std=0.1308, n_years=8)
 oos_begin_year     n        IC
           2014 25981  0.068301
           2015 26012  0.294372
           2016 25999  0.237898
           2017 25994  0.007161
           2018 25982  0.106012
           2019 25879  0.043192
           2020 25869 -0.117723
           2021 25827  0.031617

Decile spread (trung bình các năm, KHÔNG pooled):
 decile     n  mean_pred  mean_actual_ret
      1 20758  -0.476405        -0.142752
      2 20754  -0.275857        -0.032841
      3 20753  -0.164823        -0.079122
      4 20755  -0.097058        -0.044680
      5 20754  -0.054876         0.017388
      6 20750  -0.025951         0.011827
      7 20755  -0.001503         0.036289
      8 20753   0.022890         0.040332
      9 20754   0.052692         0.052615
     10 20757   0.113834         0.068745

Long-short spread (trung bình theo năm): 0.2115, t-stat: 1.2203
Best validation IC (V3, No FinCast): 0.1542
Mean yearly test IC: 0.0839
Kiểm tra: tồn tại=True, shape=(207543, 4)
```

**Điều đọc được từ mã và nhật ký:** dừng sớm ở epoch 16 với kiên nhẫn 15 nghĩa là **epoch tốt nhất là epoch 1**. Mô hình được chọn gần như chưa kịp học gì: nó là mô hình sau một epoch huấn luyện.

### B8d. Mình chạy lại bước này để có dự báo cụ thể cho từng hàng [R]

Dùng dữ liệu của dự án (978.868 hàng, train 691.497 / val 79.034 / test 208.337), cùng mã (cùng cấu trúc, siêu tham số, seed 42). Kết quả gần chứ không trùng với notebook:

| | Notebook [N] | Chạy lại [R] |
|---|---|---|
| Số tham số | 15.361 | 15.361 |
| Dừng sớm ở epoch | 16 | 17 |
| Epoch tốt nhất | 1 | 2 |
| Val IC tốt nhất | 0,1542 | 0,1509 |
| IC test trung bình theo năm | 0,0839 | 0,0673 |
| Spread t-stat | 1,22 | 0,61 |

Nhật ký chạy lại: val IC 0,148 (epoch 1), **0,151 (epoch 2)**, rồi dao động trong khoảng 0,05–0,14 trong khi MSE huấn luyện giảm đều từ 1,30 xuống 0,95. Giống notebook: IC validation đạt đỉnh ngay từ epoch 1–2.

Khác biệt nhỏ giữa hai lần chạy cho thấy kết quả có **độ nhiễu lớn giữa các lần chạy** (dữ liệu lọc hơi khác nên số hàng khác đi chút ít). IC test từng năm của lần chạy lại: 2014 0,013; 2015 0,327; 2016 0,238; 2017 −0,025; 2018 0,031; 2019 0,042; **2020 −0,303**; 2021 0,215.

## B9. So sánh với EB và Naive (ô 15, họ Accounting EW)

Đây là bước tạo bảng danh mục top 1/5/10%.

### B9a. Dựng bảng ghép

1. Đọc lợi suất tháng của họ kế toán EW, đọc file ước lượng của EB (`Nelly_Predict_SignalYear.csv.gzip`), lọc họ `acct_ew`.
2. Ghép lại như B1b: **12.066.984 dòng** `df_est_oos` (= 1.005.582 chiến lược-năm × 12 tháng) [N].
3. Cho EB và Naive: `df_signed = df_est_oos` với **`oos_begin_year <= 2020`**.

### B9b. Bốn cách chọn

| Cách | Dấu | Điểm xếp hạng | Xếp hạng trong |
|---|---|---|---|
| **EB Mining** | dấu của `pred_ret` của EB | **Sharpe dự báo** `SRahat = dấu × thetahat ÷ √nobs × √12` | từng tháng |
| **Naive Mining** | dấu của t-stat quá khứ | \|t-stat quá khứ\| | từng tháng |
| **FinCast Mining** | dấu của dự báo mô hình (i) | **\|dự báo\|** (độ lớn dự báo, **không chia cho rủi ro**) | từng tháng |
| **No-FinCast Mining** | dấu của dự báo mô hình (ii) | \|dự báo\| | từng tháng |

Với hai cách AI: ghép dự báo (file parquet ở B7, B8) với `df_est_oos`, rồi **`query('oos_begin_year >= 2014')` và không có giới hạn trên**. Số dòng sau ghép [N]: FinCast **2.493.600** (= 207.800 × 12), No-FinCast **2.490.516** (= 207.543 × 12). Nên mô hình AI có **8 năm (2014–2021)** còn EB và Naive chỉ có **7 năm (2014–2020)**. Đây là điều mình đọc ra từ mã (`YEAR_MAX = 2020` chỉ lọc ở nhánh EB/Naive) và khớp với số dòng.

### B9c. Lấy top k% và tính lãi

1. Mỗi tháng xếp hạng phần trăm điểm của tất cả chiến lược đang có hàng trong tháng đó (~26.000 chiến lược).
2. Giữ `rank_pct ≥ 100 − k` (k = 1, 5, 10): ~261, ~1.300, ~2.600 chiến lược.
3. Lãi danh mục tháng = trung bình đều của `dấu × lãi tháng` trên các chiến lược được chọn.
4. Gom các tháng: lãi/năm = TB tháng × 12; t-stat = TB ÷ độ lệch chuẩn × √(số tháng); Sharpe = TB ÷ độ lệch chuẩn × √12.

### B9d. Bảng cuối [N] (Accounting EW, test ≥ 2014)

```
=== Bảng so sánh — family: Accounting EW, test years >= 2014 ===
                           nstrat_mean ret_mean tstat    sr
Panel A
EB Mining Top 1%                  261     1.53  1.05  0.40
EB Mining Top 5%                 1300     1.74  1.48  0.56
EB Mining Top 10%                2600     1.96  1.92  0.73

Panel B
Naive Mining Top 1%               261     1.81  1.26  0.47
Naive Mining Top 5%              1300     1.67  1.44  0.54
Naive Mining Top 10%             2600     1.83  1.81  0.68

Panel C
FinCast Mining Top 1%             260     0.78  0.80  0.28
FinCast Mining Top 5%            1299     0.56  0.95  0.34
FinCast Mining Top 10%           2598     0.61  1.16  0.41

Panel D
No-FinCast Mining Top 1%          260     2.16  0.99  0.35
No-FinCast Mining Top 5%         1298     2.17  1.39  0.49
No-FinCast Mining Top 10%        2595     1.71  1.32  0.47
```

(Cột `ret_mean` là %/năm, `sr` là Sharpe quy năm.) Notebook cũng lưu `beststrats_acct_ew.tex` và vẽ lợi suất cộng dồn top 1% và top 5% của 4 cách (`beststrats-cret_acct_ew.pdf`).

### B9e. Mình tái lập EB và Naive để kiểm tra bảng này [R]

Dùng dự báo EB tính bằng mã của dự án (đã kiểm khớp với paper) trên cùng 7 năm 2014–2020:

| Cách | Top | Lãi %/năm | t-stat | Sharpe | Notebook [N] |
|---|---|---|---|---|---|
| EB | 1% | 1,54 | 1,05 | 0,40 | 1,53 / 1,05 / 0,40 |
| EB | 5% | 1,76 | 1,50 | 0,57 | 1,74 / 1,48 / 0,56 |
| EB | 10% | 1,96 | 1,92 | 0,73 | 1,96 / 1,92 / 0,73 |
| Naive | 1% | 1,81 | 1,26 | 0,48 | 1,81 / 1,26 / 0,47 |
| Naive | 5% | 1,67 | 1,44 | 0,55 | 1,67 / 1,44 / 0,54 |
| Naive | 10% | 1,83 | 1,81 | 0,68 | 1,83 / 1,81 / 0,68 |

Khớp với số của notebook (sai lệch ở chữ số thập phân thứ hai). Điều này cho thấy (1) dữ liệu và EB của dự án giống dữ liệu của notebook; (2) nhánh EB/Naive của notebook phù hợp với việc chỉ có 2014–2020 (khi tái lập đúng 7 năm này thì ra đúng số của notebook).

### B9f. Một tháng cụ thể: tháng 1/2020 [R]

Dùng dự báo của mô hình (ii) mình chạy lại (B8d), EB và Naive. Năm 2020 có **26.017** chiến lược trong tập test; top 1% = **261** chiến lược.

Phân phối dự báo năm 2020 của mô hình (ii) [R]: trung bình −0,104, độ lệch chuẩn 0,268, nhỏ nhất −1,958, lớn nhất +1,028. Lợi suất thật năm 2020: trung bình −0,452, độ lệch chuẩn 1,711.

**Chiến lược 18144 năm 2020** (một hàng cụ thể qua cả ba cách) [R]:

| | Con số |
|---|---|
| Quá khứ 2000–2019 | mean_ret_IS −0,667, t-stat −4,01 |
| Dự báo mô hình (ii) | −0,243 (hạng 75,4% theo \|dự báo\|, **không vào top 1%**) |
| EB | θ̂ = −2,742; dự báo −0,456; Sharpe dự báo = 2,742 ÷ √240 × √12 = 0,613; hạng 98,5% (**không vào top 1%**) |
| Naive | \|t\| = 4,01, hạng 98,4% (không vào top 1%) |
| Thật 2020 | −0,253 |

**8 chiến lược đứng đầu top 1% tháng 1/2020** của từng cách [R]:

*Mô hình (ii)*, xếp theo |dự báo|:

| signalid | mean_ret_IS | t-stat quá khứ | dự báo | dấu | lãi tháng 1/2020 (gốc) | lãi thật cả năm (gốc) |
|---|---|---|---|---|---|---|
| 25211 | −0,451 | −0,854 | −1,958 | − | −8,009 | −1,809 |
| 26611 | −0,620 | −1,106 | −1,885 | − | −7,351 | −2,217 |
| 26578 | −0,619 | −1,104 | −1,885 | − | −7,351 | −2,221 |
| 26595 | −0,572 | −1,433 | −1,834 | − | −5,096 | +1,526 |
| 10472 | −0,120 | −0,315 | −1,822 | − | −13,779 | +0,121 |
| 26621 | −0,368 | −0,746 | −1,803 | − | −7,048 | −0,922 |
| 26594 | −0,357 | −0,721 | −1,799 | − | −7,051 | −0,928 |
| 26591 | −0,831 | −1,715 | −1,794 | − | −6,061 | +1,723 |

*EB*, xếp theo Sharpe dự báo:

| signalid | mean_ret_IS | t-stat quá khứ | θ̂ | dự báo | dấu | Sharpe dự báo | lãi tháng 1/2020 (gốc) | lãi thật cả năm (gốc) |
|---|---|---|---|---|---|---|---|---|
| 16227 | −0,953 | −6,392 | −4,162 | −0,620 | − | 0,931 | −3,402 | +0,439 |
| 15281 | −0,858 | −6,318 | −4,117 | −0,559 | − | 0,921 | +0,227 | +1,117 |
| 15305 | −0,737 | −5,921 | −3,881 | −0,483 | − | 0,868 | +1,091 | +1,266 |
| 11271 | −1,272 | −5,824 | −3,823 | −0,835 | − | 0,855 | −4,991 | +2,186 |
| 20658 | −0,824 | −5,716 | −3,759 | −0,542 | − | 0,840 | +1,135 | +1,263 |
| 26328 | −0,898 | −5,585 | −3,681 | −0,592 | − | 0,823 | +6,225 | +2,114 |
| 20684 | −1,031 | −5,582 | −3,679 | −0,680 | − | 0,823 | +4,747 | +1,749 |
| 15314 | −0,763 | −5,540 | −3,654 | −0,503 | − | 0,817 | −0,092 | +0,804 |

(Lãi gốc dương nghĩa là chiến lược gốc lãi; vì dấu là −, **lãi của mình = −lãi gốc**.)

Số liệu danh mục top 1% năm 2020 [R]:

| | Mô hình (ii) | EB | Naive |
|---|---|---|---|
| Số chiến lược | 261 | 261 | 261 |
| Lãi tháng 1/2020 của danh mục | +1,66 | +0,41 | +0,50 |
| Lãi trung bình cả năm 2020 (%/tháng) | **+0,83** | **−0,68** | **−0,61** |
| \|t-stat quá khứ\| trung bình của các chiến lược được chọn | **1,05** | 4,62 | 4,63 |
| Phần chiến lược bị đảo dấu | 99,6% | | |
| Số chiến lược trùng với EB / với Naive | 1 / 1 | | |

Đọc bằng lời:
- Mô hình (ii) xếp hạng bằng |dự báo| nên chọn những chiến lược có **vài tháng lỗ rất nặng gần đây** (ví dụ −13,8% hay −8%) và dự báo là lỗ tiếp (rồi đảo dấu). Các chiến lược này có t-stat quá khứ **nhỏ** (|t| trung bình 1,05).
- EB và Naive chọn chiến lược có |t| lớn (≈ 4,6).
- Trong riêng năm 2020 (một năm bất thường), EB và Naive lỗ và mô hình (ii) lãi. Không thể kết luận gì từ một năm; 8 năm test nói điều khác (B9d).

**Lãi danh mục top 1% từng tháng của năm 2020** (%/tháng) [R]:

| | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | T11 | T12 | TB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mô hình (ii) | 1,66 | 0,53 | −4,81 | 2,44 | 1,87 | 0,87 | −2,09 | 1,55 | −1,05 | 1,93 | 4,91 | 2,14 | +0,83 |
| EB | 0,41 | −1,90 | 1,91 | −2,29 | 0,26 | −3,04 | −2,46 | 1,43 | 1,72 | 0,32 | −3,52 | −0,96 | −0,68 |
| Naive | 0,50 | −1,87 | 1,86 | −2,14 | 0,37 | −2,90 | −2,28 | 1,36 | 1,50 | 0,37 | −3,35 | −0,70 | −0,61 |

**Bảng top-k của lần chạy lại mô hình (ii)** [R], để thấy cùng bảng trong B9d sinh ra từ dữ liệu cụ thể:

| Cách | Giai đoạn | Top | Lãi %/năm | t-stat | Sharpe |
|---|---|---|---|---|---|
| (ii) chạy lại | 2014–2021 (8 năm, 96 tháng) | 1% | 3,78 | 1,33 | 0,47 |
| | | 5% | 1,57 | 1,08 | 0,38 |
| | | 10% | 1,63 | 1,56 | 0,55 |
| (ii) chạy lại | 2014–2020 (7 năm, 84 tháng) | 1% | 3,64 | 1,16 | 0,44 |
| | | 5% | 1,16 | 0,75 | 0,28 |
| | | 10% | 1,10 | 1,02 | 0,39 |

(Lần chạy lại này cho top 1% là 3,78%/năm so với 2,16% của notebook: cho thấy bảng top-k của mô hình (ii) dao động nhiều giữa các lần chạy.)

## B10. Họ Past Return EW (ô 17–22)

Cùng quy trình (i), (ii) và so sánh như B2–B9, đổi `FAMILY = 'PastReturnSignalsLongShort_ew'`.

### Dữ liệu và chia [N]

```
hidden_arr shape: (756678, 4, 1280)
[Chẩn đoán] Số dòng is_df (toàn bộ family có trong rollsignal): 776080
[Chẩn đoán] Số dòng keys_df (đã extract FinCast embedding):    756678
[Chẩn đoán] Tỷ lệ đã extract / tổng cần có: 97.5%
Matched: 756678/5113700 (14.8%)
Số signal-window: 756678
Split: train < 2011 (n=543256), val [2011, 2014) (n=58206), test >= 2014 (n=155216)
```

(Họ này có 19.402 chiến lược mỗi năm, nên mỗi năm test có đúng 19.402 hàng.)

### Mô hình (i) [N]

Nhật ký chọn lọc: epoch 5 Val IC 0,3365; epoch 10 0,3515; **epoch 15 0,4339**; epoch 20 0,3297; epoch 25 0,3703; epoch 30 0,2523, dừng sớm ở epoch 30. Train MSE từ 0,2787 (epoch 1) xuống 0,1274 (epoch 30). Val MSE từ 0,522 lên 0,541.

```
Mean yearly IC: -0.0364 (std=0.2644, n_years=8)
 oos_begin_year     n        IC
           2014 19402 -0.055372
           2015 19402  0.360566
           2016 19402  0.050196
           2017 19402 -0.201539
           2018 19402 -0.178541
           2019 19402  0.220061
           2020 19402 -0.493394
           2021 19402  0.006548

 decile     n  mean_pred  mean_actual_ret
      1 15528  -0.447977         0.317006
      2 15520  -0.238405         0.193942
      3 15520  -0.169025         0.078406
      4 15520  -0.122227        -0.057299
      5 15520  -0.083129        -0.145769
      6 15520  -0.045285        -0.166867
      7 15520  -0.002800        -0.155250
      8 15520   0.051681        -0.139693
      9 15520   0.139201        -0.123438
     10 15528   0.381015        -0.082908
Long-short spread: -0.3999, t-stat: -0.8765
Best validation IC: 0.4339
```

Nhóm dự báo thấp nhất có lãi thật **cao nhất** (+0,317) và nhóm cao nhất âm (−0,083): dự báo **ngược chiều** thực tế. IC validation 0,43 nhưng IC test −0,04: chọn checkpoint trên chỉ 3 năm validation rất bất ổn.

### Mô hình (ii) [N]

```
seq_arr shape: (756678, 20)
⏹ Early stopping tại epoch 29, best_val_ic=0.0191
Mean yearly IC: 0.0556 (std=0.2983, n_years=8)
 oos_begin_year     n        IC
           2014 19402 -0.356002
           2015 19402  0.094183
           2016 19402  0.362636
           2017 19402 -0.051181
           2018 19402 -0.288451
           2019 19402  0.083177
           2020 19402  0.531895
           2021 19402  0.068425

 decile     n  mean_pred  mean_actual_ret
      1 15528  -0.658730        -0.239849
      2 15520  -0.346153        -0.105048
      3 15520  -0.234233        -0.072024
      4 15520  -0.149673        -0.100042
      5 15520  -0.102991        -0.063486
      6 15520  -0.055395        -0.061690
      7 15520   0.031119        -0.113430
      8 15520   0.156318        -0.038024
      9 15520   0.343057         0.146284
     10 15528   0.814763         0.365497
Long-short spread: 0.6053, t-stat: 0.7909
Best validation IC: 0.0191
```

(Dừng sớm ở epoch 29 với kiên nhẫn 15: epoch tốt nhất là 14.)

### Bảng so sánh Past Return EW [N]

```
=== Bảng so sánh — family: Past Return EW, test years >= 2014 ===
                          nstrat_mean ret_mean  tstat    sr
Panel A
EB Mining Top 1%                  195    -4.69  -2.57  -0.97
EB Mining Top 5%                  971    -3.45  -2.42  -0.91
EB Mining Top 10%                1941    -2.66  -2.20  -0.83

Panel B
Naive Mining Top 1%               195    -4.71  -2.59  -0.98
Naive Mining Top 5%               971    -3.46  -2.57  -0.97
Naive Mining Top 10%             1941    -2.47  -2.32  -0.88

Panel C
FinCast Mining Top 1%             195     2.60   0.76   0.27
FinCast Mining Top 5%             971    -0.70  -0.31  -0.11
FinCast Mining Top 10%           1941    -1.96  -0.92  -0.33

Panel D
No-FinCast Mining Top 1%          195     6.18   1.27   0.45
No-FinCast Mining Top 5%          971     6.18   1.36   0.48
No-FinCast Mining Top 10%        1941     6.24   1.42   0.50
```

Số dòng ghép: `df_est_oos` 9.312.960; FinCast 1.862.592 và No-FinCast 1.862.592 (= 155.216 × 12) [N]. Cùng điểm lệch 2014–2020 của EB/Naive so với 2014–2021 của AI như B9b.

Đọc bằng lời: EB và Naive lỗ trong giai đoạn 2014–2020 ở họ này (Sharpe −0,97 đến −0,83), còn cách No-FinCast có lãi dương (0,45–0,50). Với 8 năm test, các t-stat đều dưới 2,6 và kết luận rất yếu.

### Ô 21

`print(sorted(tmp['signal_family'].unique()))` in ra 6 họ: `DataMinedLongShortReturnsEW`, `DataMinedLongShortReturnsVW`, `PastReturnSignalsLongShort_ew`, `PastReturnSignalsLongShort_vw`, `TickerSignalsLongShort_ew`, `TickerSignalsLongShort_vw` [N].

## B11. Thử họ Accounting VW (ô 24–25, chưa xong)

Quy trình (i) cho `DataMinedLongShortReturnsVW` [N]:

```
hidden_arr shape = (1033882, 4, 1280)
[Chẩn đoán] Số dòng is_df: 1005558 ; keys_df: 1033882 (102.8%)
Matched: 979356/5113700 (19.2%)
Split: train < 2011 (n=692423), val (n=79133), test >= 2014 (n=207800)
Epoch 5/100  | Train MSE: 0.803273 | Val MSE: 0.773860 | Val IC: -0.0262
Epoch 10/100 | Train MSE: 0.681170 | Val MSE: 0.807809 | Val IC: -0.0479
Epoch 15/100 | Train MSE: 0.622332 | Val MSE: 0.797979 | Val IC: -0.0142
Epoch 20/100 | Train MSE: 0.584699 | Val MSE: 0.815765 | Val IC: -0.0388
Epoch 22: Train MSE (running) 0.610320     ← nhật ký lưu trong notebook dừng ở đây
```

Notebook **không lưu kết quả test** của họ này. Val IC luôn âm (−0,05 đến −0,01).

## B12. Mô hình (iii): "New version to submit" (ô 28–39)

Một cách làm khác hẳn: **không** dùng FinCast, **không** chia theo năm, mà chạy kiểu walk-forward theo năm nhưng chia K-fold **ngẫu nhiên các chiến lược trong cùng một năm**.

### B12a. Kiểm tra dữ liệu lợi suất (ô 29–31)

Đọc `DataMinedLongShortReturnsEW.csv` và in thống kê [N]:

| Mục | Kết quả |
|---|---|
| Kích thước | 16.941.504 dòng × 5 cột (`signalid, year, month, ret, nstock`) |
| Số chiến lược | 29.314; năm 1951–2022 |
| Thiếu giá trị | 0 ở mọi cột |
| `ret` (%/tháng) | trung bình −0,107; độ lệch chuẩn 3,504; nhỏ nhất −296,8; lớn nhất +228,3 |
| Phân vị của `ret` | 1%: −9,68; 5%: −4,66; 50%: −0,064; 95%: +4,35; 99%: +8,72 |
| `nstock` | trung bình 928,8; trung vị 554; nhỏ nhất 2; lớn nhất 7.573 |
| Độ dài chuỗi mỗi chiến lược (tháng) | trung bình 577,9; trung vị 594; nhỏ nhất 12; lớn nhất 858 |
| Chiến lược có ≥ 61 tháng | 29.299 / 29.314 |
| Chiến lược có ít nhất 1 chỗ đứt khoảng (mẫu 500) | 47 chiến lược; trung bình 0,192 chỗ đứt mỗi chiến lược |
| Dòng \|ret\| > 50% | 3.492 (0,021%); `nstock` trung vị của chúng chỉ **3** (toàn bộ: 554) |
| Chiến lược bắt đầu trước 1963 | 7.286 |
| Chiến lược kết thúc năm 2022 | 26.055 / 29.314 |
| `nstock` trung vị theo năm | 66 (1951), 78 (1963), 112 (1965) … 614 (2013), 630 (2018–2020), 660 (2022) |
| Số chiến lược có dữ liệu mỗi năm | 2.048 (1951), 5.942 (1952), … 26.144 (2018), 26.055 (2022) |
| Chỗ đứt khoảng (đầy đủ) | trung bình 0,187 mỗi chiến lược; lớn nhất 14 chỗ; khoảng đứt dài nhất 336 tháng |
| Chiến lược có khoảng đứt > 3 tháng | 2.420 |

Đọc bằng lời: ngoại lai cực lớn (±200–300%) hầu hết đến từ chiến lược trong tháng chỉ có 2–3 cổ phiếu. Notebook sau đó lọc **`year ≥ 1963`** và **`nstock ≥ 10`**, rồi cắt thành các đoạn liên tục (ô 31). Kết quả [N]:

```
Số (signal, segment) đoạn liên tục: 34633
Số đoạn đủ dài (>=61 tháng): 29471 / 34633
Số dòng dữ liệu sau lọc: 15921773 / 15976783
Số signal unique còn lại: 29263
```

### B12b. Đọc lợi suất (ô 32)

In: `dict_keys(['DataMinedLongShortReturnsEW'])`, shape `(16941504, 4)`; 5 dòng đầu: signal 1 từ 1990-07-01, lãi 1,542175; 0,476569; 0,274472; 4,070882; 2,742471 [N].

### B12c. Dựng dữ liệu cho từng năm Y (ô 35, hàm `build_lstm_dataset`)

Với mỗi năm Y từ 1983 đến 2019:
1. Lấy 240 tháng trước Y (`is_begin = Y − 20 ≤ year < Y`).
2. Nhãn: `mean_ret` của dòng `OOS` của chiến lược trong năm Y.
3. Bỏ chiến lược có dưới 36 tháng; nếu thiếu thì **đệm giá trị thiếu ở đầu chuỗi** cho đủ 240.
4. Mặt nạ: ô nào có dữ liệu thật; ô thiếu thay bằng 0.
5. **Chuẩn hóa bằng trung bình và độ lệch chuẩn của chính năm Y** (trên các ô có dữ liệu).
6. Đầu vào LSTM: chuỗi 240 bước, mỗi bước 2 kênh (giá trị, mặt nạ).

Số in ra (ô 36) [N]: `build_lstm_dataset(1983)` cho `X.shape = (20477, 240)`, `y.shape = (20477,)`.

### B12d. Mô hình `ReturnLSTM` và huấn luyện kiểu out-of-fold (ô 35)

- Mô hình: LSTM một chiều, ẩn 32, đầu vào 2 kênh; lấy trạng thái ẩn cuối rồi `Linear(32, 1)`. **4.641 tham số** (mình tính từ mã).
- **Với mỗi năm Y:** chia **ngẫu nhiên 5 phần** (`KFold(5, shuffle=True)`) các chiến lược **của năm Y**. Huấn luyện trên 4 phần, dự báo phần còn lại, lặp 5 lần để mọi chiến lược có đúng một dự báo "out-of-fold".
- Loss MSE, Adam 1e-3, batch 512, tối đa 200 epoch, dừng sớm theo MSE của phần còn lại (kiên nhẫn 10).
- Kết quả: `lstm_predictions_acct_ew_oof.csv`.

Nhật ký thật của một số năm [N] (MSE phần còn lại của 5 fold):

| Năm | n | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---|---|---|---|---|---|
| 1983 | 20.477 | 0,2923 | 0,2900 | 0,3187 | 0,2830 | 0,3195 |
| 1990 | 24.228 | 0,8461 | 0,7636 | 0,8573 | 0,9373 | 0,8658 |
| 1991 | 24.096 | 1,3950 | 1,3987 | 1,1406 | 1,7685 | 1,3459 |
| 1999 | 25.725 | 2,2120 | 1,8644 | 2,3025 | 2,6047 | 1,7650 |
| 2000 | 25.603 | 1,5992 | 1,5058 | 1,4690 | 1,5157 | 1,4358 |
| 2006 | 26.488 | 0,1670 | 0,1724 | 0,1756 | 0,1857 | 0,1905 |
| 2019 | 25.908 | 0,4506 | 0,3259 | 0,4307 | 0,3454 | 0,4139 |

(Notebook in đủ 37 năm; MSE thay đổi mạnh theo năm vì phương sai lợi suất thay đổi.)

### B12e. Chấm điểm (ô 38) và đối chứng (ô 39) [N]

Với mỗi năm, tương quan **Pearson** giữa dự báo out-of-fold và lợi suất thật của năm đó:

```
    year  corr      n
0   1983  0.75  20477     19  2002  0.86  25926
1   1984  0.86  20572     20  2003  0.84  25942
2   1985  0.77  20689     21  2004  0.57  26143
3   1986  0.70  20773     22  2005  0.75  26390
4   1987  0.71  21269     23  2006  0.59  26488
5   1988  0.62  21644     24  2007  0.64  26541
6   1989  0.54  23123     25  2008  0.64  26523
7   1990  0.53  24228     26  2009  0.80  26432
8   1991  0.62  24096     27  2010  0.61  26478
9   1992  0.56  25114     28  2011  0.70  26438
10  1993  0.37  26057     29  2012  0.59  26376
11  1994  0.65  26211     30  2013  0.66  26376
12  1995  0.54  26346     31  2014  0.63  26061
13  1996  0.50  26453     32  2015  0.64  26048
14  1997  0.54  26224     33  2016  0.65  26018
15  1998  0.46  25481     34  2017  0.57  26011
16  1999  0.70  25725     35  2018  0.68  26004
17  2000  0.59  25603     36  2019  0.42  25908
18  2001  0.58  25770
Correlation trung bình: 0.633
```

Ô 39 tính mốc đối chứng: **tương quan giữa lợi suất trong mẫu (IS) và ngoài mẫu (OOS) của chính chiến lược**, từng năm [N]: 1983 0,523; 1984 −0,057; 1985 0,000; 1986 0,265; 1987 0,265; 1988 0,358; 1989 0,183; 1990 0,135; 1991 −0,079; 1992 0,154; 1993 0,177; 1994 0,363; 1995 0,158; 1996 0,297; 1997 0,249; 1998 0,324; 1999 0,029; 2000 0,325; 2001 0,128; 2002 0,443; 2003 0,027; 2004 0,073; 2005 0,003; 2006 0,261; 2007 0,029; 2008 0,236; 2009 −0,156; 2010 0,229; 2011 0,092; 2012 0,175; 2013 0,207; 2014 0,081; 2015 0,188; 2016 0,243; 2017 0,108; 2018 0,131; 2019 0,006; 2020 −0,231; 2021 0,038; 2022 0,419. **Trung bình 0,160.**

### B12f. Vì sao mô hình (iii) không dùng làm baseline

Tương quan 0,633 cao gấp 4 lần mốc "dùng thẳng lợi suất quá khứ" (0,160). Lý do khả dĩ, từ thiết kế huấn luyện (mình chưa tự chạy lại để chứng minh):

- Trong mỗi fold, mạng được huấn luyện bằng **nhãn của chính năm Y** của các chiến lược khác (4/5 số chiến lược của năm Y).
- Các chiến lược cùng năm có lợi suất tương quan mạnh với nhau (cùng một môi trường thị trường). Mạng có thể học "năm Y lên hay xuống" từ nhãn của chiến lược khác. Một dự báo thật vào cuối năm Y−1 **không có** thông tin đó.
- Cách chia đúng cho bài toán dự báo là theo **thời gian** (chỉ dùng dữ liệu trước Y), như B3.

Nên mô hình (iii) không phải dự báo ngoài mẫu hợp lệ; ô 37 (so sánh top-k với Naive) có mã nhưng **không lưu kết quả**.

---

# TỔNG KẾT

## Chuỗi các bước của notebook B theo thứ tự (mô hình (i), Accounting EW)

| # | Bước | Đầu vào | Đầu ra | Số thật |
|---|---|---|---|---|
| 1 | Tải embedding | Hugging Face | [1.033.883, 4, 1280] | B2a |
| 2 | Ghép nhãn | file thống kê IS/OOS + khóa | 979.380 hàng | B2b |
| 3 | Chia theo năm | `oos_begin_year` | 692.447 / 79.133 / 207.800 | B3 |
| 4 | Chuẩn hóa (chỉ train) | embedding + 2 số | [B, 4, 1280] + [B, 2] | B4 |
| 5 | Bi-LSTM + attention + ghép + FC | | `pred_ret` [B] | B5 |
| 6 | Huấn luyện MSE, chọn checkpoint theo IC val | | mô hình epoch 25 (IC val 0,0169) | B6 |
| 7 | Dự báo test, IC từng năm, 10 nhóm | 207.800 hàng | IC 0,0393; spread 0,0748 (t 0,74) | B7 |
| 8 | Ghép dự báo với lợi suất tháng, xếp theo \|dự báo\|, top 1/5/10% | 2.493.600 dòng | chuỗi lãi danh mục theo tháng | B9 |
| 9 | Lãi/năm, t-stat, Sharpe | 96 tháng | Sharpe 0,28 / 0,34 / 0,41 (EB: 0,40 / 0,56 / 0,73) | B9d |

## Kết quả đã lưu của notebook (test 2014–2021)

| Họ | IC (i) FinCast | IC (ii) chuỗi thô |
|---|---|---|
| Accounting EW | 0,0393 (std 0,0977) | 0,0839 (std 0,1308) |
| Past Return EW | −0,0364 (std 0,2644) | 0,0556 (std 0,2983) |

Sharpe top 1% / 5% / 10%:

| Họ | EB | Naive | (i) FinCast | (ii) không FinCast |
|---|---|---|---|---|
| Accounting EW | 0,40 / 0,56 / 0,73 | 0,47 / 0,54 / 0,68 | 0,28 / 0,34 / 0,41 | 0,35 / 0,49 / 0,47 |
| Past Return EW | −0,97 / −0,91 / −0,83 | −0,98 / −0,97 / −0,88 | 0,27 / −0,11 / −0,33 | 0,45 / 0,48 / 0,50 |

## Những chỗ cần cẩn thận khi dùng notebook làm mốc so sánh

1. **Cửa sổ khác nhau:** EB/Naive chỉ 2014–2020 (7 năm), mô hình AI 2014–2021 (8 năm). Bảng trong ô 15/22 không cùng giai đoạn (B9b, đã xác nhận bằng số dòng và bằng tái lập EB ở B9e).
2. **Luật chọn khác nhau:** EB xếp theo Sharpe dự báo (có chia cho rủi ro), mô hình AI theo |dự báo|. Một phần chênh lệch đến từ luật chọn chứ không phải chất lượng dự báo. B9f cho thấy cách xếp theo |dự báo| chọn chiến lược có t-stat quá khứ rất thấp (|t| trung bình 1,05 so với 4,6).
3. **Chọn checkpoint theo IC trên 3 năm validation (2011–2013):** rất bất ổn. Mô hình (ii) Accounting EW chọn epoch 1; Past Return (i) có IC val 0,43 mà IC test −0,04.
4. **Một lần chia, một seed, 8 năm test:** kết quả lần chạy lại của mình (B8d) lệch đáng kể so với notebook, cho thấy độ nhiễu giữa các lần chạy lớn. Không có p-value.
5. **MSE không phải thước đo báo cáo.** Notebook chỉ in MSE lúc huấn luyện/validation; thước báo cáo là IC, spread nhóm và bảng top-k.
6. **Mô hình (iii) có nguy cơ rò rỉ** (B12f), không dùng làm baseline.

Dự án này giữ kiến trúc của mô hình (i) làm khung và đánh giá lại bằng cách làm giống paper (walk-forward 1983–2020, cùng luật xếp hạng với EB): xem [5_FINAL_MODEL_VS_BASELINE.md](5_FINAL_MODEL_VS_BASELINE.md). Cách paper gốc đánh giá: [2_ORIGINAL_PAPER.md](2_ORIGINAL_PAPER.md). Chi tiết thêm về baseline: [1_BASELINE.md](1_BASELINE.md).

---

## Phụ lục: Notebook A tham khảo làm gì: tuần tự từng bước, kèm dữ liệu thật

Notebook: `Research/reference_notebooks/rep_original_R_LSTM_Attention_highthroughput_PointB.ipynb` (24 ô). Không có huấn luyện mô hình: chỉ chạy lại phân tích EB của paper. Phần chính của dự án nằm ở notebook B: xem [3_REFERENCE_NOTEBOOK.md](3_REFERENCE_NOTEBOOK.md).

Nhãn nguồn: **[N]** là số in nguyên văn trong notebook; **[R]** là số mình tính lại bằng dữ liệu của dự án.

Ô 2 mở đầu bằng "Created on Mon Feb 5 2024, author cdim": đây là mã Python của Chen & Dim, được chép vào notebook. Đầu ra là các bảng và hình của paper. **Không có mô hình học nào.** Tên file có chữ "LSTM_Attention" nhưng trong notebook không có LSTM hay Attention.

### A1. Tải mã gốc (ô 0–1)

`git clone https://github.com/chenandrewy/high-throughput-ap.git` rồi `cd high-throughput-ap/`. Chỉ để lấy mã và đường dẫn.

### A2. Đọc lợi suất tháng của 6 họ (ô 2, phần "load family strategies")

Đọc 4 file, tách ra 6 họ:

| File | Họ tạo ra |
|---|---|
| `DataMinedLongShortReturnsVW.csv` | Accounting VW |
| `DataMinedLongShortReturnsEW.csv` | Accounting EW |
| `PastReturnSignalsLongShort.csv.gzip` | Past Return EW và Past Return VW |
| `TickerSignalsLongShort.csv.gzip` | Ticker EW và Ticker VW |

Xử lý từng file:
- File kế toán có cột `year`, `month`: ghép thành ngày đầu tháng. Giữ `signalid, date, year, ret`.
- File Past Return và Ticker có cột `date`, `ret_ew`, `ret_vw`, `nshort`, `nlong`: **bỏ dòng có `nshort < 2` hoặc `nlong < 2`**, rồi tách `ret_ew` và `ret_vw` thành hai họ riêng.
- Chỉ giữ `year >= 1963`.

**Dữ liệu thật của file gốc** (dòng đầu và cuối của `DataMinedLongShortReturnsEW.csv`) [N]:

```
   signalid  year  month       ret  nstock
0         1  1990      7  1.542175    3160
1         1  1990      8  0.476569    3131
2         1  1990      9  0.274472    3116
3         1  1990     10  4.070882    3102
4         1  1990     11  2.742471    3084
...
16941499     29315  2022      8   1.322090     680
16941500     29315  2022      9  -0.093359     672
16941501     29315  2022     10 -12.580153     644
16941502     29315  2022     11  -4.037717     636
16941503     29315  2022     12   1.798057     627
```

Kích thước file này: **16.941.504 dòng × 5 cột**, 29.314 chiến lược, năm 1951–2022 [N, từ ô 29 của notebook B].

Ví dụ của mình [R]: chiến lược 18144 có 3 dòng đầu năm 2000: tháng 1 lỗ 7,43%, tháng 2 lỗ 21,14%, tháng 3 lãi 2,64%.

### A3. Đọc thống kê trong mẫu / ngoài mẫu và dự báo của EB, rồi ghép với lợi suất tháng (ô 2, phần "read annual estimates")

Đọc file `Nelly_Predict_SignalYear.csv.gzip`: mỗi chiến lược, mỗi năm Y có **hai dòng**:

| Cột | Ý nghĩa |
|---|---|
| `s_flag` = `IS` | 240 tháng (20 năm) **trước** Y. `mean_ret`, `tstat`, `nobs` = 240 |
| `s_flag` = `OOS` | 12 tháng của **năm Y**. `mean_ret`, `tstat`, `nobs` = 12 |
| `thetahat` | t-stat của EB sau khi co |
| `pred_ret` | dự báo lợi suất tháng của EB |

**Dữ liệu thật** (5 dòng đầu của `OOS_signal_tstat_OosNyears1.csv.gzip`, in từ ô 39 của notebook B) [N]:

```
                 signal_family  oos_begin_year  signalid  mean_ret     tstat  nobs s_flag  is_begin_year
0  DataMinedLongShortReturnsVW            1983        66  0.143098  0.825577 240.0     IS           1963
1  DataMinedLongShortReturnsVW            1983        66 -0.103663 -0.149915  12.0    OOS           1963
2  DataMinedLongShortReturnsVW            1983        67 -0.140025 -0.648427 240.0     IS           1963
3  DataMinedLongShortReturnsVW            1983        67 -0.249557 -0.238849  12.0    OOS           1963
4  DataMinedLongShortReturnsVW            1983        68 -0.039744 -0.242839 240.0     IS           1963
```

(Toàn bộ file: **10.227.400 dòng × 8 cột**: `signal_family, oos_begin_year, signalid, mean_ret, tstat, nobs, s_flag, is_begin_year` [N].)

**Ví dụ của mình** [R] với chiến lược 18144, hai năm, theo đúng định dạng của file:

| Năm | s_flag | nobs | mean_ret | tstat | thetahat | pred_ret |
|---|---|---|---|---|---|---|
| 2000 | IS (1980–1999) | 240 | −1,2527 | −9,7901 | −8,0705 | **−1,0327** |
| 2000 | OOS (2000) | 12 | −2,3967 | −1,2559 | | |
| 2020 | IS (2000–2019) | 240 | −0,6669 | −4,0115 | −2,7415 | **−0,4557** |
| 2020 | OOS (2020) | 12 | −0,2526 | | | |

(Giá trị OOS tstat năm 2000 = mean ÷ (độ lệch chuẩn ÷ √12), độ lệch chuẩn chia cho n.)

**Ghép với lợi suất tháng.** Notebook ghép bảng thống kê (mỗi chiến lược-năm 1 dòng) với lợi suất tháng của đúng năm đó, theo `(signal_family, signalid, oos_begin_year)`. Mỗi chiến lược-năm trở thành **12 dòng** (12 tháng của năm Y), mỗi dòng mang theo `ret_is`, `tstat_is`, `pred_ret`, `ret_oos` (lãi **tháng đó**). Bảng này tên là `df_est_oos`.

Ví dụ [R], chiến lược 18144 năm 2020 thành 12 dòng, lãi các tháng:

```
 T1     T2     T3     T4     T5     T6     T7     T8     T9    T10    T11    T12
-0,04  -0,19  -7,55  +0,91  -0,57  +0,87  -0,93  +1,53  +1,64  -0,89  +2,80  -0,62     trung bình −0,2526
```

Con số "trung bình −0,2526" chính là nhãn `mean_ret_OOS` của hàng (18144, 2020) mà notebook B sẽ dùng.

### A4. Tóm tắt dữ liệu (ô 2, phần "summarize strategies data")

Với mỗi họ, tính lợi suất trung bình **năm** của từng chiến lược (trung bình tháng × 12), rồi in thống kê phân phối giữa các chiến lược. Kết quả [N]:

| Họ | Số chiến lược | Phân vị 5% | Trung vị | Phân vị 95% |
|---|---|---|---|---|
| acct_ew | 29.314 | −7,0 | −1,1 | 3,7 |
| acct_vw | 29.314 | −4,5 | −0,4 | 3,9 |
| past_ret_ew | 19.402 | −5,3 | −0,4 | 2,1 |
| past_ret_vw | 19.402 | −3,4 | 0,1 | 4,3 |
| ticker_ew | 19.380 | −0,9 | −0,0 | 0,8 |
| ticker_vw | 19.380 | −2,2 | −0,2 | 1,6 |

(Đơn vị: % mỗi năm. Lưu thành `summary_stat.csv`.) Họ Ticker có độ trải rất hẹp (−0,9 đến 0,8) so với kế toán (−7,0 đến 3,7): đó là dấu hiệu Ticker gần như nhiễu.

### A5. Hình phân phối t-stat và mô hình hỗn hợp của EB (ô 2, phần cuối)

1. Đọc `Nelly_QML_FamilyYear.csv.gzip`: 5 tham số hỗn hợp của EB cho mỗi họ-năm.
2. Với mỗi họ, vẽ **histogram t-stat thật** (xanh) cạnh **histogram t-stat mô phỏng từ mô hình** (cam) và **phân phối chuẩn N(0,1)** (đen), cho 3 năm: **1983, 2004, 2020**. Mỗi năm một hình 6 ô (6 họ).
3. Mô phỏng: lấy θ từ hỗn hợp `p·N(mua, siga²) + (1−p)·N(mub, sigb²)`, cộng thêm N(0,1) ra t-stat.

Notebook **chỉ ra hình, không in số**. Số tham số của họ kế toán EW năm 2000 mà mình tính lại bằng cùng công thức [R]:

| Nhóm | Chiếm | Trung bình θ | Độ rộng |
|---|---|---|---|
| A | 33,5% | +0,08 | 0,62 |
| B | 66,5% | −1,44 | 1,96 |

### A6. Hình lợi suất theo 20 nhóm (ô 3)

Cho hai giai đoạn **1983–2004** và **2004–2020**:
1. Mỗi họ, mỗi tháng, chia chiến lược thành **20 nhóm theo lợi suất trong mẫu** (`pd.qcut`).
2. Với mỗi nhóm, tính trung bình ba đại lượng (đều nhân 12 để ra %/năm): lợi suất trong mẫu `ret_is`, lợi suất ngoài mẫu `ret_oos`, lợi suất EB dự báo `pred_ret`.
3. Vẽ 3 đường: In-Sample (tím đứt), OOS (chấm xanh, có thanh sai số ±1,96 SE), Predicted (đỏ).

Đây chính là Figure 4–5 của paper. Hình, không có số in ra.

### A7. Chọn danh mục theo luật của paper và tính Table 2 (ô 3)

Từng bước trong mã:

1. Chỉ giữ 6 họ và `oos_begin_year <= 2020`.
2. **Naive Mining:** dấu = dấu của t-stat quá khứ; xếp hạng bằng |t-stat| quá khứ.
3. **EB Mining:** dấu = dấu của `pred_ret`; `SRahat = dấu × thetahat ÷ √nobs × √12` (Sharpe dự báo quy năm); xếp hạng bằng `SRahat`.
4. **Xếp hạng theo từng tháng** (`groupby('date')`) trên **cả 6 họ gộp**, tính phần trăm.
5. Lấy chiến lược có `rank_pct ≥ 100 − k` với k = 1, 5, 10. Lãi danh mục tháng = trung bình đều lãi (đã nhân dấu).
6. Gom các tháng: lãi/năm = TB tháng × 12; t-stat = TB ÷ độ lệch chuẩn × √(số tháng); Sharpe = TB ÷ độ lệch chuẩn × √12.
7. Thêm Panel C: các anomaly đã đăng báo (`PredictorLSretWide.csv`, `PredictorSummary.xlsx`) làm đối chứng.

**Kết quả in ra, 1983–2020** [N]:

```
                      nstrat_mean ret_mean  tstat    sr
Panel A
EB Mining Top 1%            1278     5.70   9.01  1.46
EB Mining Top 5%            6389     4.02   8.24  1.34
EB Mining Top 10%          12777     2.76   7.14  1.16

Panel B
Naive Mining Top 1%         1278     5.72   8.91  1.45
Naive Mining Top 5%         6389     3.72   7.73  1.25
Naive Mining Top 10%       12777     2.61   6.77  1.10

Panel C
Pub Anytime                  204     5.99  12.73  2.06
Pub Pre-2004                  82     5.27   9.76  1.58
```

Đây là Table 2 của paper. Notebook cũng lưu bản LaTeX `beststrats.tex` và vẽ lợi suất cộng dồn (log của 1 đô la đầu tư năm 1983) cho EB, Naive và published.

### A8. Họ nào chiếm top 1% và 20 chiến lược đứng đầu từng năm (ô 3)

1. Đếm mỗi năm có bao nhiêu chiến lược từng họ nằm trong top 1% (`rank_pct ≥ 99`), rồi tính tỉ lệ (Table 3 Panel A của paper).
2. Ghép tên chiến lược: tên Accounting dựng từ `DataMinedSignalList.csv` (thay `v1`, `v2` bằng tên biến), tên Past Return từ `PastReturnSignalNames.csv.gzip`.
3. Với 4 năm **1993, 2003, 2010, 2013**: lấy 20 chiến lược có Sharpe dự báo cao nhất; tính **Sharpe thực tế của chính chúng trong 10 năm sau** (`yr+1` đến `yr+10`).

**Dữ liệu in ra** [N]. Cột "Sign" = `-` là chiến lược bị đảo dấu.

**Năm 1993:**

| Hạng | Sign | Sharpe dự báo | Sharpe thật 10 năm sau | Họ | Tên chiến lược |
|---|---|---|---|---|---|
| 1 | − | 1,56 | 1,32 | Acct EW | diff(intpn)/lag(ceq) |
| 2 | − | 1,51 | 0,84 | Acct EW | dxd2/dpact |
| 3 | − | 1,43 | 0,92 | Acct EW | dm/sale |
| 4 | − | 1,37 | 1,60 | Acct EW | dm/dpact |
| 5 | − | 1,37 | 1,64 | Acct EW | diff(intpn)/lag(seq) |
| 6 | + | 1,36 | 0,54 | Past Ret EW | ret_5_9_17_18 |
| 7 | − | 1,35 | 0,68 | Acct EW | dxd3/dpact |
| 8 | − | 1,35 | 0,69 | Acct EW | dm/cogs |
| 9 | − | 1,34 | 1,00 | Acct EW | diff(intpn)/lag(invt) |
| 10 | + | 1,33 | 0,47 | Past Ret EW | ret_17 |
| 11 | + | 1,33 | 0,62 | Past Ret EW | ret_9_17_18_19 |
| 12 | − | 1,33 | 0,62 | Acct EW | dm/xopr |
| 13 | − | 1,30 | 0,43 | Acct EW | diff(lt)/lag(dp) |
| 14 | + | 1,30 | 0,44 | Past Ret EW | ret_9_13_17_18 |
| 15 | + | 1,29 | 0,68 | Past Ret EW | ret_3_9_17_18 |
| 16 | − | 1,29 | 1,24 | Acct EW | diff(intpn)/lag(ceql) |
| 17 | − | 1,25 | 0,49 | Acct EW | dxd4/dpact |
| 18 | − | 1,25 | 1,01 | Acct EW | sstk/gp |
| 19 | − | 1,25 | 0,55 | Acct EW | dxd2/dp |
| 20 | + | 1,24 | 0,70 | Past Ret EW | ret_5_17_18_19 |

**Năm 2003** (chiến lược 18144 là hạng 1, tên `dltis/dpact`):

| Hạng | Sign | Sharpe dự báo | Sharpe thật 10 năm sau | Họ | Tên chiến lược |
|---|---|---|---|---|---|
| 1 | − | 1,60 | 0,64 | Acct EW | dltis/dpact |
| 2 | − | 1,53 | 1,27 | Acct EW | diff(xint)/lag(lt) |
| 3 | − | 1,47 | 1,63 | Acct EW | diff(xint)/lag(at) |
| 4 | − | 1,46 | 0,90 | Acct EW | diff(xint)/lag(seq) |
| 5 | − | 1,45 | 0,78 | Acct EW | diff(intpn)/lag(seq) |
| 6 | − | 1,45 | 0,87 | Acct EW | diff(lt)/lag(at) |
| 7 | − | 1,43 | 1,38 | Acct EW | diff(xint)/lag(dpact) |
| 8 | − | 1,41 | 0,96 | Acct EW | diff(lt)/lag(lct) |
| 9 | − | 1,40 | 0,68 | Acct EW | diff(lt)/lag(aco) |
| 10 | − | 1,39 | 0,71 | Acct EW | diff(intpn)/lag(ceq) |
| 11 | − | 1,39 | 0,81 | Acct EW | diff(xint)/lag(ceq) |
| 12 | − | 1,38 | 1,57 | Acct EW | diff(xint)/lag(icapt) |
| 13 | − | 1,38 | 1,15 | Acct EW | sstk/che |
| 14 | − | 1,38 | 0,64 | Acct EW | diff(lt)/lag(lt) |
| 15 | − | 1,38 | 0,17 | Acct EW | dm/dpact |
| 16 | − | 1,37 | 0,80 | Acct EW | diff(xint)/lag(ceql) |
| 17 | − | 1,37 | 0,99 | Acct EW | diff(dltt)/lag(dpact) |
| 18 | − | 1,37 | 0,75 | Acct EW | diff(lt)/lag(act) |
| 19 | − | 1,37 | 0,58 | Acct EW | diff(dltt)/lag(aco) |
| 20 | − | 1,36 | 0,43 | Acct EW | diff(dltt)/lag(ceqt) |

**Năm 2010:**

| Hạng | Sign | Sharpe dự báo | Sharpe thật 10 năm sau | Họ | Tên chiến lược |
|---|---|---|---|---|---|
| 1 | − | 1,38 | 0,64 | Acct EW | diff(xint)/lag(at) |
| 2 | − | 1,27 | 0,52 | Acct EW | diff(xint)/lag(icapt) |
| 3 | − | 1,23 | 0,46 | Acct EW | diff(xint)/lag(lt) |
| 4 | − | 1,22 | 0,32 | Acct EW | diff(xint)/lag(dpact) |
| 5 | − | 1,22 | 0,65 | Acct EW | diff(xint)/lag(me_datadate) |
| 6 | − | 1,18 | 0,44 | Acct EW | diff(dcvt)/lag(xsga) |
| 7 | − | 1,16 | 0,36 | Acct EW | diff(lt)/lag(at) |
| 8 | − | 1,14 | −0,08 | Acct EW | aqc/dpact |
| 9 | − | 1,13 | 0,54 | Acct EW | diff(gdwl)/lag(ceqt) |
| 10 | − | 1,13 | 0,26 | Acct EW | diff(xint)/lag(capx) |
| 11 | − | 1,13 | −0,24 | Acct EW | gdwl/sale |
| 12 | − | 1,12 | 0,33 | Acct EW | diff(xint)/lag(seq) |
| 13 | − | 1,12 | 0,72 | Acct EW | dltis/dpact |
| 14 | − | 1,11 | 0,88 | Acct EW | aqs/invt |
| 15 | − | 1,10 | 0,34 | Acct EW | diff(xint)/lag(dp) |
| 16 | − | 1,09 | 0,71 | Acct EW | diff(ppent)/lag(dp) |
| 17 | − | 1,09 | 0,44 | Acct EW | aqs/sale |
| 18 | − | 1,09 | 0,87 | Acct EW | sstk/che |
| 19 | − | 1,09 | 0,40 | Acct EW | aqs/ppegt |
| 20 | − | 1,09 | 0,55 | Acct EW | diff(dcvt)/lag(capxv) |

**Năm 2013:**

| Hạng | Sign | Sharpe dự báo | Sharpe thật 10 năm sau | Họ | Tên chiến lược |
|---|---|---|---|---|---|
| 1 | − | 1,40 | 0,49 | Acct EW | diff(xint)/lag(at) |
| 2 | − | 1,35 | 0,29 | Acct EW | diff(xint)/lag(icapt) |
| 3 | − | 1,25 | −0,01 | Acct EW | diff(xint)/lag(dpact) |
| 4 | − | 1,22 | 0,49 | Acct EW | fincf/che |
| 5 | − | 1,20 | 0,52 | Acct EW | diff(xint)/lag(me_datadate) |
| 6 | − | 1,18 | 0,34 | Acct EW | diff(xint)/lag(lt) |
| 7 | − | 1,18 | 0,10 | Acct EW | diff(lt)/lag(at) |
| 8 | − | 1,17 | 0,32 | Acct EW | diff(ppent)/lag(dp) |
| 9 | − | 1,16 | 0,62 | Acct EW | diff(cogs)/lag(cogs) |
| 10 | − | 1,16 | −0,37 | Acct EW | diff(intpn)/lag(dpact) |
| 11 | − | 1,15 | 0,60 | Acct EW | sstk/che |
| 12 | − | 1,14 | 0,17 | Acct EW | diff(intpn)/lag(at) |
| 13 | − | 1,14 | −0,37 | Acct EW | diff(xint)/lag(ppegt) |
| 14 | − | 1,14 | −0,11 | Acct EW | diff(xint)/lag(dp) |
| 15 | − | 1,13 | −0,32 | Acct EW | diff(dltt)/lag(at) |
| 16 | − | 1,13 | 0,26 | Acct EW | diff(lt)/lag(lct) |
| 17 | − | 1,13 | −0,20 | Acct EW | diff(dltt)/lag(dpact) |
| 18 | − | 1,13 | 0,19 | Acct EW | diff(xint)/lag(seq) |
| 19 | − | 1,11 | 0,12 | Acct EW | diff(xint)/lag(capx) |
| 20 | − | 1,11 | 0,42 | Acct EW | fincf/me_datadate |

Đọc bằng lời: Sharpe thật của những chiến lược này thấp hơn Sharpe dự báo ở hầu hết hàng, và giảm dần theo thời gian (năm 2013 nhiều hàng âm). Notebook lưu các bảng này vào `top_strategies_list.xlsx`.

### A9. Ngưỡng t-stat kiểm soát phát hiện sai (FDR) (ô 5)

Phần phụ của paper (Figure 6, 7). Ba bước:

1. **BY (Benjamini–Yekutieli):** với mỗi họ-năm, xếp |t| giảm dần. Với mỗi chiến lược: `Pr_emp` = tỉ lệ chiến lược có |t| lớn hơn hoặc bằng nó; `Pr_null = 2·Φ(−|t|)`; `FDRmax = Pr_null ÷ Pr_emp × Σ(1/k)` (tổng k = 1…N, gọi là penalty). Ngưỡng = |t| nhỏ nhất mà `FDRmax ≤ 1%` (hoặc 5%, 10%).
2. **Storey:** thay penalty bằng `pFmax = min(1, tỉ lệ |t| ≤ 1 ÷ 0,6827)`. Ngưỡng ứng với FDR 5%, 10%, 20%.
3. Chia 20 nhóm theo t-stat, vẽ lợi suất ngoài mẫu theo nhóm cùng các đường ngưỡng thẳng đứng. Mỗi kiểu một hình.

Notebook **chỉ ra hình**. Mình tính lại ngưỡng cho họ kế toán EW năm 2000 (25.603 chiến lược) [R]:

| Phương pháp | FDR | Ngưỡng \|t\| |
|---|---|---|
| BY (N = 25.603) | 1% / 5% / 10% | 3,93 / 3,42 / 3,18 |
| Storey (pFmax = 0,572) | 5% / 10% / 20% | 2,24 / 1,86 / 1,37 |

Chú thích khi đọc mã: hàm `get_BY_tstat_hurdles` đếm `Nstrat` bằng số **dòng** của `df_est_oos`, mà bảng này có 12 dòng cho mỗi chiến lược-năm (A3). Nếu đúng như vậy thì penalty được tính với N = 12 × số chiến lược (13,21 thay vì 10,73), và ngưỡng BY 1% / 5% / 10% là 4,00 / 3,49 / 3,25 [R]. Đây là nhận xét từ việc đọc mã, mình chưa kiểm xem mã gốc của tác giả có làm vậy không.

### A10. Dọn dẹp (ô 6–23)

Sao chép dữ liệu vào repo `ngandh/asset-pricing-exp`, `git commit`, `git push`, rồi xóa các thư mục kết quả cũ. Tên các thư mục bị xóa (`output_v2_A_LSTM`, `output_v3_C_Stacking`, `output_v4_E2E_LSTM`, `pointB_lb20_attention_output`) gợi ý nhóm từng thử các hướng LSTM khác, nhưng **không có mã hay kết quả của chúng trong notebook này**. Lệnh `git clone git@github.com:...` ở ô 6 lỗi "Host key verification failed" [N].

---

