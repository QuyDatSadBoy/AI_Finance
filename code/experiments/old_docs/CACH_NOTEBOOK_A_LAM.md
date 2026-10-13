# Notebook A tham khảo làm gì: tuần tự từng bước, kèm dữ liệu thật

Notebook: `Research/NoteBook_Refer/rep_original_R_LSTM_Attention_highthroughput_PointB.ipynb` (24 ô). Không có huấn luyện mô hình: chỉ chạy lại phân tích EB của paper. Phần chính của dự án nằm ở notebook B: xem [CACH_NOTEBOOK_B_LAM.md](CACH_NOTEBOOK_B_LAM.md).

Nhãn nguồn: **[N]** là số in nguyên văn trong notebook; **[R]** là số mình tính lại bằng dữ liệu của dự án.

Ô 2 mở đầu bằng "Created on Mon Feb 5 2024, author cdim": đây là mã Python của Chen & Dim, được chép vào notebook. Đầu ra là các bảng và hình của paper. **Không có mô hình học nào.** Tên file có chữ "LSTM_Attention" nhưng trong notebook không có LSTM hay Attention.

## A1. Tải mã gốc (ô 0–1)

`git clone https://github.com/chenandrewy/high-throughput-ap.git` rồi `cd high-throughput-ap/`. Chỉ để lấy mã và đường dẫn.

## A2. Đọc lợi suất tháng của 6 họ (ô 2, phần "load family strategies")

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

## A3. Đọc thống kê trong mẫu / ngoài mẫu và dự báo của EB, rồi ghép với lợi suất tháng (ô 2, phần "read annual estimates")

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

## A4. Tóm tắt dữ liệu (ô 2, phần "summarize strategies data")

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

## A5. Hình phân phối t-stat và mô hình hỗn hợp của EB (ô 2, phần cuối)

1. Đọc `Nelly_QML_FamilyYear.csv.gzip`: 5 tham số hỗn hợp của EB cho mỗi họ-năm.
2. Với mỗi họ, vẽ **histogram t-stat thật** (xanh) cạnh **histogram t-stat mô phỏng từ mô hình** (cam) và **phân phối chuẩn N(0,1)** (đen), cho 3 năm: **1983, 2004, 2020**. Mỗi năm một hình 6 ô (6 họ).
3. Mô phỏng: lấy θ từ hỗn hợp `p·N(mua, siga²) + (1−p)·N(mub, sigb²)`, cộng thêm N(0,1) ra t-stat.

Notebook **chỉ ra hình, không in số**. Số tham số của họ kế toán EW năm 2000 mà mình tính lại bằng cùng công thức [R]:

| Nhóm | Chiếm | Trung bình θ | Độ rộng |
|---|---|---|---|
| A | 33,5% | +0,08 | 0,62 |
| B | 66,5% | −1,44 | 1,96 |

## A6. Hình lợi suất theo 20 nhóm (ô 3)

Cho hai giai đoạn **1983–2004** và **2004–2020**:
1. Mỗi họ, mỗi tháng, chia chiến lược thành **20 nhóm theo lợi suất trong mẫu** (`pd.qcut`).
2. Với mỗi nhóm, tính trung bình ba đại lượng (đều nhân 12 để ra %/năm): lợi suất trong mẫu `ret_is`, lợi suất ngoài mẫu `ret_oos`, lợi suất EB dự báo `pred_ret`.
3. Vẽ 3 đường: In-Sample (tím đứt), OOS (chấm xanh, có thanh sai số ±1,96 SE), Predicted (đỏ).

Đây chính là Figure 4–5 của paper. Hình, không có số in ra.

## A7. Chọn danh mục theo luật của paper và tính Table 2 (ô 3)

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

## A8. Họ nào chiếm top 1% và 20 chiến lược đứng đầu từng năm (ô 3)

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

## A9. Ngưỡng t-stat kiểm soát phát hiện sai (FDR) (ô 5)

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

## A10. Dọn dẹp (ô 6–23)

Sao chép dữ liệu vào repo `ngandh/asset-pricing-exp`, `git commit`, `git push`, rồi xóa các thư mục kết quả cũ. Tên các thư mục bị xóa (`output_v2_A_LSTM`, `output_v3_C_Stacking`, `output_v4_E2E_LSTM`, `pointB_lb20_attention_output`) gợi ý nhóm từng thử các hướng LSTM khác, nhưng **không có mã hay kết quả của chúng trong notebook này**. Lệnh `git clone git@github.com:...` ở ô 6 lỗi "Host key verification failed" [N].

---

