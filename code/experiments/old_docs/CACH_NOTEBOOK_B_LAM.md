# Notebook B: từ dữ liệu gốc đến kết quả cuối, từng bước

Mỗi bước có 3 dòng: **DỮ LIỆU VÀO**, **XỬ LÝ**, **DỮ LIỆU RA**. Ví dụ xuyên suốt: **chiến lược 18144**, năm dự báo **2000** (dùng để học) và **2020** (dùng để thi).

Nhãn: **[N]** = số notebook tự in. **[R]** = số mình tính lại bằng dữ liệu dự án (embedding FinCast của notebook không có trên máy này, nên các số đi qua mô hình là từ bản chạy lại của mình; chi tiết ở cuối).

**Mục tiêu của cả notebook:** đoán lãi trung bình mỗi tháng của năm tiếp theo cho từng chiến lược, rồi chọn danh mục theo dự báo và so với EB.

---

## Bước 1. Dữ liệu gốc: lãi từng tháng của từng chiến lược

- **VÀO:** file `DataMinedLongShortReturnsEW.csv`, **16.941.504 dòng**, mỗi dòng là một chiến lược trong một tháng [N]:

| signalid | year | month | ret (%) | nstock |
|---|---|---|---|---|
| 18144 | 2000 | 1 | −7,43 | 2.087 |
| 18144 | 2000 | 2 | −21,14 | 2.016 |
| 18144 | 2000 | 3 | +2,64 | 1.960 |

- **XỬ LÝ:** không làm gì, đây là điểm bắt đầu. Có 29.314 chiến lược.
- **RA:** nguyên dữ liệu trên.

## Bước 2. Tính thống kê của 20 năm trước và lãi thật của năm Y (làm sẵn, ngoài notebook B)

- **VÀO:** lãi tháng ở Bước 1. Với năm Y = 2000:
  - 240 tháng 1980–1999 của 18144: `1980: +0,03 +1,62 −5,62 +2,59 +1,12 +1,73 … 1999: −4,18 −2,48 −0,28 +1,23 −2,30 −2,06 −1,26 −4,82 −4,75 −5,50 −10,30 −7,88`
  - 12 tháng của năm 2000: `−7,43 −21,14 +2,64 +1,20 +0,51 −5,14 −0,31 −5,22 −0,61 +0,58 +0,93 +5,22`
- **XỬ LÝ:** từ 240 tháng đầu tính lãi trung bình và t-stat. Từ 12 tháng sau tính lãi trung bình của năm 2000.
- **RA:** file `OOS_signal_tstat_OosNyears1.csv.gzip` (**10.227.400 dòng**, mỗi chiến lược-năm có 2 dòng) [N]:

| signalid | năm Y | s_flag | nobs | mean_ret | tstat |
|---|---|---|---|---|---|
| 18144 | 2000 | IS (20 năm trước) | 240 | **−1,2527** | **−9,79** |
| 18144 | 2000 | OOS (chính năm 2000) | 12 | **−2,3967** | −1,26 |

(Giá trị của 18144 mình tính lại theo đúng định dạng [R].) Dòng OOS chính là **đáp án**: lãi thật của năm Y. Lúc dự báo thì chưa biết.

## Bước 3. Tạo "dấu vân tay" FinCast (làm sẵn, ngoài notebook B)

- **VÀO:** chuỗi lãi tháng của chiến lược (Bước 1).
- **XỬ LÝ:** mô hình đã học sẵn **FinCast** đọc chuỗi và nén thành 4 vector × 1.280 số. Người khác tính sẵn rồi đưa lên Hugging Face; notebook chỉ tải về. Cửa sổ lịch sử đưa vào FinCast không được ghi (mình suy ra khoảng 128 tháng).
- **RA:** **1.033.883 mảng, mỗi mảng 4 × 1280 số** [N], kèm bảng khóa cho biết mảng nào của chiến lược nào, năm nào (hàng đầu: chiến lược 1, năm 1994, `emb_idx` 0). Ví dụ 5 số đầu của vector 1 cho 18144 năm 2000, lấy từ bản trích xuất riêng của dự án [R]: `0,406  −0,001  0,143  0,061  0,732`.

## Bước 4. Ghép thành một bảng (notebook B, ô 11)

- **VÀO:** mảng dấu vân tay (Bước 3) và 2 loại dòng IS/OOS (Bước 2).
- **XỬ LÝ:** lấy từ dòng IS hai số `tstat_IS`, `mean_ret_IS` (làm đầu vào); lấy từ dòng OOS số `mean_ret_OOS` (làm đáp án). Ghép với mảng dấu vân tay theo `(chiến lược, họ, năm)`. Bỏ dòng thiếu.
- **RA:** bảng **979.380 dòng** [N]. Một dòng:

| signalid | năm | dấu vân tay | tstat_IS | mean_ret_IS | **đáp án** mean_ret_OOS |
|---|---|---|---|---|---|
| 18144 | 2000 | mảng 4×1280 | −9,79 | −1,2527 | **−2,3967** |
| 18144 | 2020 | mảng 4×1280 | −4,01 | −0,6669 | **−0,2526** |

(Họ có 1.005.582 dòng thống kê và 1.033.883 mảng nhưng chỉ ghép được 979.380; số "19,2%" notebook in ra là chia cho cả 6 họ nên đừng đọc là thiếu dữ liệu.)

## Bước 5. Chia theo năm

- **VÀO:** 979.380 dòng.
- **XỬ LÝ:** chia **một lần** theo năm.
- **RA** [N]:

| Phần | Năm | Số dòng | Dùng để |
|---|---|---|---|
| Học | trước 2011 | 692.447 | cập nhật mô hình |
| Kiểm tra giữa chừng | 2011–2013 | 79.133 | chọn mô hình, biết lúc nào dừng |
| Thi | 2014–2021 | 207.800 | chấm kết quả cuối |

18144 năm 2000 vào tập học; 18144 năm 2020 vào tập thi.

## Bước 6. Chuẩn hóa

- **VÀO:** mảng 4×1280 và 2 số `tstat_IS`, `mean_ret_IS`.
- **XỬ LÝ:** trừ trung bình, chia độ lệch chuẩn, **chỉ tính trên tập học** (để không nhìn trộm tập thi). Mảng chuẩn hóa riêng từng trong 1.280 chiều. **Đáp án không chuẩn hóa.**
- **RA** [R]: với 18144 năm 2020:

| | Trước | Sau |
|---|---|---|
| Vector 1 (5 số đầu) | `0,399  0,115  0,137  0,060  0,524` | `0,716  −0,277  0,775  0,310  0,275` |
| t-stat | −4,012 | **−0,063** |
| Lãi TB | −0,667 | **−0,186** |

## Bước 7. Đưa vào mô hình, lấy ra dự báo

- **VÀO** (cho một dòng): mảng chuẩn hóa **[4, 1280]** + 2 số chuẩn hóa **[2]**. Với 18144 năm 2020: 4 vector (như bảng Bước 6) và `[−0,063; −0,186]` [R].
- **XỬ LÝ** (từng lớp, số thật của 18144 năm 2020 [R]):

| Lớp | Ra |
|---|---|
| Bi-LSTM (đọc 4 vector theo thứ tự) | [4, 256]. Vector 1: `−0,003  0,039  0,047  −0,003  0,249 …` |
| Attention (chấm điểm từng vector) | trọng số `0,105  0,076  0,407  0,412` |
| Trộn có trọng số | context [256]: `−0,090  0,453  −0,127  −0,286  −0,101 …` |
| Ghép thêm 2 số | [258]; 3 số cuối `−0,159  −0,063  −0,186` |
| Tầng ẩn 64 rồi tầng cuối 1 | **1 số** |

- **RA:** **dự báo lãi trung bình mỗi tháng của năm Y**: 18144 năm 2020 → **−0,947** (đáp án thật −0,253). 18144 năm 2000 → **−2,203** (đáp án thật −2,397) [R].

## Bước 8. Học (so dự báo với đáp án, sửa mô hình)

- **VÀO:** dự báo và đáp án của 692.447 dòng học.
- **XỬ LÝ:** tính sai số bình phương trung bình (MSE), cập nhật mô hình (Adam), lặp. Cứ 5 vòng thử trên tập kiểm tra giữa chừng và giữ mô hình có IC cao nhất.
- **RA:** mô hình đã học. Notebook giữ **vòng 25** (IC kiểm tra 0,0169) [N]:

| Vòng | MSE tập học | MSE tập kiểm tra |
|---|---|---|
| 5 | 0,593 | 0,733 |
| 25 | 0,477 | 0,759 |
| 40 | 0,438 | 0,853 |

MSE tập học giảm, MSE tập kiểm tra tăng: mô hình học thuộc.

## Bước 9. Dự báo cho tập thi

- **VÀO:** 207.800 dòng thi và mô hình đã giữ.
- **XỬ LÝ:** chạy Bước 7 cho từng dòng.
- **RA:** file `predictions_test_FinCast_...parquet`, **207.800 dòng × 4 cột** (`signal_family, signalid, oos_begin_year, pred_ret_fincast`) [N].

## Bước 10. Chấm bằng IC

- **VÀO:** dự báo và đáp án của tập thi.
- **XỬ LÝ:** mỗi năm, tính độ khớp giữa xếp hạng dự báo và xếp hạng đáp án (IC; 1 = khớp hoàn toàn, 0 = đoán mò).
- **RA** [N]:

| 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | **TB** |
|---|---|---|---|---|---|---|---|---|
| 0,108 | 0,198 | 0,120 | −0,065 | 0,037 | 0,015 | −0,090 | −0,009 | **0,039** |

## Bước 11. Chuyển dự báo thành danh mục và tính lãi từng tháng

- **VÀO:** dự báo (Bước 9) và bảng **mỗi chiến lược-năm × 12 tháng** (12.066.984 dòng [N]). Ví dụ 18144 năm 2020, 12 dòng lãi tháng: `−0,04  −0,19  −7,55  +0,91  −0,57  +0,87  −0,93  +1,53  +1,64  −0,89  +2,80  −0,62`.
- **XỬ LÝ:**
  1. **Đảo dấu** nếu dự báo âm. 18144 năm 2020 có dự báo −0,947 nên bị đảo: tháng 1 từ −0,04 thành +0,04, tháng 3 từ −7,55 thành +7,55, v.v.
  2. **Xếp hạng** mỗi tháng ~26.000 chiến lược theo **|dự báo|**.
  3. **Lấy top 1%, 5%, 10%** (khoảng 261 / 1.300 / 2.600 chiến lược).
  4. **Lãi danh mục tháng** = trung bình lãi (đã đảo dấu) của các chiến lược được chọn.
- **RA:** một chuỗi lãi theo tháng cho mỗi mức top-k. Ví dụ tháng 1/2020, top 1% (dự báo của bản không FinCast mình chạy lại [R]):

| chiến lược | dấu | lãi gốc | sau đảo dấu |
|---|---|---|---|
| 25211 | − | −8,009 | +8,009 |
| 26611 | − | −7,351 | +7,351 |
| 10472 | − | −13,779 | +13,779 |
| … 258 chiến lược nữa | | | |
| **Danh mục** | | | **TB = +1,66%** |

EB và Naive đi qua bước giống hệt, chỉ khác điểm xếp hạng: EB xếp theo **Sharpe dự báo**, Naive xếp theo **|t-stat quá khứ|**.

## Bước 12. Bảng cuối

- **VÀO:** chuỗi lãi theo tháng của từng cách chọn.
- **XỬ LÝ:** lãi/năm = trung bình tháng × 12; t-stat = trung bình ÷ độ lệch chuẩn × √(số tháng); Sharpe = trung bình ÷ độ lệch chuẩn × √12.
- **RA** [N], họ kế toán EW, thi từ 2014. Sharpe top 1% / 5% / 10%:

| EB | Naive | **AI FinCast (mô hình chính)** | AI không FinCast |
|---|---|---|---|
| 0,40 / 0,56 / 0,73 | 0,47 / 0,54 / 0,68 | **0,28 / 0,34 / 0,41** | 0,35 / 0,49 / 0,47 |

**Kết quả cuối: mô hình chính thua EB ở cả 3 mức.**

---

## Ghi chú về nguồn số

- Số đi qua mô hình (Bước 6, 7: mảng chuẩn hóa, đầu ra từng lớp, dự báo −0,947 và −2,203) là từ bản mình chạy lại đúng mã notebook bằng embedding FinCast của dự án (95.487 dòng, 10% chiến lược có |t-stat| lớn nhất mỗi năm). Bản này cho IC thi trung bình 0,075 (notebook toàn dữ liệu: 0,039).
- Ví dụ Bước 11 dùng dự báo của bản không FinCast mình chạy lại (IC 0,067; notebook 0,084), vì file dự báo của notebook không có trên máy này.
- Hai điểm yếu khi so với EB: EB chỉ có 2014–2020 (7 năm), AI có 2014–2021 (8 năm); EB xếp theo Sharpe dự báo, AI xếp theo |dự báo|.
- Phiên bản (iii) trong notebook học bằng đáp án của chính năm cần dự báo nên có nguy cơ rò rỉ; không dùng làm baseline.
- Bản chi tiết từng ô và từng dòng log: [CACH_NOTEBOOK_B_CHI_TIET.md](CACH_NOTEBOOK_B_CHI_TIET.md).
