# Paper gốc làm gì: từng bước, mỗi bước có dữ liệu thật

Paper: Chen & Dim, *High-Throughput Asset Pricing* (arXiv 2311.10685).

Mọi con số dưới đây tính lại từ dữ liệu của dự án (`data_raw/*.csv`, mã `code/data.py` giống mã tác giả). Cả bài đi theo **một ví dụ**: đứng ở 31/12/1999, dự báo **năm 2000**, soi kỹ **chiến lược số 18144**.

## Paper hỏi gì, trả lời bằng cách nào

Có hàng trăm nghìn chiến lược đầu tư "đào" ra từ dữ liệu. Nhiều cái trông đẹp trong quá khứ chỉ vì **may mắn**. Câu hỏi: nếu đầu mỗi năm ta chọn các chiến lược tốt nhất bằng một cách thống kê cẩn thận (EB, tức Empirical Bayes), thì năm sau chúng có kiếm được tiền thật không?

```
① Dữ liệu thô ─► ② Tóm tắt 20 năm quá khứ ─► ③ EB trừ phần may, dự báo năm sau
                                                      │
⑥ Lặp 38 năm, chấm điểm ◄─ ⑤ Đầu tư 12 tháng ◄─ ④ Đảo dấu, xếp hạng, lấy top 1%
```

---

## ① Dữ liệu thô

**Chiến lược 18144 làm gì mỗi tháng:**
1. Lấy hai biến kế toán của công ty: phát hành nợ dài hạn (`dltis`) chia khấu hao lũy kế (`dpact`).
2. Xếp mọi cổ phiếu theo tỉ số đó, **mua** nhóm 10% cao nhất, **bán khống** nhóm 10% thấp nhất.
3. Lãi tháng của chiến lược = lãi nhóm mua − lãi nhóm bán. Nên lãi có thể dương hoặc âm.

**Dữ liệu thật:** mỗi dòng là một chiến lược trong một tháng (`data_raw/DataMinedLongShortReturnsEW.csv`).

| signalid | year | month | ret (%/tháng) | nstock |
|---|---|---|---|---|
| 18144 | 2000 | 1 | −7,43 | 2.087 |
| 18144 | 2000 | 2 | −21,14 | 2.016 |
| 18144 | 2000 | 3 | +2,64 | 1.960 |

Tháng 1/2000 chiến lược này **lỗ 7,43%**, với 2.087 cổ phiếu tham gia.

**Cả kho dữ liệu:** 136.192 chiến lược, 6 họ. Năm 2000 có **128.770** chiến lược đủ dữ liệu.

| Nguồn chỉ số | EW (chia tiền đều) | VW (theo vốn hóa) | Năm 2000 (mỗi bản) |
|---|---|---|---|
| Kế toán | 29.314 | 29.314 | 25.603 |
| Lãi quá khứ của cổ phiếu | 19.402 | 19.402 | 19.402 |
| Ticker (chữ cái tên mã, **cố tình vô nghĩa**) | 19.380 | 19.380 | 19.380 |

Họ Ticker là nhóm đối chứng: phương pháp tốt không được chọn chúng.

---

## ② Tóm tắt 20 năm quá khứ

**Đầu vào:** 240 tháng 1/1980–12/1999 của từng chiến lược (không nhìn gì sau đó).

**Dữ liệu thật của 18144:**

```
1980:  +0,03  +1,62  −5,62  +2,59  +1,12  +1,73  +0,72  −1,21  −0,50  −0,20  +3,90  −2,12
 ...                              (còn 18 năm nữa)
1999:  −4,18  −2,48  −0,28  +1,23  −2,30  −2,06  −1,26  −4,82  −4,75  −5,50 −10,30  −7,88
```

**Tính 3 con số từ 240 số đó:**

| Con số | Công thức | Với 18144 |
|---|---|---|
| Lãi trung bình `m` | cộng 240 số, chia 240 | **−1,2527 %/tháng** |
| Độ dao động `s` | độ lệch chuẩn của 240 số | **1,9823** |
| t-stat | `m ÷ (s ÷ √240)` = −1,2527 ÷ (1,9823 ÷ 15,49) = −1,2527 ÷ 0,1280 | **−9,79** |

**Nói thường:** t-stat cho biết lãi trung bình lớn gấp mấy lần "sai số do ngẫu nhiên". −9,79 là rất lớn: 18144 lỗ đều đặn suốt 20 năm, khó mà do may rủi. Lỗ đều cũng có ích, vì làm ngược lại sẽ lãi đều.

**Làm cho cả họ:** năm 2000, họ kế toán EW có **25.603 chiến lược**, cho ra 25.603 giá trị t-stat. Nhìn tổng thể các t-stat này:

| Thống kê của 25.603 t-stat | Giá trị |
|---|---|
| Trung bình | −0,93 |
| Độ lệch chuẩn | 2,05 |
| Nhỏ nhất (là 18144) | **−9,79** |
| Lớn nhất | +7,68 |
| Tỉ lệ \|t\| > 2 | 33,4% |
| Tỉ lệ \|t\| > 3 | 17,9% |
| Tỉ lệ \|t\| > 5 | 3,6% |

Nếu không có tín hiệu thật nào, t-stat sẽ dồn quanh 0 với độ lệch chuẩn khoảng 1 và chỉ ~5% vượt 2 (theo giá trị tuyệt đối). Ở đây 33% vượt 2, nên có cả tín hiệu thật lẫn may mắn lẫn vào nhau. EB sinh ra để tách hai thứ này.

**Dòng dữ liệu của 18144 sau bước ② (dữ liệu thật):**

| họ | signalid | năm | mean_is | tstat | nobs | sd_is | mean_oos (thật) |
|---|---|---|---|---|---|---|---|
| acct_ew | 18144 | 2000 | −1,2527 | −9,7901 | 240 | 1,9823 | −2,3967 |

`mean_oos` là lãi thật của năm 2000, trung bình 12 tháng. Lúc dự báo (cuối 1999) chưa ai biết; nó chỉ dùng để chấm điểm ở bước ⑤–⑥.

---

## ③ EB dự báo năm 2000: "trừ phần may mắn"

### 3a. Mô tả cả đám bằng hai nhóm

EB giả định mỗi chiến lược thuộc một trong hai nhóm:
- **nhóm A:** "gần như không có tín hiệu thật" (t-stat thật gần 0);
- **nhóm B:** "có tín hiệu thật", mạnh yếu khác nhau.

Rồi dùng 25.603 t-stat để **ước lượng** xem mỗi nhóm chiếm bao nhiêu và trông ra sao. Kết quả ước lượng năm 2000 (5 tham số):

| Nhóm | Chiếm | t-stat thật trung bình | Độ rộng | Nói thường |
|---|---|---|---|---|
| A | **33,5%** | +0,08 | 0,62 | gần như vô tín hiệu |
| B | **66,5%** | −1,44 | 1,96 | có tín hiệu, phần lớn âm, nhiều mức độ |

Cộng thêm nhiễu đo lường (độ lệch chuẩn 1) vì ta chỉ thấy t-stat qua 240 tháng hữu hạn.

### 3b. Với từng chiến lược: "nó thuộc nhóm nào, và t-stat thật có thể là bao nhiêu?"

Đưa t-stat quan sát vào, EB trả lời hai câu:
1. **Xác suất thuộc mỗi nhóm.**
2. **Nếu thuộc nhóm đó thì t-stat thật nên là bao nhiêu** (kéo t-stat quan sát về phía trung tâm của nhóm).

Rồi lấy trung bình có trọng số. Kết quả là **θ̂** (đọc "théta mũ"): t-stat đã "co" về mức hợp lý.

**Dữ liệu thật cho 4 t-stat mẫu của năm 2000:**

| t-stat quan sát | Xác suất A / B | Nếu A thì θ̂ = | Nếu B thì θ̂ = | **θ̂ (trung bình có trọng số)** | Giữ lại |
|---|---|---|---|---|---|
| +1,0 | 56% / 44% | 0,34 | 0,50 | **0,41** | 41% |
| +2,0 | 46% / 54% | 0,61 | 1,29 | **0,98** | 49% |
| +5,0 | 1% / 99% | 1,45 | 3,67 | **3,65** | 73% |
| **−9,79 (18144)** | **0% / 100%** | −2,66 | −8,07 | **−8,07** | 82% |

**Đọc bảng:**
- t-stat nhỏ (+1) thì vừa có thể là nhóm A vừa có thể là B, và dù là nhóm nào cũng bị kéo mạnh về 0. Kết quả giữ lại chưa đến nửa.
- t-stat rất lớn (−9,79) thì gần như chắc thuộc nhóm B, và chỉ bị kéo nhẹ. Giữ lại 82%.

**Tính tay cho 18144:** với nhóm B, t-stat thật = −1,44 + 0,794 × (−9,79 − (−1,44)) = −1,44 + 0,794 × (−8,35) = **−8,07**. Số 0,794 là "mức tin vào quan sát" của nhóm B (= 1,96² ÷ (1,96² + 1)).

### 3c. Đổi t-stat đã co thành dự báo lãi

Co lãi quá khứ **đúng theo tỉ lệ đã co t-stat**:

```
dự báo (%/tháng) = θ̂ ÷ t-stat × lãi quá khứ
                 = (−8,07 ÷ −9,79) × (−1,2527)
                 = 0,824 × (−1,2527)
                 = −1,033 %/tháng
```

**Dòng dữ liệu đầy đủ của 18144 sau bước ③:**

| mean_is | tstat | thetahat (θ̂) | pred_ret | mean_oos (thật, chưa biết) |
|---|---|---|---|---|
| −1,2527 | −9,7901 | −8,0705 | **−1,0327** | −2,3967 |

**Đọc bằng lời:** EB đoán 18144 năm 2000 **vẫn lỗ** khoảng 1,03% mỗi tháng, nhẹ hơn 1,25% của quá khứ vì đã trừ phần may.

⚠️ EB **không dùng lãi của các năm sau** để học. Nó chỉ nhìn t-stat quá khứ của cả họ.

---

## ④ Đảo dấu, xếp hạng, lấy top 1%

### 4a. Đảo dấu

Dự báo lỗ thì **làm ngược lại**: bán nhóm cổ phiếu mà chiến lược gốc mua, mua nhóm nó bán. Lỗ của chiến lược gốc thành lãi của mình.

> dấu = −1 nếu dự báo âm, +1 nếu dự báo dương. Lãi của mình = dấu × lãi gốc.

Dữ liệu thật của 18144 tháng 1–3/2000:

| Tháng | Lãi gốc | Dấu | **Lãi của mình** |
|---|---|---|---|
| 1/2000 | −7,43 | −1 | **+7,43** |
| 2/2000 | −21,14 | −1 | **+21,14** |
| 3/2000 | +2,64 | −1 | **−2,64** |

### 4b. Xếp hạng bằng "Sharpe dự báo"

Paper xếp theo **lãi dự báo trên mỗi đơn vị rủi ro**, không phải lãi dự báo thuần:

> Sharpe dự báo = |dự báo| ÷ độ dao động = |θ̂| ÷ √240

18144: 8,0705 ÷ 15,49 = **0,521**, **cao nhất trong 128.770 chiến lược** năm 2000.

### 4c. Lấy 1% đầu bảng

Gộp **cả 6 họ**, xếp theo Sharpe dự báo giảm dần, lấy 1% = **1.288 chiến lược**. Ngưỡng cắt: Sharpe dự báo ≥ **0,2836**.

**Dữ liệu thật quanh bảng xếp hạng năm 2000:**

| Hạng | Họ | signalid | Lãi quá khứ | t-stat | EB dự báo | Sharpe dự báo | Làm gì | **Lãi thật 2000 (sau đảo dấu)** |
|---|---|---|---|---|---|---|---|---|
| 1 | kế toán EW | 18144 | −1,25 | −9,79 | −1,03 | 0,521 | ngược | **+2,40** |
| 2 | lãi quá khứ EW | 19246 | +2,09 | +8,77 | +1,77 | 0,478 | giữ | +0,29 |
| 3 | kế toán EW | 27689 | −2,53 | −4,43 | −2,18 | 0,469 | ngược | −0,47 |
| 4 | kế toán EW | 18135 | −1,00 | −8,77 | −0,83 | 0,468 | ngược | +0,39 |
| 5 | kế toán EW | 18131 | −1,09 | −8,76 | −0,90 | 0,468 | ngược | +0,63 |
| 101 | kế toán EW | 27721 | −2,69 | | −2,34 | 0,400 | ngược | −0,71 |
| 501 | kế toán EW | 25437 | −0,81 | | −0,68 | 0,337 | ngược | +0,55 |
| 1.001 | kế toán EW | 24620 | +1,41 | | +1,06 | 0,300 | giữ | −1,95 |
| 1.288 | lãi quá khứ EW | 13594 | +1,05 | | +0,90 | 0,2836 | giữ | +4,80 |

(Lãi thật tính %/tháng, trung bình 12 tháng 2000.)

**Danh mục 1.288 chiến lược gồm:**

| Họ | Số chiến lược | Lãi thật TB sau đảo dấu |
|---|---|---|
| kế toán EW | 1.060 | +1,46 %/tháng |
| lãi quá khứ EW | 228 | +0,96 %/tháng |
| kế toán VW, lãi quá khứ VW, Ticker EW, Ticker VW | **0** | |

- 78,6% số chiến lược được chọn bị đảo dấu.
- **Không có Ticker nào.** Năm 2000 t-stat của 19.380 Ticker nằm trong khoảng −3,7 đến +4,6. EB co chúng mạnh, |θ̂| lớn nhất chỉ 1,67, tức Sharpe dự báo tối đa 1,67 ÷ 15,49 = 0,108, thấp hơn ngưỡng 0,2836 nhiều. EB đã loại họ nhiễu mà không cần ai dặn.

---

## ⑤ Đầu tư năm 2000 và ghi lãi từng tháng

- **Chia tiền đều** cho 1.288 chiến lược, giữ 12 tháng.
- **Lãi danh mục mỗi tháng** = trung bình lãi (đã đảo dấu) của 1.288 chiến lược.

**Tháng 1/2000:**

```
(+7,43 [18144] − 10,20 [19246] + 12,60 [27689] + 7,91 [18135] + 6,51 [18131] + ... 1.283 số khác) ÷ 1.288
= +3,37%
```

**Cả năm 2000** (%/tháng):

| T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | T11 | T12 | **TB** | Độ dao động |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| +3,37 | +7,02 | −1,04 | −1,47 | −1,27 | +6,13 | −0,98 | +2,71 | +0,99 | +0,60 | −0,27 | +0,63 | **+1,37** | 2,87 |

Năm 2000 lãi trung bình 1,37%/tháng, nhưng tháng thì nhảy mạnh (từ −1,47 đến +7,02).

**Soi riêng 18144:**

| | Mức (%/tháng, sau đảo dấu) |
|---|---|
| EB đoán | +1,03 |
| Thật năm 2000 | **+2,40** |

---

## ⑥ Lặp 38 năm và chấm điểm

Làm ②–⑤ cho từng năm 1983, 1984, …, 2020. Mỗi năm EB được ước lượng lại từ đầu và chỉ dùng dữ liệu trước năm đó. Kết quả: **38 × 12 = 456 tháng** lãi của danh mục top 1%.

**Danh mục top 1% từng năm** (lãi trung bình tháng %, độ dao động trong năm %, số chiến lược):

| Năm | Lãi TB | Dao động | Số CL | | Năm | Lãi TB | Dao động | Số CL |
|---|---|---|---|---|---|---|---|---|
| 1983 | 0,98 | 0,81 | 1.186 | | 2002 | 1,55 | 1,74 | 1.295 |
| 1984 | 0,24 | 0,74 | 1.188 | | 2003 | 0,80 | 1,08 | 1.295 |
| 1985 | −0,03 | 1,15 | 1.190 | | 2004 | 0,26 | 1,40 | 1.299 |
| 1986 | 0,86 | 0,61 | 1.192 | | 2005 | −0,21 | 0,53 | 1.304 |
| 1987 | 0,86 | 0,83 | 1.202 | | 2006 | 0,30 | 0,45 | 1.306 |
| 1988 | 0,37 | 0,73 | 1.209 | | 2007 | −0,31 | 0,63 | 1.307 |
| 1989 | 0,38 | 0,41 | 1.239 | | 2008 | 0,39 | 0,92 | 1.307 |
| 1990 | 0,41 | 0,93 | 1.261 | | 2009 | 0,31 | 1,51 | 1.305 |
| 1991 | 0,06 | 0,79 | 1.258 | | 2010 | 0,47 | 0,76 | 1.306 |
| 1992 | 0,67 | 0,51 | 1.278 | | 2011 | 0,24 | 0,69 | 1.305 |
| 1993 | 0,77 | 0,57 | 1.297 | | 2012 | 0,37 | 0,74 | 1.304 |
| 1994 | 0,82 | 0,51 | 1.300 | | 2013 | 0,46 | 0,34 | 1.304 |
| 1995 | 0,53 | 0,58 | 1.303 | | 2014 | 0,02 | 0,73 | 1.298 |
| 1996 | 0,69 | 0,54 | 1.305 | | 2015 | 0,29 | 0,72 | 1.298 |
| 1997 | 0,87 | 1,28 | 1.301 | | 2016 | 0,71 | 1,13 | 1.299 |
| 1998 | 1,03 | 1,25 | 1.286 | | 2017 | −0,02 | 0,79 | 1.299 |
| 1999 | 1,21 | 1,34 | 1.291 | | 2018 | 0,31 | 0,51 | 1.299 |
| 2000 | 1,37 | 2,87 | 1.288 | | 2019 | −0,05 | 0,70 | 1.297 |
| 2001 | 0,61 | 2,59 | 1.292 | | 2020 | −0,24 | 1,33 | 1.298 |

Nhìn bảng thấy ngay: **đầu giai đoạn hầu hết các năm lãi đều, sau 2004 lãi mỏng và có nhiều năm âm.**

**Chấm điểm bằng 3 con số** (trên 456 tháng):

| Con số | Công thức | Nói thường |
|---|---|---|
| Lãi %/năm | lãi TB tháng × 12 | mỗi năm kiếm bao nhiêu |
| t-stat | lãi TB ÷ dao động × √456 | lãi có chắc khác 0 không; trên 2 là chắc |
| Sharpe/năm | lãi TB ÷ dao động × √12 | lãi trên mỗi đơn vị rủi ro; trên 1 là rất tốt |

**Kết quả (Table 2 của paper, 6 họ gộp, 1983–2020; mã dự án tính lại):**

| Danh mục | Số chiến lược mỗi năm | Lãi %/năm | t-stat | Sharpe |
|---|---|---|---|---|
| **Top 1%** | 1.279 | **5,79** (paper: 5,70) | 9,17 | **1,49** (paper: 1,46) |
| Top 5% | 6.391 | 3,99 | 8,15 | 1,32 |
| Top 10% | 12.784 | 2,82 | 7,18 | 1,17 |

Chia hai giai đoạn (top 1%):

| Giai đoạn | Lãi %/năm | Sharpe |
|---|---|---|
| 1983–2004 | 8,34 | 1,93 |
| 2005–2020 | 2,29 | 0,78 |

Paper giải thích: sau 2004 nhà đầu tư chú ý và khai thác các tín hiệu này nhiều hơn.

**Hai đối thủ cùng bảng:**
- **Chọn ngây thơ** (xếp theo thành tích quá khứ, không co): top 1% 5,72%/năm, Sharpe 1,45. Gần như bằng EB.
- **Anomaly đã đăng báo:** 5,88%/năm, Sharpe 2,03.

---

## ⑦ Dự báo có "đúng mức" không (Figure 4–5)

Câu hỏi khác với ⑥: EB đoán một nhóm lãi X% thì thật có lãi khoảng X% không?

**Cách làm:**
1. Mỗi năm, chia các chiến lược kế toán EW thành 20 nhóm theo dự báo của EB.
2. Với mỗi nhóm, tính dự báo trung bình và lãi thật trung bình.
3. Gộp các năm.

**Dữ liệu thật, 1983–2004 (%/tháng):**

| Nhóm (theo dự báo) | Nếu tin hẳn quá khứ | EB đoán | **Thực tế** |
|---|---|---|---|
| 1 (thấp nhất) | −0,92 | −0,75 | **−0,76** |
| 5 | −0,36 | −0,30 | **−0,32** |
| 10 | −0,13 | −0,11 | **−0,13** |
| 15 | +0,05 | 0,00 | **−0,06** |
| 20 (cao nhất) | +0,69 | +0,37 | **+0,18** |

**Đọc bằng lời:**
- **Tin hẳn quá khứ thì lạc quan quá:** nhóm cao nhất quá khứ +0,69, thật chỉ +0,18.
- **EB đoán sát thực tế hơn nhiều.** Độ dốc giữa thực tế và EB là **0,88**; 1 là đúng mức hoàn toàn.
- **Sau 2004 độ dốc còn 0,31:** tín hiệu yếu đi nên EB cũng đoán cao quá.

---

## Paper kết luận gì

1. Chọn bằng EB **kiếm được tiền thật** ở năm sau (⑥) và không dùng thông tin tương lai.
2. Chọn ngây thơ "thành tích đẹp nhất" cho lãi **gần như vậy**. Điểm hơn riêng của EB là **dự báo đúng mức** (⑦).
3. Tín hiệu tập trung ở **chỉ số kế toán, trước 2004**. Ticker gần như bằng 0.

## Bảng, hình trong paper ứng với bước nào

| Trong paper | Nội dung | Bước |
|---|---|---|
| **Table 2** | lãi, t-stat, Sharpe top 1/5/10%; EB so với chọn ngây thơ và anomaly đã đăng | ⑥ |
| Figure 1 | lãi cộng dồn theo thời gian | ⑥ |
| **Figure 4, 5** | EB đoán so với thực tế theo 20 nhóm | ⑦ |
| Table 3 Panel A | mỗi họ chiếm bao nhiêu trong top 1%: kế toán EW 91,0%, lãi quá khứ EW 8,6%, Ticker 0% | ④ |
| Table A.1, A.2 | 20 chiến lược đứng đầu 1993 / 2003 / 2013; 18144 (đảo dấu) hạng 1 năm 2003 | ④ |
| Table 1, Figure 2, 3 | thống kê dữ liệu, phân phối t-stat | ①② |
| Figure 6, 7 | so với các cách kiểm soát phát hiện sai | phụ |

Dự án thay bước ③ bằng mô hình AI và giữ nguyên các bước còn lại: xem [5_FINAL_MODEL_VS_BASELINE.md](5_FINAL_MODEL_VS_BASELINE.md).
