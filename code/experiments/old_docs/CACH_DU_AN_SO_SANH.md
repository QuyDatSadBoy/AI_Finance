# Dự án làm gì, kể bằng lời thường: theo chân một chiến lược qua từng bước

Đọc trước nếu cần: [CACH_PAPER_DANH_GIA.md](CACH_PAPER_DANH_GIA.md) (paper gốc làm gì).

## Tóm tắt trong 4 câu

1. Có hàng chục nghìn **chiến lược đầu tư** (mua nhóm cổ phiếu này, bán khống nhóm kia). Mỗi tháng, mỗi chiến lược lãi hoặc lỗ một ít.
2. Đầu mỗi năm, ta phải **đoán**: năm nay chiến lược nào sẽ lãi nhất?
3. Paper đoán bằng một công thức thống kê tên là **EB**. Dự án đoán bằng **mô hình AI**.
4. Hai bên **dùng cùng một luật** để chọn chiến lược: chọn ra 1% tốt nhất, chia tiền đều, giữ 12 tháng. Sau đó xem **bên nào kiếm tiền tốt hơn**.

```
 Lịch sử 20 năm      ──►  Người đoán  ──►  Xếp hạng  ──►  Chọn 1% tốt nhất  ──►  Năm sau lãi bao nhiêu?
 của mỗi chiến lược       (EB hoặc AI)                    chia tiền đều          (chấm điểm)
```

Để dễ theo dõi, cả bài đi theo **một ví dụ thật**: đứng ở **ngày 1/1/2000**, nhìn vào **chiến lược số 18144** (họ kế toán EW).

---

## Bước 0. Dữ liệu thô: mỗi tháng chiến lược lãi bao nhiêu

File gốc ghi lãi/lỗ của từng chiến lược trong từng tháng. Ba dòng thật của chiến lược 18144:

| Chiến lược | Tháng | Lãi trong tháng |
|---|---|---|
| 18144 | 1/2000 | −7,43% |
| 18144 | 2/2000 | −21,14% |
| 18144 | 3/2000 | +2,64% |

Ta xếp lại thành **một bảng lớn**:
- mỗi hàng là một chiến lược (họ kế toán EW có **29.314** chiến lược);
- mỗi cột là một tháng (**756** tháng, từ 1960 đến 2022).

Nhìn vào một hàng là thấy toàn bộ lịch sử lãi/lỗ của một chiến lược.

---

## Bước 1. Ngày 1/1/2000: chấm điểm quá khứ và giữ lại những chiến lược đáng chú ý

**Làm gì:** với mỗi chiến lược, chỉ nhìn **20 năm trước đó** (1980–1999). Từ đó tính 3 con số:

- **Lãi trung bình mỗi tháng**
- **Độ dao động**: lãi lên xuống thất thường đến đâu.
- **t-stat**: lãi trung bình so với độ dao động, nhân với √(số tháng). Con số này lớn nghĩa là chiến lược **lãi đều đặn**, không phải ăn may vài tháng.

**Ví dụ 18144:**
- 20 năm qua, trung bình mỗi tháng **lỗ 1,25%**, dao động khoảng 1,98%.
- t-stat = **−9,79**, đều đặn nhất trong cả họ.
- Lỗ đều đặn cũng là thông tin quý: **làm ngược lại** chiến lược này thì sẽ lãi đều đặn.

**Lọc bớt:**
- Năm 2000 có **25.603** chiến lược đủ dữ liệu.
- Ta chỉ giữ **10% có |t-stat| lớn nhất**, tức **2.561 chiến lược**. Gọi đây là **tập ứng viên**.
- Lý do: top 1% của EB nằm trong nhóm này tới 95% (tính trên cả 6 họ, 1983–2020). Nhưng với top 5% và top 10% của EB thì chỉ 54% và 38% nằm trong nhóm này (họ kế toán EW: 46% và 25%). Nên chỉ giữ 10% có \|t\| lớn nhất là một giới hạn của cách làm, đặc biệt khi so top 5% và 10%.

| Hạng | Chiến lược | Lãi TB/tháng | Dao động | t-stat | Kết quả |
|---|---|---|---|---|---|
| 1 | 18144 | −1,25% | 1,98% | −9,79 | giữ |
| 2 | 18135 | −1,00% | 1,77% | −8,77 | giữ |
| 3 | 18131 | −1,09% | 1,93% | −8,76 | giữ |
| … | | | | | |
| 2.561 | 18384 | −0,66% | 2,67% | −3,809 | giữ (người cuối cùng) |
| 2.562 | 14953 | −0,95% | 3,84% | −3,808 | **loại** |

**Công bằng:** EB cũng chỉ được chọn trong **đúng 2.561 chiến lược này**.

---

## Bước 2. AI "đọc" lịch sử của 18144 như thế nào

**2a. Quay đầu cho thuận chiều.** 18144 lỗ trong quá khứ, nên ta **đổi dấu toàn bộ** lịch sử của nó: lỗ thành lãi, lãi thành lỗ. Như vậy mọi chiến lược đều được nhìn theo chiều "có lời trong quá khứ".

**2b. Chuỗi 240 tháng mà AI nhìn thấy** (đã đổi dấu):

```
1980: −0,03  −1,62  +5,62  −2,59  −1,12  −1,73 ...
 ...
1999: ... +1,26  +4,82  +4,75  +5,50  +10,30  +7,88   ← cuối 1999 lãi rất mạnh
```

**2c. Nén chuỗi thành "dấu vân tay".** Một mô hình chuỗi thời gian **đã học sẵn** (pretrained: FinCast hoặc Chronos-2) đọc chuỗi trên. Nó tóm chuỗi lại thành vài dãy số: 4 vector, mỗi vector được nén còn 16 số. Các dãy số này mô tả "hình dáng" của lịch sử: xu hướng, độ giật, chu kỳ… Con người không đọc trực tiếp được; ví dụ 5 số đầu là `0,406  −0,001  0,143  0,061  0,732`.

**2d. Thêm hai con số quen thuộc**, đưa về cùng thang đo:
- **t-stat 9,79.** Ứng viên trung bình có t-stat 4,06, nên 18144 **cao hơn trung bình khoảng 6 lần độ lệch**.
- **Lãi TB 1,25%.** Ứng viên trung bình lãi 0,66%, nên 18144 **cao hơn khoảng 2,5 lần độ lệch**.

**AI không được thấy:** con số dự báo của EB, và bất cứ thứ gì từ năm 2000 trở đi.

---

## Bước 3. AI học từ đâu: "đề thi" của các năm trước

AI học bằng cách làm lại các năm trước như làm đề có đáp án. Với mỗi chiến lược ở mỗi năm cũ:
- **Câu hỏi:** lịch sử 20 năm (như Bước 2).
- **Đáp án:** năm sau chiến lược đó lãi thật bao nhiêu.

Lúc dự báo năm 2000, AI được học:

| Phần | Năm | Số câu hỏi |
|---|---|---|
| Học | 1966–1998 | 61.399 |
| Kiểm tra để biết lúc nào dừng học | 1999 | 2.573 |
| **Dự báo (chưa biết đáp án)** | **2000** | **2.561** |

Sang năm 2001, AI học lại từ đầu, thêm dữ liệu năm 1999. Việc này lặp lại 38 lần, cho 1983–2020.

**Đáp án được viết thế nào.** Ta không dùng thẳng "lãi bao nhiêu %". Ta đổi sang "lãi bao nhiêu **so với độ dao động của chính nó**", gọi là **đơn vị Sharpe**. Sau đó trừ đi mức trung bình của cả năm. Nhờ vậy AI chỉ phải học "chiến lược này **hơn hay kém** các chiến lược khác bao nhiêu".

Ví dụ: đây là đáp án thật của năm 2000. Nó chỉ dùng để chấm điểm, AI không được thấy.

| Chiến lược | Lãi thật 2000 (đã đổi dấu) | ÷ dao động | Trừ mức chung (0,59) = **đáp án** | Nói thường |
|---|---|---|---|---|
| 18144 | +2,40%/tháng | 1,21 | **+0,62** | thắng đậm |
| 18135 | +0,39% | 0,22 | −0,37 | kém trung bình |
| 18131 | +0,63% | 0,33 | −0,26 | kém trung bình |
| 18178 | −0,12% | −0,06 | −0,65 | thua |

---

## Bước 4. AI đoán năm 2000 cho 18144

Có thể hiểu dự báo của AI là **ba lớp cộng lại**, đơn vị là Sharpe:

```
  Mức chung của năm      0,288   "năm nay các ứng viên nói chung lãi cỡ này"
+ Thưởng cho t-stat cao  0,239   "chiến lược càng đều đặn thì càng đáng tin"
+ AI tự chỉnh            −0,115  "nhìn kỹ chuỗi và so với hàng xóm, nên hạ bớt"
= Dự báo                 0,412   → nhân với dao động 1,98% = 0,82% mỗi tháng
```

- **Mức chung, 0,288.** Trong quá khứ, các ứng viên trông như có Sharpe 0,327. Nhưng 20 năm qua cho thấy thực tế chỉ giữ được khoảng **88%** mức trông thấy trong quá khứ. Vì vậy: 0,327 × 0,88 ≈ 0,288.
- **Thưởng t-stat, 0,239.** 18144 có t-stat cao hơn trung bình ứng viên 5,06 lần độ lệch. Mỗi lần độ lệch được thưởng 0,047, hệ số này học từ 15 năm trước. Vậy thưởng = 5,06 × 0,047 ≈ 0,239.
- **AI tự chỉnh, −0,115.** Đây là phần mạng nơ-ron học được. Nó đọc "dấu vân tay" của chuỗi (Bước 2c) và so 18144 với 2.560 ứng viên khác cùng năm, bằng khối attention. Ở đây AI cho rằng nên **hạ bớt** kỳ vọng.

**So ba người đoán với thực tế:**

| Ai đoán | Đoán (%/tháng) | Hạng trong 2.561 |
|---|---|---|
| EB (paper) | 1,03 | 1 |
| AI ROBUST | 0,94 | 2 |
| AI FINAL | 0,82 | 30 |
| **Thực tế 2000** | **2,40** | |

Với riêng 18144, cả ba đều đoán thấp hơn thực tế. Nhưng **điều quan trọng không phải đoán đúng từng con số**. Điều quan trọng là **xếp hạng đúng**: chiến lược nào được đưa vào danh mục.

---

## Bước 5. Chọn danh mục năm 2000: hai bên chọn khác nhau ra sao

**Luật chọn (giống paper):**
1. Nếu dự báo âm thì làm ngược chiến lược.
2. Xếp hạng theo **dự báo ÷ dao động**: ưu tiên lãi cao nhưng ít giật.
3. Lấy **26 chiến lược đứng đầu** (1% của 2.561).
4. Chia tiền đều, giữ suốt năm 2000.

**8 lựa chọn đầu tiên của mỗi bên, kèm kết quả thật năm 2000:**

| AI FINAL chọn | t-stat quá khứ | Lãi thật 2000 | | EB chọn | t-stat quá khứ | Lãi thật 2000 |
|---|---|---|---|---|---|---|
| 8878 | 8,2 | ✅ +2,54% | | 18144 | 9,8 | ✅ +2,40% |
| 26328 | 8,5 | ✅ +2,46% | | 27689 | **4,4** | ❌ −0,47% |
| 15313 | 8,6 | ✅ +2,29% | | 18135 | 8,8 | +0,39% |
| 3587 | 8,1 | ✅ +1,03% | | 18131 | 8,8 | +0,63% |
| 2481 | 7,8 | ✅ +2,28% | | 27771 | **4,4** | ❌ −0,10% |
| 2525 | 8,0 | ✅ +2,26% | | 27677 | **4,4** | +0,11% |
| 3586 | 7,8 | ✅ +1,66% | | 18155 | 8,7 | +0,48% |
| 2482 | 7,5 | ✅ +2,37% | | 18178 | 8,6 | ❌ −0,12% |

(Lãi tính theo %/tháng, đã đổi dấu theo hướng được chọn.)

**Đọc bảng bằng lời:**
- Hai bên chọn **rất khác nhau**: chỉ **trùng 4/26** chiến lược.
- **EB** bị hấp dẫn bởi vài chiến lược **lãi quá khứ rất to nhưng thất thường** (t-stat chỉ ~4,4). Năm 2000, những chiến lược này lỗ hoặc gần như hòa.
- **AI** nghiêng về các chiến lược **lãi đều đặn** (t-stat 7,5–8,6), và năm 2000 chúng lãi thật.

**Kết quả cả danh mục năm 2000**, mỗi tháng (%):

```
            T1    T2    T3    T4    T5    T6    T7    T8    T9   T10   T11   T12  │ Trung bình
EB        +8,6 +22,6  −6,7  −5,4  −3,2  +8,2  −2,5  +3,7  −1,5  −2,8  −5,8  −8,0  │   0,62
AI ROBUST +6,6 +12,6  −0,7  −1,6  −2,2  +2,3  −0,1  +2,8  +0,0  +0,9  −0,6  −2,8  │   1,43
AI FINAL  +3,9  +6,4  +0,5  −0,3  +0,6  +1,8  +2,0  +0,8  +1,8  +3,2  +4,0  +0,2  │   2,08
```

**EB** lên xuống dữ dội: có tháng +22,6%, có tháng −8%. **AI FINAL** đi đều hơn nhiều: chỉ 1 tháng âm nhẹ, cuối năm lãi trung bình gấp hơn 3 lần.

⚠️ Đây chỉ là **một năm** để minh họa. Kết luận phải dựa trên tất cả các năm (Bước 6).

---

## Bước 6. Làm lại cho mọi năm 1983–2020 và chấm điểm

Lặp Bước 1–5 cho từng năm, ta có **456 tháng** lãi/lỗ cho mỗi bên. Thời gian được chia hai phần:

- **1983–2004 ("dev"):** dùng để **chọn** mô hình nào tốt nhất.
- **2005–2020 ("holdout"):** **không được nhìn** khi chọn, chỉ dùng để kiểm tra cuối cùng. Giống như đề thi thật sau khi đã ôn bằng đề thử.

### Thước đo 1 (quan trọng nhất): danh mục kiếm tiền tốt cỡ nào

Ta dùng **Sharpe**: lãi trung bình so với độ giật. Sharpe 1 nghĩa là lãi đủ bù rủi ro. Cao hơn là tốt hơn.

| Danh mục | Giai đoạn | Sharpe AI (ROBUST) | Sharpe EB | AI hơn có chắc không? |
|---|---|---|---|---|
| Top 1% | 1983–2004 | **1,61** | 1,06 | chắc (p = 0,008) |
| Top 5% | 1983–2004 | **1,89** | 1,48 | chắc (p = 0,024) |
| Top 10% | 1983–2004 | **2,01** | 1,72 | chắc (p = 0,023) |
| Top 10% | 2005–2020 | **1,03** | 0,82 | chắc (p = 0,005) |

"p" là xác suất chênh lệch này chỉ do may rủi. Dưới 0,05 thì coi là chắc.

**Nói thường:**
- Lãi thô hai bên **gần bằng nhau**, khoảng 9,3%/năm ở 1983–2004.
- AI hơn ở chỗ **ít giật hơn**: kiếm cùng mức tiền nhưng ít rủi ro hơn.

### Thước đo 2: đoán có bị lệch không

Chia các chiến lược thành 20 nhóm theo mức dự báo. Với mỗi nhóm, so "đoán bao nhiêu" với "thật bao nhiêu". Đường lý tưởng có độ dốc 1.

| | 1983–2004 | 2005–2020 |
|---|---|---|
| AI | 1,08 | 0,42 |
| EB | 1,11 | 0,28 |

Hai bên ngang nhau, và cả hai đều đoán quá lạc quan ở giai đoạn sau.

### Thước đo 3: xếp hạng đúng thứ tự đến đâu (IC)

Mỗi năm, ta đo mức khớp giữa thứ tự dự báo và thứ tự thật: 1 là khớp hoàn toàn, 0 là như đoán bừa.

| | 1983–2004 | 2005–2020 |
|---|---|---|
| AI | 0,215 | **0,118** |
| EB | 0,238 | 0,070 |

Giai đoạn đầu hai bên ngang nhau. Giai đoạn sau AI hơn rõ (p = 0,016).

### Các mốc đặt cạnh để không tự lừa mình

- **Cách ngây thơ "cứ chọn t-stat cao nhất", không cần mô hình:** Sharpe 1,59 / 1,85 / 1,95. Ngang AI, cũng hơn EB.
- **Mô hình gốc trong notebook tham khảo:** Sharpe 1,24 / 1,41 / 1,49. Thua AI hiện tại rõ ràng.

### Kiểm tra bẫy: chiến lược "Ticker"

Ticker là các chiến lược **vô nghĩa**, sắp cổ phiếu theo chữ cái tên mã. Một người đoán tốt **không được** chọn chúng.

| | % Ticker lọt vào top 1% |
|---|---|
| EB | 0% |
| **AI FINAL** | **0,6%** |
| AI ROBUST | 1,4% |
| Mô hình gốc | 87% (bị lừa nặng) |

---

## Bước 7. Những điều phải nói thật

1. **Không so thẳng với con số trong paper** (Sharpe 1,46). Paper gộp 6 họ chiến lược, còn ở đây chỉ làm 1 họ. Vì vậy EB được **chấm lại trên đúng dữ liệu của dự án** rồi mới so.
2. **Đã thử 115 cấu hình AI.** Thử nhiều thì dễ gặp một cái "đẹp nhờ may". Sau khi trừ hao yếu tố này:
   - câu "có AI vượt EB" vẫn đứng vững (p = 0,010);
   - riêng mô hình được chọn thì chưa đủ chắc (p = 0,09–0,20).
3. **AI hơn chủ yếu nhờ ít giật, không phải nhờ lãi nhiều hơn.**
4. **AI dùng mô hình đã học sẵn** (pretrained). Bản tự học từ đầu kém hơn rõ.
5. **Mọi lựa chọn mô hình chỉ dựa trên 1983–2004.** Giai đoạn 2005–2020 không hề được nhìn khi chọn.
