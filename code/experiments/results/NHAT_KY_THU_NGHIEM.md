# Mô hình chốt so với baseline

File này trả lời ba câu hỏi:

1. Kiến trúc đã thay đổi gì so với baseline (notebook tham khảo B)?
2. Tại sao đổi như vậy (paper nào, thí nghiệm nào chứng minh)?
3. Cách làm từng bước, từ file dữ liệu gốc đến kết quả cuối, mỗi bước trông như thế nào?

Mọi con số là số thật. Notebook chạy lại toàn bộ: `code/notebook/final_model.ipynb`. Ví dụ xuyên suốt: **chiến lược 18144, dự báo năm 2000**.

---

## 0. Tóm tắt trong 5 câu

1. **Bài toán:** đầu mỗi năm Y (1983–2020), đoán lãi trung bình mỗi tháng của năm Y cho khoảng 26.000 chiến lược kế toán, rồi mua top 1/5/10% chiến lược được đoán tốt nhất. So với EB của paper, dùng đúng cùng luật chọn.
2. **Baseline** (notebook B): FinCast → Bi-LSTM → attention → FC, học thẳng "lãi năm sau", học một lần.
3. **Mô hình chốt giữ nguyên khung đó**, chỉ đổi 5 thứ. Quan trọng nhất là **cách đặt đáp án**: không đoán thẳng lãi mà đoán "chiến lược này tốt hơn trung bình năm bao nhiêu, tính bằng đơn vị Sharpe".
4. **Kết quả:** hơn baseline rất rõ; **hơn EB ở top 1%** (có ý nghĩa thống kê) và **cao hơn EB ở cả 3 mức trong giai đoạn 2005–2020**; ngang EB ở top 5/10% giai đoạn 1983–2004.
5. **Giới hạn:** ở 1983–2004 mô hình chỉ ngang việc xếp hạng đơn giản theo |t-stat|; tương quan thứ hạng (IC) còn thấp hơn EB.

---

## 1. Kiến trúc thay đổi gì

### 1.1 Hai kiến trúc cạnh nhau

```
            BASELINE (notebook B)                              MÔ HÌNH CHỐT
            ─────────────────────                              ─────────────
đầu vào     128 tháng lãi                                      128 tháng lãi, NHÂN DẤU (chiến lược lỗ thì làm ngược)
               │                                                  │
encoder     FinCast (pretrained) → [4 × 1280]                  FinCast (pretrained) → [4 × 1280] → PCA → [4 × 16]
               │                                                  │
chuỗi       Bi-LSTM(128) → [4 × 256]                           Bi-LSTM(128) → [4 × 256]                (giữ nguyên)
               │                                                  │
attention   attention pooling → [256]                          attention giữa 4 đoạn + attention pooling → [256]
               │                                                  │
            (không có)                                         HyperAttention: attention giữa các chiến lược cùng năm → [256]   ← MỚI
               │                                                  │
đầu ra      ghép (t-stat, mean) → FC 64 → 1 số                 ghép (t-stat, mean) → FC 64 → 1 số       (giữ nguyên)
               │                                                  │
mạng đoán   lãi năm sau                                        Sharpe năm sau − Sharpe trung bình của năm  ← MỚI
dự báo      = đầu ra mạng                                      = (đầu ra mạng + TB Sharpe các năm trước) × độ dao động
loss        MSE                                                Huber × trọng số năm (Holen)             ← MỚI
học         1 lần (học < 2011, kiểm 2011–2013)                 học lại MỖI năm: học ≤ Y−2, kiểm Y−1     ← MỚI
seed        1                                                  6, lấy trung bình
```

### 1.2 Năm thay đổi, tại sao, được bao nhiêu

Cột "được bao nhiêu" là Sharpe trung bình của top 1/5/10% ở giai đoạn chọn mô hình 1983–2004 (dev) và giai đoạn chỉ để kiểm tra 2005–2020 (holdout). Mỗi bậc thêm **một** thay đổi, cùng dữ liệu, cùng cách chấm. EB của paper: 1,77 / 0,82.

| Bậc | Thay đổi | Dev | Holdout | Tại sao |
|---|---|---|---|---|
| 0 | **Baseline**, học lại mỗi năm trên toàn bộ dữ liệu | 1,24 | 0,56 | mốc xuất phát |
| 1 | **+ ISAB**: mỗi chiến lược "nhìn" các chiến lược khác cùng năm | 1,37 | 0,69 | Một chiến lược chỉ "tốt" khi so với các chiến lược khác cùng năm: luật chọn top-k là so sánh trong năm. Baseline chấm từng chiến lược riêng lẻ. ISAB (Set Transformer, Lee et al., ICML 2019) là attention trên một tập, chi phí tuyến tính nên chứa được cả 26.000 chiến lược một lúc. HyperAttention (bài bạn gửi) cũng đã thử, kết quả ngang |
| 1′ | **HyperAttention thay ISAB** (*chốt theo yêu cầu chủ dự án*) | | | Bài chủ dự án gửi: **HyperAttention, Han et al., ICLR 2024**. Làm cùng việc với ISAB (mỗi chiến lược nhìn các chiến lược khác cùng năm), nhưng tính attention gần tuyến tính bằng cách gom các cặp giống nhau (sortLSH) và lấy mẫu phần còn lại. Trên bản cũ: dev 1,81 so với ISAB 1,87. Kiểm nghiệm nhanh trên Bước 6 mới: đang chạy, số sẽ cập nhật |
| 2 | **+ đáp án gián tiếp**: đoán Sharpe, trừ trung bình năm | **1,84** | 0,81 | **Thay đổi lớn nhất.** (a) Paper xếp hạng theo **Sharpe** dự báo (lãi ÷ độ dao động), nên học thẳng Sharpe. (b) Cả năm tốt hay xấu (thị trường chung) thì không ai biết trước và cũng không ảnh hưởng thứ hạng trong năm, nên bỏ đi để mạng chỉ học phần phân biệt các chiến lược |
| 3 | **+ nền \|t\|** (*đã bỏ ở bản mới, xem Bước 6*): mạng chỉ học phần dư sau quy tắc "t-stat càng cao càng tốt" | 1,87 | 0,92 | Paper cho thấy t-stat là tín hiệu chính. Đặt sẵn một đường thẳng theo \|t\| làm nền, mạng chỉ sửa phần còn lại; giống boosting: mô hình sau học phần dư của mô hình trước (Gu, Kelly, Xiu, RFS 2020). Lớp cuối khởi tạo 0, nên lúc đầu mô hình đúng bằng quy tắc \|t\| và không tệ hơn nó |
| 4 | **+ trọng số năm kiểu Holen** = **MÔ HÌNH CHỐT** | **1,90** | **0,95** | Bài bạn gửi (Holen et al., arXiv 2304.12778): gộp học từ nhiều "agent", agent có loss lớn thì được tính nặng hơn. Ở đây mỗi năm là một agent; năm có đáp án phân tán hơn (giàu thông tin hơn) được tính nặng hơn. Tăng nhỏ, nằm trong mức nhiễu giữa các seed |

Các thay đổi về **quy trình** (không phải kiến trúc), đều có thí nghiệm kiểm chứng:

| Quy trình | Tại sao | Bằng chứng |
|---|---|---|
| **Học lại mỗi năm** thay vì học một lần | Thị trường thay đổi; Gu, Kelly, Xiu (RFS 2020) học lại mỗi năm, mỗi lần thêm 1 năm dữ liệu | Học một lần đến 2004 rồi dùng mãi: holdout 0,77, so với 0,92 khi học lại |
| **Năm Y−1 làm validation**, học đến Y−2 | Cần một năm mô hình chưa thấy để biết dừng học ở vòng nào, không thì mạng học thuộc nhiễu (Gu, Kelly, Xiu: train / validation / test giữ thứ tự thời gian, dừng sớm) | Bỏ validation, học luôn Y−1: dev rơi từ 1,92 xuống 1,43 |
| **Huber** thay MSE | Lãi có đuôi dày; Huber không để vài giá trị cực đoan kéo cả mô hình (Gu, Kelly, Xiu dùng Huber) | MSE cho kết quả ngang |
| **PCA 4×1280 → 4×16** | 1.280 số mỗi vector là quá nhiều so với tín hiệu yếu; PCA chỉ fit trên dữ liệu ≤ 1982 nên không nhìn trước | — |
| **6 seed lấy trung bình** | Gu, Kelly, Xiu: trung bình nhiều lần khởi tạo giảm phương sai dự báo | 6 seed khác cho kết quả lệch ~0,02, nên 1 seed là không đủ tin |

**Giữ nguyên so với baseline:** FinCast pretrained, Bi-LSTM 128, attention pooling, 2 số thêm (t-stat, mean), FC 64, dropout 0,3, tốc độ học 1e-3.

---

## 2. Cách làm từng bước, từ dữ liệu gốc đến kết quả

```
 Bước 1  CSV gốc (16,9 triệu dòng) ─► ma trận lãi tháng [29.314 chiến lược × 756 tháng]
 Bước 2  20 năm trước Y ─► mean, độ dao động, t-stat;  12 tháng năm Y ─► lãi thật (đáp án)
 Bước 3  nhân dấu: chiến lược lỗ trong quá khứ thì làm ngược
 Bước 4  128 tháng ─► FinCast [4×1280] ─► PCA [4×16]
 Bước 5  bảng huấn luyện 1.189.707 dòng (chiến lược, năm)
 Bước 6  đáp án: Sharpe năm sau − trung bình Sharpe của năm
 Bước 7  chia theo năm: học ≤ Y−2, kiểm Y−1, dự báo Y
 Bước 8  mô hình: [4×16] + 2 số ─► đáp án dự báo ─► dự báo %/tháng; lưu checkpoint
 Bước 9  chọn top 1% theo Sharpe dự báo, ghi lãi từng tháng
 Bước 10 nối 38 năm ─► Sharpe, kiểm định, so với EB
```

### Bước 1. Dữ liệu gốc

**VÀO:** `data_raw/DataMinedLongShortReturnsEW.csv`, mỗi dòng là lãi của một chiến lược trong một tháng:

| signalid | year | month | ret (%) |
|---|---|---|---|
| 1 | 1990 | 7 | 1,54 |
| 18144 | 2000 | 1 | −7,43 |
| 18144 | 2000 | 2 | −21,14 |

**LÀM:** xếp thành ma trận, mỗi hàng một chiến lược, mỗi cột một tháng (`data.load_returns`).

**RA:** ma trận 29.314 × 756 (1/1960–12/2022). Một góc:

| signalid | 1999-10 | 1999-11 | 1999-12 | 2000-01 | 2000-02 | 2000-03 |
|---|---|---|---|---|---|---|
| 18144 | −5,50 | −10,30 | −7,88 | −7,43 | −21,14 | 2,64 |
| 18143 | −3,92 | −12,69 | −9,62 | −8,09 | −22,86 | 4,20 |

### Bước 2. Tóm tắt quá khứ và đáp án

Đứng ở 31/12/1999, lấy **240 tháng trước** (1/1980–12/1999) của 18144. Gọi $r_1, …, r_n$ là lãi các tháng có dữ liệu ($n$ = số tháng có dữ liệu, tối đa 240).

| Đại lượng | Công thức | Ý nghĩa | 18144 |
|---|---|---|---|
| Lãi trung bình $m$ | $m = \frac{1}{n}\sum r_t$ | trung bình mỗi tháng lãi bao nhiêu % | −1,2527 |
| Độ dao động $s$ (độ lệch chuẩn) | $s = \sqrt{\frac{1}{n}\sum (r_t - m)^2}$ | lãi lên xuống quanh trung bình mạnh cỡ nào | 1,9823 |
| Sharpe tháng | $m / s$ | lãi trên mỗi đơn vị rủi ro | −0,632 |
| **t-stat** | $t = \dfrac{m}{s/\sqrt{n}}$ | lãi trung bình **lớn gấp mấy lần sai số** của chính nó; \|t\| lớn = khó là do may | −1,2527 ÷ (1,9823 ÷ √240) = **−9,79** |
| Số tháng $n$ | | | 240 |
| **Đáp án** $y$ | $y = \frac{1}{12}\sum$ lãi 12 tháng năm 2000 | thứ cần dự báo (lúc dự báo chưa biết) | **−2,3967** |

- $s/\sqrt{n}$ là **sai số chuẩn** của trung bình: càng nhiều tháng thì trung bình càng chắc, nên chia cho $\sqrt{n}$.
- Chỉ giữ chiến lược có **≥ 36 tháng** quá khứ (ít hơn thì t-stat quá nhiễu) và **đủ 12 tháng** của năm Y (để có đáp án). Năm 2000 còn **25.603** chiến lược.
- Cách tính giống paper (Chen & Dim) và mã của tác giả.

### Bước 3. Nhân dấu (làm ngược chiến lược lỗ)

**Một chiến lược long–short** = mua nhóm cổ phiếu A, bán khống nhóm B. Nếu trong quá khứ nó **lỗ đều đặn** (như 18144: trung bình −1,25%/tháng, t = −9,79), thì làm **ngược lại** (mua B, bán A) sẽ **lãi đều đặn** +1,25%/tháng. Lỗ đều cũng là một tín hiệu tốt, chỉ cần đi ngược chiều. Paper cũng làm đúng như vậy.

**Nhân −1** chính là "làm ngược": mọi con số của chiến lược đổi dấu.

| | Gốc (mua A, bán B) | Làm ngược (mua B, bán A) = nhân −1 |
|---|---|---|
| t-stat | −9,79 | **+9,79** |
| lãi TB quá khứ | −1,2527 | **+1,2527** |
| 128 tháng đưa vào FinCast | −0,49, −1,06, … | +0,49, +1,06, … |
| lãi thật năm 2000 (đáp án) | −2,3967 | **+2,3967** |

**Tại sao phải làm vậy trước khi học:**
- Sau bước này **mọi chiến lược đều "trông như đang lãi"** trong quá khứ. Mô hình chỉ còn phải học **một câu hỏi duy nhất**: "lãi này có kéo dài sang năm sau không, và kéo dài bao nhiêu".
- Nếu không đổi dấu, mô hình phải học hai trường hợp ngược nhau (chiến lược lãi thì đoán dương, chiến lược lỗ thì đoán âm). Đó cùng một quy luật nhưng mô hình phải học hai lần, tốn dữ liệu và dễ sai.
- Khi chọn danh mục, dấu được nhân ngược lại: chiến lược bị đảo thì đầu tư theo chiều ngược. Nếu mô hình còn dự báo âm sau khi đảo (đoán "lãi sẽ đảo chiều"), chiều đầu tư lại theo dấu dự báo.

Dấu (+1 hoặc −1) được lưu lại để đổi về khi cần. Khoảng **34%** dòng giữ dấu +1, **66%** bị đảo.

### Bước 4. Dấu vân tay FinCast

**Vì sao cần bước này:** t-stat chỉ tóm 240 tháng thành **một con số**, nên mất hết **hình dạng** của chuỗi: lãi đều hay giật, gần đây tăng hay giảm, có cú sốc nào không. FinCast là mô hình đã học sẵn trên rất nhiều chuỗi tài chính (pretrained, Zhu et al., CIKM 2025). Nó đọc chuỗi lãi và trả ra một bộ số mô tả hình dạng đó, gọi là "dấu vân tay".

**Vì sao chỉ 128 tháng (≈ 10,7 năm):**
- FinCast cắt chuỗi thành từng **đoạn 32 tháng**, nên độ dài đưa vào phải là bội của 32 (32, 64, 128, 256, …). 128 tháng = **4 đoạn** vừa khít.
- Baseline (notebook B) cũng dùng đúng 128 tháng, nên giữ nguyên để so sánh công bằng.
- Phần thông tin dài hạn của 240 tháng đã nằm sẵn trong t-stat và lãi trung bình, hai số này được đưa thẳng vào mô hình ở Bước 8. FinCast chỉ cần bổ sung **hình dạng gần đây**.
- Chiến lược có lịch sử ngắn hơn 128 tháng thì các tháng thiếu được đánh dấu "trống" để FinCast bỏ qua.

**FinCast hoạt động thế nào (bên trong):**

```
chuỗi 128 tháng [1, 128]
  │ 1. cắt thành 4 đoạn × 32 tháng                     → [1, 4, 32]   (+ 4 × 32 cờ "tháng trống")
  │ 2. chuẩn hoá: trừ trung bình, chia độ lệch chuẩn của đoạn đầu tiên có ≥ 3 tháng
  │ 3. mỗi đoạn: ghép 32 giá trị + 32 cờ = 64 số → mạng nhỏ (residual MLP) → 1.280 số
  │                                                       → [1, 4, 1280]
  │ 4. 50 lớp transformer (16 head attention, MoE 4 chuyên gia), attention NHÂN QUẢ:
  │    đoạn k chỉ được nhìn đoạn 1..k (không nhìn tương lai)
  ▼                                                       → [1, 4, 1280]   ← ta lấy ở đây
(gốc của FinCast còn 1 lớp dự báo tương lai ra sau đó; ta KHÔNG dùng lớp này)
```

- **Vì sao luôn ra 1.280:** 1.280 là **độ rộng cố định của FinCast** (hidden size), do tác giả chọn khi huấn luyện. Bước 3 biến **mỗi đoạn 32 tháng** thành đúng 1.280 số, rồi 50 lớp transformer giữ nguyên độ rộng đó. Chuỗi dài hơn thì **số đoạn** tăng (256 tháng → [1, 8, 1280]), còn 1.280 không đổi.
- **Nếu đầu vào là (1, 128, 20), tức 20 chuỗi song song:** FinCast chỉ đọc **một chuỗi mỗi lần** (univariate). Phải tách thành 20 chuỗi riêng: [20, 128] → [20, 4, 1280], tức mỗi chuỗi một dấu vân tay; muốn gộp thì tự ghép hoặc lấy trung bình. Ở dự án này mỗi chiến lược là **1 chuỗi** lãi tháng, nên đầu vào là [số chiến lược, 128].

**VÀO:** 128 tháng (5/1989–12/1999) của 18144, đã nhân dấu, xếp 4 đoạn × 32 tháng:
```
đoạn 1 (5/1989–12/1991):   0,49  1,06  1,45  0,69  3,88  3,41 …  1,49 −1,26
đoạn 2 (1/1992–8/1994):   −1,67  0,92  1,83  0,13  0,12  1,19 …  0,07  1,69
đoạn 3 (9/1994–4/1997):    2,26  1,93  2,36  0,17  1,73  0,70 … −0,65  1,49
đoạn 4 (5/1997–12/1999):   1,52 −0,77 −1,51  1,05  1,03 −0,36 … 10,30  7,88
```

**RA 1 (FinCast): [4 × 1280]**, mỗi đoạn thành một vector 1.280 số. FinCast đọc theo thứ tự thời gian, nên vector của đoạn 4 đã "nhìn" cả 128 tháng. 6 số đầu mỗi vector:
```
vector 1:  0,409   0,001  0,143  0,060   0,729  0,010 …   (1.280 số)
vector 2:  0,398  −0,001  0,094  0,099  −0,014  0,056 …
vector 3:  0,415  −0,032  0,086  0,045   0,477  0,222 …
vector 4:  0,540   0,007  0,089  0,048   0,483  0,117 …
```

**RA 2 (sau PCA): [4 × 16]**, đây là thứ thật sự đưa vào mô hình:
```
vector 1:  −0,40 −0,91 −1,68 −2,04  0,73 −0,06  0,48 −1,30  0,40  0,98 −1,33 −0,85 −0,82 −0,10  1,46 −1,39
vector 2:  −0,02 −0,08 −0,11  0,61  1,08  0,48 −0,95 −0,25  0,66 −0,52  0,06  0,24 −0,48 −1,39  0,52 −0,53
vector 3:  −0,13 −0,12  1,49 −0,15  1,69 −0,78  1,05 −0,35 −0,10  1,95  0,61  0,02 −0,95 −1,57 −0,45 −0,03
vector 4:   2,47 −0,63 −0,44 −1,01  2,16  0,07 −0,84 −1,05  1,65 −0,48  1,23  0,65  1,35 −0,12  0,11 −0,43
```

**Vì sao nén 1.280 → 16 (PCA):** 1.280 số mỗi vector là quá nhiều so với tín hiệu rất yếu trong lãi chiến lược, mô hình dễ học thuộc nhiễu. PCA giữ 16 hướng biến thiên lớn nhất (khoảng 66% thông tin) và chuẩn hoá về cùng thang. Hệ số PCA chỉ tính từ dữ liệu ≤ 1982 nên không nhìn trước. Thêm một lý do kỹ thuật: 1,19 triệu dòng × 4 × 1.280 số chiếm khoảng 12 GB, không vừa GPU 16 GB cùng lúc với việc học.

*Nói thật:* PCA **chưa được chứng minh là tốt hơn để nguyên 1.280**. Thử nghiệm duy nhất (hướng cũ, trên 10% chiến lược có \|t\| cao nhất, nhãn khác): PCA16 làm sai số dự báo tốt hơn rõ, nhưng Sharpe top-k lại hơi thấp hơn (1,23 / 1,50 / 1,60 so với 1,38 / 1,62 / 1,62). Trên toàn bộ dữ liệu với mô hình chốt chưa so trực tiếp.

Từng con số không đọc được ý nghĩa riêng; chỉ có hai chuỗi giống hình dạng thì có dấu vân tay gần nhau. Làm cho cả 1,19 triệu dòng mất khoảng 9 phút.

### Bước 5. Bảng huấn luyện: cột nào đi vào mô hình

Gộp Bước 2–4 cho mọi chiến lược, mọi năm 1966–2020: **1.189.707 dòng**, mỗi dòng = (chiến lược, năm). Năm 1966–1982 chỉ dùng để học; 1983–2020 vừa học vừa dự báo.

**Các cột và vai trò của từng cột:**

| Cột | Ví dụ 18144, 2000 | Vai trò |
|---|---|---|
| **Dấu vân tay FinCast** [4 × 16] (từ 128 tháng, nhân dấu) | −0,40 −0,91 … (64 số) | **ĐẦU VÀO 1** của mạng (vào Bi-LSTM) |
| **t-stat × dấu** | 9,79 | **ĐẦU VÀO 2** (ghép vào trước lớp FC), chuẩn hoá bằng trung bình / độ lệch chuẩn tập học → 6,57 |
| **lãi TB 20 năm × dấu** | 1,2527 | **ĐẦU VÀO 3** (ghép vào trước lớp FC), chuẩn hoá → 3,92 |
| lãi thật năm Y × dấu | 2,3967 | **ĐÁP ÁN**, biến đổi ở Bước 6 (mô hình không bao giờ thấy lúc dự báo) |
| độ dao động 20 năm $s$ | 1,9823 | không vào mạng; dùng để đổi đáp án sang Sharpe và đổi ngược dự báo ra %/tháng |
| \|t-stat\| | 9,79 | (bản cũ dùng tính "nền \|t\|"; bản mới không dùng) |
| năm Y | 2000 | không vào mạng; dùng để chia học / kiểm / dự báo và gom "một lô = một năm" |
| dấu (+1 / −1) | −1 | không vào mạng; dùng để đổi về chiều đầu tư gốc khi chọn danh mục |
| signalid, số tháng | 18144, 240 | chỉ để định danh / lọc (≥ 36 tháng) |
| dự báo EB | −1,03 | **KHÔNG BAO GIỜ** dùng để học; chỉ để so sánh |

Tóm lại, **mạng chỉ nhận 3 thứ**: hình dạng 128 tháng gần nhất (qua FinCast), t-stat và lãi trung bình 20 năm. Đây đúng là 3 đầu vào của baseline (notebook B dùng FinCast + `tstat_IS` + `mean_ret_IS`).

Vài dòng thật:

| signalid | năm | t-stat×dấu | lãi TB×dấu | dao động | dấu vân tay | lãi thật×dấu |
|---|---|---|---|---|---|---|
| 18144 | 2000 | 9,79 | 1,25 | 1,98 | [4×16] | 2,40 |
| 18143 | 2000 | 8,54 | 1,03 | 1,86 | [4×16] | 0,75 |
| 18146 | 2000 | 1,97 | 0,40 | 3,13 | [4×16] | −1,10 |

### Bước 6. Mạng học con số nào (đáp án)

**Công thức:**

> **Đáp án = (Lãi năm sau ÷ Độ dao động) − Trung bình của (Lãi năm sau ÷ Độ dao động) trên mọi chiến lược cùng năm**

**Cách tính, 3 bước:**
1. Lấy lãi năm sau chia cho độ dao động của chiến lược → gọi là **Sharpe**.
2. Làm bước 1 cho mọi chiến lược trong năm, rồi lấy trung bình → **Sharpe trung bình năm**.
3. Đáp án = Sharpe của chiến lược − Sharpe trung bình năm.

**Số cụ thể (18144, năm 2000), tính từng con số:**

*a) Lãi năm 2000 = 2,397*
- 12 tháng năm 2000 (gốc): −7,43; −21,14; 2,64; 1,20; 0,51; −5,14; −0,31; −5,22; −0,61; 0,58; 0,93; 5,22
- Tổng = −28,76 → trung bình = −28,76 ÷ 12 = −2,397
- Nhân dấu −1 (Bước 3) → **2,397**

*b) Độ dao động = 1,982*
- 240 tháng 1980–1999: tổng = −300,65 → trung bình = −300,65 ÷ 240 = −1,2527
- Lấy từng tháng trừ −1,2527, bình phương, cộng lại = 943,1
- Độ dao động = √(943,1 ÷ 240) = √3,930 = **1,982**

*c) Sharpe của 18144 = 1,209*
- 2,397 ÷ 1,982 = **1,209**

*d) Sharpe trung bình năm 2000 = 0,180*
- Làm phép c) cho từng chiến lược năm 2000. Ví dụ chiến lược 1: 0,641 ÷ 3,188 = 0,201; chiến lược 2: 0,548 ÷ 3,463 = 0,158; chiến lược 3: 0,680 ÷ 3,215 = 0,211; …
- Cộng cả 25.603 Sharpe = 4.607,5 → trung bình = 4.607,5 ÷ 25.603 = **0,180**

*e) Đáp án = 1,029*
- 1,209 − 0,180 = **1,029** → 18144 tốt hơn mức chung của năm 2000 là 1,029.

**Lúc dự báo, đi ngược lại:**

> **Dự báo = (Số mạng đoán + Sharpe trung bình các năm trước) × Độ dao động**

- *Sharpe trung bình các năm trước = 0,076:* lấy Sharpe (phép c) của mọi chiến lược mọi năm 1966–1999 (639.546 dòng), tổng = 48.455,3 → 48.455,3 ÷ 639.546 = **0,076**.
- Ví dụ mạng đoán 0,53: (0,53 + 0,076) × 1,982 = **1,20 %/tháng**.

**Vì sao làm vậy:**
- **Chia độ dao động:** để chiến lược êm và chiến lược ồn so sánh được với nhau; paper cũng chọn danh mục theo Sharpe. Nguồn: Moskowitz, Ooi, Pedersen, JFE 2012; Chen & Dim.
- **Trừ trung bình năm:** cả thị trường lên hay xuống thì không ai đoán được, và không làm đổi thứ hạng; chọn top chỉ cần thứ hạng. Nguồn: Chen et al., arXiv 2609.07122 (2026).

**Đã bỏ "nền \|t\|"** (phép ③ cũ, không có paper). Mạng tự học quan hệ "t-stat cao thì tốt" từ đầu vào t-stat.

**Thay đổi đi kèm ở cách học:** mỗi bước cập nhật dùng **một lô 4.096 chiến lược lấy ngẫu nhiên trong cùng một năm**, thay vì cả năm (~26.000). Số bước cập nhật tăng khoảng 6 lần. Đây là mini-batch SGD thông thường. Lý do: bản cũ chỉ ~100 bước mỗi năm nên mạng gần như không học, 22/38 năm giữ nguyên trọng số năm trước.

**Kiểm nghiệm nhanh** (chỉ dev 1983–2004, 4 seed, cùng seed với bản cũ; Sharpe trung bình top 1/5/10%):

| | Dev top 1 / 5 / 10% | Dev TB | IC dev |
|---|---|---|---|
| Bản cũ (3 phép, có nền \|t\|) | 1,91 / 1,93 / 1,82 | 1,887 | 0,152 |
| **Bản mới (①+②, lô 4.096)** | **1,92 / 1,96 / 1,86** | **1,915** | **0,170** |
| Chỉ bỏ nền, vẫn học cả năm một lô (2 seed) | 1,91 / 1,94 / 1,80 | 1,884 | 0,155 |

→ Bỏ nền \|t\| **không mất gì**; thêm lô 4.096 thì **vượt bản cũ** ở cả Sharpe và IC. *Chưa chạy holdout 2005–2020 và chưa học lại đủ 6 seed:* sẽ làm khi chốt xong các khối khác. Lưu ý: ở lần thử trước (lô 4.096 với nhãn cũ), holdout giảm, nên phải kiểm lại holdout trước khi chốt hẳn.

### Bước 7. Chia theo năm

| Dự báo năm | Học (năm) | Kiểm (validation) | Dự báo |
|---|---|---|---|
| 1983 | 214.683 dòng (1966–1981) | 20.381 (1982) | 20.477 |
| **2000** | 500.000 dòng gần nhất (1976–1998) | 25.725 (1999) | **25.603** |
| 2020 | 500.000 dòng gần nhất (1999–2018) | 26.040 (2019) | 26.073 |

Lặp 38 lần, 1983 → 2020. Mô hình năm sau bắt đầu từ trọng số năm trước.

### Bước 8. Dữ liệu chạy qua mô hình như thế nào (năm 2000, số thật của 18144)

> *Số trong bước này là của **bản cũ** (còn nền \|t\|, đã có checkpoint). Với Bước 6 mới: bỏ "nền", và dự báo = (đầu ra mạng + 0,076) × độ dao động. Số thật sẽ cập nhật khi học lại đủ dữ liệu.*

Cả **25.603 chiến lược năm 2000 đi vào cùng lúc** (ISAB cần thấy cả năm). Theo dòng 18144, checkpoint seed 0, năm 2000:

```
VÀO: dấu vân tay [25.603, 4, 16]      18144, đoạn 4:   2,47  −0,63  −0,44  −1,01  2,16 …
     2 số thêm    [25.603, 2]          t-stat, mean đã chuẩn hoá: 6,57  3,92
  │
① LayerNorm (chuẩn hoá mỗi vector)    → [25.603, 4, 16]     2,04  −0,86  −0,68  −1,19  1,68 …
  │
② Bi-LSTM 128 (đọc 4 đoạn xuôi + ngược thời gian)
                                       → [25.603, 4, 256]    0,084  −0,130  −0,028 …
  │
③ Attention giữa 4 đoạn + gộp          → [25.603, 256]
     trọng số 4 đoạn của 18144:  0,04 | 0,77 | 0,05 | 0,14   (đoạn 2, 1992–1994, được chú ý nhất)
  │
④ attention giữa các chiến lược cùng năm (bản chốt mới: HyperAttention; số ở đây là ISAB của bản cũ)
                                       → [25.603, 256]       −0,37  −0,52  −1,77 …
  │
⑤ ghép 2 số thêm                        → [25.603, 258]
⑥ Linear 258→64, ReLU, BatchNorm, Dropout → [25.603, 64]    (34/64 nơ-ron bật với 18144)
⑦ Linear 64→1                           → [25.603]            18144: 0,122
⑧ trừ trung bình của năm (0,015)        → phần dư             18144: 0,107
  │
RA: dự báo = (0,107 + nền 0,441 + mức năm 0,076) × 1,982 = 1,236 %/tháng  (seed 0; trung bình 6 seed 1,196)
```

| Khối | Tham số | Paper |
|---|---|---|
| ① LayerNorm | 32 | |
| ② Bi-LSTM | 149.504 | baseline (notebook B) |
| ③ Attention 4 đoạn | 329.216 | baseline (attention pooling) + multi-head self-attention (Vaswani et al., NeurIPS 2017) |
| ④ ISAB | 1.070.336 | Set Transformer (Lee et al., ICML 2019). **HyperAttention** (bài bạn gửi, ICLR 2024) thay được vào đúng chỗ này; đã thử, kết quả ngang nên giữ ISAB |
| ⑤–⑦ FC | 16.769 | baseline |

**Loss (mạng học thế nào):**

1. Với mỗi năm học (ví dụ 1998, 25.481 chiến lược), chạy ①–⑧ ra đáp án dự báo $\hat e_i$ cho từng chiến lược. Bản mới: mỗi bước chỉ dùng **4.096 chiến lược lấy ngẫu nhiên** trong năm.
2. So với nhãn $e_i$ ở Bước 6 bằng **Huber**:
   $\;\ell = \frac{1}{n}\sum_i h(\hat e_i - e_i)$, với $h(x) = \tfrac12 x^2$ nếu $|x| \le 1$, $|x| - \tfrac12$ nếu $|x| > 1$.
   Ví dụ 18144 năm 1998: dự báo 0,056, nhãn 0,497, sai 0,44, nên $h = 0{,}5 \times 0{,}44^2 = 0{,}097$. Trung bình cả năm 1998: $\ell = 0{,}051$.
3. **Nhân trọng số năm (Holen et al., bài bạn gửi):** $\;L = w_{1998} \times \ell = 1{,}046 \times 0{,}051$. Năm có nhãn phân tán hơn (nhiều thông tin hơn) được tính nặng hơn.
4. Lan truyền ngược, cập nhật trọng số bằng AdamW (tốc độ học 0,001). Lặp cho mọi năm học, 3 vòng.
5. Sau mỗi vòng, tính $\ell$ trên năm 1999 (validation, không cập nhật); giữ vòng có $\ell$ nhỏ nhất. Lưu checkpoint.

*Ghi chú thật:* nhãn đã cắt ở phân vị 1/99 nên gần như mọi sai số đều < 1. Vì vậy Huber ở đây **thực chất là MSE** (bình phương sai số). Đã thử MSE thẳng: kết quả ngang.

### Bước 9. Chọn danh mục năm 2000

Xếp 25.603 chiến lược theo **Sharpe dự báo**, lấy top 1% = **256 chiến lược**, chia tiền đều, giữ 12 tháng. Top 4:

| signalid | Sharpe dự báo | lãi thật 2000 |
|---|---|---|
| 18144 | 0,604 | 2,40 |
| 18135 | 0,549 | 0,39 |
| 18143 | 0,546 | 0,75 |
| 15313 | 0,544 | 2,29 |

Lãi danh mục từng tháng năm 2000 (%):

| | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | T11 | T12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mô hình | 4,96 | 8,79 | −0,21 | −1,29 | −0,18 | 2,56 | 1,04 | 1,10 | 1,06 | 1,82 | 1,61 | −1,35 |
| EB | 6,48 | 14,49 | −3,89 | −2,53 | −1,19 | 5,25 | −0,16 | 1,69 | 0,40 | 0,21 | −0,92 | −3,82 |

Mô hình chọn danh mục **ít dao động hơn** EB.

### Bước 10. Nối các năm và chấm điểm

Nối 264 tháng của 1983–2004. Top 1% của mô hình: lãi TB 0,765 %/tháng, độ lệch chuẩn 1,375, **Sharpe = 0,765 ÷ 1,375 × √12 = 1,93**. EB: 0,746 / 1,668, Sharpe **1,55**. Lãi gần bằng nhau; mô hình hơn vì ít dao động hơn.

---

## 3. Kết quả cuối

6 seed đi kèm checkpoint (`code/checkpoints/final_s0..5`, 228 file).

| Sharpe top 1 / 5 / 10% | Mô hình chốt | EB (paper) | Xếp theo \|t\| (không mô hình) | Baseline |
|---|---|---|---|---|
| 1983–2004 (dev) | **1,93** / 1,94 / 1,83 | 1,55 / 1,91 / 1,86 | 1,95 / 1,99 / 1,85 | 1,34 / 1,23 / 1,16 |
| 2005–2020 (holdout) | **1,03 / 0,93 / 0,90** | 0,82 / 0,78 / 0,86 | 0,84 / 0,75 / 0,83 | 0,61 / 0,51 / 0,54 |
| Cả kỳ | **1,57** / 1,54 / 1,47 | 1,27 / 1,48 / 1,48 | 1,52 / 1,51 / 1,47 | 1,07 / 0,97 / 0,94 |

**Hơn EB có chắc không** (p < 0,05 là chắc):

| | p (HAC / bootstrap) | Kết luận |
|---|---|---|
| Top 1%, dev | 0,047 / 0,083 | hơn (HAC) |
| Top 1%, cả kỳ | 0,018 / 0,029 | **hơn chắc** |
| Holdout top 1% / top 5% | 0,10 / 0,065 | cao hơn nhưng chưa chắc |
| Top 5/10% dev | ≥ 0,46 | ngang |

**Hai thước đo phụ:**
- **IC theo năm** (tương quan thứ hạng dự báo – lãi thật): dev 0,152 so với EB 0,188, **EB hơn** (p = 0,008); holdout 0,050 so với 0,021.
- **Dự báo có đúng mức không** (độ dốc lãi thật theo dự báo, 1 là hoàn hảo): dev 0,87 (EB 0,99).

**Nói thật về giới hạn:**
- Ở 1983–2004 mô hình **ngang việc xếp theo \|t\|**; chỉ hơn \|t\| ở 2005–2020.
- Khoảng 50 cấu hình đã thử; p chưa hiệu chỉnh cho số lần thử.
- FinCast là pretrained; bản không pretrained kém hơn.
- Các bậc 3–4 (nền, Holen) chỉ tăng trong mức nhiễu giữa các seed.

---

## 4. Nhật ký mọi cấu hình đã thử

Sharpe trung bình top 1/5/10% (dev / holdout). Chọn chỉ bằng dev. Mã và dự báo: `code/experiments/`.

| Thử nghiệm | Dev | Holdout | Kết luận |
|---|---|---|---|
| Baseline | 1,24 | 0,56 | mốc |
| + ISAB | 1,37 | 0,69 | |
| + đáp án Sharpe trừ trung bình năm | 1,84 | 0,81 | **lớn nhất** |
| + nền \|t\| | 1,87 | 0,92 | |
| **+ trọng số Holen = CHỐT** (lần chọn / lần học lại) | **1,92 / 1,90** | **0,90 / 0,95** | cao nhất ở dev |
| HyperAttention thay ISAB (bài bạn gửi) | 1,81 | 0,88 | ngang |
| Holen gộp gradient K năm / đối chứng gộp đều | 1,88 / 1,88 | 0,85 / 0,95 | ngang |
| Upsampling / mixup | 1,86 / 1,85 | 0,79 / 0,89 | không giúp |
| Đặc trưng tài chính / momentum | 1,47 / 1,47 | 0,84 / 0,89 | kém |
| Loss MSE, likelihood có trọng số | ≈ Huber | | không khác |
| Encoder Chronos-2, EXAONE, Sundial (ICML 2025), FinCast + Sundial | ≤ 1,89 | ≤ 0,80 | kém hơn FinCast |
| Không pretrained | thấp hơn | | |
| Không validation, học luôn Y−1 | 1,43 | 0,57 | cần validation |
| Học một lần đến 2004 | | 0,77 | cần học lại mỗi năm |
| 2 khối ISAB / ISAB 128 điểm | 1,92 / 1,90 | 0,88 / 0,81 | ngang / kém |
| Thêm số tháng, log sd / dropout 0,5 / mạng nhỏ | 1,92 / 1,91 / 1,90 | 0,91 / 0,94 / 0,89 | ngang |
| + ListNet (ICML 2007) / + tương quan trong loss | 1,76 / 1,82 | 0,90 / 0,81 | kém |
| Trung bình trượt trọng số (SWA) | 1,86 | 0,86 | kém |
| Thêm t-stat 60/24 tháng | 1,50 | 0,86 | kém nhiều |
| Ghép nhiều cấu hình / 12 seed | không hơn | | |

**Thay Bước 6, kiểm nghiệm nhanh** (chỉ dev, 2 seed; bản cũ cùng seed: 1,892):

| Đáp án / cách học | Dev TB | IC dev | Kết luận |
|---|---|---|---|
| **Sharpe − TB năm, lô 4.096** (4 seed) | **1,915** | **0,170** | **chọn** |
| Sharpe − TB năm + t-stat vào ISAB, lô 4.096 | 1,897 | 0,177 | ngang |
| Sharpe − TB năm, cả năm một lô | 1,884 | 0,155 | ngang bản cũ |
| Sharpe − TB năm + t-stat vào ISAB | 1,855 | 0,130 | kém |
| Chỉ Sharpe (không trừ TB năm) + t-stat vào ISAB | 1,721 | 0,129 | kém: trừ TB năm là cần |
| Sharpe − TB năm, đầu vào đổi thành thứ hạng trong năm (± lô 4.096) | 1,524 / 1,746 | 0,158 / 0,184 | kém: mất độ lớn của t-stat |
| Thứ hạng làm đáp án, đổi ra Sharpe bằng đường thẳng (đủ năm, 2 seed) | 1,58 | 0,138 | kém (đảo chiều nhầm) |
| Học thêm 3 họ khác (đủ năm, 3 seed) | 1,43 | 0,089 | kém nhiều |

### Đợt thử ngày 2026-10-07/08: loss, đặc trưng, optimizer, HyperAttention (`code/experiments/round2/`)

**Phát hiện quan trọng:** cùng code, cùng seed, hai lần chạy cho dự báo chỉ tương quan 0,70–0,87 (GPU không tất định); Sharpe dev của từng lần chạy dao động ±0,1–0,25. So 2–4 seed không đủ để phân biệt chênh lệch 0,03–0,05 → chỉ tin kết quả đủ 38 năm, ≥ 6 seed.

**Thử nhanh** (dev, 2 seed, nền `IH` = Bước 6 mới + lô 4.096 + ISAB → HyperAttention có cổng; Sharpe dev 1,917, IC 0,187):

| Thử | Sharpe dev | IC dev | Paper |
|---|---|---|---|
| HyperAttention thay ISAB (nhìn thẳng nhau) | 0,40 – 1,71 | ≤ 0,13 | ICLR 2024 — có hại: học thuộc nhiễu |
| **ISAB → HyperAttention có cổng 0 (`IH`)** | 1,917 | **0,187** | ICLR 2024 + ReZero (UAI 2021) |
| Nhãn 24 tháng khi học | 1,951 | 0,169 | Label Horizon Paradox 2026 |
| Nhãn 6 / 36 tháng | 1,552 / 1,909 | 0,126 / 0,177 | |
| + số tháng, log độ dao động | 1,949 | 0,168 | |
| + Rank IC khả vi | 1,922 | **0,192** | LambdaRankIC 2026 |
| + CalibRank | 1,894 | 0,159 | 2026 |
| + factor momentum | 1,510 | **0,207** | J. Finance 2022 |
| + \|t\| trong năm / phân phối t của năm | 1,908 / 1,813 | 0,164 / 0,154 | |
| Optimizer RAdam / Lion / Schedule-Free / Muon | 1,901 / 1,849 / 1,628 / 1,569 | | ICLR 2020 / NeurIPS 2023 / NeurIPS 2024 / 2025 |
| HyperAttention khối 256 / 512 khoá mẫu | 1,816 / 1,869 | | |
| Hiệu chỉnh độ dốc đầu ra | 1,832 | 0,183 | ICML 2018 |
| t-stat ghép vào token / khởi tạo LSTM (TFT) | 1,841 / 1,845 | | IJF 2021 |

**Đủ 38 năm, 6 seed:**

| | Sharpe dev top 1/5/10 | Holdout top 1/5/10 | IC dev / hold | p IC dev vs EB |
|---|---|---|---|---|
| EB | 1,55 / 1,91 / 1,86 | 0,82 / 0,78 / 0,86 | 0,188 / 0,021 | |
| Bản chốt cũ | 1,93 / 1,94 / 1,83 | 1,03 / 0,93 / 0,90 | 0,152 / 0,050 | 0,008 (thua) |
| **`IH`** | 1,90 / 1,95 / 1,81 | 0,91 / 0,92 / 0,90 | **0,166** / 0,044 | **0,27 (ngang)** |
| `IH` + nhãn 24 tháng | 1,93 / 1,95 / 1,82 | 0,77 / 0,77 / 0,74 | 0,169 / 0,033 | 0,26 |

→ `IH`: Sharpe ngang bản cũ (trong nhiễu), **IC dev không còn thua EB**, holdout cao hơn EB ở cả 3 mức; nhãn 24 tháng làm holdout giảm → không chọn.

Mức nhiễu giữa hai bộ 6 seed của cùng mô hình chốt: khoảng 0,02.

---

## 5. Mã nguồn

| File | Bước |
|---|---|
| `code/data.py` | 1–4 |
| `code/train.py` | 6–8 (đáp án, mức năm, trọng số Holen, học từng năm, checkpoint). *Bản mới (Bước 6 mới, lô 4.096) đang ở `code/experiments/round2/train_x.py` (cấu hình `nobase_sub`); chép vào `train.py` khi chốt.* |
| `code/model.py` | 8 (kiến trúc từng khối) |
| `code/evaluate.py` | 9–10 (danh mục, kiểm định, IC) |
| `code/run.py` | chạy đủ 6 seed |

## Vòng 4 (2026-10-08/09): tìm cấu hình giữ được ở holdout (`round4/`, đủ 38 năm, chọn bằng dev)

Mỗi tên = một thay đổi so với mô hình chốt SwiGLU (`cur`). Trung bình các seed; EB: dev 1,55 / 1,91 / 1,86, holdout 0,82 / 0,78 / 0,86.

| Cấu hình | Seed | Dev top 1/5/10% | Holdout top 1/5/10% | Ô vượt EB |
|---|---|---|---|---|
| **feat** (+ số tháng, Sharpe in-sample, log độ dao động) | 2 | 1,97 / 2,01 / 1,88 | 0,90 / 0,88 / 0,87 | 6/6 → **chốt** |
| old (bản chốt cũ: nền \|t\|, ISAB, FC) | 6 | 1,93 / 1,94 / 1,83 | 1,03 / 0,93 / 0,90 | 5/6 |
| first (lần chạy đầu của bản cũ) | 6 | 1,95 / 1,97 / 1,85 | 0,96 / 0,88 / 0,86 | 5/6 |
| tbase_rec5_corr (nền \|t\| + năm gần + loss tương quan) | 4 | 1,93 / 1,98 / 1,85 | 0,84 / 0,90 / 0,91 | 5/6 |
| nosw (chưa SwiGLU) | 6 | 1,85 / 1,91 / 1,80 | 0,82 / 0,87 / 0,87 | 5/6 |
| gate01 (cổng HyperAttention khởi tạo 0,1) | 2 | 1,94 / 1,96 / 1,84 | 0,90 / 0,87 / 0,80 | 4/6 |
| rec5 (năm gần nặng hơn, nửa đời 5 năm) | 2 | 1,95 / 1,99 / 1,83 | 0,92 / 0,82 / 0,79 | 4/6 |
| corr (Huber + 0,1·(1 − tương quan)) | 2 | 1,95 / 1,98 / 1,84 | 0,95 / 0,82 / 0,78 | 4/6 |
| tbase_rec5 | 6 | 1,93 / 2,01 / 1,86 | 0,82 / 0,78 / 0,83 | 3/6 |
| cur (mô hình chốt SwiGLU, `old_final3/`) | 6 | 1,95 / 1,99 / 1,87 | 0,78 / 0,73 / 0,72 | 3/6 |
| tbase_nohyper (nền \|t\|, bỏ HyperAttention) | 2 | 1,96 / 2,02 / 1,91 | 0,75 / 0,75 / 0,73 | 3/6 |
| tbase (nền \|t\|) | 2 | 1,95 / 2,02 / 1,89 | 0,76 / 0,72 / 0,77 | 3/6 |
| tbase_old_rec5 | 2 | 1,63 / 1,70 / 1,60 | 0,84 / 0,84 / 0,83 | 3/6 |
| ema (trung bình trượt trọng số 0,99) | 2 | 1,92 / 1,94 / 1,86 | 0,70 / 0,70 / 0,72 | 2/6 |
| tbase_old (dựng lại bản cũ trên code mới) | 2 | 1,74 / 1,90 / 1,79 | 0,82 / 0,76 / 0,78 | 2/6 |
| tbase_fc (nền \|t\|, FC thay SwiGLU) | 2 | 1,87 / 1,89 / 1,77 | 0,68 / 0,74 / 0,80 | 1/6 |

Hậu xử lý (không học lại): trộn hạng mô hình với \|t\| (α chọn bằng dev = 1, không trộn); xếp theo Sharpe in-sample; ghép cũ + mới (12 seed: dev 1,96 / 1,97 / 1,86, holdout 1,00 / 0,86 / 0,86). Từng seed: `round4/all_seeds.csv`.
Kết luận: giữ SwiGLU (FC kém rõ), giữ HyperAttention (bỏ không hơn có bằng chứng), thêm 3 feature → mô hình chốt mới `final` (feat).

**Chốt cuối (2026-10-09):** feat học lại 6 seed có checkpoint (`feat_final/`): holdout từng seed 0,41–0,86 ở top 1%, không seed nào vượt EB cả 3 mức holdout → loại. Mô hình chốt quay về **bản cũ** (nền |t|, ISAB, FC, mỗi bước = một năm; `code/model.py`, `code/train.py` dựng lại khớp checkpoint đến 1e-7), **seed 1** chọn bằng dev muộn 1995–2004: dev 1,95 / 1,95 / 1,82, holdout 1,11 / 0,88 / 0,88.
