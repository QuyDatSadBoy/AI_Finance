# Kiến trúc nền (FinCast → Bi-LSTM → Attention) và các cải tiến

Mô hình **tự dự báo** `pred_ret`, tức lợi suất trung bình tháng của một chiến lược trong năm Y, từ lịch sử của chính nó trước năm Y. Không dùng đầu ra của Empirical Bayes (EB) làm đầu vào hay làm điểm neo; EB chỉ là đối thủ để so sánh. Mã: [code/ht_base.py](../code/ht_base.py). Baseline của notebook tham khảo: [BASELINE_NOTEBOOK_THAM_KHAO.md](BASELINE_NOTEBOOK_THAM_KHAO.md).

## 0. Hướng nghiên cứu (cập nhật 2026-10-03)

- **Chỉ học từ các nguồn có tính thống kê:** kế toán và lợi suất quá khứ, tức 4 họ `acct_ew`, `acct_vw`, `past_ret_ew`, `past_ret_vw`. Không học từ Ticker. Theo Table 3 Panel A của paper, hai họ EW chiếm 91,0% và 8,6% top 1% của EB, còn Ticker chiếm 0,0%.
- **Thử mọi tập huấn luyện:** 1 họ, 2, 3 hoặc cả 4 họ (15 tổ hợp). Với mỗi tập, so mô hình với **mô hình gốc** (cấu hình của notebook tham khảo) và **EB trên cùng tập họ đó**.
- **Tinh chỉnh từng khối một** (dữ liệu, tiền xử lý, encoder, attention, loss, quy trình huấn luyện): mỗi bước xem đầu vào và đầu ra của khối, tìm nguyên nhân, rồi mới đổi tiếp. Dựa trên các paper bạn đưa và các phương pháp SOTA tìm qua MCP.
- **Bước cuối:** dùng mô hình đã huấn luyện để **suy luận trên Ticker** (không huấn luyện trên Ticker), xem nó có nhận ra Ticker gần như không có tín hiệu như paper không.

## 1. Kiến trúc nền và các khối thay được

```
chuỗi lợi suất tháng của 1 chiến lược (đảo dấu theo dấu trung bình trong mẫu)
   │  [B,128] hoặc [B,240]
   ▼
[A] Encoder (đóng băng)            FinCast (sơ đồ gốc) | Chronos-Bolt | Chronos-2 | TiRex | chuỗi thô (không pretrained)
   │  [B,4,1280] (FinCast)
   ▼
[B] Sequence                       Bi-LSTM 128 (gốc) | GRU | Transformer | trung bình
   │  [B,4,256]
   ▼
[C] Attention trên 4 patch         attention pooling (gốc) | mha (patch tự chú ý rồi pooling) | query | trung bình
   │  [B,256]
   ▼
[D] Attention giữa các chiến lược  xs: Transformer trên tập chiến lược CÙNG họ-năm (tối đa 2048)
    (mới, không có trong gốc)      | hyper: HyperAttention trên cả họ-năm | isab: Set Transformer
   │  [B,256]
   ▼
[E] Ghép đặc trưng                 E0: tstat_IS, mean_ret_IS (gốc) | F1 (35) | F2 (46)
   │  [B,258] .. [B,302]
   ▼
FC 64 → ReLU → BatchNorm → Dropout → FC 1      →  pred_ret (%/tháng)
```

Sơ đồ gốc không đổi; mỗi cải tiến là một khối thay được hoặc chèn thêm, bật tắt bằng một khóa cấu hình, nên đo được đóng góp riêng của từng khối.

## 2. Attention

| Khối | Cách làm | Vì sao |
|---|---|---|
| C: `mha` | trước khi pooling, 4 patch tự chú ý lẫn nhau (4 đầu, cộng phần dư) | một patch được đọc trong ngữ cảnh các patch khác, ví dụ patch gần nhất yếu đi so với patch cũ |
| D: `xs` | Transformer trên tập chiến lược cùng họ-năm, mỗi batch là một họ-năm | EB co dự báo vì nhìn cả phân phối t-stat của họ; `xs` cho mạng nhìn phân phối đó và tự học cách co. Tiền lệ: attention giữa các cổ phiếu trong MASTER (arXiv 2312.15235) |
| D: `hyper` | HyperAttention (Han và cộng sự, ICLR 2024): sortLSH xếp các chiến lược giống nhau cạnh nhau, attention chính xác trong các khối 256, phần còn lại ước lượng từ 256 khóa lấy mẫu đều nhân với `(n − khối)/m`; chi phí gần tuyến tính nên một batch chứa cả họ-năm (đến 16.384) | Guo (arXiv 2609.08106) đo được attention giữa các cổ phiếu gần đồng đều cộng một phần hạng thấp; lấy mẫu đều nắm phần đồng đều, LSH nắm các phần tử lớn. Đã kiểm tra: khi khối bao trùm cả tập, kết quả trùng attention chính xác (sai khác 6e-8) |
| D: `isab` | Set Transformer với m điểm cảm ứng | chi phí tuyến tính; **đã thử với 6 họ: không ổn định** (một số seed phân kỳ, median 3 seed kém EB 22% đến 24%) |

## 3. Hàm mất mát

Mô hình chính chỉ dùng **sai số dự báo hoặc likelihood của lợi suất** (nguyên tắc A). Các biến thể có lợi suất danh mục trong loss được gắn nhãn thăm dò.

| Khóa | Công thức (y: nhãn, ŷ: dự báo) | Vì sao |
|---|---|---|
| `huber` (mặc định) | bình phương khi sai nhỏ, tuyến tính khi sai lớn; `hdelta` là ngưỡng | nhãn có đuôi dày |
| `wls` = τ² | trọng số `1/(σ²/12 + τ²)`, σ là độ lệch chuẩn trong mẫu | nhãn là trung bình 12 tháng nên phương sai nhiễu là σ²/12; chiến lược nhiễu cao nên được tính ít hơn (EB tránh vấn đề này vì làm việc trong đơn vị t-stat) |
| `corrw` = λ | cộng `λ·(1 − tương quan(ŷ, y))` trong mỗi họ-năm | danh mục chỉ dùng thứ hạng; thành phần này thưởng cho việc xếp hạng đúng |
| `lw` = K | **L-Weighted** (Holen và cộng sự, arXiv 2304.12778): mỗi họ-năm là một "tác tử", K họ-năm mỗi bước, gradient của tác tử i nhân `loss_i / tổng loss + 1/K` | họ-năm khó học hơn được ưu tiên |
| `rw` = K | **R-Weighted** (cùng bài): trọng số `(r_i − r_min)/tổng + 1/K`, reward `r_i` là lợi suất thực tế của 10% chiến lược được dự đoán cao nhất (nhãn huấn luyện) | **thăm dò, vi phạm nguyên tắc A** theo yêu cầu của chủ dự án; cùng ý với bài metafunctor (REINFORCE là cross-entropy nhân trọng số bằng reward) |
| `mse`, `mae`, `wmse`, `studentt` | các loss khác đã cài | đã thử với 6 họ: không tốt hơn `huber` |

## 4. Dữ liệu và tiền xử lý

| Khóa | Cách làm | Vì sao |
|---|---|---|
| `fams` | tập họ dùng để huấn luyện, ví dụ `acct_ew+past_ret_ew` | hướng mới: chỉ học từ họ có tính thống kê |
| `infer` | họ chỉ dùng để suy luận (Ticker ở bước cuối) | kiểm tra mô hình có nhận ra họ không có tín hiệu |
| đảo dấu, chuẩn hóa | đảo theo dấu trung bình trong mẫu; chuẩn hóa bằng thống kê tập huấn luyện | chỉ một bài toán; không rò rỉ |
| `ydm` | nhãn trừ trung bình của chính họ-năm; khi dự báo cộng trung bình họ của tập huấn luyện | bỏ phần "cả năm lên hay xuống" mà không ai dự báo được, để mạng học xếp hạng |
| `fx=rank` | xếp hạng chuẩn hóa từng đặc trưng trong họ-năm | đặc trưng so sánh được giữa các năm |
| `shuf` | duyệt các họ-năm theo thứ tự ngẫu nhiên mỗi epoch | trước đó thứ tự cố định (lỗi thiết kế), các cập nhật cuối mỗi epoch luôn rơi vào cùng một họ-năm |
| `enc=raw240`, `chronos2_240` | đưa cả 240 tháng vào encoder | EB nhìn 20 năm; FinCast hiện chỉ nhìn 128 tháng |

## 5. Quy trình thí nghiệm

- **Đơn vị lặp là lần huấn luyện (seed).** Hai lần chạy cùng cấu hình lệch nhau 2 đến 3 điểm phần trăm sai số, nên mỗi biến thể chạy **3 seed** và so **theo cặp seed** với cấu hình gốc (`code/paired.py`); các ứng viên cuối chạy 5 seed. Dự báo cuối là median các seed.
- **Chấm trên đúng tập họ huấn luyện** (`eval_subset`): sai số dự báo so với EB (kiểm định Diebold–Mariano theo năm), và danh mục top 1/5/10% so với EB trên hai vũ trụ: chỉ các chiến lược ứng viên (AI tự quyết mọi lựa chọn), và toàn bộ chiến lược của các họ đó (kiểu paper).
- Chọn cấu hình **chỉ bằng dev 1983–2004**; holdout 2005–2020 chỉ báo cáo. Kiểm định cuối: Jobson–Korkie–Memmel và bootstrap theo khối năm cho Sharpe, Newey–West cho lợi suất (`code/stats_final.py`), kèm hiệu chỉnh nhiều phép thử.
- Mọi cấu hình, kể cả thất bại, nằm trong `code/_cache/fam_results.tsv` (hướng mới) và `code/_cache/base_results.tsv` (hướng cũ).

## 6. Kết quả hướng mới (cập nhật 2026-10-03)

Mọi số là median của 3 seed. "Dự đoán so với EB" là mức giảm sai số bình phương so với EB trên nhóm ứng viên (dương là tốt hơn EB). Sharpe là top 1 / 5 / 10%, vũ trụ ứng viên (AI tự quyết mọi lựa chọn), cả kỳ 1983–2020, cùng họ với EB. Toàn bộ trong `code/_cache/fam_results.tsv`.

### 6.1 F1: mọi tổ hợp họ × ba cấu hình

**Từng họ đơn:**

| Họ | Cấu hình | Dự đoán so với EB (dev / holdout) | Sharpe AI | Sharpe EB |
|---|---|---|---|---|
| Accounting EW | mô hình gốc | −16,3% / +4,8% | 1,09 / 1,22 / 1,25 | 0,95 / 1,25 / 1,38 |
| Accounting EW | `xs` + `ydm` + `mha` | −10,4% / +5,9% | **1,11 / 1,49 / 1,52** | 0,95 / 1,25 / 1,38 |
| Accounting EW | `xs` + `mha` + F2 | −17,9% / **+18,5%** | 1,16 / 1,35 / 1,38 | 0,95 / 1,25 / 1,38 |
| Accounting VW | `xs` + `mha` + F2 | −4,5% / −12,3% | −0,02 / −0,07 / 0,04 | 0,36 / 0,40 / 0,42 |
| Past Return EW | mô hình gốc | **+6,9%** / −20,0% | 0,22 / 0,23 / 0,21 | 0,40 / 0,34 / 0,27 |
| Past Return VW | `xs` + `ydm` + `mha` | −4,9% / −15,0% | 0,21 / **0,54 / 0,62** | 0,55 / 0,42 / 0,38 |

**Nhiều họ cùng lúc:** với `xs` + `ydm` + `mha`, gần như mọi tổ hợp đều thua EB về danh mục (ví dụ 4 họ: 0,79 / 1,18 / 1,15 so với 1,11 / 1,42 / 1,50). Học trên một họ cho kết quả tốt hơn học gộp, khớp với nhận định rằng mỗi họ có cấu trúc thống kê riêng.

### 6.2 Truy vết từng khối (Accounting EW, `trace_blocks.py`)

Thông tin dự báo còn lại sau mỗi khối (thăm dò tuyến tính, tương quan hạng trong họ-năm với lợi suất thật, trung bình 1995, 2000, 2004): đầu vào abs(t-stat) 0,154; encoder FinCast 0,071; Bi-LSTM 0,022; attention pooling 0,079; `xs` 0,070; FC 0,098; **đầu ra 0,111**; EB 0,173. Đầu ra của mạng kém hơn chính t-stat đầu vào, tức mạng không giữ được thông tin của đầu vào mạnh nhất.

### 6.3 Phân tích sai số theo năm (Accounting EW)

Phần lớn sai số vượt EB ở dev đến từ **mức trung bình của cả họ-năm**, không phải từ thứ hạng: các năm có lợi suất thật trung bình cao (1983, 1992, 1993, 2002) chiếm phần lớn khoảng cách. Thử thay mức trung bình bằng một mô hình mức học từ lịch sử không thu hẹp được khoảng cách, nên mức trung bình của họ-năm về bản chất rất khó dự báo. Kết luận: phép so chính giữa AI và EB nên là danh mục trong vũ trụ ứng viên (thứ hạng), còn MSE báo kèm.

### 6.4 Sửa lỗi ổn định và các khối mới (Accounting EW, mốc `T2` = `xs` + `ydm` + `mha` + F2)

| Biến thể | Dự đoán so với EB (dev / holdout) | Sharpe AI | Kết luận |
|---|---|---|---|
| `T2` | −7,4% / +10,2% | 1,14 / 1,34 / 1,39 | mốc |
| A: khối dự báo cố định + cả họ-năm trong một khối (`detev`, `xset` 3072) | −11,4% / +5,7% | 1,03 / 1,24 / 1,33 | xấu hơn |
| **A + LayerNorm thay BatchNorm** (`hnorm=ln`) | **−4,2%** / +4,1% | 1,21 / 1,29 / 1,35 | tốt nhất về dự đoán ở dev |
| A + xáo thứ tự + cosine | −6,1% / +4,3% | 0,97 / 1,35 / 1,42 | |
| A + đưa thống kê vào token `xs` (`xfuse`) | −11,6% / +3,0% | 0,87 / 0,95 / 1,05 | xấu hơn |
| A + `wls` τ² = 0,5 | −12,6% / +6,3% | 1,02 / 1,28 / 1,33 | không giúp |
| A + `wls` τ² = 2 | −10,0% / +4,9% | 1,02 / 1,30 / 1,41 | không giúp |
| `T` + `wide` (nhánh tuyến tính) | −14,1% / +15,3% | 0,91 / 1,40 / 1,50 | không giúp dev |

Thay BatchNorm bằng LayerNorm là thay đổi có tác dụng rõ: trong các batch `xs`, mỗi batch là một họ-năm nên BatchNorm học thống kê của từng họ-năm, còn lúc dự báo lại dùng thống kê trung bình chạy. Bước tiếp theo (giai đoạn S2, `code/run_fam2.py`): lấy `T2` + LayerNorm làm mốc, tách từng thay đổi còn lại, rồi áp dụng cho các họ khác.

### 6.5 Kiểm định (Accounting EW, `code/stats_sub.py`)

`xs` + `ydm` + `mha` so với EB, vũ trụ toàn bộ chiến lược của Accounting EW, **holdout 2005–2020**: Sharpe top 5% 1,07 so với 0,78 (JKM p = 0,001; bootstrap p = 0,021), top 10% 1,02 so với 0,86 (JKM p = 0,001; bootstrap p = 0,033); lợi suất thô ngang EB. Nhưng ở dev, top 5% và 10% thấp hơn EB có ý nghĩa, và vũ trụ này trộn dự báo AI với dự báo EB cho các chiến lược ngoài nhóm ứng viên. Ở vũ trụ ứng viên, Sharpe cao hơn EB ở mọi top-k trên cả kỳ nhưng p từ 0,06 đến 0,38, còn **lợi suất thô thấp hơn EB có ý nghĩa** (top 1% dev 5,1% so với 9,4%/năm, p = 0,017). **Chưa đạt tiêu chí SOTA.**

### 6.6 Nguyên nhân gốc: lệch về chiến lược biến động thấp (Accounting EW)

Luật của paper xếp hạng theo Sharpe dự báo, tức chia dự báo cho độ lệch chuẩn. Nếu dự báo gần như không phân biệt được các chiến lược (gần bằng mức trung bình họ-năm), thứ hạng sẽ do 1/sd quyết định: danh mục gồm chiến lược biến động thấp, Sharpe cao nhưng lợi suất thấp. Đo trên `T2`: tương quan giữa điểm xếp hạng và 1/sd là 0,321 (EB 0,112), với abs(t) chỉ 0,234 (EB 0,869); lợi suất thật trung bình của top 1% là 0,339 so với 0,584 của EB.

Tương quan hạng trong họ-năm giữa dự báo và lợi suất thật (median 3 seed, trung bình theo năm, dev / holdout):

| Dự báo | IC dev | IC holdout |
|---|---|---|
| EB | 0,238 | 0,070 |
| Lợi suất trung bình trong mẫu (không mô hình) | 0,236 | 0,067 |
| abs(t) | 0,149 | 0,023 |
| `xs` + `ydm` + `mha` (Sharpe dev cao nhất) | **0,020** | 0,118 |
| `T3` = `T2` + `ydms` | 0,152 | 0,079 |
| `T3` + `hpca16` | 0,210 | 0,105 |
| Ridge trên đặc trưng thống kê F2 (walk-forward) | 0,187 | 0,142 |
| LightGBM trên F2 | 0,203 | 0,076 |

Hai kết luận. (1) Cấu hình có Sharpe dev cao nhất (`xs` + `ydm` + `mha`) gần như không dự báo được lợi suất trong họ-năm (IC dev 0,020); lợi thế Sharpe của nó đến từ luật chia cho sd, không phải từ khả năng dự báo. Vì vậy Sharpe danh mục không đủ để gọi là thắng; phải báo kèm IC và lợi suất thô. (2) Ở dev, thứ hạng của EB gần như trùng với thứ hạng theo lợi suất trung bình trong mẫu, và chưa mô hình nào (kể cả ridge, LightGBM) vượt được nó; ở holdout, các mô hình học đều có IC cao hơn EB (tín hiệu kế toán suy yếu sau 2004 như paper ghi nhận, mô hình dựa vào các đặc trưng khác).

`ydms` (nhãn là lợi suất trừ trung bình họ-năm theo đơn vị Sharpe) sửa một phần lệch này: tương quan với 1/sd còn −0,212, lợi suất thật của top 1% tăng lên 0,480 (top 5% 0,505, EB 0,547). Nhưng `ydms` làm xấu các họ khác (Past Return VW, Accounting VW, Past Return EW).

### 6.7 Hàng đợi 2 (Accounting EW, mốc `T3`)

| Biến thể | Dự đoán so với EB (dev / holdout) | Sharpe dev (top 1/5/10) | Sharpe cả kỳ |
|---|---|---|---|
| `T3` | −7,7% / −6,2% | 1,38 / 1,62 / 1,62 | 1,26 / 1,38 / 1,38 |
| + bỏ chuẩn hóa ở đầu ra (`hnone`) | −2,7% / −11,8% | 1,26 / 1,43 / 1,44 | 1,07 / 1,30 / 1,30 |
| + loss tương quan đơn vị Sharpe λ = 0,2 | −6,6% / −5,4% | 1,16 / 1,41 / 1,50 | 1,04 / 1,25 / 1,30 |
| + loss tương quan λ = 1,0 | −7,7% / −1,4% | 1,24 / 1,36 / 1,52 | 1,16 / 1,24 / 1,35 |
| + `hnone` + tương quan λ = 0,5 | −3,8% / −7,1% | 1,09 / 1,39 / 1,45 | 0,98 / 1,29 / 1,32 |
| + đặc trưng chế độ (`regime`) | −6,9% / −2,7% | 1,38 / 1,41 / 1,46 | 1,25 / 1,24 / 1,27 |
| + PCA 16 chiều trên token FinCast (`hpca16`) | **+0,6%** / −26,1% | 1,23 / 1,50 / 1,60 | 1,08 / 1,31 / 1,36 |
| + khởi động ấm có nhiễu (`warm_sp`) | −6,6% / −29,5% | 1,07 / 1,31 / 1,40 | 0,84 / 1,06 / 1,10 |
| + `setbn` + `xset` 3072 + `detev` | −12,3% / −40,0% | 1,38 / 1,53 / 1,59 | 1,17 / 1,36 / 1,40 |
| EB | | 1,06 / 1,48 / 1,72 | 0,95 / 1,25 / 1,38 |

`hpca16` là lần đầu mô hình ngang EB về MSE ở dev, và có IC cao nhất trong các mạng. Sai số holdout lớn của nó và của `warm_sp`, `setbn` do **dự báo bị phân tán quá mức**: độ lệch chuẩn dự báo 0,36–0,53 so với 0,195 của EB, hệ số hồi quy lợi suất thật lên dự báo trong họ-năm chỉ 0,2–0,6 (EB 0,83 ở dev), và một số seed nổ ở vài năm (ví dụ `warm_sp` seed 0 năm 2013). Phương sai giữa các seed rất lớn (từng seed `hpca16`: −6,6% và −5,7% ở dev, median 3 seed +0,6%), nên bước tiếp theo là trung bình 10 seed (`code/run_queue3.py`) và thử độ rộng của nút cổ chai PCA (`code/run_queue4.py`).

### 6.8 Mười seed và kiểm định (Accounting EW, `code/run_queue3.py`, `NSEED=10 code/stats_sub.py`)

Median 10 seed, vũ trụ ứng viên. DM dương nghĩa là AI có sai số lớn hơn EB.

| Cấu hình | Dự đoán so với EB, dev (DM) | Holdout (DM) | Sharpe dev top 1/5/10 | Lợi suất top 1% dev (%/năm) |
|---|---|---|---|---|
| `xs` + `ydm` + `mha` | −7,9% (3,40) | +12,8% (−2,00) | 1,20 / 1,55 / 1,69 | 5,1 |
| `xs` + `ydm` + `mha` + `hpca16` | −1,0% (0,55) | +3,6% (−1,37) | 1,14 / 1,51 / 1,76 | 5,6 |
| `T3` | −6,0% (2,60) | −1,0% (0,23) | 1,42 / 1,50 / 1,51 | 6,8 |
| `T3` + `hpca16` | +0,7% (−0,26) | −11,2% (2,86) | 0,97 / 1,23 / 1,38 | 7,9 |
| EB | | | 1,06 / 1,48 / 1,72 | 9,4 |

- Tăng từ 3 lên 10 seed giảm rõ sai số của các cấu hình có `hpca16` (với `xs` + `ydm` + `mha` + `hpca16`: dev từ −6,1% lên −1,0%), xác nhận nguyên nhân là phương sai giữa các seed.
- Kiểm định Sharpe (chạy cho `xs` + `ydm` + `mha`, bản `hpca16` của nó, và `T3` + `hpca16`): hai cấu hình `ydm` không khác EB có ý nghĩa ở bất kỳ top-k và giai đoạn nào (JKM p ≥ 0,14); `T3` + `hpca16` thấp hơn EB ở top 10% dev (JKM p = 0,007, bootstrap p = 0,076).
- Cấu hình dùng `ydm` có **lợi suất thô thấp hơn EB có ý nghĩa** (Newey–West p ≤ 0,02 ở dev). `T3` + `hpca16` là cấu hình duy nhất có lợi suất ngang EB (top 10% dev 9,09% so với 9,07%; holdout 4,20% so với 2,79%, p = 0,075), tức là cấu hình thật sự dự báo lợi suất giống EB nhất, nhưng sai số holdout của nó lớn hơn EB có ý nghĩa.

**Mức trung bình của dự báo.** Mức của họ được lấy bằng trung bình nhãn các năm ≤ Y−2, bỏ qua năm Y−1 dù đã biết trước năm Y. Thử lại trên dự báo đã lưu (mức chỉ cộng vào sau mạng nên phép thử là chính xác) với cả năm Y−1, cửa sổ 10 hoặc 20 năm, trung bình mũ: ở dev mọi lựa chọn đều ngang EB (từ −1,0% đến +0,2% với `xs` + `ydm` + `mha` + `hpca16`); lựa chọn tốt nhất theo dev là cửa sổ 20 năm, cho holdout +4,4% (DM −2,23) và cả kỳ +1,8% (DM −1,33, không có ý nghĩa). Ý tưởng này nảy ra **sau khi đã xem sai số holdout**, nên kết quả holdout của nó không được coi là bằng chứng ngoài mẫu.

**Hàng đợi 4 (`T3` + PCA, 5 seed).** Độ rộng PCA 4 / 8 / 16 cho dev ngang EB (+0,7% / −0,6% / −2,5%), 32 chiều kém hơn (−2,9%); holdout của mọi biến thể `T3` kém EB 9–22%. `fx=rank` (−4,0%, DM 2,06), `wide` (−5,7%, DM 2,29) và loss tương quan λ = 1 (−0,2%, holdout −21,7%) không giúp.

**Không pretrained (hàng đợi 5, encoder `raw`, 10 seed).** `xs` + `ydm` + `mha` + `hpca16`: dev −13,1% (DM 2,26), holdout +5,2%, Sharpe cả kỳ 0,90 / 1,20 / 1,35, so với bản FinCast −1,0% / +3,6% / 0,95 / 1,31 / 1,46. `xs` + `ydm` + `mha`: −9,4% (DM 3,13) / +7,1% so với FinCast −7,9% / +12,8%. FinCast (pretrained) có đóng góp vào độ chính xác ở dev khi kết hợp với PCA.

**Kết luận đến lúc này (Accounting EW):** mô hình ngang EB về dự báo và về Sharpe, chưa hơn có ý nghĩa ở thước đo nào. Chưa đạt SOTA. Nguyên nhân chính (mục 6.6): ở dev, thứ hạng theo lợi suất trung bình trong mẫu gần như là tốt nhất có thể với các đặc trưng hiện có, và mọi mô hình học (kể cả tuyến tính) đều không vượt được nó.

### 6.9 Kiểm tra suy luận trên Ticker (`code/run_ticker.py`)

Mô hình học trên các họ thống kê, dự báo cả hai họ Ticker (không học), gộp ứng viên của các họ đó với Ticker, chọn top-k theo Sharpe dự báo. Họ chưa thấy nhận mức trung bình chung của các họ đã học (trước đây nhận mức 0, tức bị loại khỏi top-k một cách giả tạo; đã sửa). EB trên cùng tập tái lập đúng paper: top 1% gồm Accounting EW 92,4%, Past Return EW 7,0%, Ticker 0,0% (paper 91,0 / 8,6 / 0,0).

| Mô hình (median seed) | Học trên | Ticker trong top 1 / 5 / 10% |
|---|---|---|
| Mô hình gốc (3 seed) | Accounting EW | 87,3 / 75,1 / 69,0% |
| `xs` + `ydm` + `mha` + `hpca16` (10 seed) | Accounting EW | 76,2 / 72,7 / 70,2% |
| + F3 (độ phủ tín hiệu của họ) | Accounting EW | 66,8 / 67,2 / 67,1% |
| + F3 + mức theo độ phủ (`lvl=breadth`) | Accounting EW | 83,4 / 83,1 / 82,5% |
| `xs` + `ydm` + `mha` + `hpca16` | 4 họ thống kê | 53,3 / 46,6 / 43,6% |
| + F3 + `lvl=breadth` | 4 họ thống kê | 40,7 / 37,6 / 36,0% |
| EB | | 0,0 / 0,0 / 0,0% |

Tỷ lệ Ticker trong nhóm gộp là 60,7% (gộp với Accounting EW) và 30,3% (gộp với 4 họ). **Mô hình chưa nhận ra Ticker là nhiễu.** Nguyên nhân:

1. Ứng viên Ticker là 10% abs(t) cao nhất của họ nên từng chuỗi trông giống chiến lược tốt; thông tin "cả họ là nhiễu" nằm ở phân phối t của toàn họ (Ticker: 4% chiến lược có abs(t) > 2, gần mức 4,6% của nhiễu; Accounting EW 29%). EB dùng đúng thông tin này. F3 đưa nó vào mạng, nhưng khi học trên một họ, độ phủ chỉ dao động 18–34% nên mạng không ngoại suy được tới 4% (mức theo độ phủ học từ một họ còn làm tệ hơn). Học trên 4 họ (độ phủ 2,7–34%) cải thiện rõ.
2. Ticker EW biến động thấp (sd trung vị 1,2 so với 2,3 của Accounting EW). Với `ydm`, mức dự báo tính bằng lợi suất nên khi chia cho sd để xếp hạng, Ticker EW được đẩy lên (cùng cơ chế lệch biến động thấp ở mục 6.6). Bài tiếp theo dùng `ydms` (mức theo đơn vị Sharpe).

Ở 4 họ, mô hình đã tự học được rằng hai họ VW yếu (gần 0% top-k, giống EB), nhưng Accounting EW chỉ chiếm 32–49% top 1% so với 92% của EB.

### 6.10 Truy vết lại và sửa tận gốc (Accounting EW, hàng đợi 6–7, `code/run_q6.py`, `code/run_q7.py`, `code/diag_acct.py`)

**Truy vết từng khối** (`xs` + `ydm` + `mha` + `hpca16`, các năm dev 1988–2004, IC thăm dò trong họ-năm): đầu vào lợi suất trung bình trong mẫu 0,233, t-stat 0,193; encoder FinCast −0,021, Bi-LSTM −0,012, pooling −0,019, `xs` −0,044, FC 0,052, đầu ra 0,154; EB 0,241. Nhánh chuỗi (FinCast → … → `xs`) không mang thông tin xếp hạng; mạng chỉ dựa vào hai số đầu vào và làm mất một phần tín hiệu của lợi suất trung bình.

**Nguyên nhân gốc** (rà soát code, kiểm chứng trên dự báo đã lưu):
1. BatchNorm ở đầu ra: mỗi batch huấn luyện là một họ-năm và nhãn đã trừ trung bình họ-năm, nên hệ số chặn theo năm không được học; lúc dự báo, thống kê chạy cộng một mức dịch ngẫu nhiên theo năm (độ lệch chuẩn 0,18–0,31). Đây là phần lớn phương sai giữa các seed, và là nguồn của nhiều kết quả "đẹp" trước đó (ví dụ holdout +12,8% của `xs` + `ydm` + `mha` còn +0,5% khi khử). Sửa: `rc` (trừ trung bình đầu ra trong mỗi họ-năm ở huấn luyện, validation và dự báo; mỗi họ-năm một batch, `xset` 4096, dự báo tất định).
2. Hai đầu vào mạnh nhất bị pha loãng trong FC cùng vector 256 chiều. Sửa: `base=lin4` (hồi quy dạng đóng trên [m, m/sd, |t|, sd] trong họ-năm, nhãn các năm Y−15..Y−1, không dùng EB), mạng học phần dư với lớp cuối khởi tạo 0.
3. Thước đo: ở dev 52% sai số bình phương của EB là nhiễu, 19% là mức trung bình năm (22 con số), phần xếp hạng của EB chỉ 3,6%; DM có ý nghĩa cần khoảng +3,5%.

| Cấu hình (trung bình 10 seed) | Dev MSE vs EB (DM) | Phần mức / trong năm | IC dev (EB 0,238) | Sharpe dev top 1/5/10 (EB 1,06/1,48/1,72) |
|---|---|---|---|---|
| A0: + `rc` | −4,1% (2,24) | −1,8 / −2,3 | 0,132 | 1,17 / 1,50 / 1,67 |
| A1: + `base=lin4` | −2,1% (1,76) | −1,8 / −0,3 | 0,214 | 1,27 / 1,82 / 1,95 |
| A2: + mức dùng năm Y−1 | −1,7% | | | 1,30 / 1,70 / 1,92 |
| A3–A5: + `ep0`/`seed2`/`optkeep`, `fx=dm`, bỏ LayerNorm | −2,1% / −1,8% / −1,8% | | | không khác A2 ngoài nhiễu |
| A1 cửa sổ nền 10 / 20 năm | −2,2% / −2,9% | | | kém hơn 15 năm |
| **A2_r10: A1 + mức theo 10 năm gần nhất** | **−1,5% (0,83)** | | | **1,33 / 1,81 / 1,96** |

Độ lệch chuẩn sai số dev giữa các seed giảm từ 3,6 xuống 0,7 điểm sau `rc`.

**Kiểm định A2_r10** (10 seed, vũ trụ ứng viên): Sharpe top 5% dev 1,81 so với 1,48 (JKM p = 0,006, bootstrap p = 0,24), top 10% dev 1,96 so với 1,72 (JKM 0,016, bootstrap 0,21); cả kỳ top 5% JKM 0,001 (bootstrap 0,10), top 10% JKM 0,001 (bootstrap 0,055); holdout không đảo chiều (top 10% 1,12 so với 0,82, JKM 0,019). Lợi suất thô vẫn thấp hơn EB ở dev (top 10% 8,15% so với 9,07%, p = 0,018). **Chưa đạt tiêu chí** (bootstrap chưa có ý nghĩa).

**Tín hiệu chuỗi thời gian** (CPU, cửa sổ 240 tháng trước Y): momentum của chính chiến lược (tháng 2–12) có IC riêng phần 0,092 / 0,094 (dev / holdout) ngoài lợi suất trung bình trong mẫu; lịch sử xa hơn 2 năm không có hình dạng ổn định; mô hình học gộp (ridge, LightGBM) kém công thức cố định vì các năm trước 1980 có cửa sổ ngắn và momentum đổi lợi suất mạnh theo năm. Dự báo zero-shot của Chronos-2 (học chéo trong họ-năm) có IC 0,213, riêng phần 0,106 ngoài lợi suất trung bình (dev). Thí nghiệm đang chạy: hàng đợi 8 (`extras=E2` momentum, `trmin`, encoder không pretrained `lp7`), trích dự báo zero-shot (`code/tsfm_zs/`), hàng đợi 9 (`base=blend` với trọng số cố định chọn trên dev, nhãn đơn vị Sharpe, dự báo zero-shot làm đầu vào phụ — PRETRAINED).

**Nhãn đơn vị Sharpe (B2 = A2 + `prep=ysdm`, 10 seed, `NSEED=10 stats_sub.py`).** Dev MSE −1,0% (DM 1,34); Sharpe dev 1,40 / 1,84 / 1,90 (JKM p 0,007 / 0,004 / 0,030; bootstrap 0,064 / 0,21 / 0,21); lợi suất thô ngang EB (top 10% dev 9,04% so với 9,07%); holdout Sharpe 0,94 / 1,18 / 1,18 so với 0,86 / 0,90 / 0,82; cả kỳ top 10% JKM 0,001, bootstrap 0,027. Nhưng holdout MSE −9,3% (DM 3,84): mức theo đơn vị Sharpe (nhân sd) quá cao sau 2004. Với nhãn đơn vị Sharpe, mức không đổi thứ hạng theo dự báo/sd trong một họ, nên kết quả danh mục không phụ thuộc lựa chọn mức; thử lại mức trên dự báo đã lưu: 10 năm gần nhất dev −1,4% / holdout −4,2% (DM 0,56), 20 năm dev −0,3% / holdout −7,3% (DM 2,80) — ở dev các lựa chọn mức chỉ khác nhau trong nhiễu.

**Quy tắc chọn cho tuyên bố về danh mục** (viết sau khi đã thấy kết quả B2, B3 — kể cả số cả kỳ có chứa holdout — nên KHÔNG phải khai báo trước thật sự với B3; chỉ là khai báo trước với hàng đợi 8–9): trong nhánh nhãn đơn vị Sharpe, chọn cấu hình có trung bình Sharpe dev (top 1/5/10, vũ trụ ứng viên, trung bình 10 seed) cao nhất; tuyên bố chỉ khi JKM và bootstrap theo khối năm cùng p < 0,05 ở dev cho ít nhất một top-k, holdout không đảo chiều Sharpe, lợi suất thô không thấp hơn EB có ý nghĩa; báo kèm tổng số cấu hình đã thử và hiệu chỉnh nhiều phép thử.

**Ứng viên theo quy tắc trên (đến hết hàng đợi 8): B3 = B2 + `ep0`, `seed2`, `optkeep`** (trung bình Sharpe dev cao nhất trong nhánh đơn vị Sharpe: 1,771; B2 1,693). Trung bình 10 seed (`AGG=mean NSEED=10 stats_sub.py`), vũ trụ ứng viên:

| Top-k | Giai đoạn | Sharpe AI | Sharpe EB | p JKM | p bootstrap | Lợi suất AI / EB (%/năm, p) |
|---|---|---|---|---|---|---|
| 1% | dev | 1,459 | 1,056 | **0,002** | **0,031** | 8,65 / 9,37 (0,51) |
| 5% | dev | 1,898 | 1,481 | 0,001 | 0,084 | 9,16 / 9,19 (0,97) |
| 10% | dev | 1,955 | 1,719 | 0,005 | 0,077 | 9,03 / 9,07 (0,92) |
| 1% | holdout | 0,929 | 0,861 | 0,64 | 0,79 | 3,92 / 3,85 |
| 5% | holdout | 1,104 | 0,898 | 0,13 | 0,14 | 3,78 / 3,09 |
| 10% | holdout | 1,133 | 0,816 | **0,015** | **0,048** | 3,75 / 2,79 (0,044) |
| 1 / 5 / 10% | cả kỳ | 1,25 / 1,58 / 1,63 | 0,95 / 1,25 / 1,38 | 0,003 / 0,000 / 0,000 | 0,044 / 0,025 / 0,006 | |

So với mô hình gốc của notebook (10 seed ghép cặp, `paired_sub.py`, từng seed đơn): dev MSE so với EB −19,3% → −1,9% (+17,3 điểm, t 22,7), Sharpe dev top 1 / 5 / 10% từ 1,24 / 1,41 / 1,45 lên 1,52 / 1,87 / 1,94 (+0,28 / +0,46 / +0,48, t 4,5 / 8,0 / 10,4). Dev MSE so với EB −1,4% (DM 1,64, ngang); holdout MSE −9,2% (DM 3,85, kém có ý nghĩa — do mức theo đơn vị Sharpe).

**Kiểm tra phản biện (workflow 3 agent độc lập, thư mục scratchpad/verify):** (1) không rò rỉ: thử nghiệm phá hoại tương lai trên đúng cấu hình B3 cho dự báo 1990–1991 giống hệt từng bit, đối chứng dương phát hiện thay đổi; không cột EB nào vào mô hình; (2) số liệu tái lập độc lập khớp đến 1e-14; (3) nhưng: JKM giả định iid — kiểm định HAC kiểu Ledoit–Wolf cho p = 0,018, bootstrap 20.000 lần p ≈ 0,035 và khoảng tin cậy phân vị 95% [−0,05; 0,68] chứa 0; B3 được chọn trong 55 cấu hình đã huấn luyện trên acct_ew (342 lượt chạy): bootstrap max-statistic theo khối năm cho p ≈ 0,054 một phía (0,15 hai phía) ở top 1%, 0,145 / 0,31 khi tính cả ba mức top-k; holdout ở top 1% không xác nhận (0,93 so với 0,86, p 0,64–0,79), top 10% holdout p = 0,048 với trung bình seed nhưng 0,10 với trung vị seed; chênh Sharpe đến từ biến động thấp hơn trong vài năm EB biến động mạnh (1985, 1987, 1999–2001), không phải lợi suất cao hơn; B3 không hơn B2 (+0,078, p 0,27); chưa có bản đối chứng không pretrained.

**Phát biểu đã hiệu chỉnh:** B3 (FinCast pretrained) ngang EB về lợi suất thô và có Sharpe dev cao hơn ở danh mục top 1% của tập ứng viên (1,46 so với 1,06; p danh nghĩa 0,002–0,035 tùy kiểm định), nhưng sau khi hiệu chỉnh cho quá trình tìm kiếm cấu hình, chứng cứ còn yếu (p ≈ 0,05–0,15 một phía) và holdout không xác nhận ở cùng mức top-k. **Chưa đạt SOTA theo tiêu chí.** Việc tiếp theo: bản không pretrained của B3, seed 10–19 của B3 như một tập riêng, kiểm định chính chuyển sang HAC / bootstrap ≥ 10.000 lần, hiệu chỉnh max-statistic trên toàn bộ cấu hình sau khi chốt.

**Độ bền theo seed của B3 (HAC / bootstrap 10.000 lần, dev, vũ trụ ứng viên):** seed 0–9: top 1% 1,46 (HAC 0,018, bootstrap 0,034), top 5% 1,90 (0,045 / 0,073), top 10% 1,95 (0,053 / 0,067); **seed 10–19 (tập độc lập về seed): top 1% 1,34 (0,090 / 0,111), top 5% 1,89 (0,075 / 0,132), top 10% 1,92 (0,120 / 0,188)**, holdout 0,78 / 0,86 / 0,89; 20 seed gộp: 1,37 / 1,92 / 1,93 (top 1% HAC 0,057, bootstrap 0,078). Kết quả có ý nghĩa của seed 0–9 một phần là may mắn của seed (winner's curse): với 20 seed, **B3 không còn hơn EB có ý nghĩa ở mức 5%**.

**Mốc tầm thường phải vượt: xếp hạng theo abs(t) trong mẫu (không mô hình).** Trên vũ trụ ứng viên acct_ew, chỉ xếp theo abs(t) cho Sharpe dev 1,59 / 1,85 / 1,95 (trung bình 1,797), holdout 0,83 / 0,94 / 0,84 — ngang hoặc hơn B3 (dev 1,46 / 1,90 / 1,96, trung bình 1,771). Xếp theo Sharpe trong mẫu (t/√n, gần với cách EB xếp θ̂/√n) chỉ cho 1,10 / 1,39 / 1,66. Nghĩa là phần lớn lợi thế của B3 so với EB đến từ việc thứ hạng của nó gần với abs(t) (tương quan hạng 0,87), không phải từ thông tin chuỗi thời gian mà mạng học thêm. Riêng nền tuyến tính lin4 theo đơn vị Sharpe (không mạng) cho dev 1,54 / 1,82 / 1,98, top 1% HAC p < 0,001, bootstrap 0,003; mạng không khác nền có ý nghĩa (HAC p 0,30–0,58). Thêm momentum hoặc dự báo zero-shot vào quy tắc cố định làm IC theo Sharpe tăng (0,139 → 0,163) nhưng Sharpe top-k giảm. **Tiêu chuẩn mới cho tuyên bố "mô hình học sâu có giá trị":** phải vượt cả mốc abs(t), không chỉ EB. Kiểm định của chính mốc abs(t) so với EB (vũ trụ ứng viên acct_ew): top 1% dev 1,59 so với 1,06 (HAC p = 0,001, bootstrap 0,005; lợi suất 10,75 so với 9,36 %/năm), holdout 0,83 so với 0,86 (p 0,61), cả kỳ 1,31 so với 0,95 (HAC 0,004, bootstrap 0,010); top 5% / 10% dev p ≈ 0,06–0,09. Tức là luật xếp hạng θ̂/√n của EB (chia cho √n) không tối ưu trên tập ứng viên của một họ; một quy tắc không học đã hơn EB ở top 1% dev, và không đảo chiều ở holdout.

**Dt0 = B3 với nền cố định z(abs(t)) (một hệ số dốc, `base=blend`, `bw=0`), mạng học phần dư** (hàng đợi 11; trung bình Sharpe dev cao nhất trong 67 cấu hình acct_ew: 1,817). Trung bình 10 seed, vũ trụ ứng viên, HAC / bootstrap 10.000 lần:

| Top-k | Dev AI / EB (p HAC / boot) | Holdout AI / EB (p HAC / boot) | Lợi suất dev AI / EB (%/năm) |
|---|---|---|---|
| 1% | 1,58 / 1,06 (0,015 / 0,030) | 0,92 / 0,86 (0,69 / 0,72) | 9,23 / 9,37 |
| 5% | 1,89 / 1,48 (0,026 / 0,031) | 0,94 / 0,90 (0,61 / 0,51) | 9,43 / 9,19 |
| 10% | 1,98 / 1,72 (0,031 / 0,028) | 1,00 / 0,82 (0,036 / 0,068) | 9,23 / 9,07 |

Dev MSE −0,8% (DM 1,22, ngang); holdout MSE −5,6% (DM 2,67, kém có ý nghĩa — phần mức). **Hiệu chỉnh cho việc chọn** (`TARGET=Dt0 verify/maxstat.py`, bootstrap khối năm chung cho 67 cấu hình; lợi suất vượt EB của các cấu hình tương quan trung bình 0,75, số phép thử hiệu dụng ≈ 1,7): top 1%, max-statistic chuẩn hóa một phía p = 0,068 (67 cấu hình), 0,19 (67 × 3 mức top-k); Romano–Wolf 0,067 / 0,19. **So với mốc abs(t):** Sharpe không khác (HAC p 0,55–0,91), lợi suất thô dev thấp hơn (9,2 so với 10,8 %/năm ở top 1%, p = 0,012); holdout top 10% cao hơn (1,00 so với 0,84, p 0,076). Kết luận: Dt0 hơn EB có ý nghĩa danh nghĩa ở cả ba mức top-k trong dev và không đảo chiều ở holdout, nhưng sau hiệu chỉnh cho việc chọn chỉ còn p ≈ 0,07, và không hơn quy tắc abs(t).

**Độ bền của Dt0 theo seed và kiểm định 20 seed.** Seed 10–19 (tập độc lập về seed, không dùng khi chọn): top 1 / 5 / 10% dev 1,68 / 1,87 / 1,99 (HAC 0,003 / 0,018 / 0,021; bootstrap 0,015 / 0,020 / 0,016), holdout 1,04 / 1,00 / 1,04 — **lặp lại được**, khác với B3. Trung bình 20 seed (`AGG=mean NSEED=20 stats_sub.py`, log `_cache/stats_Dt0_n20.log`):

| Top-k | Dev AI / EB (HAC / boot) | Holdout AI / EB (HAC / boot) | Lợi suất dev / holdout AI vs EB (%/năm) |
|---|---|---|---|
| 1% | 1,61 / 1,06 (0,008 / 0,024) | 0,95 / 0,86 (0,53 / 0,55) | 9,30 vs 9,37 / 4,25 vs 3,85 |
| 5% | 1,89 / 1,48 (0,024 / 0,028) | 0,99 / 0,90 (0,28 / 0,19) | 9,31 vs 9,19 / 3,68 vs 3,09 |
| 10% | 2,01 / 1,72 (0,023 / 0,022) | **1,03 / 0,82 (0,005 / 0,010)** | 9,28 vs 9,07 / **3,61 vs 2,79 (p 0,009)** |

So với mô hình gốc (10 seed ghép cặp): MSE dev +17,7 điểm (t 21,9), Sharpe dev +0,32 / +0,47 / +0,54 (t 5,0 / 8,6 / 11,4). So với mốc abs(t) (20 seed): dev Sharpe ngang (HAC 0,29–0,91), lợi suất dev thấp hơn (top 1% 9,3 so với 10,8, p 0,008); **holdout top 10% cao hơn có ý nghĩa** (1,03 so với 0,84, HAC 0,023; lợi suất 3,61 so với 2,92, p 0,033). Tức là phần mạng học thêm trên nền abs(t) không thay đổi kết quả dev nhưng làm danh mục bền hơn sau 2004.

**MSE của Dt0: chỉ thua ở phần mức.** Tách MSE (20 seed): dev −0,92% = mức −0,65 + trong năm −0,26; holdout −5,83% = mức −6,27 + trong năm **+0,44** (trong năm AI tốt hơn EB ở holdout). Thay mức bằng ước lượng tỉ số không dùng EB (`lvl=ratio<w>`): mức Sharpe của họ-năm = c × Sharpe trong mẫu trung bình của các ứng viên, c = Σ Sharpe thực / Σ Sharpe trong mẫu của w năm trước (theo dõi được sự suy giảm sau 2004). Chọn w trên dev trong {tất cả, 10, 20, 30}: dev −1,07 / −1,54 / **+0,40** / −1,35; chọn w = 20 (các số holdout của lưới này đã được in cùng lúc, cần ghi rõ). Dt0 + `ratio20` (20 seed; mức không đổi thứ hạng trong họ-năm nên Sharpe danh mục gần như y hệt): **MSE dev +0,4% (DM −0,39), holdout +0,4% (DM −0,17)** — không còn thua EB ở cả hai giai đoạn, nhưng chưa hơn có ý nghĩa. Thắng MSE có ý nghĩa (cần ~+3,5% ở dev) không đạt được: phần trong năm của EB chỉ là 3,6% MSE và phần mức là 22 con số nhiễu.

**Hàm mất mát (hàng đợi 11, trên B3, trung bình 10 seed; Sharpe dev trung bình top 1/5/10, MSE dev so với EB).** Thiết kế dựa trên hai paper người dùng gửi và phân tích phần dư (workflow `loss-from-papers-and-data`): trong đơn vị Sharpe phần dư gần chuẩn, sát nền nhiễu (nhãn ~97% nhiễu), Huber δ = 1 trên thực tế là MSE. Kết quả: B3 1,771 / −1,4%; MSE 1,721 / −1,2%; Huber δ 0,3 1,694 / −1,0%; δ 0,1 thấp hơn; likelihood Gaussian có phương sai nhiễu theo s24 (`nw` 0,05) 1,767 / **−0,4%** (DM 0,45, MSE tốt nhất của nhánh); Holen L-weighting thay loss bằng kỳ vọng (`fyw=vH`) 1,746; liều đầy đủ `v` 1,754 (holdout giảm mạnh); nghịch đảo `inv` thấp hơn; ghép 4 họ-năm trọng số đều (`lw=4, lwm=eq`) MSE −3,1% và Sharpe giảm; Holen L-weighted đúng nguyên bản (`lw=4`) MSE −3,5%, Sharpe giảm thêm; tương quan toàn dự báo λ 0,5 / 2 (`corru=tot`) 1,7 / −1,1–1,8%; trọng số nhóm đầu `hw=3` thấp hơn trên cả B3 và Dt0. **Không loss nào hơn B3 / Dt0 ngoài nhiễu giữa các seed**; cơ chế L-/R-weighting của Holen et al. không cải thiện bài toán này. Bảng xếp hạng theo Sharpe dev trung bình (đến hết hàng đợi 11): Dt0 1,818 > B3 1,771 > B3_nw05 1,767 > …

### 6.11 Thay từng khối trên Dt0 (hàng đợi 12, 10 seed, Sharpe dev trung bình top 1/5/10; Dt0 = 1,818)

Encoder Chronos-2 ẩn 240 tháng (pretrained) **1,849**; GRU 1,792; bỏ khối chuỗi 1,770; Transformer 1,734; pooling trung bình 1,758 / query 1,753 / attention thường 1,582; HyperAttention 1,730; ISAB 1,723; **bỏ set attention 1,611** (MSE dev −3,9%); upsampling 1,736; mixup 1,742; Chronos-2 ẩn 128 tháng 1,613; không pretrained: chuỗi thô 1,570, token lp7 1,397; R-weighted (thăm dò) 1,601. Khối quan trọng nhất là set attention giữa các chiến lược cùng họ-năm; Bi-LSTM + `mha` là tốt nhất trong các lựa chọn; HyperAttention không hơn attention đầy đủ (họ-năm chỉ ~2.600 chiến lược, attention đầy đủ vẫn rẻ); tăng dữ liệu không giúp; bỏ pretrained làm kém rõ.

### 6.12 Kết quả cuối (Accounting EW, `code/final_acct.py`, log `_cache/final_acct.log`)

**Chọn trên dev (20 seed, trung bình Sharpe dev top 1/5/10):** Dt0 (FinCast) 1,835; Dt0_chr2_240 (Chronos-2 240 tháng) 1,836 → theo quy tắc chọn Dt0_chr2_240, nhưng hai cấu hình **hòa trong nhiễu** (chênh 0,001) nên báo cả hai. Mức `ratio20` (chọn trên dev) gắn cho cả hai.

| | Dt0_chr2_240 + ratio20 | Dt0 + ratio20 | EB |
|---|---|---|---|
| MSE dev / holdout so với EB (DM) | **+1,2% (−1,0)** / −0,3% (0,15) | +0,4% (−0,39) / +0,4% (−0,17) | — |
| IC trong năm dev / holdout | **0,243** / 0,085 | 0,215 / 0,116 | 0,238 / 0,070 |
| Sharpe dev top 1 / 5 / 10% | **1,75** / 1,88 / 1,88 | 1,61 / 1,89 / **2,01** | 1,06 / 1,48 / 1,72 |
| p HAC / bootstrap dev | 0,020 / 0,050; 0,11 / 0,19; 0,31 / 0,44 | 0,008 / 0,024; 0,024 / 0,028; 0,023 / 0,022 | |
| Sharpe holdout top 1 / 5 / 10% | 0,85 / 0,97 / 0,93 | 0,95 / 0,99 / **1,03** (p 0,005 / 0,011) | 0,86 / 0,90 / 0,82 |
| Lợi suất dev top 1% (%/năm) | 10,25 | 9,30 | 9,37 |

So với mô hình gốc (10 seed ghép cặp, Dt0): Sharpe dev +0,32 / +0,47 / +0,54 (t 5,0 / 8,6 / 11,4), MSE +17,7 điểm (t 21,9). So với quy tắc abs(t): Sharpe dev ngang (p ≥ 0,2), lợi suất dev thấp hơn ở top 5–10%; holdout top 10% Dt0 cao hơn (HAC 0,023). Bản không pretrained của cùng cấu hình: Sharpe dev trung bình 1,57 (raw) / 1,40 (lp7).

**Hiệu chỉnh cho việc tìm kiếm cấu hình** (`verify/maxstat.py`, 99 cấu hình acct_ew, bootstrap khối năm chung, các cấu hình tương quan 0,75, số phép thử hiệu dụng ≈ 1,7): (a) giả thuyết "không cấu hình nào hơn EB ở top 1% dev" bị bác bỏ: max-statistic chuẩn hóa p = **0,009** (99 phép thử top 1%), 0,044 (297 phép thử top 1/5/10) — kiểu Reality Check của White; (b) với riêng cấu hình được chọn theo quy tắc: Dt0 p = 0,084, Dt0_chr2_240 p = 0,22 (Romano–Wolf 0,08 / 0,21).

**Kết luận trung thực:** (1) Mô hình học sâu (FinCast/Chronos-2 pretrained → Bi-LSTM → attention `mha` → set attention trong họ-năm → FC, nhãn đơn vị Sharpe, nền abs(t), mức theo tỉ số) **vượt EB về Sharpe danh mục top-k trong dev**, không đảo chiều ở holdout, và với Dt0 vượt cả ở holdout top 10%; giả thuyết "không cấu hình nào vượt EB" bị bác bỏ sau hiệu chỉnh tìm kiếm (p = 0,009), nhưng ý nghĩa của riêng cấu hình được chọn sau hiệu chỉnh chỉ ở mức 0,08–0,22. (2) **MSE ngang hoặc nhỉnh hơn EB** ở cả dev và holdout (+0,4% đến +1,2%), không có ý nghĩa — giới hạn do phần trong năm của EB chỉ 3,6% MSE. (3) Lợi thế danh mục phần lớn trùng với quy tắc abs(t); phần mạng học thêm làm danh mục bền hơn sau 2004 (Dt0) hoặc nâng IC trong năm lên ngang/hơn EB (Dt0_chr2_240). (4) Dùng mô hình pretrained (FinCast, Chronos-2); bản không pretrained kém hơn rõ.

### 6.13 Kết quả cuối với các kỹ thuật người dùng gửi (hàng đợi 14–17, notebook `code/notebook/02_ai_vs_eb_accounting_ew.ipynb`)

Thước đo kết quả theo paper/notebook: danh mục top-k (Table 2), dự báo không chệch (Figure 4–5), IC theo năm; MSE chỉ là loss.

**Thử thêm (10 seed, Sharpe dev trung bình top 1/5/10; Dt0 = 1,818):** HyperAttention 1024/512 1,699; Holen L-weighting (`fyw=vH`) 1,757; HyperAttention + Holen 1,769 (20 seed 1,782); loss + tương quan λ=2 1,664; GRN 1,729; HyperAttention + GRN 1,758; HyperAttention + Holen + GRN 1,751; TiRex 128/240 1,580/1,658; TiRex-2 (2026, cần CUDA toolkit trong user + GCC 14) 128/240 1,792/1,757; Chronos-2 240 + Holen **1,861 (20 seed)**; Chronos-2 240 + HyperAttention + Holen 1,852 (20 seed); Chronos-2 240 + HyperAttention 1,785 (10 seed); Chronos-2 240 + tương quan ≈ Chronos-2 240.

**Chọn trên dev (20 seed):** Dt0_chr2_240_vyH (Chronos-2 240 + Holen) 1,861 — FINAL; Dt0 1,835 — ROBUST (hòa trong nhiễu, báo cả hai). Mức `ratio20`.

| | FINAL | ROBUST | EB | abs(t) |
|---|---|---|---|---|
| Sharpe dev top 1/5/10 | 1,78 / 1,88 / 1,92 | 1,61 / 1,89 / 2,01 | 1,06 / 1,48 / 1,72 | 1,59 / 1,85 / 1,95 |
| p HAC/boot dev (AI vs EB) | 0,028/0,056; 0,15/0,24; 0,23/0,33 | 0,008/0,024; 0,024/0,028; 0,023/0,022 | | |
| Sharpe holdout top 1/5/10 | 0,85 / 0,87 / 0,87 | 0,95 / 0,99 / 1,03 (top 10%: p 0,005/0,010) | 0,86 / 0,90 / 0,82 | 0,83 / 0,94 / 0,84 |
| Độ dốc không chệch dev / holdout | 1,26 / 0,33 | 1,08 / 0,42 | 1,11 / 0,28 | |
| IC dev / holdout | 0,242 / 0,089 | 0,215 / **0,118 (p 0,016)** | 0,238 / 0,070 | |
| Ticker trong top 1/5/10% (chỉ học kế toán) | **0,6 / 1,6 / 2,6%** | 1,4 / 2,6 / 3,9% | 0 / 0 / 0% | |

So với mô hình gốc: Ticker 87 / 75 / 69%, Sharpe dev 1,24 / 1,41 / 1,49, IC 0,07. Hiệu chỉnh 115 cấu hình (`verify/maxstat.py`): "không cấu hình nào vượt EB" p = 0,010 (top 1%) / 0,047 (3 mức); riêng ROBUST 0,089, FINAL 0,20.

### 6.14 Toàn bộ dữ liệu Accounting EW (2026-10-05/06; `code/full_cfgs.py`, `code/run_full.py`, notebook `code/notebook/03_full_data_ai_vs_eb_accounting_ew.ipynb`, cách làm `CACH_LAM_FULL_DATA.md`)

Học trên cả 1.189.707 dòng (~26.000 chiến lược/năm), không dùng tập ứng viên 10%; học đến Y−2, validation Y−1, học lại mỗi năm; chọn chỉ bằng dev 1983–2004. Sharpe trung bình top 1/5/10% (dev / holdout), EB: 1,77 / 0,82.

| Cấu hình | Dev | Holdout | Ghi chú |
|---|---|---|---|
| `goc` (baseline kiểu notebook B) | 1,24 | 0,56 | thua rõ (p ≤ 0,005) |
| `sh_isab_nobase` (nhãn gián tiếp, ISAB, chưa nền) | 1,84 | 0,81 | nhãn gián tiếp là yếu tố chính (`dir_isab` 1,37 / 0,69 → 1,84) |
| `Dt0_isab` (+ ISAB) | 1,87 | 0,92 | |
| `Dt0_hyp` (HyperAttention) | 1,81 | 0,88 | ngang attention đầy đủ |
| **`I_vyH` (ISAB + Holen `fyw=vH`) — FINAL** | **1,92** | **0,90** | 6 seed |
| `I_lweq4` (đối chứng gộp gradient đều, 4 seed) | 1,88 | 0,95 | lợi ích Holen có thể do gộp K năm/bước |
| encoder Chronos-2, EXAONE, `lp7`, `raw` | không hơn FinCast | | `lp7` (không pretrained) IC dev cao nhất 0,192, Sharpe thấp hơn |
| upsampling/mixup, đặc trưng tài chính, momentum | không giúp | | |
| không validation (`nval=0`) | 1,43 | 0,57 | validation Y−1 là cần thiết |
| đóng băng sau 2004 | | 0,77 | học lại mỗi năm giúp holdout (0,92) |

Kết luận: FINAL hơn EB ở top 1% (dev HAC 0,037, bootstrap 0,059; cả kỳ 0,022 / 0,034), ngang ở top 5/10%, IC dev thấp hơn (p 0,012), ngang quy tắc |t| ở dev. Chưa hiệu chỉnh nhiều cấu hình (~40 cấu hình), chưa kiểm tra Ticker.

## 7. Bài học từ hướng cũ (huấn luyện trên cả 6 họ, gồm Ticker)

82 lần chạy (58 cấu hình), chi tiết trong `code/_cache/base_results.tsv`.

- Khối có tác dụng: `xs` (sai số dự báo từ kém EB 10,9% xuống khoảng 4%), `mha` cùng với `ydm` (bỏ `mha` làm sai số xấu đi 4 điểm, t = −8 theo cặp 5 seed), và lấy trung bình nhiều seed (từng mạng kém EB 19% đến 32% ở holdout, trung bình 5 seed khoảng 0%).
- Không giúp hoặc làm xấu: đổi sequence, pooling `query`, các loss `mse`, `mae`, `wmse`, `studentt`, tăng dữ liệu gấp đôi bằng nhiễu hoặc mixup, chia nhãn cho σ, ISAB, hiệu chỉnh bằng một năm validation, ghép nhiều cải tiến cùng lúc. Chronos-2 ngang FinCast theo cặp seed.
- Chẩn đoán: sau 2004 chính EB cũng không còn xếp hạng đúng trong nhóm ứng viên (tương quan hạng −0,045), nên holdout chủ yếu thưởng cho dự báo gần 0. Hiệu chỉnh độ dốc theo các năm trước không đổi khoảng cách với EB, nên nút thắt là thông tin để xếp hạng chứ không phải biên độ. Mô hình thua EB nặng nhất ở các họ Ticker, là lý do hướng mới bỏ Ticker khỏi tập huấn luyện.
- Tốt nhất ở hướng cũ (`xs` + `ydm` + `mha`, 5 seed): ngang EB về dự báo; Sharpe top 5% và 10% cao hơn EB trên cả kỳ (p khoảng 0,001) nhưng chưa có ý nghĩa riêng ở holdout; chưa phải SOTA.

## 8. Paper đã dùng (xác minh qua MCP hoặc đọc toàn văn)

- Chen và Dim, High-Throughput Asset Pricing, arXiv 2311.10685 (paper gốc).
- Han và cộng sự, HyperAttention: Long-context Attention in Near-Linear Time, ICLR 2024 (file trong `docs/`).
- Holen và cộng sự, Loss- and Reward-Weighting for Efficient Distributed Reinforcement Learning, arXiv 2304.12778 (file trong `docs/`).
- Li và cộng sự, MASTER, arXiv 2312.15235; Guo, arXiv 2609.08106 (attention giữa các cổ phiếu).
- Auer và cộng sự, TiRex, arXiv 2505.23719; Noguer i Alonso và Franklin, arXiv 2606.27100 (mô hình nền chuỗi thời gian cho lợi suất).
