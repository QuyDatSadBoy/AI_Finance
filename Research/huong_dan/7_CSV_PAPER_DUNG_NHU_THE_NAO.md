# Luồng repo: nhìn input → output bằng số cụ thể

**Ví dụ:** cuối 2019 dự báo năm 2020 cho hai chiến lược **kế toán EW**, ID **66 và 67**. Lợi suất quá khứ/tương lai lấy từ CSV thật. **Chỉ hai giá trị thetahat của EB được đặt giả**, để thấy hết luồng. Các bước sau tính theo công thức code gốc. Bảng làm tròn 6 chữ số; CSV giữ số đầy đủ.

**Đọc từ trên xuống là đủ:** tất cả số cần hiểu ví dụ đã có trong các bảng ở đây. Link CSV/code chỉ để đối chiếu thêm. Một **signal** là quy tắc xếp hạng cổ phiếu; từ quy tắc đó tạo một **chiến lược** mua một nhóm và bán khống nhóm khác. ret là lợi suất **nhóm mua trừ nhóm bán**. EW chia trọng số đều giữa các mã, VW chia theo vốn hóa.

| Đứng ở cuối 2019 | Dữ liệu nào? | Được dùng làm gì? |
|---|---|---|
| **Đã biết** | Tháng 2000–2019: **IS** (in-sample) | Tính thành tích quá khứ và tạo dự báo. |
| **Chưa biết** | Tháng 2020: **OOS** (out-of-sample) | Chỉ dùng sau để kiểm tra dự báo. |

## 0. Repo đọc file nào?

**Câu hỏi chính của paper:** nhìn quá khứ của rất nhiều chiến lược, ta có thể dự báo mức lời/lỗ hợp lý và tìm được chiến lược có hiệu quả ở năm sau không? Hiệu quả đó nằm ở dữ liệu kế toán, lợi suất quá khứ hay chữ ticker?

**Target** là lợi suất trung bình tương lai của **chiến lược**. Output gồm dự báo từng chiến lược, danh sách được chọn và lợi suất thật của danh mục sau đó. Điều quan trọng là **chọn trước khi xem đáp án tương lai**.

| File tên/quy tắc: dùng để giải thích ID | File lợi suất: dùng cho thống kê và EB |
|---|---|
| DataMinedSignalList.csv | DataMinedLongShortReturnsEW.csv; DataMinedLongShortReturnsVW.csv |
| PastReturnSignalNames.csv | PastReturnSignalsLongShort.csv |
| TickerSignalNames.csv | TickerSignalsLongShort.csv |

Nối theo signalid **trong cùng họ**. Past-return/ticker chứa hai cột ret_ew/ret_vw, code tách thành hai nhóm. Kế toán đã có hai file riêng. Tổng cộng **6 nhóm** được ước lượng EB riêng theo từng năm. Ví dụ dưới chỉ theo nhóm kế toán EW.

| File kết quả | Format cột gốc | Một dòng thật, đọc bằng lời |
|---|---|---|
| Kế toán EW/VW | signalid,year,month,ret,nstock | ID 1, 07/1990: EW **+1,542175%**, VW **+5,726635%**, nstock=3160 = **tổng số mã trong nhóm mua và nhóm bán**. Không biết riêng mỗi nhóm. |
| Past-return | signalid,date,ret_ew,ret_vw,nshort,nlong | ID 0, 01/1962: EW **−1,387352%**, VW **+0,630737%**; 92 mã bán, 92 mã mua. |
| Ticker | signalid,date,ret_ew,ret_vw,nlong,nshort | ID 0, 01/1963: EW **−1,007404%**, VW **−0,141816%**; 196 mã mua, 196 mã bán. |

Ba file tên tương ứng cho biết ID 1 kế toán = diff(acchg)/lag(aco); ID 0 past-return = ret_1; ID 0 ticker = L1_lng_1_2_sht_3_4. Chúng là bảng tra quy tắc, không phải lợi suất hoặc target.

nstock không phải số mã ban đầu được xếp hạng. Code tạo dữ liệu bản cũ cộng `nstock_long + nstock_short`; [lịch sử code tác giả xác nhận](https://github.com/chenandrewy/flex-mining/commit/8e49bc51a1b3f3a7937633bf813097dc3d5c44df).

## 1. CSV gốc → chuỗi tháng để xử lý

[get_rolling_signal_stats.py](../code_goc_paper/high-throughput-ap/get_rolling_signal_stats.py) lấy ID, tháng và ret từ [CSV kế toán EW gốc](../../data_raw/DataMinedLongShortReturnsEW.csv). Nó bỏ nstock khỏi dữ liệu dùng để tính thống kê. [Bảng tên gốc](../../data_raw/DataMinedSignalList.csv) ghi ID 66 là diff(aco)/lag(aco).

| signalid | year | month | ret | nstock |
|---:|---:|---:|---:|---:|
| 66 | 2019 | 12 | 2,047543 | 608 |
| 66 | 2020 | 1 | −6,737422 | 573 |

Output trong biến signal_dic, [xem CSV xuất để minh họa](../output_minh_hoa/MINH_HOA_signal_dic.csv):

| signalid | date | year | ret |
|---:|---|---:|---:|
| 66 | 2019-12-01 | 2019 | 2,047543 |
| 66 | 2020-01-01 | 2020 | −6,737422 |

**ret=2,047543 nghĩa là +2,047543% trong tháng đó.** Để dự báo 2020, chỉ dùng các tháng 2000–2019. Dòng năm 2020 được giữ để chấm sau.

## 2. Chuỗi tháng → thống kê quá khứ và tương lai

get_rolling_signal_stats.py tính trung bình và t-stat theo từng signal. **IS** = quá khứ để dự báo; **OOS** = năm sau để kiểm tra.

Output chuẩn của script là OOS_signal_tstat_OosNyears1.csv.gzip. [Bản CSV dễ mở của ví dụ](../output_minh_hoa/OOS_signal_tstat_OosNyears1.csv):

| signal_family | oos_begin_year | signalid | mean_ret | tstat | nobs | s_flag | is_begin_year |
|---|---:|---:|---:|---:|---:|---|---:|
| DataMinedLongShortReturnsEW | 2020 | 66 | −0,549202 | −3,761010 | 240 | IS | 2000 |
| DataMinedLongShortReturnsEW | 2020 | 66 | −1,169952 | −1,087265 | 12 | OOS | 2000 |
| DataMinedLongShortReturnsEW | 2020 | 67 | −0,431650 | −2,915257 | 240 | IS | 2000 |
| DataMinedLongShortReturnsEW | 2020 | 67 | −0,503880 | −0,524270 | 12 | OOS | 2000 |

**Tất cả số trên tính từ dữ liệu thật**, dùng sai số chuẩn HAC lag 0 như cấu hình repo. Đọc dòng đầu: ID 66 có 240 tháng quá khứ, trung bình −0,549202%/tháng. Hai dòng OOS **không đi vào EB**.

Đọc tên cột: signal_family = nhóm chiến lược; oos_begin_year = năm cần dự báo; mean_ret = lợi suất tháng trung bình; nobs = **số tháng**, không phải số cổ phiếu; s_flag = IS hoặc OOS; is_begin_year = năm bắt đầu cửa sổ quá khứ. tstat đo độ nổi bật của trung bình so với dao động: **−3,761010 là một điểm thống kê, không phải −3,761010%**.

### tstat ở bước 2 tính từ đâu?

**Đúng: mỗi tstat ở đây thuộc về một signal ID, trong một nhóm EW/VW và một khoảng thời gian cụ thể.** Code lấy nhiều tháng của **chính ID đó** để tính một tstat. Không gộp ret của ID 66 với ID 67. Đến bước EB mới dùng tstat của nhiều ID trong cùng nhóm.

### Tự tính từ đầu với 4 tháng cho dễ hiểu

**Bảng sau là số giả lập để học cách tính, không phải dữ liệu thật của ID 66.** Giả sử một ID có lợi suất như sau; các số ret dùng đơn vị %, nên ret=1 nghĩa là lời 1% trong tháng.

| Tháng | ret | Lệch so với trung bình 2 | Bình phương độ lệch |
|---|---:|---:|---:|
| 1 | 1 | 1 − 2 = −1 | (−1)² = 1 |
| 2 | 3 | 3 − 2 = +1 | (+1)² = 1 |
| 3 | 1 | 1 − 2 = −1 | (−1)² = 1 |
| 4 | 3 | 3 − 2 = +1 | (+1)² = 1 |

**1. Tính trung bình:** cộng lợi suất rồi chia số tháng: **(1 + 3 + 1 + 3) / 4 = 2%/tháng**. Đây là mean_ret.

**2. Tính độ dao động:** mỗi tháng lệch khỏi trung bình bao nhiêu? Bảng trên cho thấy tháng nào cũng lệch 1 điểm phần trăm. Bình phương để độ lệch âm và dương không triệt tiêu nhau; lấy trung bình rồi căn bậc hai:

**Độ lệch chuẩn = √[(1 + 1 + 1 + 1) / 4] = √1 = 1 điểm phần trăm.** Căn bậc hai đưa kết quả trở lại đơn vị ban đầu. Đây là cách dùng mẫu số N để khớp cấu hình repo này.

**3. Tính sai số chuẩn:** đây là mức không chắc chắn của **trung bình**, khác với độ dao động của **từng tháng**. Theo phép tính trong repo, lấy độ lệch chuẩn chia căn bậc hai số tháng:

**Sai số chuẩn = 1 / √4 = 1 / 2 = 0,5 điểm phần trăm.** Khi độ dao động giữ nguyên, có nhiều tháng hơn thì sai số chuẩn nhỏ hơn.

**4. Tính tstat:** lấy trung bình chia sai số chuẩn:

**tstat = 2 / 0,5 = 4.** Nghĩa là trung bình +2% cách mức 0 bằng 4 lần sai số chuẩn. Số 4 không có đơn vị %, cũng không phải xác suất chiến lược sẽ thắng.

**Toàn bộ luồng:** 4 số ret của một ID → trung bình 2 → độ lệch chuẩn 1 → sai số chuẩn 0,5 → tstat 4. ID khác có 4 số riêng thì tính riêng theo đúng các bước đó.

### Áp dụng đúng các bước trên vào dữ liệu thật ID 66

**Nó chỉ được tính từ các số ret từng tháng trong CSV**, không từ nstock và không từ dự báo EB. ID 66 có 240 số ret quá khứ; dưới đây là ba số cuối, không phải toàn bộ chuỗi:

| Tháng quá khứ của ID 66 EW | ret thật, %/tháng |
|---|---:|
| 10/2019 | −1,894562 |
| 11/2019 | +0,723975 |
| 12/2019 | +2,047543 |

**Từ cả 240 số**, code tính lần lượt:

| Phép tính | Số thật của ID 66 | Hiểu bằng lời |
|---|---:|---|
| Cộng 240 số ret | −131,808470 | Tổng các lợi suất tháng để tính trung bình; không phải lợi suất ghép lãi. |
| Chia tổng cho 240 | **−0,549202** | Trung bình mỗi tháng lỗ khoảng 0,55%. Đây là mean_ret. |
| Đo độ dao động của 240 số | **2,262212** | Các tháng lên xuống quanh trung bình bao nhiêu. Đây là độ lệch chuẩn trong phép tính HAC lag 0 của ví dụ. |
| Độ dao động / √240 | **0,146025** | Mức không chắc chắn khi ước lượng trung bình. Gọi là sai số chuẩn. |
| Trung bình / sai số chuẩn | **−3,761010** | Đây là tstat. Trung bình nằm dưới 0 khoảng 3,76 lần sai số chuẩn. |

**Công thức ngắn:** `tstat = mean_ret / sai số chuẩn`. Trong cấu hình repo này, `sai số chuẩn = độ lệch chuẩn / √số tháng`. Cụ thể: **−0,549202 / 0,146025 ≈ −3,761010**. Để khớp HAC lag 0 của code, độ lệch chuẩn ở phép tính này dùng mẫu số 240 khi tính phương sai; đổi sang độ lệch chuẩn mẫu với mẫu số 239 sẽ cho t-stat hơi khác.

**Ví dụ dễ hiểu hơn, dùng số minh họa:** hai chiến lược đều trung bình lời 1%/tháng trong 100 tháng:

| | A: ít dao động | B: dao động mạnh |
|---|---:|---:|
| Trung bình | +1% | +1% |
| Độ dao động tháng | 2% | 10% |
| Sai số chuẩn: độ dao động / √100 | 0,2 | 1 |
| tstat: 1 / sai số chuẩn | **5** | **1** |

A có mức trung bình dương rõ hơn; B lên xuống mạnh nên trung bình +1% dễ xuất hiện do nhiễu hơn. **Dấu của tstat** cho biết trung bình âm/dương; **độ lớn** đo mức rõ của trung bình so với sai số ước lượng. Nó không phải % lời, không phải xác suất thắng và không bảo đảm tương lai.

**Vì sao tstat cao vẫn chưa đủ?** Khi thử hàng chục nghìn chiến lược, vài cái có tstat rất cao chỉ do may mắn. Vì vậy paper còn dùng EB và kiểm tra bằng năm sau.

**Vì sao một ID có hai dòng tstat trong bảng output?** ID 66 có dòng IS dùng **240 tháng 2000–2019**, nên tstat=−3,761010; dòng OOS dùng **12 tháng 2020**, nên tstat=−1,087265. Cùng ID nhưng khác các tháng đầu vào nên kết quả khác nhau. Chỉ dòng IS được dùng để tạo dự báo 2020; dòng OOS để kiểm tra sau. Đổi từ EW sang VW hoặc đổi cửa sổ năm cũng phải tính lại.

## 3. Thống kê IS → dự báo EB

### EB “nhìn cả nhóm” cụ thể là làm gì?

Hãy hình dung bước 2 tạo **một tstat cho mỗi chiến lược**. EB gom các tstat quá khứ của **cùng nhóm, cùng năm dự báo** — ví dụ tất cả kế toán EW cho năm 2020 — rồi xem các điểm đó phân bố như thế nào. Bước này không chỉ nhìn ID 66, cũng không lấy lợi suất trung bình của cả nhóm để gán cho mọi ID.

| EB thấy ở cả nhóm — minh họa bằng lời | EB suy xét điều gì? | Tác động có thể có |
|---|---|---|
| Phần lớn tstat gần 0; một số ít rất cao/thấp | Những điểm nổi bật có thể được giải thích phần lớn bằng nhiễu thống kê. | Điều chỉnh dự báo về gần 0 mạnh hơn. |
| Có nhiều tstat lớn hơn mức nhiễu dự kiến có thể giải thích | Có dấu hiệu cả nhóm chứa hiệu quả thật. | Có thể giữ lại nhiều hơn thành tích của từng chiến lược. |

**Cách code thực hiện:** giả thiết `tstat quan sát = hiệu quả thật đã chuẩn hóa + nhiễu`. Nó ước lượng phân phối hiệu quả thật bằng hỗn hợp hai phân phối chuẩn, giả thiết nhiễu có phân phối chuẩn với độ lệch chuẩn 1. Từ phân phối đã học và tstat riêng của ID 66, code tính **thetahat** — ước lượng phần hiệu quả thật — rồi chuyển về **pred_ret**, lợi suất dự báo. Đây là mô hình thống kê dựa trên giả thiết, không phải máy biết chắc phần nào là may mắn.

Đọc giản dị: **tstat là điểm quá khứ → thetahat là điểm sau điều chỉnh bằng cả nhóm → pred_ret là dự báo %/tháng**. Không phải mọi chiến lược đều bị nhân cùng một hệ số; mức điều chỉnh phụ thuộc tstat và phân phối đã học của nhóm.

[estimate_EB_distr.r](../code_goc_paper/high-throughput-ap/estimate_EB_distr.r) đọc **chỉ IS**, học phân phối t-stat của cả nhóm kế toán EW, rồi tạo thetahat và pred_ret. Ở ví dụ nhỏ, **đặt giả thetahat = −2,5 cho ID 66 và −1,5 cho ID 67**, không chạy ước lượng EB.

Output chuẩn là PREFIX_Predict_SignalYear.csv.gzip. [CSV minh họa cùng tên cột](../output_minh_hoa/MINH_HOA_Predict_SignalYear.csv):

| signal_family | oos_begin_year | signalid | mean_ret | tstat | nobs | s_flag | is_begin_year | thetahat **giả** | pred_ret |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| DataMinedLongShortReturnsEW | 2020 | 66 | −0,549202 | −3,761010 | 240 | IS | 2000 | −2,5 | −0,365063 |
| DataMinedLongShortReturnsEW | 2020 | 67 | −0,431650 | −2,915257 | 240 | IS | 2000 | −1,5 | −0,222099 |

**Phép tính đúng trong repo:** pred_ret = thetahat × mean_ret / tstat. Với ID 66: −2,5 × (−0,549202) / (−3,761010) ≈ **−0,365063%/tháng dự báo**. Đây là số giả lập có tính toán, không phải dự báo thật đã chạy của paper.

thetahat là điểm hiệu quả đã điều chỉnh, **không có đơn vị %**. pred_ret mới là lợi suất dự báo. Vì thế output bước này trả lời: “ID 66 dự kiến trung bình **−0,365063%/tháng** trong 2020”, thay vì lấy nguyên kết quả quá khứ **−0,549202%/tháng**.

Script R còn xuất PREFIX_QML_FamilyYear.csv.gzip chứa tham số phân phối theo nhóm–năm. Ví dụ này đặt thẳng thetahat nên **không có bước fit QML và không tạo output QML**.

## 4. Dự báo → đảo chiều, xếp hạng và ghép kết quả thật

[create_plots_from_est.py](../code_goc_paper/high-throughput-ap/create_plots_from_est.py) ghép dự báo và CSV tháng theo **signal_family, signalid, oos_begin_year**. Nó đổi tên nhóm thành acct_ew. Nếu pred_ret âm, code nhân pred_ret, thetahat và lợi suất thật ret_oos với **−1**. Trong bảng dưới, pred_ret, thetahat và ret_oos **đã được đảo dấu**; tên cột giữ nguyên như code.

Các cột cần đọc của df_signed đã nằm trong bảng dưới; [CSV đầy đủ để đối chiếu](../output_minh_hoa/MINH_HOA_df_signed.csv):

| signal_family | signalid | date | sign | pred_ret | thetahat | SRahat | rank_pct | ret_oos |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| acct_ew | 66 | 2020-01-01 | −1 | 0,365063 | 2,5 | 0,559017 | 100 | 6,737422 |
| acct_ew | 67 | 2020-01-01 | −1 | 0,222099 | 1,5 | 0,335410 | 50 | 2,102018 |

**SRahat = thetahat sau đảo / √nobs × √12.** ID 66 có điểm dự báo cao hơn nên đứng đầu. Hai signal được xếp hạng 100 và 50 theo cách rank(pct=True) của repo. Với điều kiện top 1%: rank_pct ≥ 99, ví dụ này chọn ID 66. Đây là top trong **ví dụ hai signal**, không phải danh sách top thật của toàn bộ paper.

Đọc ID 66: sign=−1 nghĩa là **đổi nhóm mua thành bán và nhóm bán thành mua**. Dự báo **−0,365063%** đổi thành **+0,365063%**; kết quả thật tháng 1 **−6,737422%** đổi thành **+6,737422%**. SRahat=0,559017 là điểm Sharpe dự báo, không phải 0,559017% lợi suất. rank_pct=100 chỉ nghĩa là đứng đầu trong ví dụ.

## 5. Chiến lược được chọn → lợi suất danh mục mỗi tháng

Code lấy trung bình ret_oos của các chiến lược được chọn trong từng tháng. Ví dụ chỉ chọn ID 66 nên lợi suất danh mục bằng lợi suất ID 66 sau đảo chiều.

Output trong biến df_best_ret, [CSV minh họa](../output_minh_hoa/MINH_HOA_df_best_ret.csv):

| date | ret_oos | nstrat | pctmin | sort_var |
|---|---:|---:|---:|---|
| 2020-01-01 | 6,737422 | 1 | 1 | SRahat |
| 2020-02-01 | 1,635572 | 1 | 1 | SRahat |
| 2020-03-01 | 4,306392 | 1 | 1 | SRahat |
| 2020-04-01 | −8,030583 | 1 | 1 | SRahat |
| 2020-05-01 | 3,118492 | 1 | 1 | SRahat |
| 2020-06-01 | 1,783722 | 1 | 1 | SRahat |
| 2020-07-01 | 1,382905 | 1 | 1 | SRahat |
| 2020-08-01 | −0,885959 | 1 | 1 | SRahat |
| 2020-09-01 | −2,468820 | 1 | 1 | SRahat |
| 2020-10-01 | 1,017614 | 1 | 1 | SRahat |
| 2020-11-01 | 5,414981 | 1 | 1 | SRahat |
| 2020-12-01 | 0,027682 | 1 | 1 | SRahat |

nstrat=1 là **một chiến lược**, không phải một cổ phiếu. pctmin=1 là nhóm top 1%; sort_var=SRahat nghĩa là chọn theo điểm Sharpe dự báo. CSV còn có các nhóm top 5%, 10% và nhánh chọn theo t-stat quá khứ, như vòng lặp của repo.

## 6. 12 tháng → bảng kết quả cuối

Code tổng hợp danh mục: lợi suất bình quân tháng × 12; t-stat và Sharpe từ chuỗi lợi suất tháng. Bảng tabdat trước khi đổi tên cột để xuất LaTeX, [CSV minh họa](../output_minh_hoa/MINH_HOA_tabdat.csv). name là nhãn dòng (index) được đưa thành cột khi xuất CSV để dễ xem:

| name | nstrat_mean | ret_mean | tstat | sr |
|---|---:|---:|---:|---:|
| EB Mining Top 1% | 1 | 14,039420 | 1,040977 | 1,040977 |

**Đọc bằng lời:** danh mục minh họa có một chiến lược, năm 2020 trung bình +1,169952%/tháng; nhân 12 thành **14,039420%/năm**. Đây là kết quả tính từ lợi suất thật **với lựa chọn dựa trên thetahat giả**, không phải kết quả paper và không phải lợi suất ghép lãi.

Các cột cuối: nstrat_mean = số chiến lược trung bình; ret_mean = lợi suất năm quy đổi; tstat = điểm thống kê của danh mục; sr = Sharpe năm. Với 12 số ở bước 5, code lấy trung bình **1,169952**, độ lệch chuẩn mẫu khoảng **3,893295**; tstat = trung bình / độ lệch chuẩn × √12 ≈ **1,040977**. Sharpe năm cũng ≈1,040977 trong ví dụ đúng 12 tháng này.

| So sánh cuối cùng cho danh mục minh họa | %/tháng | %/năm, nhân 12 |
|---|---:|---:|
| **Dự báo**, sau đảo chiều | +0,365063 | +4,380754 |
| **Thực tế**, sau đảo chiều | +1,169952 | +14,039420 |

**Đây là output bạn cần hình dung:** trước năm 2020, chọn ID 66 đảo chiều và kỳ vọng +0,365063%/tháng; sau năm 2020, kiểm tra thấy thực tế +1,169952%/tháng. Hai con số khác nhau vì một bên là dự báo, một bên là kết quả xảy ra.

Trong **paper thật**, làm trên toàn bộ chiến lược và nhiều năm, Bảng 2 báo cáo top 1% EB: trung bình **1.278 chiến lược**, lợi suất **5,70%/năm trong 1983–2020**.

Các CSV signal_dic, df_signed, df_best_ret, tabdat ở đây là **bản xuất thêm để xem các bảng trong bộ nhớ**. Repo không tự lưu chúng với những tên minh họa này. Hai file thống kê/dự báo giữ đúng schema xuất của repo, dùng đuôi CSV thường để bạn mở dễ.

**Idea:** quá khứ → dự báo → chọn chiến lược → kiểm tra bằng tương lai. Dữ liệu [data_bo_sung/](../../data_bo_sung/README.md) chỉ dùng thêm cho phần **so sánh chiến lược đã công bố**, không đưa vào bước EB chính.

Muốn tạo lại đúng ví dụ này, chạy python Research/output_minh_hoa/tao_output_minh_hoa.py ở thư mục repo.

## 7. Ngoài chọn chiến lược, paper muốn kiểm tra những gì?

Ví dụ ID 66 chỉ cho thấy **cách dữ liệu đi qua code**. Những kết luận dưới đây lấy từ paper thật, không lấy từ dự báo giả lập của ví dụ. Bạn không cần mở paper để hiểu bảng; tên bảng/hình cho biết chỗ đối chiếu.

| Câu hỏi của paper | Làm ở bước nào? | Bằng chứng thật và ý nghĩa |
|---|---|---|
| Dùng quá khứ để chọn có tìm được lợi suất ở tương lai không? | Bước 4–6: chọn trước, giữ trong năm sau, lặp qua các năm. | **Bảng 2:** top 1% EB đạt **5,70%/năm**, Sharpe **1,46** trong 1983–2020. Đây là kết quả ngoài mẫu. |
| Có phải chỉ vài lựa chọn nổi bật tạo ra kết quả? | Bước 4–6: thử danh mục rộng hơn. | **Bảng 2:** top 5% EB đạt **4,03%/năm**, top 10% đạt **2,77%/năm**. Kết quả không chỉ nằm ở top 1%. |
| EB có dự báo được mức lợi suất hợp lý không? | Bước 3 tạo dự báo; thêm phân tích sau bước 5 để so dự báo với đáp án. | **Hình 4:** mỗi họ chia thành 20 nhóm theo lợi suất quá khứ; tổng **6 × 20 = 120 nhóm**. Với 1983–2004, dự báo nhìn chung sát lợi suất thật; ví dụ nhóm kế toán EW ở đầu âm có dự báo và thực tế đều khoảng **−8%/năm**. Đây là nhóm trong hình, không phải ID 66. |
| Dữ liệu nào có thông tin dự báo? | Bước 2–3 xem phân phối tstat; sau bước 4–6 so sánh giữa các họ. | **Hình 2–3** so tstat với mô hình chỉ có nhiễu. **Bảng 3:** top 1% gồm **91,0% kế toán EW**, **8,6% past-return EW**, **0,3% kế toán VW**, **0,1% past-return VW**, **0% ticker**. **Hình 4–5** kiểm tra lợi suất tương lai theo họ. |
| Điều chỉnh EB có nhất thiết kiếm lời hơn chọn theo quá khứ không? | Bước 4–6 chạy thêm nhánh naive dùng độ lớn tstat quá khứ. | **Bảng 2:** top 1% naive đạt **5,72%/năm**, gần EB **5,70%**. Paper không chứng minh EB luôn thắng; EB còn cung cấp dự báo hiệu quả đã điều chỉnh. |
| Vì sao naive có thể chọn gần giống EB? | Phần lý thuyết, **Proposition 1**, giải thích kết quả so sánh sau bước 6. | Dưới các điều kiện thống kê nêu trong paper, hai cách có thể chọn cùng tập chiến lược. **Chọn đúng chiến lược** và **ước lượng đúng nó sẽ lời bao nhiêu** là hai việc khác nhau; EB giúp điều chỉnh việc thứ hai. Kết luận lý thuyết có điều kiện, không phải luôn đúng với mọi dữ liệu. |
| Phương pháp có hạn chế không? | So dự báo và thực tế theo các giai đoạn. | **Hình 5:** sau 2004, hiệu quả giảm và một số dự báo EB kế toán EW quá cao so với thực tế. Cửa sổ quá khứ 20 năm có thể thích nghi chậm khi thị trường thay đổi. |
| Chỉ đặt ngưỡng tstat rất cao có bỏ sót chiến lược tốt không? | Nhánh so sánh kiểm soát phát hiện sai, ngoài chuỗi ví dụ chính. | **Hình 6 / mục 5:** trong 5 nhóm có lợi suất OOS ít nhất 3%/năm, BY1.3 bắt được **1**, RW **3**, Storey **4**. Trong thí nghiệm này, ngưỡng quá chặt bỏ sót nhiều nhóm có hiệu quả. |

**Ý nghĩa kinh tế mà tác giả rút ra:** thông tin dự báo tập trung ở kế toán, cổ phiếu nhỏ và giai đoạn trước 2004. EW đặt trọng số tương đối lớn hơn vào công ty nhỏ so với VW. Các kết quả phù hợp với cách giải thích rằng nhà đầu tư không chú ý đủ và thông tin được phản ánh vào giá chậm; đây là **cách diễn giải phù hợp với bằng chứng**, không phải bằng chứng xác định chắc chắn nguyên nhân.

**“Không dựa vào may mắn” nên hiểu đúng:** họ điều chỉnh nhiễu bằng EB và kiểm tra trên dữ liệu tương lai chưa dùng để chọn. Đây là bằng chứng mạnh hơn việc chỉ khoe thành tích quá khứ, **không phải chứng minh hoàn toàn không có may mắn**. tstat cao đơn lẻ cũng không đủ khi thử rất nhiều chiến lược; đảo chiều một chiến lược từng lỗ cũng không bảo đảm nó sẽ lời sau này.

**Điều quan trọng nhất:** dự báo và lựa chọn dùng **quá khứ**; tương lai chỉ dùng để **kiểm tra**. Giá trị riêng của EB là đánh giá **nên kỳ vọng bao nhiêu** từ thành tích quá khứ và **nhóm dữ liệu nào có thông tin**, chứ không chỉ tạo một danh sách chiến lược lời cao.

Nguồn: [paper gốc, mục 3–5, Bảng 2–3 và Hình 2–6](../paper%20ch%C3%ADnh/High-Throughput%20Asset%20Pricing%20-%20Chen%20%26%20Dim.pdf); thuật toán nằm trong get_rolling_signal_stats.py, estimate_EB_distr.r và create_plots_from_est.py của repo tác giả.
