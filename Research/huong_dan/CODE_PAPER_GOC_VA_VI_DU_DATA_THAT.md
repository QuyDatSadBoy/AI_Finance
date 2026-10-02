# Code paper gốc và 7 file data: đọc từ đầu bằng số thật

Tài liệu này dành cho người mới. Mục tiêu là trả lời: **mỗi file chứa gì, nối các file ra sao, paper dùng chúng làm input/output gì, và “đáp án” (target) cần dự báo là gì?** Bắt đầu bằng các file nhỏ trong [`data_sample/`](../../data_sample/README.md) nếu máy không mở nổi CSV gốc. Bản đầy đủ nằm ở `data_raw/`.

## 1. Bức tranh trước khi đọc file

Một **cổ phiếu** là một mã của công ty. Một **signal** là *quy tắc* để xếp hạng các mã, chẳng hạn xếp theo một tỷ số kế toán hoặc lợi suất quá khứ. Từ một signal, tác giả chọn một nhóm mã để **mua (long)** và một nhóm khác để **bán khống (short)**. Kết quả của chiến lược trong tháng được ghi bằng lợi suất **long trừ short**. File chiến lược chỉ lưu kết quả đã gộp; nó không chứa bảng giá/lợi suất của từng công ty.

**EW** (equal-weighted) chia trọng số đều cho từng mã trong mỗi nhóm; **VW** (value-weighted) chia trọng số theo vốn hóa. Chẳng hạn nhóm mua có A lời 10%, B lời 0%, vốn hóa A:B = 90:10: EW của nhóm mua là `(10% + 0%)/2 = 5%`, VW là `10% × 90% + 0% × 10% = 9%`. Tính tương tự cho nhóm bán, rồi lấy **lợi suất nhóm mua − lợi suất nhóm bán**. Các số A/B này chỉ minh họa công thức; **không phải** quan sát cấp cổ phiếu trong repo.

**Điểm bắt đầu trên máy này:** tác giả đã tính sẵn lợi suất tháng của chiến lược. Bảy CSV trong `data_raw/` gồm **3 bảng tra tên/quy tắc** và **4 bảng lợi suất**. Chúng đủ để bắt đầu bước thống kê và dự báo EB, nhưng không đủ để tự tính lại `ret` từ từng cổ phiếu.

```text
Dữ liệu cổ phiếu/biến kế toán cấp công ty (không có ở local)
    → tạo signal, xếp mã, chọn nhóm mua/bán, tính EW/VW
7 CSV ở data_raw/: bảng tên + lợi suất chiến lược theo tháng
    → thống kê quá khứ từng chiến lược/năm
    → Empirical Bayes (EB) dự báo lợi suất năm tới
    → chọn/đảo chiều chiến lược trước khi xem năm tới
    → dùng lợi suất thực tế năm tới để chấm kết quả
```

## 2. Bảy file, chia thành ba họ chiến lược

| Họ | File tra tên signal | File lợi suất tháng | Một dòng lợi suất là gì? |
|---|---|---|---|
| **Kế toán** (`accounting`) | `DataMinedSignalList.csv` | `DataMinedLongShortReturnsEW.csv` và `DataMinedLongShortReturnsVW.csv` | Một signal, một tháng, **một** cách EW hoặc VW tùy file |
| **Lợi suất quá khứ** (`past_return`) | `PastReturnSignalNames.csv` | `PastReturnSignalsLongShort.csv` | Một signal, một tháng, **cả** EW và VW trong cùng dòng |
| **Ký tự mã cổ phiếu** (`ticker`) | `TickerSignalNames.csv` | `TickerSignalsLongShort.csv` | Một signal, một tháng, **cả** EW và VW trong cùng dòng |

**Không có 7 nguồn độc lập:** bảng tên và bảng lợi suất là hai phần của cùng một họ. Một signal có nhiều tháng; EW/VW là hai phiên bản của signal đó. Theo Bảng 1 của paper, số phiên bản chiến lược có lợi suất là `29.314 × 2 + 19.402 × 2 + 19.380 × 2 = 136.192`. Bảng *tên* kế toán có 29.315 ID, nhưng ID `25144` không có lợi suất ở cả hai file kế toán, nên không cộng nó vào 136.192.

### 2.1. Ba file **tra tên/quy tắc**

| File | Cột | Ý nghĩa và cách dùng |
|---|---|---|
| `DataMinedSignalList.csv` | `signalid,signal_form,v1,v2` | `signalid` là mã quy tắc kế toán; `signal_form` là dạng công thức; `v1/v2` là **tên biến** kế toán, không phải số liệu của từng công ty. Ví dụ thật `1,diff(v1)/lag(v2),acchg,aco`: signal 1 dùng thay đổi của `acchg` chia giá trị trễ của `aco`. |
| `PastReturnSignalNames.csv` | `signalid,signalname,root` | Tra mã thành quy tắc dựa trên lợi suất trong quá khứ. Ví dụ thật `0,ret_1,ret`. `ret_1` là **tên signal**, còn `ret_ew` trong file kết quả mới là lợi suất chiến lược tháng. |
| `TickerSignalNames.csv` | `signalid,signalname` | Tra mã thành quy tắc nhóm ký tự ticker. Ví dụ thật `0,L1_lng_1_2_sht_3_4`: xét ký tự thứ nhất (`L1`), mua nhóm ký tự 1 và 2, bán nhóm 3 và 4. Các số nhóm **không phải tên cổ phiếu**. |

Ba file này giúp **giải mã ID**, không phải biến mục tiêu (target) và không chứa lợi suất để huấn luyện EB. Không ghép `signalid=0` của past-return với `signalid=0` của ticker: hai ID thuộc hai họ khác nhau.

### 2.2. Hai file **lợi suất kế toán**

Hai file `DataMinedLongShortReturnsEW.csv` và `DataMinedLongShortReturnsVW.csv` đều có header:

```text
signalid,year,month,ret,nstock
```

Một khóa `(signalid, year, month)` cho biết **cùng quy tắc, cùng tháng**; tên file quyết định `ret` là EW hay VW. `nstock` là **tổng số mã trong nhóm mua và nhóm bán**. Code tạo dữ liệu bản cũ tính `nstock = nstock_long + nstock_short`; bản mới tách thành hai cột riêng. [Bằng chứng trong lịch sử code tác giả](https://github.com/chenandrewy/flex-mining/commit/8e49bc51a1b3f3a7937633bf813097dc3d5c44df).

Ví dụ **hai dòng thật**, cùng signal 1, tháng 7/1990:

| File | `ret` | `nstock` | Cách đọc |
|---|---:|---:|---|
| Kế toán EW | `1.54217517336206` | `3160` | Chiến lược long–short lời khoảng **1,542%** theo EW; file ghi nhận **3.160 mã**. |
| Kế toán VW | `5.72663514827621` | `3160` | Cùng signal và tháng, lời khoảng **5,727%** theo VW. |

Từ hai dòng này **không thể tính ngược** lợi suất từng mã, lợi suất riêng nhóm mua/bán hoặc danh sách mã đã chọn. `nstock=3160` nghĩa là **nhóm mua cộng nhóm bán có 3.160 mã**, không phải mua 3.160 mã. File không giữ số lượng riêng của hai nhóm, nên không suy ra hai nhóm bằng nhau và không lấy 10% của 3160 để tính số mã mỗi nhóm.

### 2.3. File **lợi suất quá khứ**

`PastReturnSignalsLongShort.csv` có header:

```text
signalid,date,ret_ew,ret_vw,nshort,nlong
```

Ví dụ **dòng thật**:

```text
0,1962-01-31,-1.3873515111576442,0.6307365039511064,92,92
```

Ghép `signalid=0` với bảng tên ta được `ret_1`. Vào tháng 1/1962, nhóm **bán khống có 92 mã** (`nshort`), nhóm **mua có 92 mã** (`nlong`). Kết quả cả chiến lược là khoảng **−1,387% EW** nhưng **+0,631% VW**. Dấu khác nhau là có thể xảy ra vì hai cách đặt trọng số khác nhau. `ret_ew`/`ret_vw` đều là lợi suất **mua trừ bán**, **không phải** một cột mua và một cột bán. Code tạo file tính trung bình hoặc trung bình theo vốn hóa trong mỗi nhóm rồi trừ hai nhóm; trước khi lưu CSV, nó bỏ các cột lợi suất riêng của hai nhóm.

File có 19.402 signal và 732 tháng (01/1962–12/2022), tổng 14.202.264 dòng. `date` thường là ngày cuối tháng/ngày giao dịch cuối gần đó.

### 2.4. File **lợi suất theo ký tự ticker**

`TickerSignalsLongShort.csv` có header:

```text
signalid,date,ret_ew,ret_vw,nlong,nshort
```

Ví dụ **dòng thật**:

```text
0,1963-01-01,-1.00740388187827,-0.14181616981004286,196,196
```

Ghép với `TickerSignalNames.csv`, ID 0 là `L1_lng_1_2_sht_3_4`. Tháng 1/1963, **196 mã mua**, **196 mã bán**; chiến lược lỗ khoảng **1,007% EW** hoặc **0,142% VW**. Cẩn thận: **thứ tự cột đếm đảo ngược file past-return**: ticker là `nlong,nshort`, past-return là `nshort,nlong`. Đọc theo *header*, đừng đọc theo vị trí. `date` ở đây là ngày đầu tháng.

File có 19.380 signal và 690 tháng (01/1963–12/2022), tổng 13.372.200 dòng. Tên ticker mô tả **quy tắc phân nhóm ký tự**, không cung cấp tên 196 mã thật trong hai chân danh mục.

## 3. Các file liên kết với nhau thế nào?

Để tra một dòng lợi suất ra ý nghĩa của nó, nối **bảng tên với bảng lợi suất trong cùng họ theo `signalid`**:

```text
DataMinedSignalList.signalid       → DataMinedLongShortReturnsEW/VW.signalid
PastReturnSignalNames.signalid     → PastReturnSignalsLongShort.signalid
TickerSignalNames.signalid         → TickerSignalsLongShort.signalid
```

Khi gom tất cả thành một bảng phân tích, khóa đúng là **`(family, weight, signalid, month)`**. `family` phân biệt ba họ; `weight` phân biệt EW/VW. Riêng hai file kế toán có thể ghép theo `(signalid, year, month)` để đặt hai lợi suất cạnh nhau, nhưng **không ghép theo thứ tự dòng** vì file EW/VW không có hoàn toàn cùng tập khóa.

Để ghép theo tháng, chuẩn hóa `date` thành `(year, month)`: past-return `1962-01-31` và ticker `1963-01-01` dùng quy ước ngày khác nhau, nhưng cùng trường `year, month` mới là đơn vị thời gian paper phân tích. Ba họ **không cần ghép từng signal với nhau**: EB ước lượng từng họ/phiên bản EW-VW riêng, sau đó mới so sánh và chọn chiến lược.

## 4. Input, target và output của paper

Từ “target” dễ bị hiểu sai vì ở đây có **hai cấp**: paper không dự báo giá của từng cổ phiếu mà dự báo **hiệu quả tương lai của chiến lược**.

| Giai đoạn | Input được phép thấy | Output | Target/đáp án ở đâu? |
|---|---|---|---|
| **Tạo chiến lược** | Dữ liệu cổ phiếu và biến kế toán cấp công ty của tác giả; **không có trong local** | Bốn CSV lợi suất tháng, ba CSV tên signal | Ở bước này `ret` là **kết quả tính từ cổ phiếu**, không phải dự báo EB. |
| **Thống kê trước năm dự báo** | Lợi suất tháng của signal trong tối đa **20 năm trước** | `mean_ret`, `tstat`, `nobs` cho IS; cùng các cột cho OOS để chấm sau này | Khi chọn chiến lược, **chỉ dùng IS**. OOS là dữ liệu tương lai được giữ riêng. |
| **Dự báo EB** | `tstat`/`mean_ret` IS của **cả một họ chiến lược trong cùng năm** | `thetahat`, `pred_ret` cho từng signal–năm; tham số phân phối theo họ–năm | `pred_ret` là **dự báo lợi suất bình quân của chiến lược trong năm tới**. |
| **Chọn và chấm** | `pred_ret` đã tạo trước năm mới; sau đó đọc `ret` thực tế trong năm mới | Danh sách top 1%/5%/10%, lợi suất thực tế, bảng và hình | **Lợi suất OOS thực tế** là đáp án để kiểm tra dự báo và danh mục. |

Ví dụ đứng ở **cuối năm 2019**: muốn đánh giá năm **2020**, dùng các tháng 2000–2019 (nếu signal có đủ dữ liệu) để tính IS; **không dùng một tháng nào của 2020 để dự báo hoặc chọn**. Sau khi đã chọn xong, lợi suất tháng 1–12/2020 mới được dùng để xem chọn đúng hay sai. Script chấp nhận tối thiểu **36 tháng IS** và yêu cầu **đủ 12 tháng OOS** cho cửa sổ một năm; không phải signal nào cũng có trọn 240 tháng quá khứ.

Ví dụ signal kế toán **66** trong CSV thật: trung bình EW 2000–2019 là khoảng **−0,5492 điểm %/tháng**, t-stat IS khoảng **−3,7610**; trung bình EW **thực tế** năm 2020 là **−1,1700 điểm %/tháng**. Số −1,1700 là **đáp án sau này**, không được đưa vào EB ở cuối 2019. Không thể suy ra `pred_ret` của signal 66 chỉ từ chuỗi riêng nó: EB còn học phân phối t-stat của cả họ kế toán EW cùng năm.

**Phân biệt 4 tên gần giống nhau:** `ret`/`ret_ew`/`ret_vw` là lợi suất **từng tháng đã quan sát**; `mean_ret` IS là trung bình các tháng **quá khứ**; `pred_ret` là **ước lượng trước năm tới**; `mean_ret` OOS là **lợi suất tương lai đã xảy ra**, chỉ dùng để đánh giá. `nstock`, `nlong`, `nshort` là số mã, **không phải target**.

## 5. File code gốc nào dùng dữ liệu ở bước nào?

Code gốc công khai ở [chenandrewy/high-throughput-ap](https://github.com/chenandrewy/high-throughput-ap), đã clone về [`Research/code_goc_paper/high-throughput-ap/`](../code_goc_paper/high-throughput-ap/). Đọc theo thứ tự công việc:

| Script | Input | Output và vai trò |
|---|---|---|
| [`get_past_return_signals.py`](../code_goc_paper/high-throughput-ap/get_past_return_signals.py), [`get_ticker_signals.py`](../code_goc_paper/high-throughput-ap/get_ticker_signals.py) | Dữ liệu CRSP cấp cổ phiếu của tác giả, **không có ở local** | Tạo hai bảng lợi suất và hai bảng tên cho họ past-return/ticker. Code xếp nhóm rồi tính long−short EW/VW. |
| [`get_rolling_signal_stats.py`](../code_goc_paper/high-throughput-ap/get_rolling_signal_stats.py) | Bốn CSV lợi suất tháng; tách EW/VW của past-return/ticker thành phiên bản riêng | `OOS_signal_tstat_OosNyears1.csv.gzip`: mỗi `(signal_family, oos_begin_year, signalid)` có dòng `s_flag=IS` và `s_flag=OOS`, cùng `mean_ret,tstat,nobs`. |
| [`estimate_EB_distr.r`](../code_goc_paper/high-throughput-ap/estimate_EB_distr.r) | **Chỉ dòng IS** từ file thống kê; nhóm theo họ và năm | `*_QML_FamilyYear.csv.gzip` chứa tham số họ–năm; `*_Predict_SignalYear.csv.gzip` chứa `thetahat,pred_ret` của từng signal–năm. Đây là dự báo EB. |
| [`create_plots_from_est.py`](../code_goc_paper/high-throughput-ap/create_plots_from_est.py) / notebook `rep_original_R_LSTM_Attention_highthroughput_PointB.ipynb` | CSV tháng + file dự báo EB + các file so sánh nếu cần | Ghép dự báo với OOS thực tế, đảo chiều signal khi dự báo âm, xếp hạng/chọn top, xuất bảng/hình. Notebook chủ yếu làm đoạn cuối, **không tự sinh EB chỉ từ 7 CSV**. |

Code R có quan hệ `pred_ret = thetahat × mean_ret / tstat`: nó điều chỉnh độ tin cậy của t-stat quá khứ bằng EB thay vì lấy nguyên lợi suất quá khứ. Đây là **bài toán dự báo/đánh giá ở cấp chiến lược–năm**, không phải bài toán có nhãn “giá cổ phiếu ngày mai tăng hay giảm”.

Các file trung gian `OOS_signal_tstat...`, `*_QML_FamilyYear...`, `*_Predict_SignalYear...` **chưa có sẵn ở local**. Code gốc còn dùng đường dẫn máy tác giả, tên `.csv.gzip` khác đuôi `.csv` local và có nhánh RavenPack/news hoặc file so sánh chiến lược công bố không nằm trong 7 CSV. Vì vậy có thể **hiểu luồng và dữ liệu đầu vào**, nhưng không nên nói đã tái lập nguyên bảng paper nếu chưa điều chỉnh môi trường và chạy đủ pipeline.

## 6. Cách tự kiểm tra bằng file mẫu

1. Mở `data_sample/DataMinedSignalList_sample.csv`, tìm `signalid=1`; sau đó mở hai sample `DataMinedLongShortReturnsEW_sample.csv`/`VW_sample.csv`, tìm cùng `(1,2000,1)`. So `ret`: **−12,413% EW** và **−6,238% VW**, `nstock` cùng **3800**. Dòng 7/1990 ở ví dụ phía trên chỉ nằm trong file gốc đầy đủ, không nằm trong hai sample này.
2. Mở `data_sample/PastReturnSignalNames_sample.csv`, tra `signalid=0` là `ret_1`; tìm ID 0, tháng 1/1983 trong `PastReturnSignalsLongShort_sample.csv`. Đọc đúng thứ tự `nshort,nlong`: **323 mã mỗi chân**, `ret_ew≈1,756%`, `ret_vw≈3,447%`.
3. Làm tương tự với `TickerSignalNames_sample.csv` và `TickerSignalsLongShort_sample.csv`; ID 0, tháng 1/1983 có **498 mã mỗi chân**. Đọc đúng thứ tự `nlong,nshort`.
4. Khi viết code phân tích, luôn giữ `family` và `weight`; chỉ dùng `ret` **trước** năm dự báo để huấn luyện, giữ `ret` **trong** năm dự báo để chấm.

Muốn xem toàn bộ số dòng, khoảng thời gian, cách tạo mẫu và lưu ý về ZIP, đọc thêm [`README_DATA.md`](README_DATA.md). Muốn hiểu ý tưởng paper không cần code, đọc [`PAPER_GOC_DE_HIEU_INPUT_OUTPUT.md`](../paper%20ch%C3%ADnh/PAPER_GOC_DE_HIEU_INPUT_OUTPUT.md).
