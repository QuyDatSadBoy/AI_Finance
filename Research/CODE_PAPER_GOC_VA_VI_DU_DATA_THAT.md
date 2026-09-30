# Code công khai của paper gốc: đọc bằng dữ liệu thật

Tài liệu này giải thích **paper Chen–Dim và notebook `rep_original_R_LSTM_Attention_highthroughput_PointB.ipynb`**, không phải notebook thử FinCast. Đọc [bản nhập môn paper](paper%20chính/PAPER_GOC_DE_HIEU_INPUT_OUTPUT.md) trước nếu chưa biết chiến lược long–short là gì.

## 1. Có public code không?

**Có.** Tác giả công bố [repo `chenandrewy/high-throughput-ap`](https://github.com/chenandrewy/high-throughput-ap) trên GitHub. Toàn bộ repo đã được clone về [`code_goc_paper/high-throughput-ap/`](code_goc_paper/high-throughput-ap/) với lịch sử Git riêng. Bản paper cũng chỉ tới repo này. Đây là **code gốc công khai của tác giả**; notebook trong `NoteBook_Refer/` là bản tham chiếu đã chép/chỉnh một phần code để chạy trên Colab/Drive của nhóm. Vì vậy không nên coi notebook là toàn bộ pipeline chính thức.

**Đường đi của dữ liệu — đọc từ trên xuống:**

1. **CSV lợi suất tháng trong `data_raw/`** → dữ liệu đầu vào đã có trên máy.
2. **`get_rolling_signal_stats.py`** → tính trung bình và t-stat của mỗi signal theo năm; xuất file thống kê IS/OOS.
3. **`estimate_EB_distr.r`** → dùng **chỉ phần IS** để tạo dự báo EB cho từng signal–năm.
4. **`create_plots_from_est.py` hoặc notebook `rep`** → đọc dự báo, chọn top chiến lược; sau đó mới ghép **lợi suất thực tế của năm sau** từ CSV ở bước 1 để làm bảng và hình.

Notebook `rep` chủ yếu làm **bước 4**. Nó đọc dự báo EB đã được tính ở bước 3. Nếu chỉ có CSV trong `data_raw/`, mở notebook lên **chưa thể tự có ngay** bảng kết quả EB.

## 2. Input thực sự là gì?

`data_raw/` chứa bảy CSV gốc. Bốn CSV là **lợi suất tháng đã dựng sẵn cho chiến lược**: `DataMinedLongShortReturnsEW.csv`, `DataMinedLongShortReturnsVW.csv`, `PastReturnSignalsLongShort.csv`, `TickerSignalsLongShort.csv`. Ba CSV còn lại giúp tra tên/công thức từ `signalid`. File mẫu nằm riêng trong `data_sample/`; ZIP gốc ở thư mục chính.

Một dòng kế toán EW thật trong [`data_raw/DataMinedLongShortReturnsEW.csv`](../data_raw/DataMinedLongShortReturnsEW.csv) có dạng:

```text
signalid,year,month,ret,nstock
1,1990,7,1.54217517336206,3160
```

Nó nói: **chiến lược kế toán số 1, tháng 7/1990, có giá trị lợi suất được lưu là 1,542..., và dữ liệu ghi 3.160 cổ phiếu**. Đây không phải một cổ phiếu tên “1”. Muốn biết công thức của số 1, tra `signalid=1` trong `DataMinedSignalList.csv`. Khi gom ba họ dữ liệu, phải dùng cặp **(họ signal, signalid)** vì cùng ID ở họ khác là quy tắc khác.

Trong hai file kế toán, `ret` là lợi suất của từng cách EW hoặc VW. File past-return/ticker gộp **cả `ret_ew` và `ret_vw`** trong một dòng. Code tách chúng thành sáu họ chiến lược, rồi xử lý mỗi họ theo năm. Theo cách tác giả tính bảng, `ret` được diễn giải theo **điểm phần trăm mỗi tháng**, và lợi suất năm trong bảng được tính xấp xỉ bằng trung bình tháng × 12.

## 3. Theo chân một signal thật: cuối 2019 dự báo năm 2020

Lấy **signal kế toán 66** từ dữ liệu local. Bảng tên ghi `signal_form=diff(v1)/lag(v2)`, `v1=aco`, `v2=aco`: công thức dùng thay đổi của biến `aco` chia cho giá trị trễ của chính biến đó. CSV hiện không giải thích nội dung kinh tế của mã `aco`; không nên đoán thêm.

Đứng ở **cuối 2019**, code dùng cửa sổ lịch sử **2000–2019** để tính thống kê, rồi để riêng **2020** để kiểm tra:

| Dữ liệu thật của signal 66 | Kế toán EW | Kế toán VW |
|---|---:|---:|
| Số tháng quá khứ 2000–2019 | 240 | 240 |
| Lợi suất trung bình quá khứ, điểm %/tháng | −0,5492 | −0,2982 |
| t-stat quá khứ, theo cấu hình `MAX_LAGS=0` | −3,7610 | −1,5660 |
| Số tháng kiểm tra năm 2020 | 12 | 12 |
| Lợi suất trung bình **thực tế** năm 2020, điểm %/tháng | −1,1700 | −0,9429 |

Ví dụ hai dòng **EW** gần ranh giới thời gian: tháng 12/2019 có `ret=2.04754252015271`, tháng 1/2020 có `ret=-6.73742218306261`. Khi chọn ở cuối 2019, **được phép dùng tháng 12/2019, không được xem tháng 1/2020**.

Signal 66 EW có t-stat quá khứ âm. Cách chọn **naive** trong notebook dùng dấu của t-stat để đảo chiều chiến lược, rồi xếp hạng bằng độ lớn t-stat. Nếu riêng signal này được chọn, lợi suất năm 2020 sau khi đảo chiều sẽ đổi dấu từ trung bình **−1,1700** thành **+1,1700 điểm %/tháng** trong dữ liệu lịch sử, trước mọi chi phí. Đây chỉ là phép minh họa trên **một** signal đã biết, **không phải kết quả danh mục của paper**.

Với **EB**, chưa thể điền một con số dự báo của signal 66 chỉ từ 240 dòng của nó. EB phải nhìn **phân phối t-stat của toàn bộ các signal kế toán EW ở cùng năm 2020**, ước lượng mô hình, rồi mới điều chỉnh t-stat của signal 66. Cột OOS 2020 chỉ dùng để chấm điểm sau khi dự báo xong.

## 4. Code công khai làm gì, file nào nhận/trả gì?

| File code trong repo tác giả | Input chính | Output chính | Hiểu đơn giản |
|---|---|---|---|
| `get_past_return_signals.py`, `get_ticker_signals.py` | Dữ liệu cổ phiếu gốc của tác giả (ví dụ CRSP), không có trong `data_raw/` | CSV lợi suất chiến lược past-return/ticker | **Tạo ứng viên** từ cổ phiếu. Các CSV bạn có là kết quả của giai đoạn này. |
| `get_rolling_signal_stats.py` | Các CSV lợi suất tháng | `OOS_signal_tstat_OosNyears1.csv.gzip` | Với mỗi signal và năm, tính trung bình/t-stat trong quá khứ (IS) và năm sau (OOS). |
| `estimate_EB_distr.r` | File thống kê signal–năm vừa tạo | `*_QML_FamilyYear.csv.gzip`, `*_Predict_SignalYear.csv.gzip` | Học phân phối thành tích của **cả họ signal**, rồi xuất dự báo EB cho từng signal–năm. |
| `create_plots_from_est.py` | CSV lợi suất tháng **và** file dự báo EB | Bảng `.tex`, danh sách top strategy, hình `.pdf` | Ghép dự báo với lợi suất thực tế tương lai; chọn top 1/5/10%; báo cáo kết quả. |
| `create_plots.py` | File thống kê signal–năm | Bản ước lượng/vẽ bằng Python theo cấu hình riêng | Một nhánh phân tích khác trong repo; không đồng nhất với pipeline R `estimate_EB_distr.r`. |

Trong `get_rolling_signal_stats.py`, cửa sổ là **tối đa 20 năm**, nhưng code chấp nhận signal có **ít nhất 36 tháng** dữ liệu trong cửa sổ đó. Năm OOS một năm cần **đủ 12 tháng**. Vì vậy không đúng nếu nói *mọi* signal đều có trọn 240 tháng dữ liệu IS. `MAX_LAGS=0` là lựa chọn khi tính sai số chuẩn/t-stat trong script hiện tại.

Trong `estimate_EB_distr.r`, bước cốt lõi là: lấy t-stat IS của **một họ trong một năm**, ước lượng phân phối mức hiệu quả thật bằng hỗn hợp hai phân phối chuẩn, rồi tính giá trị đã điều chỉnh `thetahat` từ t-stat quan sát. Code chuyển nó về dự báo lợi suất bằng quan hệ `pred_ret = thetahat * mean_ret / tstat`. Đó là lý do dự báo EB không phải chỉ lấy nguyên lợi suất quá khứ.

## 5. Đọc notebook `rep_original...ipynb` theo các nhóm cell

| Cell | Đọc để hiểu điều gì? | Input → output |
|---|---|---|
| 0–1 | Clone repo công khai của tác giả trên Colab. | URL GitHub → thư mục code. |
| 2 | Khai báo đường dẫn Google Drive, đọc CSV chiến lược và file EB dự báo đã tính sẵn; ghép theo `(signal_family, signalid, oos_begin_year)` rồi tính thống kê/biểu đồ ban đầu. | CSV tháng + `Nelly_Predict_SignalYear.csv.gzip` → bảng đã ghép, thống kê, hình. |
| 3–4 | Hai cell dài gần lặp nhau: vẽ IS/OOS/dự báo, chọn EB top 1/5/10% và naive top 1/5/10%, làm bảng, hình tích lũy, top strategy. | Bảng đã ghép → kết quả so sánh. |
| 5 | So các ngưỡng kiểm soát phát hiện sai như BY/Storey. | Bảng thống kê/dự báo → hình về ngưỡng t-stat. |
| 6–23 | Lệnh quản lý repo/Colab, có cả lệnh xóa thư mục. | Không thuộc mô hình EB và không cần chạy để hiểu paper. |

**Điểm quan trọng:** tên notebook có “LSTM_Attention”, nhưng các cell nghiên cứu chính ở đây là **phân tích EB của paper gốc**. Notebook không tự chạy toàn bộ từ CSV gốc tới EB; nó **đòi file trung gian** như `OOS_signal_tstat_OosNyears1.csv.gzip`, `Nelly_Predict_SignalYear.csv.gzip`, và vài file so sánh chiến lược công bố. Các file trung gian này **chưa nằm trong repo local**. Một cell đã lưu lỗi đường dẫn `TypeError`; các bảng/hình lưu trong notebook là đầu ra của những lần chạy trước, không chứng minh toàn bộ notebook chạy sạch từ đầu đến cuối ở máy hiện tại.

## 6. Đọc output cuối cùng cho đúng

Mỗi năm code dùng thông tin lịch sử để **đổi chiều nếu dự báo âm**, xếp hạng các chiến lược và chọn top 1%, 5%, 10%. Sau đó mới lấy lợi suất tháng thực tế trong năm kế tiếp để tạo danh mục và tính lợi suất bình quân, t-stat danh mục, Sharpe và biểu đồ tích lũy. Trong notebook, bảng top 1% gộp các họ cho giai đoạn **1983–2020** cho **EB 5,70%/năm, Sharpe 1,46**; naive là **5,72%/năm, Sharpe 1,45**. Hai cách gần như ngang nhau ở top 1%; không thể kết luận EB thắng rõ rệt bằng riêng hai con số này.

Đừng trộn **ba loại output**: `mean_ret` IS là kết quả **đã xảy ra trong quá khứ**; `pred_ret` EB là **dự báo tạo trước năm OOS**; `ret_oos` là kết quả **thực tế dùng để chấm dự báo**. Nhìn ba cột này theo đúng thứ tự thời gian là cách nhanh nhất để hiểu toàn bộ paper và code.

## 7. Nếu muốn chạy lại bằng folder hiện tại

Cần chỉnh đường dẫn Windows/Colab/Drive trong code gốc sang `data_raw/`; đổi tên đầu vào `.csv.gzip` cho past-return/ticker thành `.csv` vì file local là CSV văn bản; và bỏ/điều chỉnh RavenPack do dữ liệu đó không có trong `data_raw/`. Tiếp đó mới chạy bước thống kê theo năm, bước R Empirical Bayes, và cuối cùng bước vẽ bảng. Code tạo chiến lược từ CRSP không chạy được chỉ bằng các CSV chiến lược hiện có, nhưng **không cần chạy lại bước tạo chiến lược** để bắt đầu tái lập EB từ những CSV này.

Đây là các thay đổi môi trường cần làm trước khi chạy; tài liệu này **không tuyên bố đã tái lập chính xác toàn bộ bảng của paper** trên máy local.
