# Hướng dẫn đọc toàn bộ dữ liệu của High-Throughput Asset Pricing

Tài liệu dành cho người mới, đối chiếu với [paper chính](paper%20chính/High-Throughput%20Asset%20Pricing%20-%20Chen%20%26%20Dim.pdf). Kiểm tra các file trong `data_raw/` ngày 29/09/2026. [Bảy file mẫu nhỏ tương ứng với từng CSV gốc](../data_sample/README.md) nằm trong `data_sample/`, còn ZIP gốc `Strategy Returns.zip` nằm ở thư mục chính. Các số đếm, khoảng thời gian và cột trống dưới đây được tính từ **file thực tế**, không suy ra từ tên file.

**Đọc nhanh trong 30 giây:** Input chính của EB là **lợi suất tháng của từng chiến lược** trong `data_raw/`. Mỗi cuối năm, code lấy tối đa 20 năm đã qua để tính thành tích và t-stat, nhìn cả nhóm chiến lược để điều chỉnh dự báo, rồi chọn top 1%/5%/10%. **Lợi suất năm sau là đáp án để kiểm tra, tuyệt đối không đưa vào bước chọn.** Output cuối là dự báo theo chiến lược–năm, danh sách được chọn và bảng lợi suất thực tế của các danh mục. Nếu cần hiểu từng script, đọc thêm [hướng dẫn code với một signal thật](CODE_PAPER_GOC_VA_VI_DU_DATA_THAT.md).

**Luồng dữ liệu — đọc từ trên xuống:**

1. **Dữ liệu của từng cổ phiếu và biến kế toán** → tác giả tạo **136.192 chiến lược**. Dữ liệu cấp cổ phiếu này **không có trong folder hiện tại**.
2. **Lợi suất tháng của các chiến lược** → nằm trong các CSV ở `data_raw/`. **Đây là điểm bắt đầu từ dữ liệu bạn đang có.**
3. **Các tháng quá khứ** → code tính trung bình và t-stat của từng chiến lược, rồi EB tạo **dự báo trước năm kiểm tra**.
4. **Các dự báo** → code chọn top 1%, 5% hoặc 10% chiến lược để lập danh mục.
5. **Các tháng của năm sau** → code mới đọc từ chính CSV lợi suất để xem **danh mục được chọn đã lãi/lỗ thật bao nhiêu**. Những tháng này **không được dùng ở bước 3–4**.

File thống kê và dự báo tạo ở bước 3 **chưa được lưu sẵn ở local**; cần chạy code để sinh ra.

## 1. Đọc nhanh trước khi xem bảng

**Một signal** là công thức dùng để xếp hạng cổ phiếu. Từ signal đó, tác giả xây chiến lược **long–short**: mua một nhóm cổ phiếu và bán khống nhóm còn lại. Mỗi tháng chiến lược tạo ra một lợi suất.

Thư mục `data_raw/` chứa:

1. **Bảng định nghĩa signal:** ID nào có nghĩa là công thức nào.
2. **Bảng lợi suất tháng:** signal đó lãi/lỗ bao nhiêu ở từng tháng.
3. **Hai cách tính danh mục:** `EW` cho trọng số cổ phiếu bằng nhau; `VW` cho trọng số theo vốn hóa.

**Một dòng lợi suất = một signal × một tháng × một cách tính EW/VW** ở nhóm kế toán. Ở nhóm ticker và past-return, cùng một dòng chứa cả hai cột `ret_ew` và `ret_vw`. Đây **không phải** bảng giá của từng cổ phiếu. Bạn sẽ không thấy giá cổ phiếu, khoản mục kế toán gốc của từng công ty hoặc lệnh giao dịch chi tiết trong các file này.

Paper dùng khoảng **136.000 chiến lược**, chính xác là **136.192** trong Bảng 1. Các bảng lợi suất trong folder hiện chứa đủ ba *họ signal* mà Bảng 1 mô tả: kế toán, lợi suất quá khứ và ký tự ticker. Điều này **chưa có nghĩa** chỉ cần mở CSV là tự tái lập được các kết quả EB: còn phải làm đúng quy tắc cửa sổ 20 năm, ước lượng mô hình, xếp hạng, đảo dấu và đánh giá OOS.

## 2. Các file dữ liệu gốc trong `data_raw/`

| File | Vai trò | Số dòng dữ liệu, không tính header | Số signal có lợi suất | Khoảng tháng quan sát |
|---|---|---:|---:|---|
| `DataMinedSignalList.csv` | Định nghĩa signal **kế toán** | 29.315 | Không áp dụng | Không có ngày |
| `DataMinedLongShortReturnsEW.csv` | Lợi suất kế toán, danh mục EW | 16.941.504 | 29.314 | 07/1951–12/2022 |
| `DataMinedLongShortReturnsVW.csv` | Lợi suất kế toán, danh mục VW | 16.941.358 | 29.314 | 07/1951–12/2022 |
| `PastReturnSignalNames.csv` | Tên signal từ lợi suất quá khứ | 19.402 | Không áp dụng | Không có ngày |
| `PastReturnSignalsLongShort.csv` | Lợi suất past-return, **cả EW và VW** | 14.202.264 | 19.402 | 01/1962–12/2022 |
| `TickerSignalNames.csv` | Tên signal dựa vào ký tự ticker | 19.380 | Không áp dụng | Không có ngày |
| `TickerSignalsLongShort.csv` | Lợi suất ticker, **cả EW và VW** | 13.372.200 | 19.380 | 01/1963–12/2022 |

Đừng cộng số **dòng tháng** để ra số chiến lược. Một signal xuất hiện ở nhiều tháng. Với mỗi signal, EW và VW là **hai phiên bản chiến lược**. Theo Bảng 1: $29.314\times2+19.402\times2+19.380\times2=136.192$ chiến lược.

### Ghi chú về ZIP gốc ở thư mục chính

ZIP gốc gọi bốn file bên trong là `*.csv.gzip`, nhưng mình đã kiểm tra byte đầu: chúng là **CSV văn bản thường**, không có cấu trúc GZIP. Vì vậy các bản đã giải nén được đặt đúng đuôi `.csv` trong folder này. Không cần chạy lệnh `gunzip` với chúng. ZIP còn có một mục thư mục tên `/`; mục này được bỏ qua khi giải nén, bốn bảng CSV đã qua kiểm tra CRC.

## 3. Khóa nối bảng: điều quan trọng nhất khi dùng dữ liệu

`signalid` là ID **chỉ có ý nghĩa trong họ signal tương ứng**. ID `1` của kế toán, ID `1` của past-return và ID `1` của ticker là **ba công thức khác nhau**. Nếu gom dữ liệu lại, hãy tạo khóa ghép **`(family, signalid)`**, trong đó `family` là `accounting`, `past_return` hoặc `ticker`.

| Muốn biết một dòng là signal nào | Nối theo |
|---|---|
| Lợi suất kế toán EW/VW ↔ tên kế toán | `signalid` trong họ `accounting` |
| Lợi suất past-return ↔ tên past-return | `signalid` trong họ `past_return` |
| Lợi suất ticker ↔ tên ticker | `signalid` trong họ `ticker` |

Muốn so EW với VW của nhóm **kế toán**, nối hai file theo `(signalid, year, month)`. Hai file không có số dòng bằng nhau, vì vậy **không nối theo số thứ tự dòng** và đừng giả định mọi khóa xuất hiện ở cả hai file.

Muốn ghép **ba họ** theo thời gian, chuẩn hóa về `(year, month)`:

- Kế toán đã có `year`, `month`.
- Ticker có `date` là **ngày đầu tháng**, ví dụ `1963-01-01`.
- Past-return có `date` thường là **ngày cuối tháng hoặc ngày giao dịch cuối gần đó**, ví dụ `1962-01-31` hay `2022-12-30`.

Vì vậy `2022-12-01` và `2022-12-30` là **cùng tháng 12/2022** trong bảng tháng. Không nối trực tiếp bằng chuỗi `date` đầy đủ.

## 4. Nhóm 1: dữ liệu kế toán

### Bảng định nghĩa `DataMinedSignalList.csv`

| Cột | Nghĩa |
|---|---|
| `signalid` | ID của công thức, từ 1 đến 29.315 trong bảng tên |
| `signal_form` | Cách kết hợp hai biến kế toán |
| `v1`, `v2` | Tên viết tắt hai biến đầu vào |

Có hai mẫu công thức:

- `v1/v2`: tỷ số giữa hai biến.
- `diff(v1)/lag(v2)`: thay đổi của `v1` chia cho giá trị trễ của `v2`.

Ví dụ dòng `1,diff(v1)/lag(v2),acchg,aco` nghĩa là signal 1 được mô tả bằng công thức **thay đổi `acchg` / giá trị trễ `aco`**. Bảng không chứa giá trị `acchg`, `aco` của từng công ty; nó chỉ cho biết công thức đã dùng để tạo signal. Muốn hiểu ý nghĩa kinh tế của những mã biến này cần từ điển biến của nguồn Compustat/nguồn xây signal tương ứng.

Trong bảng tên có 15.730 công thức `diff(v1)/lag(v2)` và 13.585 công thức `v1/v2`.

### Bảng lợi suất EW và VW

Cả `DataMinedLongShortReturnsEW.csv` và `DataMinedLongShortReturnsVW.csv` đều có:

| Cột | Nghĩa |
|---|---|
| `signalid` | Nối tới công thức kế toán |
| `year`, `month` | Tháng của quan sát |
| `ret` | Lợi suất danh mục long–short trong tháng, theo EW hoặc VW tùy file |
| `nstock` | Số cổ phiếu được dữ liệu ghi nhận cho signal-tháng; **không phải** hai cột đếm chân mua/bán riêng |

Ví dụ `1,1990,7,1.542175...,3160` trong file EW có nghĩa: signal kế toán 1, tháng 07/1990, giá trị lợi suất EW được lưu là khoảng `1.542`, `nstock=3160`. **CSV không ghi rõ đơn vị trong metadata**, nhưng code tác giả nhân lợi suất tháng trung bình với 12 và ghi bảng là `% ann`; vì vậy khi theo đúng code paper, nên đọc các giá trị `ret` như **điểm phần trăm mỗi tháng**. Nếu đem dữ liệu dùng cho một code khác, vẫn phải kiểm tra quy ước đơn vị trước.

Hai file lợi suất đều có **29.314 ID**. Bảng tên lại có **29.315 ID**: riêng `signalid=25144` (`v1/v2,pstkn,pstk`) **không có dòng lợi suất** trong file EW lẫn VW. Điều này giải thích chênh lệch 1 ID với Bảng 1 của paper; không nên coi bảng tên 29.315 dòng là 29.315 chiến lược có dữ liệu.

Các signal kế toán có độ dài chuỗi khác nhau. Khoảng 07/1951–12/2022 là phạm vi **toàn bộ file**, không có nghĩa signal nào cũng có dữ liệu xuyên suốt. Hai file EW/VW cũng chênh 146 dòng, nên phải kiểm tra dữ liệu thiếu theo khóa khi so sánh chúng.

[`data_sample/DataMinedLongShortReturnsSample.csv`](../data_sample/DataMinedLongShortReturnsSample.csv) chỉ có 10 signal đầu tiên và 2.580 dòng. Hãy dùng nó để học cách đọc cột hoặc thử code nhỏ; đừng dùng như toàn bộ tập thực nghiệm.

## 5. Nhóm 2: signal dựa trên lợi suất quá khứ

### `PastReturnSignalNames.csv`

| Cột | Nghĩa |
|---|---|
| `signalid` | ID từ 0 đến 19.401 |
| `signalname` | Tên công thức, ví dụ `ret_1`, `skew_17_18_19_20` |
| `root` | Loại thống kê nền: `ret`, `std`, `skew`, `kurt` |

Tên `ret` chỉ lợi suất quá khứ; `std` độ biến động; `skew` độ lệch phân phối; `kurt` độ nhọn/đuôi của phân phối. Paper xây nhiều cách chọn các quý trong **20 quý quá khứ**, rồi tính các thống kê này để xếp hạng cổ phiếu. Có 19.402 signal: 4.867 thuộc `ret`, và mỗi loại `std`, `skew`, `kurt` có 4.845.

### `PastReturnSignalsLongShort.csv`

Header chính xác là:

`signalid,date,ret_ew,ret_vw,nshort,nlong`

Mỗi dòng chứa **cả** lợi suất EW và VW của một signal-tháng. `nshort`, `nlong` là số cổ phiếu của chân bán khống và chân mua. File có **19.402 signal × 732 tháng = 14.202.264 dòng**, phủ đủ các tháng từ 01/1962 đến 12/2022 cho mỗi signal. Không có giá trị trống ở các cột lợi suất hoặc số lượng chân danh mục trong lần quét toàn file.

Đừng nhầm `ret_ew`/`ret_vw` là hai tháng khác nhau: chúng là hai cách cho trọng số **của cùng signal trong cùng tháng**.

## 6. Nhóm 3: signal dựa trên ký tự ticker

### `TickerSignalNames.csv`

| Cột | Nghĩa |
|---|---|
| `signalid` | ID từ 0 đến 19.379 |
| `signalname` | Quy tắc dùng vị trí và nhóm ký tự trong ticker |

Ví dụ `L1_lng_1_2_sht_3_4` nói rằng quy tắc dùng **ký tự thứ nhất** (`L1`) của mã cổ phiếu, mua (`lng`) hai nhóm ký tự số 1 và 2, bán khống (`sht`) hai nhóm số 3 và 4. `L4` dùng ký tự thứ tư. Các số nhóm là nhóm được tạo khi sắp xếp ký tự, không phải số cổ phiếu.

Đây là nhóm hữu ích để thấy *data mining bias*: nếu thử rất nhiều tổ hợp chữ cái, sẽ có tổ hợp trông tốt trong quá khứ dù chữ cái thường không phải nguồn thông tin kinh tế đáng tin về lợi suất.

### `TickerSignalsLongShort.csv`

Header chính xác là:

`signalid,date,ret_ew,ret_vw,nlong,nshort`

**Chú ý:** hai cột cuối có **thứ tự ngược** so với file past-return (`nlong,nshort` thay vì `nshort,nlong`). Dùng **tên cột**, không lấy cột thứ năm rồi mặc định luôn là chân mua.

File có 13.372.200 dòng và 19.380 signal. Không phải mọi signal có cùng số tháng:

| Nhóm tên | Số signal | Khoảng thời gian | Tháng mỗi signal |
|---|---:|---|---:|
| `L1`, `L2`, `L3` | 14.535 | 01/1963–12/2022 | 720 |
| `L4` | 4.845 | 01/1973–12/2022 | 600 |

Vì vậy sự vắng mặt của nhóm `L4` trước 1973 là đặc điểm phạm vi dữ liệu, không phải ô `ret` bị để trống. Lần quét toàn file không thấy giá trị trống ở các cột lợi suất hoặc số lượng chân danh mục.

## 7. Các bẫy dữ liệu cần tránh

1. **Trùng ID giữa các họ:** luôn giữ `family` cùng với `signalid`.
2. **Khác ngày trong cùng tháng:** chuyển `date` về `(year, month)` trước khi ghép.
3. **Khác cấu trúc EW/VW:** kế toán tách hai file; past-return và ticker gộp hai cột trong một file.
4. **Khác thứ tự `nlong`, `nshort`:** chọn theo tên cột.
5. **Bảng tên có một ID kế toán không có lợi suất:** `25144` phải được xử lý khi nối.
6. **Các họ có giai đoạn phủ khác nhau:** chọn giai đoạn chung hoặc chấp nhận mẫu không cân bằng; đừng mặc nhiên xem ô không tồn tại là lợi suất 0.
7. **Đơn vị `ret` chưa được ghi rõ trong CSV:** xác nhận trước khi tính lợi suất năm, Sharpe hay vẽ biểu đồ.
8. **Có giá trị rất cực đoan:** trong dữ liệu kế toán, giá trị `ret` quan sát được có thể thấp hơn `-300` hoặc cao hơn `240`. Chúng cần được kiểm tra theo định nghĩa lợi suất, đơn vị và xử lý outlier của nghiên cứu; không tự động xóa hoặc coi là lỗi.
9. **Không dùng dữ liệu tương lai:** tại ngày dự báo, chỉ tính mean, t-stat hoặc EB từ các tháng trước ngày đó. Paper đánh giá OOS theo năm trong 1983–2020, dù các CSV hiện kéo dài đến 2022.
10. **File rất lớn:** bốn bảng lợi suất có tổng hơn **61 triệu dòng**. Không nên mở toàn bộ bằng Excel rồi lưu lại; nên đọc theo luồng hoặc theo phần khi phân tích.

## 8. Từ CSV tới nghiên cứu: một đường đi dễ hiểu

Giả sử muốn kiểm tra một signal kế toán ở cuối năm 1993:

1. Tra `signalid` trong `DataMinedSignalList.csv` để biết công thức.
2. Lấy chuỗi `ret` từ file EW hoặc VW cho **các tháng đã có đến 12/1993**.
3. Tính lợi suất trung bình, độ bất định và t-stat từ chuỗi quá khứ.
4. Làm tương tự với các signal khác trong cùng họ; EB dùng toàn bộ phân phối này để điều chỉnh những giá trị quá khứ có thể bị may mắn làm phồng lên.
5. Xếp hạng theo dự báo, chọn chiến lược, rồi **chỉ sau đó** mới nhìn lợi suất năm 1994 để đánh giá OOS.

Đường đi này cho thấy vì sao bảng tên và bảng lợi suất đều cần thiết: bảng tên giải thích **quy tắc**, bảng lợi suất cho **lịch sử và tương lai để kiểm tra**. Chỉ có CSV không tự tạo ra dự báo EB; cần tái hiện đúng mô hình và mốc thời gian trong paper.

## 9. Paper lấy dữ liệu từ đâu và đã biến nó thành gì?

Paper có **hai tầng dữ liệu**. Tầng đầu là dữ liệu của **từng cổ phiếu** để thiết kế và tính lợi suất của các quy tắc. Tầng thứ hai là dữ liệu **lợi suất của từng quy tắc theo tháng** để chạy EB. Folder hiện có **tầng thứ hai**; điều này đủ để bắt đầu tái hiện bước thống kê/EB, nhưng không đủ để tự dựng lại từng danh mục từ giao dịch của từng cổ phiếu.

| Nguồn ban đầu tác giả dùng | Cách tạo quy tắc | File trong `data_raw/` biểu diễn kết quả |
|---|---|---|
| Các biến kế toán doanh nghiệp, cùng dữ liệu thị trường cần thiết | Kết hợp biến thành tỷ số `X/Y` hoặc thay đổi `ΔX/lag(Y)`, xếp cổ phiếu theo kết quả rồi mua/bán hai nhóm cực trị | `DataMinedSignalList.csv` mô tả công thức; hai file `DataMinedLongShortReturnsEW/VW.csv` chứa lợi suất chiến lược |
| Lợi suất lịch sử **của từng cổ phiếu** | Chọn các quý trong 20 quý trước, tính mức sinh lời, độ biến động, độ lệch/độ nhọn rồi xếp cổ phiếu; thêm một số biến dựa trên 1–3 quý gần nhất | `PastReturnSignalNames.csv` và `PastReturnSignalsLongShort.csv` |
| Ký tự trong mã giao dịch và lợi suất từng cổ phiếu | Chia cổ phiếu thành 20 nhóm theo ký tự ticker, mua hai nhóm và bán hai nhóm khác; lặp cho vị trí ký tự 1–4 | `TickerSignalNames.csv` và `TickerSignalsLongShort.csv` |

**Lưu ý về nguồn:** trong paper, chiến lược kế toán được lấy từ bộ đã tạo bởi Chen, Lopez-Lira và Zimmermann; code công khai của Chen–Dim có script tự tạo chiến lược past-return/ticker từ dữ liệu cổ phiếu của tác giả. Các file đó dùng dữ liệu cấp cổ phiếu như CRSP, **không nằm trong repo này**. Bảng tên signal cho biết *quy tắc*, còn CSV lợi suất cho biết *quy tắc đã lãi/lỗ thế nào*. `nstock`, `nlong`, `nshort` là thông tin số cổ phiếu; riêng code xử lý past-return/ticker loại tháng có dưới hai cổ phiếu ở một chân danh mục. Các cột đếm **không phải dự báo EB**.

Ba nguồn nhân hai cách trọng số tạo **sáu họ chiến lược**:

| Họ trong code | Số chiến lược theo Bảng 1 | Lấy cột nào? |
|---|---:|---|
| `DataMinedLongShortReturnsEW` | 29.314 | `ret` của file accounting EW |
| `DataMinedLongShortReturnsVW` | 29.314 | `ret` của file accounting VW |
| `PastReturnSignalsLongShort_ew` | 19.402 | `ret_ew` của file past-return |
| `PastReturnSignalsLongShort_vw` | 19.402 | `ret_vw` của file past-return |
| `TickerSignalsLongShort_ew` | 19.380 | `ret_ew` của file ticker |
| `TickerSignalsLongShort_vw` | 19.380 | `ret_vw` của file ticker |

**Một dòng past-return hoặc ticker có hai lợi suất EW/VW**, nên tác giả tách thành hai họ trong code. Các `signalid` có thể trùng số giữa ba nguồn; luôn ghép với `signal_family`. Dữ liệu ticker giống nhóm đối chứng trực giác: thử đủ nhiều tổ hợp chữ cái cũng có thể sinh ra thành tích quá khứ đẹp, nhưng paper thấy nhóm này gần như không có khả năng dự báo tương lai.

## 10. Chính xác code đưa dữ liệu nào vào EB, tạo output nào?

Hãy giả sử ta **đứng ở cuối năm 2019** để chọn chiến lược cho **2020**. Các năm 2000–2019 là cửa sổ lịch sử tối đa 20 năm; năm 2020 được giữ riêng để chấm kết quả.

| Bước | Input | Việc xử lý | Output |
|---|---|---|---|
| 1. Chuẩn hóa | Bốn CSV lợi suất tháng | Đưa ngày về cùng tháng; tách EW/VW thành sáu họ; giữ `(signal_family, signalid, year, month, ret)` | Sáu bảng lợi suất tháng có khóa rõ ràng |
| 2. Thống kê cuốn chiếu | Lợi suất của từng signal trong tối đa 20 năm trước năm dự báo, **và** năm kế tiếp chỉ để đánh giá | Tính `mean_ret`, `tstat`, `nobs` riêng cho quá khứ (`s_flag=IS`) và năm kiểm tra (`s_flag=OOS`) | `OOS_signal_tstat_OosNyears1.csv.gzip` |
| 3. Ước lượng EB | **Chỉ các dòng IS** của một họ và một năm dự báo | Nhìn phân phối t-stat của cả họ; ước lượng hỗn hợp hai phân phối chuẩn; điều chỉnh t-stat từng chiến lược | `*_QML_FamilyYear.csv.gzip` chứa tham số họ–năm; `*_Predict_SignalYear.csv.gzip` chứa `thetahat`, `pred_ret` theo chiến lược–năm |
| 4. Chọn chiến lược | Dự báo EB hoặc t-stat quá khứ của cách naive | Đảo chiều nếu dự báo âm; xếp hạng theo Sharpe dự báo; lấy top 1%, 5%, 10% và cho các chiến lược được chọn trọng số bằng nhau | Danh sách và lợi suất **dự báo** của các danh mục |
| 5. Kiểm tra OOS | Danh sách đã chọn + lợi suất tháng **thực tế** năm 2020 | Tính lợi suất danh mục theo tháng, trung bình năm, t-stat, Sharpe, hình tích lũy | Bảng/hình của paper, gồm Bảng 2 |

`get_rolling_signal_stats.py` của tác giả chọn cửa sổ **tối đa 20 năm**, nhưng cho phép signal chỉ có **ít nhất 36 tháng** trong cửa sổ; năm OOS một năm cần **12 tháng**. Vì vậy không phải signal nào cũng có đủ 240 tháng IS. `MAX_LAGS=0` là cấu hình sai số chuẩn/t-stat của script này. Phần R `estimate_EB_distr.r` **không lấy `mean_ret` OOS để học mô hình**; OOS chỉ được dùng về sau để kiểm tra dự báo.

EB được ước lượng **riêng trong từng họ–năm**, rồi tác giả **so điểm dự báo của các chiến lược để chọn danh mục**. Đây là hai cấp khác nhau: “học từ các chiến lược tương tự” và “chọn chiến lược tốt nhất trong tập ứng viên”. Code `create_plots_from_est.py` nối dự báo với lợi suất năm sau theo `(signal_family, signalid, oos_begin_year)`; sau đó lấy lợi suất thực tế của từng tháng trong năm OOS.

Các từ dễ nhầm trong file trung gian:

| Cột | Có biết ở cuối 2019 không? | Nghĩa |
|---|---|---|
| `mean_ret` ở dòng `IS` | Có | Lợi suất trung bình *đã xảy ra* trong cửa sổ quá khứ |
| `tstat` ở dòng `IS` | Có | Mức thành tích quá khứ so với độ nhiễu ước lượng |
| `thetahat`, `pred_ret` | Có, sau khi ước lượng EB bằng IS | Thành tích/dự báo đã điều chỉnh; **không phải lợi suất thật** |
| `mean_ret` ở dòng `OOS` hoặc `ret_oos` | **Chưa** | Lợi suất *thực sự xảy ra* trong 2020 để chấm dự báo |

### Một trường hợp thật từ CSV

Signal kế toán `66` có công thức `diff(aco)/lag(aco)`. Ở bản **EW**, 240 tháng 2000–2019 cho trung bình `ret = −0,5492` điểm %/tháng và t-stat khoảng `−3,7610`. Năm 2020 có đủ 12 tháng; trung bình thực tế là `−1,1700` điểm %/tháng. Riêng bản **VW** của cùng signal có trung bình quá khứ `−0,2982` và trung bình thực tế 2020 `−0,9429` điểm %/tháng. Cùng một công thức nhưng hai cách cho trọng số tạo **hai chiến lược khác nhau**.

Đây là cách đọc: tại cuối 2019, code **được dùng** trung bình/t-stat quá khứ của signal 66 và của hàng chục nghìn signal kế toán EW khác. Code **chưa được dùng** `−1,1700` của năm 2020. Sau khi có dự báo EB và chọn danh mục, `−1,1700` mới xuất hiện ở bước kiểm tra. Không thể suy ra dự báo EB của signal 66 chỉ từ bốn con số trong ví dụ; cần ước lượng trên **toàn bộ họ kế toán EW** của năm đó.

## 11. Output cuối cùng và những dữ liệu không nằm trong `data_raw/`

Bảng 2 của PDF báo cáo giai đoạn 1983–2020: **EB top 1% đạt trung bình 5,70%/năm, Sharpe 1,46**; cách chọn naive top 1% là **5,72%/năm, Sharpe 1,45**. Đây là kết quả của **danh mục gồm nhiều chiến lược được chọn lại mỗi năm**, không phải lợi suất của một signal hay giá cổ phiếu. Nó cho thấy EB không thắng rõ rệt ở cách chọn top 1%; paper còn dùng EB để ước lượng *mức* hiệu quả tương lai và phân biệt họ nào chứa thông tin đáng tin.

Để tạo hàng so sánh **chiến lược đã công bố** trong Bảng 2, code còn đọc `PredictorLSretWide.csv` và `PredictorSummary.xlsx` từ bộ Chen–Zimmermann. **Hai file này không nằm trong `data_raw/`**, nên chỉ với bảy CSV hiện tại chưa tái tạo được hàng so sánh đó. Cũng chưa có các file trung gian `OOS_signal_tstat...`, `*_QML_FamilyYear...`, `*_Predict_SignalYear...` ở local; phải tạo chúng bằng code. Một số nhánh code công khai nhắc đến RavenPack/news và các file ngưỡng FDR/RW; chúng là **phân tích mở rộng/so sánh**, không phải ba nguồn chính tạo 136.192 chiến lược trong Bảng 1. FinCast embedding và file `data_sample/` **không phải input của paper gốc**.

Folder `data_raw/` có CSV kéo đến **2022**, nhưng bảng kết quả chính paper báo cáo **1983–2020**. Các tháng sau 2020 trong file **không được tự động tính vào Bảng 2**. ZIP gốc ở thư mục chính là bản lưu để đối chiếu, không phải một đầu vào thứ bảy cho EB.

## 12. Trạng thái kiểm tra file

- ZIP gốc `Strategy Returns.zip` ở thư mục chính đã qua kiểm tra CRC; bốn CSV mới được giải nén đầy đủ và giữ kích thước byte như các thành viên trong ZIP.
- Bốn bảng lợi suất đã được quét để đếm dòng, ID, khoảng ngày và ô trống; không thấy ô trống trong các cột của bốn bảng này.
- Phạm vi `ret` có những giá trị cực đoan. Đây là dấu hiệu **cần kiểm tra định nghĩa và cách xử lý dữ liệu khi phân tích**, không đủ để kết luận file hỏng.
- Bảng này là **dữ liệu đầu vào**, không phải code hoặc kết quả EB đã chạy lại. Các con số trong paper chỉ được coi là tái lập khi có cùng lựa chọn mẫu, mô hình và quy trình đánh giá.
