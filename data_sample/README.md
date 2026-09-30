# Các file mẫu nhỏ để xem dữ liệu

Thư mục này có **một file `_sample.csv` cho mỗi CSV trong `data_raw/`**. Mỗi file giữ nguyên tên cột và giữ nguyên giá trị của những dòng được chọn từ file gốc. Đây là **trích mẫu có thật**, không phải dữ liệu giả hay kết quả EB đã tính.

Các bảng tên signal và bảng lợi suất được lấy theo những `signalid` tương ứng, để bạn có thể tra tên/công thức rồi xem lợi suất của cùng quy tắc. File lợi suất mẫu lấy các năm **1983, 2000, 2019, 2020** khi signal có dữ liệu; riêng signal kế toán **66** còn lấy toàn bộ **2000–2020** để thấy rõ cửa sổ 20 năm quá khứ và năm kiểm tra tiếp theo. Không phải signal nào cũng tồn tại ở cả bốn năm.

| Muốn xem | File mẫu | Số dòng |
|---|---|---:|
| Công thức signal kế toán | `DataMinedSignalList_sample.csv` | 7 |
| Lợi suất kế toán EW | `DataMinedLongShortReturnsEW_sample.csv` | 408 |
| Lợi suất kế toán VW | `DataMinedLongShortReturnsVW_sample.csv` | 408 |
| Tên signal lợi suất quá khứ | `PastReturnSignalNames_sample.csv` | 6 |
| Lợi suất past-return, có cả EW/VW | `PastReturnSignalsLongShort_sample.csv` | 288 |
| Tên signal ticker | `TickerSignalNames_sample.csv` | 6 |
| Lợi suất ticker, có cả EW/VW | `TickerSignalsLongShort_sample.csv` | 288 |

Mỗi file mẫu mới dưới **18 KB**, nên có thể mở trực tiếp trong trình xem văn bản hoặc bảng tính.

`DataMinedLongShortReturnsSample.csv` là **file mẫu có sẵn từ nguồn trước đó**, gồm 10 signal kế toán đầu tiên; mình giữ nguyên để bạn đối chiếu. Nó không thay thế hai mẫu EW/VW được trích trực tiếp từ các file gốc hiện tại.

Để tạo lại bảy mẫu `_sample.csv` từ `data_raw/`, chạy `python Research/create_data_samples.py` từ thư mục chính. Muốn phân tích hoặc tái lập kết quả paper, hãy dùng **file đầy đủ trong `data_raw/`**; mẫu nhỏ chỉ để đọc cấu trúc và thử code.
