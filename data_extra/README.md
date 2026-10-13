# Dữ liệu bổ sung

Folder này giữ **3 CSV cần đọc** để so sánh các chiến lược data-mined của paper với những predictor đã công bố. Nguồn là [trang dữ liệu của Chen–Zimmermann tại Federal Reserve](https://www.federalreserve.gov/econres/feds/open-source-cross-sectional-asset-pricing.htm). Đây không phải 3 nguồn dữ liệu độc lập: **hai CSV mô tả bên dưới là hai sheet của cùng một XLSX gốc**.

| File trong folder | Nguồn tải | Dùng để làm gì? |
|---|---|---|
| [`PredictorLSretWide-feds-2021037.csv`](PredictorLSretWide-feds-2021037.csv) | [CSV lợi suất gốc của Fed](https://www.federalreserve.gov/econres/feds/files/PredictorLSretWide-feds-2021037.csv) | Lợi suất tháng của 205 predictor đã công bố, để so với kết quả chiến lược được paper chọn bằng EB. Có 1.140 tháng; cột `date` và 205 cột predictor. |
| [`signaldocumentation-BasicInfo.csv`](signaldocumentation-BasicInfo.csv) | Sheet `BasicInfo` của [XLSX mô tả gốc](https://www.federalreserve.gov/econres/feds/files/signaldocumentation-feds-2021037.xlsx) | Tên predictor (`Acronym`), tác giả, năm công bố (`Year`), mô tả và năm kết thúc mẫu (`SampleEndYear`). 324 dòng dữ liệu. |
| [`signaldocumentation-AddInfo.csv`](signaldocumentation-AddInfo.csv) | Sheet `AddInfo` của cùng [XLSX mô tả gốc](https://www.federalreserve.gov/econres/feds/files/signaldocumentation-feds-2021037.xlsx) | Chi tiết phương pháp tạo predictor, kết quả lợi suất/t-stat và quy tắc danh mục. 324 dòng dữ liệu. |

**Vì sao không giữ XLSX?** File XLSX gốc hiển thị trắng trong Office Viewer của VS Code ở máy này. Hai CSV ở trên đã được xuất từ hai sheet của nó để mở trực tiếp; giữ thêm XLSX trong folder sẽ trùng nội dung. Có thể tải lại bản nguyên gốc từ link Fed bất cứ lúc nào.

`NA` trong CSV lợi suất nghĩa là predictor đó **chưa có quan sát ở tháng ấy**, không phải file rỗng. Tháng đầu 1926-01-30 chỉ có 2/205 predictor có số; tháng 2000-07-31 cả 205 predictor đều có số.

**Lưu ý khi dùng code paper:** [`create_plots_from_est.py`](../Research/code_goc_paper/high-throughput-ap/create_plots_from_est.py) tìm `PredictorLSretWide.csv` và `PredictorSummary.xlsx`; bản Fed ở đây **khác tên/định dạng** và bảng mô tả **không có cột `signalname`** mà code yêu cầu. Cần đối chiếu phiên bản/ánh xạ tên predictor trước khi tái lập bảng so sánh; đổi tên file đơn thuần sẽ không đủ.

Dữ liệu lợi suất của **136.192 chiến lược chính** đã ở [`data_raw/`](../data_raw/), nên không tải trùng vào đây. Dữ liệu cổ phiếu cấp thấp hơn để tính lại các chiến lược phải lấy qua [WRDS/CRSP](https://wrds-www.wharton.upenn.edu/pages/get-data/center-research-security-prices-crsp/) nếu có quyền truy cập; folder này không có dữ liệu CRSP. Các file dự báo EB là **output do code tạo**, không phải dữ liệu bổ sung cần tải.
