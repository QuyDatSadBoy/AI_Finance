# Paper gốc làm gì? Giải thích bằng ngôn ngữ thường

Tài liệu này nói về **High-Throughput Asset Pricing của Chen và Dim**, không nói về mô hình FinCast của nhóm. Đọc file này trước [bản giải thích chi tiết](GIAI_THICH_TU_CO_BAN_DEN_NANG_CAO.md).

## 1. Một câu chuyện đơn giản

Giả sử bạn có 100.000 **cách chọn cổ phiếu**. Mỗi cách là một quy tắc, chẳng hạn: “mua nhóm cổ phiếu có đặc điểm X cao, bán khống nhóm có X thấp”. Bạn nhìn lại quá khứ và thấy 1.000 quy tắc từng kiếm tiền rất tốt.

Nhưng có một vấn đề: **nếu thử đủ nhiều quy tắc, chắc chắn sẽ có vài quy tắc trông xuất sắc chỉ do may mắn**. Chọn đúng những quy tắc thắng nhất trong quá khứ thường cũng là chọn những quy tắc đã gặp may nhất. Khi áp dụng vào năm tới, phần may mắn đó khó lặp lại.

Paper hỏi: **trong rất nhiều quy tắc, quy tắc nào đáng tin hơn, và nó có thể sinh lời bao nhiêu trong năm tiếp theo?**

Tác dụng của paper là đưa ra một cách **đánh giá và chọn chiến lược thận trọng hơn khi đã thử rất nhiều chiến lược**. Đây là công cụ nghiên cứu chiến lược; không phải ứng dụng đưa ra mã cổ phiếu nên mua ngày mai, và không bảo đảm lợi nhuận sau chi phí giao dịch.

## 2. Một quy tắc tạo ra dữ liệu như thế nào?

Ví dụ minh họa: mỗi tháng xếp các cổ phiếu theo một đặc điểm kế toán. Mua nhóm có giá trị cao, bán khống nhóm có giá trị thấp. Cuối tháng, lấy kết quả của nhóm mua trừ kết quả của nhóm bán. Ta có **một con số lãi/lỗ của quy tắc trong tháng đó**.

Lặp lại nhiều tháng sẽ có một dãy như: “tháng 1 lãi, tháng 2 lỗ, tháng 3 lãi…”. Một quy tắc như vậy gọi là **signal**; khi biến nó thành danh mục mua–bán theo quy trình cố định, ta có một **chiến lược**. Paper phân tích chuỗi lợi suất của *chiến lược*, không trực tiếp dự báo giá của từng cổ phiếu.

Với cùng một quy tắc, tác giả tính hai bản danh mục: **EW** cho các cổ phiếu trọng số bằng nhau; **VW** cho trọng số theo vốn hóa. Vì vậy một signal có thể cho hai chiến lược với hai dãy lợi suất khác nhau.

## 3. Input: paper nhận những gì?

**Đầu vào chính của bước đánh giá là lịch sử lợi suất hằng tháng của rất nhiều chiến lược**, kèm nhãn cho biết mỗi chiến lược thuộc nhóm nào. Với các file đang có trong repo, ba nhóm là:

| Nhóm | Quy tắc dựa trên gì? | File lợi suất trong `data/` |
|---|---|---|
| Kế toán | Các công thức từ dữ liệu kế toán doanh nghiệp | `DataMinedLongShortReturnsEW.csv`, `DataMinedLongShortReturnsVW.csv` |
| Lợi suất quá khứ | Mẫu hình lợi suất/biến động ở các quý trước | `PastReturnSignalsLongShort.csv` |
| Ticker | Các ký tự trong mã cổ phiếu | `TickerSignalsLongShort.csv` |

Hai bảng tên `PastReturnSignalNames.csv`, `TickerSignalNames.csv` và `DataMinedSignalList.csv` giúp biết ID nào ứng với quy tắc nào. Bảng lợi suất cho biết **quy tắc đó lãi/lỗ bao nhiêu trong từng tháng**. Đây là dữ liệu *đã xây thành chiến lược*, không phải giá cổ phiếu hay báo cáo tài chính thô của từng doanh nghiệp.

Ba nhóm nhân với hai cách EW/VW tạo thành **sáu họ chiến lược**. Paper có **136.192 chiến lược** trong bảng mô tả dữ liệu. Tập hiện có đủ ba nhóm lợi suất này, nhưng chỉ có CSV không có nghĩa toàn bộ các ước lượng và bảng kết quả của paper đã được tính sẵn.

## 4. Paper xử lý input ra sao?

Hãy đứng ở **cuối năm 1999** và giả vờ chưa biết điều gì xảy ra trong năm 2000.

1. Xem khoảng **20 năm lợi suất đã xảy ra**, đến hết 1999, của các chiến lược.
2. Tính xem từng chiến lược trước đây có vẻ lãi tốt đến đâu. Đây là **thành tích quan sát**, gồm cả khả năng thật lẫn may mắn.
3. Dùng thông tin từ cả họ chiến lược để điều chỉnh những thành tích quá cực đoan. Nếu một chiến lược nổi bật giữa một họ đa phần giống nhiễu, paper **giảm mức dự báo** cho nó. Cách điều chỉnh này là **Empirical Bayes (EB)**.
4. Với mỗi chiến lược, tạo **dự báo** về hiệu quả tương lai. Nếu dự báo mang dấu âm, có thể đảo chiều mua/bán khi lập danh mục. Sau đó xếp hạng và lấy nhóm đứng đầu, ví dụ top 1%.
5. Bây giờ mới mở dữ liệu **năm 2000** để xem các dự báo và danh mục đã chọn đúng đến đâu.
6. Dịch sang cuối năm 2000, dự báo 2001, rồi lặp lại. Paper đánh giá các năm **1983–2020**.

Ví dụ số để hiểu bước 3, **không phải con số đo từ paper**: một chiến lược đã lãi trung bình 12%/năm trong quá khứ. Nếu nhiều chiến lược tương tự cũng có kết quả cực đoan nhưng không bền, EB có thể dự báo tương lai của chiến lược này thấp hơn 12%. Nó không đơn giản lấy nguyên 12% làm lời hứa cho năm tới.

**Quy tắc quan trọng:** để dự báo năm 2000, mọi phép tính chọn chiến lược phải dùng thông tin có đến cuối 1999. Nếu dùng lợi suất 2000 trong lúc chọn, kết quả kiểm tra sẽ không còn là dự báo thật.

## 5. Output: paper trả về những gì?

Có **ba cấp đầu ra**, dễ bị trộn lẫn:

| Cấp | Output là gì? | Dùng để trả lời câu hỏi nào? |
|---|---|---|
| Từng chiến lược, từng năm | Điểm/ước lượng hiệu quả tương lai đã được EB điều chỉnh | “Chiến lược này có vẻ đáng tin đến mức nào?” |
| Danh sách được chọn | Top 1%, 5% hoặc 10% chiến lược theo điểm dự báo, cùng chiều mua–bán | “Nếu phải chọn vào cuối năm, chọn nhóm nào?” |
| Kết quả đánh giá | Lợi suất thực tế của danh mục trong các năm sau, cùng thước đo rủi ro | “Cách chọn đó đã hoạt động ra sao ngoài mẫu?” |

Paper còn so với cách **naive**: nhìn thống kê quá khứ rồi chọn nhóm tốt nhất **mà không điều chỉnh như EB**. Kết quả nổi bật của paper: trên giai đoạn 1983–2020, danh mục **EB top 1% đạt trung bình 5,70%/năm, Sharpe 1,46**; cách naive top 1% đạt **5,72%/năm, Sharpe 1,45**. Hai kết quả rất gần nhau. Vì thế kết luận đúng **không phải** “EB thắng rõ ràng ở top 1%”. EB có ích đặc biệt ở việc cho **dự báo mức hiệu quả đã bớt lạc quan** và giúp hiểu nhóm nào chứa tín hiệu đáng tin hơn.

## 6. Bây giờ mới đến thuật ngữ

- **Signal:** công thức/quy tắc xếp hạng cổ phiếu. Ví dụ: một tỷ số kế toán.
- **Long–short:** mua một nhóm và bán khống nhóm khác; lợi suất được đo bằng chênh lệch hai nhóm.
- **In-sample (IS):** quá khứ được phép xem khi lập dự báo.
- **Out-of-sample (OOS):** giai đoạn tương lai chỉ mở ra *sau* khi đã dự báo/chọn; dùng để kiểm tra.
- **T-stat:** một cách đo lợi suất trung bình quan sát lớn đến đâu so với mức nhiễu ước lượng. T-stat cao không tự chứng minh chiến lược có khả năng thật khi đã thử rất nhiều chiến lược.
- **Empirical Bayes:** dùng kết quả của cả họ chiến lược để ước lượng mức nhiễu và điều chỉnh dự báo của từng chiến lược. Có thể hình dung là “bớt tin vào các thành tích quá đẹp”.
- **Sharpe:** lợi suất trung bình so với độ biến động; dùng để so hiệu quả trên một đơn vị rủi ro biến động.
- **High-throughput:** xây và kiểm tra rất nhiều ứng viên bằng cùng một quy trình, thay vì chỉ chọn vài quy tắc nổi tiếng.

## 7. Đừng nhầm paper gốc với notebook FinCast

**Paper gốc:** lấy lịch sử lợi suất, dùng EB để dự báo và chọn chiến lược, rồi kiểm tra năm sau. **Notebook FinCast của nhóm:** thử một cách dự báo khác bằng mô hình học sâu trên lịch sử lợi suất/embedding, sau đó so với EB. FinCast không phải thành phần của phương pháp Chen–Dim.

Nếu chỉ cần nhớ một câu để kể lại: **“Paper thử rất nhiều cách đầu tư, dùng thống kê để bớt tin vào các cách từng thắng nhờ may mắn, rồi kiểm tra xem các cách được chọn có còn hiệu quả trong năm tiếp theo không.”**
