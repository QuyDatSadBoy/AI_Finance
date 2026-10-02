# Bắt đầu từ đây: paper này làm gì?

## Ý chính trong một câu

Theo [bản thảo phương pháp hiện có](../../docs-add/Asset%20pricing/0.%20Paper_Draft_Sections_1-3.docx), nhóm muốn **dự báo một chiến lược đầu tư từng hiệu quả trong quá khứ có còn hiệu quả trong tương lai không**, bằng FinCast + Bi-LSTM + Attention.

Đây là **hướng đề xuất rõ nhất trong folder**, chưa chắc là hướng nhóm đã chốt. Folder hiện là kho tài liệu; chưa có code, dữ liệu gốc hoặc kết quả chạy mô hình FinCast.

## Hiểu bài toán qua một ví dụ

Giả sử một quy tắc nói: “Mỗi tháng, mua nhóm cổ phiếu có đặc điểm X cao và bán khống nhóm có X thấp.” Quy tắc đó gọi là **signal**; biến nó thành danh mục mua–bán gọi là **chiến lược long–short**. Chiến lược tạo ra một dãy lợi suất theo tháng.

Nó lãi tốt trong quá khứ vẫn có thể chỉ là may mắn, nhất là khi người ta đã thử hàng nghìn quy tắc. Paper hỏi: **nhìn vào lịch sử của từng chiến lược, ta dự báo được lợi suất trung bình của nó trong kỳ tiếp theo tốt đến đâu?**

| Một mẫu dữ liệu | Ví dụ minh họa |
|---|---|
| Đầu vào | Lợi suất tháng của chiến lược đến hết 12/2020; mean return và t-stat tính từ chính giai đoạn quá khứ ấy |
| Đầu ra cần dự báo | Lợi suất trung bình của chiến lược trong năm 2021 |

Mốc thời gian chỉ để minh họa; độ dài cửa sổ thật chưa được chốt.

## Mô hình dự kiến hoạt động thế nào?

`Lịch sử lợi suất → FinCast → Bi-LSTM → Attention → ghép mean và t-stat → dự báo lợi suất tương lai`

- **FinCast:** mô hình chuỗi thời gian đã được huấn luyện trước, dùng để tạo biểu diễn từ lịch sử lợi suất. Bản thảo định giữ nguyên trọng số của nó.
- **Bi-LSTM:** học thêm đặc điểm của chuỗi phục vụ đúng nhiệm vụ dự báo hiệu quả chiến lược.
- **Attention:** học cách gán trọng số cho các phần khác nhau trong lịch sử. Ở đây nó chú ý **theo thời gian của một chiến lược**, không phải giữa các cổ phiếu.
- **Mean return, t-stat:** hai con số truyền thống tóm tắt mức lợi suất và độ tin cậy thống kê trong quá khứ.
- **Đầu ra:** một số dự báo lợi suất trung bình tương lai của chiến lược.

## Phân biệt ba thứ dễ bị lẫn

1. **Asset pricing cổ điển:** giải thích vì sao tài sản có lợi suất kỳ vọng khác nhau, chẳng hạn qua các nhân tố rủi ro.
2. **High-Throughput Asset Pricing:** paper tham khảo xét khoảng 136.000 chiến lược và dùng Empirical Bayes để điều chỉnh thành tích quá khứ bị thổi phồng do lựa chọn và nhiễu.
3. **Bản thảo của nhóm:** đề xuất dự báo trực tiếp hiệu quả tương lai của **từng chiến lược** từ chuỗi lợi suất của nó. Ghép mean/t-stat với mạng học sâu **không tự động trở thành Empirical Bayes**.

## Hiện đã có gì?

- [Bản thảo Sections 1–3](../../docs-add/Asset%20pricing/0.%20Paper_Draft_Sections_1-3.docx) mô tả câu hỏi, literature và pipeline; vẫn còn chỗ trống về nguồn dữ liệu, độ dài cửa sổ, cách train và đánh giá.
- [Slide LSTM–Bayes](../../docs-add/Asset%20pricing/Slide/LSTM-Bayes%20for%20asset%20pricing.pptx) có bảng kết quả của **một hướng thử nghiệm khác**. Đó chưa phải kết quả của pipeline FinCast trong bản thảo.
- Các PDF là tài liệu nền và paper tham khảo. Chưa thấy mã nguồn hoặc dữ liệu thực nghiệm trong folder để tái lập kết quả.

## Đọc tiếp theo thứ tự nào?

1. Đọc file này để nắm câu hỏi.
2. Xem [slide High-Throughput](../../docs-add/Asset%20pricing/Slide/High-throughput%20asset%20pricing.pptx) để hiểu signal, long–short và vì sao thành tích quá khứ dễ đánh lừa.
3. Đọc Introduction và Methodology trong [bản thảo Sections 1–3](../../docs-add/Asset%20pricing/0.%20Paper_Draft_Sections_1-3.docx).
4. Khi cần chi tiết về toàn bộ file, thuật ngữ và các điểm chưa chốt, mở [bản giải thích đầy đủ](../paper%20ch%C3%ADnh/GIAI_THICH_TU_CO_BAN_DEN_NANG_CAO.md).

**Một câu để nói lại với người khác:** “Nhóm đang đề xuất mô hình dùng lịch sử lợi suất của các chiến lược long–short để dự báo chiến lược nào còn hiệu quả ngoài mẫu; kiến trúc FinCast–Bi-LSTM–Attention đã được viết trong bản thảo, còn thiết kế thực nghiệm và kết quả FinCast chưa hoàn thiện trong folder.”
