# Output minh họa — có số giả lập, không phải kết quả tái lập paper

Sáu CSV dùng **đúng tên cột trong code gốc**. Đuôi để `.csv` cho dễ mở; script gốc dùng `.csv.gzip` cho file thống kê/dự báo. Ví dụ chỉ có **hai signal kế toán EW: 66 và 67**, không phải toàn bộ dữ liệu paper. Tạo lại bằng `python Research/output_minh_hoa/tao_output_minh_hoa.py`.

| File | Số thật hay giả? | Tác dụng |
|---|---|---|
| [OOS_signal_tstat_OosNyears1.csv](OOS_signal_tstat_OosNyears1.csv) | Tính từ CSV thật, với sai số chuẩn HAC lag 0 như cấu hình code | Thống kê quá khứ IS 2000–2019 và kết quả OOS 2020. |
| [MINH_HOA_Predict_SignalYear.csv](MINH_HOA_Predict_SignalYear.csv) | **thetahat giả lập**; pred_ret tính từ thetahat theo công thức repo. Các cột lịch sử là thật. | Thấy chính xác một dòng dự báo signal–năm trông như thế nào. |
| [MINH_HOA_df_best_ret.csv](MINH_HOA_df_best_ret.csv) | **Lựa chọn danh mục giả lập**; lợi suất tháng lấy thật từ signal 66 rồi đảo dấu | Bản xuất minh họa của bảng `df_best_ret` trong bộ nhớ code. Repo không tự lưu bảng này với tên file trên. |
| [MINH_HOA_signal_dic.csv](MINH_HOA_signal_dic.csv) | Lợi suất tháng thật | 504 dòng: 2 signal × 252 tháng, tương ứng bảng input đã chuẩn hóa. |
| [MINH_HOA_df_signed.csv](MINH_HOA_df_signed.csv) | Tính theo code từ dự báo giả và lợi suất thật | 24 dòng: 2 signal × 12 tháng, có điểm xếp hạng và kết quả sau đảo chiều. |
| [MINH_HOA_tabdat.csv](MINH_HOA_tabdat.csv) | Kết quả thật dưới lựa chọn minh họa | Bảng tổng kết EB top 1% trên ví dụ: ret_mean=14,039420%/năm quy đổi, không phải kết quả paper. |

Ví dụ đặt thetahat của ID 66 là **−2,5**. Code tính `pred_ret = thetahat × mean_ret / tstat`, nên pred_ret = **−0,365063%/tháng**. Dự báo âm → đảo chiều → kỳ vọng sau đảo chiều là **+0,365063%/tháng**. Tháng 1/2020 của ID 66 thật là −6,737422%, sau đảo chiều thành **+6,737422%**.

Trong ví dụ hai signal, ID 66 có điểm Sharpe dự báo cao hơn sau đảo chiều và đứng đầu. Bảng tháng vì vậy có nstrat=1. Ngưỡng top 1% của code (rank_pct≥99) chọn ID 66 trong ví dụ nhỏ; **không đại diện việc chọn top 1% trên toàn bộ chiến lược paper**. Không dùng sáu file này làm kết quả tái lập paper. Các CSV có tên biến nội bộ là bản xuất thêm để xem, không phải tên file mà repo tự ghi ra đĩa.
