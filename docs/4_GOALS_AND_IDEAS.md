# Mục tiêu và ý tưởng nghiên cứu

## 1. Ý tưởng của paper gốc

Chen và Dim, *High-Throughput Asset Pricing* (arXiv 2311.10685, PDF ở `docs/papers/`):

- Dựng khoảng **136 nghìn chiến lược long-short** từ ba nguồn: tỉ số kế toán, lợi suất quá khứ và ký tự mã cổ phiếu (ticker). Mỗi nguồn có hai cách chia tiền (EW, VW), thành 6 họ.
- Mỗi năm Y (1983–2020), dùng 20 năm trước Y để tính t-stat của từng chiến lược, rồi dùng **Empirical Bayes (EB)** co các t-stat quá đẹp về mức hợp lý: phân phối tiên nghiệm là hỗn hợp hai phân phối chuẩn, ước lượng riêng cho từng họ-năm.
- Dự báo lợi suất năm sau `pred_ret = θ̂ · m / t`, chọn top 1%, 5%, 10% chiến lược theo Sharpe dự báo trong tập gộp 6 họ, chia đều tiền, rồi đo lợi suất thật.
- Kết quả: top 1% đạt 5,70%/năm, Sharpe 1,46 (Table 2); dự báo của EB không chệch (Hình 4, 5); khả năng dự báo tập trung ở **chiến lược kế toán EW, cổ phiếu nhỏ, và giai đoạn trước 2004**; Ticker gần như không có tín hiệu (Table 3 Panel A: Accounting EW 91,0%, Past Return EW 8,6%, Ticker 0,0% của top 1%).

## 2. Mục tiêu (target) của nghiên cứu này

**Đại lượng dự báo:** giống EB và notebook tham khảo, là `pred_ret`, lợi suất trung bình tháng của từng chiến lược trong năm sau, chỉ dùng dữ liệu trước năm đó.

**Mô hình:** kiến trúc học sâu của notebook tham khảo (encoder chuỗi thời gian → sequence → attention → FC), được cải tiến từng khối. Không dùng đầu ra EB làm đầu vào hay điểm neo.

**Dữ liệu học:** chỉ các nguồn **có tính thống kê**, là kế toán và lợi suất quá khứ (4 họ). Ticker không dùng để học.

**Thế nào là đạt (SOTA) trên một tập họ:** chọn cấu hình chỉ bằng dev 1983–2004, rồi trên cùng tập họ đó:

1. **Hơn mô hình gốc** (cấu hình của notebook tham khảo) theo cặp seed.
2. **Hơn EB** ở ít nhất một trong hai thước đo chính, có kiểm định:
   - độ chính xác dự báo (MSE thấp hơn, Diebold–Mariano theo năm, p < 0,05), hoặc
   - danh mục top-k (Sharpe cao hơn; kiểm định chính là HAC kiểu Ledoit–Wolf và bootstrap theo khối năm ≥ 10.000 lần, cùng p < 0,05; Jobson–Korkie–Memmel báo kèm vì giả định lợi suất độc lập), kèm báo cáo lợi suất thô và hiệu chỉnh cho số cấu hình đã thử (bootstrap max-statistic theo khối năm, `code/experiments/old_code/verify/maxstat.py`).
3. **Không đảo chiều ở holdout 2005–2020** (holdout chỉ dùng để báo cáo, không để chọn), và nêu rõ số cấu hình đã thử cùng hiệu chỉnh nhiều phép thử.

Nếu chỉ ngang EB thì báo là ngang, không gọi là SOTA.

## 3. Ý tưởng

- **Vì sao chỉ học từ nguồn có tính thống kê:** ở hướng thử trước (học cả 6 họ), mô hình thua EB nặng nhất ở họ Ticker, vì Ticker là nhiễu thuần mà EB co về 0 gần như hoàn toàn. Học trên nhiễu làm mạng phân tán năng lực. Kế toán và lợi suất quá khứ là nơi paper tìm thấy tín hiệu thật.
- **Thử mọi tập huấn luyện** (1, 2, 3 hoặc 4 họ): một họ thì dữ liệu đồng nhất, mạng dễ nắm cấu trúc thống kê của chuỗi; nhiều họ thì có thêm dữ liệu và có thể học điểm chung. Thí nghiệm sẽ cho biết bên nào thắng.
- **Tinh chỉnh từng khối một và tìm nguyên nhân trước khi đổi:** mỗi khối (dữ liệu, tiền xử lý, encoder, attention, loss, quy trình huấn luyện) được truy vết đầu vào và đầu ra (`model.FinalNet(trace=...)` in shape đầu ra của từng khối), so theo cặp seed, rồi mới đổi tiếp.
- **Dựa trên paper:** HyperAttention (attention gần tuyến tính để nhìn cả họ-năm), Holen và cộng sự (gộp gradient có trọng số theo loss hoặc reward), các mô hình nền chuỗi thời gian (FinCast, Chronos-2, TiRex), và các phương pháp tìm thêm qua MCP arXiv.
- **Bước kiểm tra cuối:** mô hình học trên kế toán và lợi suất quá khứ được dùng để **suy luận trên Ticker** mà không huấn luyện trên đó. Nếu mô hình thật sự nắm được tính thống kê, nó phải dự báo Ticker gần 0 và hầu như không chọn chiến lược Ticker vào top-k, giống kết quả của paper.

## 4. Đánh giá

- Walk-forward: với mỗi năm Y, huấn luyện trên các năm đến Y−2, dừng sớm theo Y−1, dự báo Y; học lại mỗi năm. Chuẩn hóa chỉ bằng thống kê tập huấn luyện.
- Chấm trên **toàn bộ** chiến lược của họ Accounting EW (khoảng 26.000 mỗi năm), cùng luật chọn top-k với EB.
- Mỗi cấu hình thử 3 seed; mô hình chốt 6 seed. Dự báo cuối là trung bình các seed.
- Kiến trúc, kết quả và nhật ký thử nghiệm: [5_FINAL_MODEL_VS_BASELINE.md](5_FINAL_MODEL_VS_BASELINE.md). Baseline: [1_BASELINE.md](1_BASELINE.md). Notebook chạy toàn bộ: `code/notebook/final_model.ipynb`.

## 5. Nguyên tắc

Không rò rỉ thời gian; chọn cấu hình chỉ bằng dev; ghi lại mọi cấu hình kể cả thất bại; báo cáo trung thực cả điểm hơn lẫn điểm kém; trích dẫn chỉ sau khi xác minh. Chi tiết trong [AGENTS.md](../AGENTS.md).
