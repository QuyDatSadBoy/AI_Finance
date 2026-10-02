# Idea nghiên cứu: dùng AI dự báo lợi suất chiến lược thay cho bước EB

**Ý tưởng trong một câu:** giữ bài toán và dữ liệu chiến lược của Chen & Dim, nhưng thử dùng mô hình AI học từ lịch sử lợi suất để dự báo năm tiếp theo, rồi kiểm tra xem dự báo và việc chọn chiến lược có tốt hơn phương pháp gốc không.

Đây là bản mô tả hướng nghiên cứu dựa trên paper và hai notebook hiện có, chưa phải kết luận rằng AI đã tốt hơn EB.

## 1. Paper gốc là gì?

**High-Throughput Asset Pricing — Andrew Y. Chen & Chukwuma Dim.** Bản PDF đang dùng là bản tháng 6/2025, arXiv v3.

- [Trang paper](https://arxiv.org/abs/2311.10685) · [PDF đúng phiên bản](https://arxiv.org/pdf/2311.10685v3) · [PDF trong folder docs](High-Throughput%20Asset%20Pricing%20-%20Chen%20%26%20Dim.pdf).
- [Code công khai của tác giả](https://github.com/chenandrewy/high-throughput-ap) đã clone tại [Research/code_goc_paper/high-throughput-ap/](../Research/code_goc_paper/high-throughput-ap/).

**Câu hỏi của họ:** khi thử rất nhiều chiến lược, một số sẽ có thành tích quá khứ đẹp chỉ vì may mắn. Ta có thể dự báo mức lợi suất hợp lý, chọn chiến lược có hiệu quả năm sau và xác định nguồn signal nào có thông tin dự báo không?

Một **signal** là quy tắc xếp hạng cổ phiếu. Từ đó tạo chiến lược mua một nhóm, bán khống nhóm khác; lợi suất long–short là lợi suất nhóm mua trừ nhóm bán. Paper nghiên cứu khoảng **136.000 chiến lược**, thuộc ba nguồn: dữ liệu kế toán, lợi suất quá khứ và chữ trong mã ticker. Mỗi nguồn có cách tính EW và VW, tạo thành sáu nhóm.

## 2. Paper gốc làm như thế nào?

| Bước | Làm gì, hiểu bằng lời | Code chính |
|---|---|---|
| 1. Đọc lịch sử | Input là lợi suất tháng đã tính sẵn của từng chiến lược, không phải dự đoán giá từng cổ phiếu. | `get_rolling_signal_stats.py` |
| 2. Tóm tắt quá khứ | Mỗi cuối năm lấy tối đa 20 năm trước đó; tính trung bình và tstat riêng cho từng ID. tstat đo mức trung bình khác 0 rõ đến đâu so với sai số ước lượng. | `get_rolling_signal_stats.py` |
| 3. Dự báo bằng EB | Học phân phối tstat của cả nhóm để điều chỉnh đánh giá từng chiến lược, rồi chuyển thành lợi suất tháng dự báo. | `estimate_EB_distr.r` |
| 4. Chọn chiến lược | Đảo chiều chiến lược có dự báo âm, xếp hạng theo Sharpe dự báo và chọn top 1%/5%/10%. | `create_plots_from_est.py` |
| 5. Kiểm tra tương lai | Ghép lợi suất thật của năm sau, tạo danh mục và tổng hợp kết quả qua nhiều năm. | `create_plots_from_est.py` |

**Ví dụ thời gian:** đứng cuối 2019, dùng dữ liệu 2000–2019 để dự báo 2020. Các số thực tế của 2020 chỉ dùng để kiểm tra sau. Dữ liệu quá khứ gọi là IS; dữ liệu kiểm tra tương lai gọi là OOS.

Bảy CSV trong [data_raw/](../data_raw/) đủ dữ liệu chiến lược cho ba nguồn chính: bốn bảng lợi suất và ba bảng tra tên/quy tắc. Muốn tái lập mọi phần so sánh trong paper còn cần dữ liệu benchmark và tạo các kết quả trung gian. Repo gốc cần chỉnh đường dẫn/đuôi CSV và bỏ nguồn RavenPack nếu chỉ chạy ba nguồn hiện có.

## 3. Paper gốc đã tìm thấy gì?

Các con số dưới đây thuộc **paper gốc**, không phải kết quả AI của mình.

| Kết quả hoặc lập luận | Bằng chứng trong paper | Hiểu đúng |
|---|---|---|
| Chọn từ quá khứ vẫn tạo được lợi suất ngoài mẫu. | Bảng 2: top 1% EB đạt **5,70%/năm**, Sharpe **1,46**, giai đoạn 1983–2020. | Có bằng chứng về khả năng dự báo trong mẫu nghiên cứu; không phải cam kết lợi nhuận tương lai. |
| EB không nhất thiết chọn chiến lược tốt hơn cách đơn giản. | Bảng 2: top 1% naive đạt **5,72%/năm**, gần EB. Proposition 1 giải thích điều kiện để lựa chọn hai cách trùng nhau. | Giá trị của EB còn nằm ở ước lượng mức lợi suất nên kỳ vọng; lý thuyết có điều kiện. |
| Thông tin dự báo tập trung ở kế toán EW. | Bảng 3: **91,0%** chiến lược trong top 1% thuộc kế toán EW; ticker EW/VW đều **0%**. | Không có nghĩa mọi signal kế toán đều tốt, hay chỉ riêng tỷ lệ này đã chứng minh ticker vô dụng. |
| Ticker không cho thấy khả năng dự báo rõ như kế toán. | Hình 2–5: phân bố tstat và dự báo/kết quả theo từng nhóm. | Thành tích ticker nổi bật phù hợp với nhiễu; không chứng minh tuyệt đối mọi signal ticker chỉ do may mắn. |
| Độ chính xác của EB thay đổi theo thời gian. | Hình 4–5: dự báo sát hơn trước 2004; hiệu quả giảm và một số dự báo quá cao sau 2004. | Đây là động lực để thử mô hình thích nghi tốt hơn, chưa chứng minh AI sẽ giải quyết được. |
| Ngưỡng thống kê quá chặt có thể bỏ sót chiến lược hữu ích. | Mục 5/Hình 6 so các cách kiểm soát phát hiện sai với kết quả OOS. | Không nên chỉ dùng một ngưỡng tstat để kết luận có/không có khả năng dự báo. |

**Một ý quan trọng: paper giúp phân biệt nguồn signal có thông tin với thành tích có thể chỉ do may mắn.** Signal ticker được tạo từ chữ trong mã cổ phiếu, thay vì thông tin kinh tế như dữ liệu kế toán. Vì thế nó đóng vai trò đối chiếu hữu ích: thử rất nhiều quy tắc ticker vẫn có thể tìm ra vài chiến lược có quá khứ đẹp, nhưng liệu chúng có dự báo được tương lai không?

Trong **top 1% chiến lược được chọn bằng EB**, ticker EW và VW đều chiếm **0,0%**, còn kế toán EW chiếm **91,0%** (Bảng 3). Kết hợp với phân bố tstat gần mô hình nhiễu và dự báo ticker gần 0 trong Hình 2–5, kết quả này **ủng hộ cách hiểu rằng thành tích nổi bật của ticker trong quá khứ chủ yếu có thể được giải thích bằng nhiễu/may mắn khi thử nhiều chiến lược**. Trái lại, nhóm kế toán, đặc biệt EW, cho thấy thông tin dự báo và hiệu quả ngoài mẫu rõ hơn.

**Lưu ý cách đọc:** 0% ở đây là **tỷ trọng ticker trong danh sách top 1%**, không phải xác suất ticker có hiệu quả bằng 0. Chỉ riêng tỷ trọng này chưa chứng minh may mắn; lập luận dựa thêm vào phân bố tstat và kiểm tra tương lai. Paper cũng không chứng minh mọi signal ticker đều vô dụng hoặc mọi signal kế toán đều hiệu quả. Đây là kết luận về các nhóm chiến lược, dữ liệu và giai đoạn đã nghiên cứu.

**Liên hệ với idea của mình:** ngoài câu hỏi “AI có chọn chiến lược lời hơn EB không?”, cần kiểm tra “AI có phân biệt được nhóm chứa thông tin như kế toán với nhóm gần nhiễu như ticker không?”. Nếu AI cho ticker dự báo rất đẹp trong quá khứ nhưng thất bại ở tương lai, đó là dấu hiệu mô hình có thể đang học nhiễu. Vì vậy, mở rộng đánh giá AI sang ticker là một phép đối chiếu hữu ích; các nhánh notebook hiện tại chưa thực hiện đầy đủ phép kiểm tra này.

Tác giả diễn giải sự tập trung ở kế toán, cổ phiếu nhỏ và trước 2004 là phù hợp với việc nhà đầu tư chú ý hạn chế. Đây là diễn giải kinh tế phù hợp bằng chứng, chưa xác định chắc chắn nguyên nhân. Các bằng chứng đều lấy từ [paper v3](https://arxiv.org/pdf/2311.10685v3), Bảng 2–3, Hình 2–6 và Proposition 1.

## 4. Idea của mình thay đổi phần nào?

**Thay cách tạo dự báo ở bước 3.** Giữ việc chọn bằng thông tin quá khứ và kiểm tra bằng kết quả tương lai.

| | Cách gốc | Hướng của mình |
|---|---|---|
| Cách tạo dự báo | EB: dùng mô hình phân phối thống kê để điều chỉnh thành tích quá khứ. | AI: học quan hệ giữa lịch sử chiến lược và lợi suất tương lai. |
| Input | Thống kê quá khứ của từng ID và phân bố tstat cùng nhóm. | Chuỗi lợi suất hoặc đặc trưng FinCast; các nhánh hiện có còn thêm tstat và trung bình quá khứ. |
| Target | Ước lượng lợi suất trung bình tương lai. | Khi train, đáp án là lợi suất trung bình tháng thực tế trong năm tiếp theo. |
| Output | Dự báo lợi suất theo chiến lược–năm. | Cùng ý nghĩa đầu ra để có thể so sánh. |
| Đánh giá | Độ sát của dự báo, lợi suất và Sharpe ngoài mẫu. | Kiểm tra các tiêu chí đó trên cùng khoảng thời gian và tập chiến lược. |

**Không nên gọi là “thay công thức tính thủ công bằng AI”:** tác giả đã tính tự động bằng Python/R. Ý mới là thay mô hình dự báo EB bằng mô hình học từ dữ liệu. AI vẫn sử dụng toán học, hàm mất mát và các phép thống kê; notebook hiện tại cũng chưa loại bỏ tstat.

Các câu hỏi cần kiểm tra: AI có dự báo sát hơn không? Có xếp hạng/chọn chiến lược tốt hơn không? FinCast có giúp hơn LSTM đọc chuỗi gốc không? Hiệu quả có giữ được trong giai đoạn sau 2004 không? Đây là **giả thuyết nghiên cứu**, chưa phải kết quả đã chứng minh.

## 5. Hai notebook hiện thực hóa idea ra sao?

### Notebook 1: tái hiện phần phân tích của paper

[rep_original_R_LSTM_Attention_highthroughput_PointB.ipynb](../Research/NoteBook_Refer/rep_original_R_LSTM_Attention_highthroughput_PointB.ipynb), cell đánh số từ 0.

Cell 2–5 đọc lợi suất chiến lược, dự báo EB `Nelly_Predict_SignalYear.csv.gzip` và tham số QML đã tạo sẵn; ghép với kết quả thật, so sánh dự báo, chọn top chiến lược, tạo bảng/hình và so phương pháp. **Notebook không tự thực hiện đầy đủ bước fit EB bằng R**, dù tên có chữ R/LSTM/Attention. Các cell phía sau còn có thao tác Git và file, không phải mô hình dự báo mới.

Vai trò: cung cấp cách đánh giá gốc để làm mốc so sánh, không phải bằng chứng toàn bộ pipeline đã được tái lập trên máy hiện tại.

### Notebook 2: thử phương pháp dự báo AI

[submit_financial_fincast_high_throughput.ipynb](../Research/NoteBook_Refer/submit_financial_fincast_high_throughput.ipynb).

| Nhánh | Input và xử lý | Target/output |
|---|---|---|
| FinCast + LSTM + Attention, cell 10–11 | Đọc embedding FinCast đã tạo sẵn; LSTM xử lý chuỗi, Attention học trọng số các phần; ghép thêm tstat IS và trung bình IS. | Học `mean_ret_OOS`; xuất dự báo `pred_ret_fincast`. |
| LSTM + Attention không FinCast, cell 13 | Đọc **20 tháng** lợi suất trước năm dự báo, kèm tstat IS và trung bình IS. Con số 20 ở đây là tháng, không phải 20 năm. | Cùng target lợi suất trung bình tháng năm tiếp theo. |
| Các nhánh theo nhóm, cell 17–25 | Có thêm thử nghiệm past-return EW và kế toán VW; phạm vi từng nhánh khác nhau. | Kiểm tra theo từng nhóm, chưa phải chạy đầy đủ cả sáu nhóm bằng một pipeline thống nhất. |
| LSTM mới ở cuối, cell 35–38 | Tối đa **240 tháng** của kế toán EW, thêm mask cho dữ liệu thiếu; dùng LSTM thường. | Xuất `lstm_predictions_acct_ew_oof.csv`: `signalid,oos_begin_year,signal_family,pred_ret`; chọn top và tổng hợp lợi suất. |

Các nhánh FinCast/raw Attention chia theo năm: **train trước 2011**, validation **2011–2013**, test **từ 2014**, rồi so EB, naive, FinCast và không FinCast. Notebook đọc embedding từ nguồn ngoài; nó không tự tạo toàn bộ embedding trong các cell này, nên cần kiểm tra lịch sử tạo embedding để xác nhận không dùng thông tin tương lai.

**Điểm cần sửa ở nhánh cuối:** nó chia 5 fold theo ID **trong cùng năm**, train bằng đáp án năm đó của các ID khác. ID được dự báo chưa đưa vào train, nhưng kết quả của chính năm cần dự báo đã được dùng để học. Vì vậy nhánh này chưa mô phỏng việc đứng cuối năm trước để dự báo năm sau. Đây là kiểm tra theo ID, không phải kiểm tra tương lai đúng nghĩa của paper.

Một khác biệt nữa: EB xếp hạng bằng Sharpe dự báo, trong khi nhánh LSTM cuối xếp hạng bằng độ lớn lợi suất dự báo. Khi so kết quả cần tách ảnh hưởng của **mô hình dự báo** và **quy tắc xếp hạng**; dùng cùng tập ID có dữ liệu và cùng thời gian để tránh so lệch.

## 6. Hướng thực hiện để kiểm tra idea

Giữ EB và naive làm hai mốc. So với LSTM + Attention và FinCast + LSTM + Attention, bắt đầu từ kế toán EW rồi mở rộng sang nhóm khác.

Mỗi mẫu là **một chiến lược × một năm dự báo**. Input phải có sẵn trước năm đó; target là trung bình lợi suất tháng của năm đó. Train/validation/test chia theo thời gian, chuẩn hóa chỉ bằng tập train; checkpoint/chọn tham số dùng validation, không dùng test. Nếu cập nhật mô hình hằng năm, chỉ dùng các năm đã kết thúc để train.

Đầu ra tối thiểu là bảng `signal_family,signalid,oos_begin_year,pred_ret`, sau đó ghép lợi suất tháng thật để đánh giá. So độ sai dự báo, khả năng xếp hạng, lợi suất/Sharpe top 1%/5%/10%, và kết quả theo giai đoạn. Với các mô hình AI, chạy nhiều seed và tách test khỏi quá trình thử nhiều cấu hình.

**Đóng góp kỳ vọng:** kiểm tra liệu học chuỗi và đặc trưng FinCast có cải thiện dự báo/chọn chiến lược so với EB hay không, đặc biệt khi hiệu quả thị trường thay đổi. Chỉ kết luận AI tốt hơn nếu kiểm tra ngoài mẫu công bằng cho bằng chứng; hiện chưa xác nhận điều đó bằng lần chạy tái lập mới.

---

Ghi chú biên soạn: tài liệu đối chiếu PDF gốc, các script trong repo tác giả và source của hai notebook; không chạy lại notebook để tạo kết quả mới. Dùng skill scientific-writing để tách kết quả, giả thuyết và giới hạn diễn giải. Nguồn công cụ: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026), [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065).
