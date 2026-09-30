# Hiểu paper “High-Throughput Asset Pricing” từ đầu

**Paper gốc:** [High-Throughput Asset Pricing - Chen & Dim.pdf](High-Throughput%20Asset%20Pricing%20-%20Chen%20%26%20Dim.pdf), Andrew Y. Chen và Chukwuma Dim, bản arXiv tháng 6/2025. Đây là **paper của Chen và Dim**, không phải kết quả thực nghiệm của nhóm mình. File này giải thích paper theo thứ tự dành cho người mới hoàn toàn.

## 1. Nếu chỉ nhớ một ý

Khi thử hàng trăm nghìn quy tắc đầu tư, một số quy tắc sẽ có lợi suất quá khứ rất cao **chỉ vì may mắn**. Paper đề xuất: hãy tạo nhiều quy tắc một cách có hệ thống, rồi dùng **Empirical Bayes (EB)** để ước lượng phần thành tích nào có khả năng là tín hiệu thật, phần nào cần điều chỉnh vì nhiễu. Sau đó kiểm tra dự báo trên các năm **chưa được dùng để chọn quy tắc**.

Một câu dễ nhớ: **“Tìm kiếm nhiều vẫn được, nhưng phải tính đến việc mình đã tìm kiếm nhiều khi đánh giá kết quả.”**

## 2. Các viên gạch tài chính cơ bản

### Cổ phiếu, lợi suất và lợi suất kỳ vọng

Cổ phiếu là một phần quyền sở hữu doanh nghiệp. Nếu mua ở giá 100 và sau một kỳ giá thành 105, lợi suất từ biến động giá là $(105-100)/100=5\%$; ví dụ này bỏ qua cổ tức và các chi phí khác.

- **Lợi suất thực tế:** kết quả đã xảy ra, có thể đo được.
- **Lợi suất kỳ vọng:** mức trung bình ta tin sẽ có trước khi tương lai xảy ra; không quan sát trực tiếp được.
- **Dự báo lợi suất:** một ước lượng về lợi suất tương lai, có thể đúng hoặc sai.

Giá cổ phiếu và lợi suất không giống nhau. Paper này làm việc chủ yếu với **lợi suất của chiến lược đầu tư**, không dự báo giá tuyệt đối của một cổ phiếu.

### Signal là gì?

**Signal** là một công thức hoặc đặc điểm được dùng để xếp hạng cổ phiếu. Ví dụ: tỷ số giữa hai khoản mục kế toán của doanh nghiệp. Giả thuyết có thể là doanh nghiệp có tỷ số cao sẽ sinh lời khác doanh nghiệp có tỷ số thấp.

Một signal chưa phải chiến lược. Cần thêm quy tắc chọn cổ phiếu, cách phân bổ vốn và thời điểm tái cân bằng để biến nó thành chiến lược.

### Chiến lược long–short là gì?

Mỗi kỳ, xếp các cổ phiếu theo signal. **Mua** nhóm ở một đầu bảng xếp hạng và **bán khống** nhóm ở đầu còn lại. Với chiến lược mua nhóm cao nhất, bán khống nhóm thấp nhất:

$$r^{LS}_m=r^{Long}_m-r^{Short}_m.$$

Nếu nhóm mua tăng 3% và nhóm bán khống tăng 1%, chênh lệch lợi suất là 2%. Nếu nhóm mua tăng ít hơn nhóm bán khống, kết quả có thể âm. Paper thường dùng **nhóm 10% cao nhất và 10% thấp nhất** (*deciles*).

Một signal có thể tạo hai phiên bản danh mục:

| Ký hiệu | Cách cho trọng số | Hệ quả thường cần chú ý |
|---|---|---|
| **EW**, equal-weighted | Mỗi cổ phiếu có trọng số ngang nhau | Cổ phiếu nhỏ có ảnh hưởng lớn hơn so với VW |
| **VW**, value-weighted | Trọng số theo vốn hóa | Cổ phiếu lớn chi phối kết quả hơn |

### Asset pricing có nghĩa gì trong paper này?

Asset pricing là lĩnh vực nghiên cứu giá và lợi suất tài sản. Trong paper này, câu hỏi cụ thể là **liệu các đặc điểm cổ phiếu có tạo ra khả năng dự báo lợi suất theo mặt cắt ngang hay không**. “Mặt cắt ngang” nghĩa là so sánh nhiều cổ phiếu ở **cùng một thời điểm**, chẳng hạn mua nhóm signal cao và bán nhóm signal thấp.

Paper không chứng minh rằng mỗi signal thắng là một *risk factor* căn bản. Một chiến lược có lợi suất ngoài mẫu vẫn có thể cần giải thích thêm về rủi ro, hành vi thị trường và chi phí giao dịch.

## 3. Vấn đề: càng thử nhiều càng dễ gặp chiến lược “thắng giả”

Hãy tưởng tượng thử 100.000 signal, trong khi phần lớn không thực sự dự báo được gì. Ngay cả như vậy, sẽ có vài signal trông cực tốt trong dữ liệu quá khứ chỉ vì biến động ngẫu nhiên.

Viết đơn giản cho chiến lược $i$:

$$\text{thành tích quan sát}_i = \text{khả năng thật}_i + \text{may mắn/nhiễu}_i.$$

Nếu chọn những chiến lược có thành tích quan sát lớn nhất, ta đồng thời chọn những chiến lược có phần **may mắn dương** lớn. Do đó, lấy nguyên lợi suất hoặc Sharpe quá khứ của nhóm vừa được chọn làm dự báo tương lai sẽ thường quá lạc quan. Đó là **data mining bias** hay **selection bias** trong ngữ cảnh này.

**In-sample (IS)** là giai đoạn quá khứ dùng để tính thành tích và chọn chiến lược. **Out-of-sample (OOS)** là giai đoạn sau khi đã chọn, dùng để kiểm tra dự báo. Một signal lãi cao IS nhưng gần 0 OOS có thể là tín hiệu đã suy giảm hoặc chỉ từng gặp may.

## 4. Ý tưởng khác biệt của Chen–Dim

Một phản ứng phổ biến với data mining là “thử ít quy tắc hơn và chỉ thử những gì có lý thuyết kinh tế”. Chen–Dim đưa ra hướng khác: **tạo một tập ứng viên rộng, có hệ thống**, rồi điều chỉnh thống kê cho việc đã tìm kiếm trên tập lớn ấy.

“High-throughput” nghĩa là xử lý một số lượng ứng viên rất lớn theo cùng quy trình, giống cách nghiên cứu khác kiểm tra hàng loạt giả thuyết. Paper dùng **136.192 chiến lược** thuộc sáu nhóm: ba nguồn signal nhân với hai cách trọng số EW/VW.

| Nguồn signal | Cách hình thành, nói ngắn gọn | Số chiến lược trong Bảng 1 |
|---|---|---:|
| **Accounting ratios** | Tỷ số và thay đổi của các biến kế toán | 29.314 EW + 29.314 VW |
| **Past returns** | Các công thức dựa trên lợi suất ở những quý trước | 19.402 EW + 19.402 VW |
| **Ticker characters** | Quy tắc từ các chữ cái trong mã cổ phiếu | 19.380 EW + 19.380 VW |

Ticker là nhóm gần với **đối chứng trực giác**: chữ cái trong mã cổ phiếu thường thiếu lý do kinh tế rõ ràng để dự báo lợi suất. Nếu nhiều quy tắc ticker trông tốt IS nhưng EB dự báo gần 0 và OOS không tốt, ta thấy vì sao không thể tin ngay thành tích quá khứ.

## 5. Empirical Bayes là gì? Đi từ trực giác đến công thức

### Mức trực giác

Giả sử một chiến lược có thành tích quá khứ rất cao. Ta nên hỏi thêm: **cả nhóm chiến lược giống nó thường có khả năng thật đến mức nào?**

- Nếu cả nhóm chủ yếu giống nhiễu, một “ngôi sao” đơn lẻ có thể là may mắn; dự báo nên **co rút mạnh về gần 0**.
- Nếu cả nhóm cho thấy nhiều tín hiệu vượt khỏi mức nhiễu, thành tích cao của từng chiến lược đáng tin hơn; dự báo có thể **co rút ít hơn**.

Việc kéo dự báo cực đoan về gần mức hợp lý hơn gọi là **shrinkage**. EB dùng toàn bộ tập chiến lược để **ước lượng phân phối của khả năng thật**, sau đó tính dự báo cho mỗi chiến lược dựa trên quan sát riêng của nó. “Empirical” là vì phân phối ban đầu được **ước lượng từ dữ liệu**, thay vì hoàn toàn ấn định trước.

### T-stat được dùng để làm gì?

Với một chiến lược có lợi suất tháng $r_1,\ldots,r_T$, trung bình quá khứ là $\bar r$. T-stat thường có dạng:

$$t=\frac{\bar r}{SE(\bar r)},$$

trong đó $SE$ là sai số chuẩn của trung bình. T-stat cao cho thấy trung bình quan sát lớn so với sai số lấy mẫu **theo giả định thống kê đang dùng**. Nhưng khi thử rất nhiều chiến lược, vẫn có t-stat cao do may mắn.

Trong mô hình của paper, t-stat quan sát được được xem như “khả năng thật ở thang đo t-stat + nhiễu”. Tác giả ước lượng phân phối khả năng thật **riêng cho sáu nhóm chiến lược** bằng hỗn hợp hai phân phối chuẩn, rồi tính kỳ vọng hậu nghiệm của khả năng thật khi đã thấy t-stat của chiến lược.

### Ví dụ toán học giản lược

Giả sử $z=\mu+\varepsilon$, với nhiễu $\varepsilon$ có phương sai 1, còn khả năng thật $\mu$ trong một nhóm có phương sai 0,25 và trung bình 0. Với mô hình chuẩn đơn giản, kỳ vọng hậu nghiệm là:

$$E(\mu\mid z)=\frac{0,25}{0,25+1}z=0,2z.$$

Nếu thấy $z=3$, dự báo đã điều chỉnh là $0,6$ ở **cùng thang đo chuẩn hóa**, chứ không giữ nguyên 3. Đây chỉ là ví dụ giúp hiểu shrinkage; paper thực tế dùng **mixture hai chuẩn**, nhiều tham số hơn, và chuyển dự báo chuẩn hóa về lợi suất hoặc Sharpe.

Một phân biệt quan trọng:

- **Chọn chiến lược nào?** Có thể xếp hạng khá tốt ngay cả bằng t-stat/Sharpe quá khứ trong vài điều kiện.
- **Dự báo nó sẽ tốt đến mức nào?** Thành tích quá khứ cực đoan có thể quá lạc quan; EB giúp điều chỉnh con số này.

Paper có một mệnh đề lý thuyết cho thấy naive và EB **có thể chọn cùng tập chiến lược dưới các điều kiện cụ thể**. Đừng diễn giải thành “EB luôn vô ích” hoặc “naive luôn đúng”.

## 6. Thực nghiệm chạy theo thời gian như thế nào?

Tại mỗi năm đánh giá, tác giả:

1. Dùng **20 năm dữ liệu quá khứ** để ước lượng mô hình EB cho từng nhóm chiến lược.
2. Tính lợi suất hoặc Sharpe **dự báo** cho từng chiến lược, chỉ dựa trên thông tin có vào thời điểm đó.
3. Đảo chiều mua/bán nếu dự báo mang dấu âm, rồi xếp hạng các chiến lược theo Sharpe dự báo.
4. Lấy top **1%, 5% hoặc 10%**, cho các chiến lược được chọn trọng số bằng nhau, giữ danh mục **một năm**.
5. Quan sát lợi suất trong năm tiếp theo để đánh giá **OOS**, rồi lặp lại cho các năm sau.

Ví dụ: ở cuối năm 1993, không thể dùng dữ liệu năm 1994 để chọn signal. Nếu chọn bằng dữ liệu 1994 rồi tuyên bố đây là lựa chọn “ở cuối 1993”, đó là **look-ahead bias**.

Tác giả còn so với cách **naive mining**: xếp hạng bằng Sharpe quá khứ mà không co rút. So sánh hai cách trên **cùng quy trình OOS** giúp biết EB thêm giá trị ở lựa chọn chiến lược và/hoặc độ chính xác dự báo.

## 7. Kết quả chính: đọc đúng các con số

Bảng 2 của PDF báo cáo danh mục trong giai đoạn **1983–2020**:

| Cách chọn | Lợi suất trung bình năm | Sharpe năm |
|---|---:|---:|
| EB, top 1% | **5,70%** | **1,46** |
| EB, top 5% | 4,03% | 1,34 |
| EB, top 10% | 2,77% | 1,16 |
| Naive, top 1% | 5,72% | 1,45 |
| Danh mục chiến lược đã công bố trong bộ so sánh | 5,88% | 2,03 |

**Đọc đúng:** EB top 1% và naive top 1% có hiệu quả danh mục rất gần nhau. Paper không cho thấy EB thắng rõ rệt về **chọn top 1%**. EB có vai trò khác: **dự báo độ lớn thành tích đã điều chỉnh và phân biệt nhóm signal nào đáng tin**. Danh mục gồm các chiến lược được biết là đã công bố sau này không phải phép so sánh hoàn toàn “thời gian thực”, vì danh sách ấy có thể mang thông tin từ tương lai.

Bảng 3 cho thấy thành phần top 1% được EB chọn: khoảng **91,0% accounting EW**, **8,6% past-return EW**; các nhóm ticker gần như bằng 0. Điều này gợi ý khả năng dự báo chủ yếu nằm ở dữ liệu kế toán và cổ phiếu nhỏ. **Đó là diễn giải phù hợp với dữ liệu, chưa chứng minh nhân quả** rằng chỉ có một cơ chế kinh tế tạo ra kết quả.

Paper cũng thấy hiệu quả suy giảm theo thời gian. Danh mục EB top 1% có lợi suất khoảng **8,17%/năm trong 1983–2005**, còn khoảng **2,03%/năm trong 2005–2020**. Tác giả chỉ ra EB với cửa sổ 20 năm dự báo khó hơn khi thị trường thay đổi cấu trúc. Đây là giới hạn lớn của phương pháp.

## 8. Phần nâng cao: paper đóng góp gì, còn giới hạn gì?

### Ba đóng góp nên hiểu

1. **Thiết kế tìm kiếm:** tạo rất nhiều signal một cách có hệ thống thay vì chỉ chọn vài signal đã nổi tiếng.
2. **Ước lượng:** dùng EB để điều chỉnh sai lệch sau khi chọn và cho dự báo định lượng về thành tích tương lai.
3. **Kiến thức về thị trường:** cho thấy nguồn và giai đoạn nào có khả năng dự báo, nguồn nào chủ yếu là nhiễu.

### Vì sao paper nói tới False Discovery Rate?

**False discovery** là kết luận một signal có ích trong khi thực ra nó chỉ là nhiễu. Các phương pháp kiểm soát FDR đặt ngưỡng để hạn chế phát hiện nhầm khi kiểm tra nhiều signal.

Paper so một số cách đặt ngưỡng. Theo kết quả của tác giả, vài ngưỡng rất bảo thủ có thể **bỏ sót** nhóm signal có lợi suất OOS đáng kể. EB cung cấp thêm điều mà ngưỡng “đạt/không đạt” không trực tiếp cho: **dự báo mức lợi suất sau điều chỉnh**. Kết luận này phụ thuộc dữ liệu, tiêu chí và cấu hình kiểm định của paper; không phải mọi phương pháp FDR đều luôn kém EB.

### Các giới hạn cần nhớ

- **Regime shift:** mối quan hệ trong 20 năm cũ có thể không còn đúng; paper tự cho thấy khó khăn sau đầu những năm 2000.
- **Lợi suất trước chi phí:** bảng lợi suất trong paper không tự chứng minh nhà đầu tư có thể thu được đúng con số ấy sau chi phí, tác động thị trường và điều kiện bán khống.
- **EW và cổ phiếu nhỏ:** khi kết quả tập trung ở EW, khả năng giao dịch thực tế của cổ phiếu nhỏ cần được xem riêng.
- **Lợi suất ngoài mẫu không chứng minh nguyên nhân:** kết quả phù hợp với câu chuyện thông tin được phản ánh chậm vào giá, nhưng chưa chứng minh cơ chế đó là nguyên nhân duy nhất.
- **EB không xóa mọi loại bias:** dự báo của nó vẫn phụ thuộc mô hình phân phối, cách tạo dữ liệu và sự ổn định của thị trường.

## 9. Các CSV trong folder `data/` liên quan thế nào?

Thư mục `data/` hiện có đủ ba nhóm dữ liệu lợi suất chiến lược của paper: **kế toán, past-return và ticker**, mỗi nhóm gồm cả cách tính EW/VW. Riêng file mẫu `DataMinedLongShortReturnsSample.csv` đã được chuyển sang `data_sample/`; nó chỉ dùng để xem cấu trúc và thử code, không dùng thay toàn bộ dữ liệu gốc.

Ví dụ một dòng trong file lợi suất có các cột:

| Cột | Ý nghĩa cần hiểu |
|---|---|
| `signalid` | ID để nối với bảng định nghĩa signal |
| `year`, `month` | Thời điểm của lợi suất |
| `ret` | Lợi suất long–short của signal ở tháng đó; cần xác nhận đơn vị trước khi tính toán |
| `nstock` | Số cổ phiếu dùng trong danh mục/tháng đó theo dữ liệu |

Bảng định nghĩa signal có `signal_form`, `v1`, `v2`: chẳng hạn `v1/v2` hoặc `diff(v1)/lag(v2)` là công thức biến kế toán tạo signal. **Một dòng lợi suất = một signal trong một tháng**, không phải dữ liệu của một cổ phiếu riêng lẻ.

Lưu ý phiên bản: file `DataMinedSignalList.csv` đang có **29.315 ID**, trong khi Bảng 1 của PDF ghi **29.314 chiến lược kế toán ở mỗi loại EW/VW**. Cần xác định bước lọc hoặc khác biệt phiên bản nếu muốn tái lập đúng bảng của paper.

## 10. Paper này liên quan hướng nghiên cứu của nhóm ra sao?

Chen–Dim là **paper nền và có thể là baseline EB**. Trong bản thảo khác của nhóm, hướng FinCast–Bi-LSTM–Attention muốn dùng **toàn bộ lịch sử lợi suất theo thời gian của một signal** để dự báo signal có tiếp tục hiệu quả hay không.

Hai câu hỏi liên quan nhưng phương pháp khác nhau:

| Chen–Dim | Hướng FinCast trong bản thảo nhóm |
|---|---|
| Ước lượng phân phối thành tích của nhiều chiến lược và EB shrinkage | Học biểu diễn chuỗi lợi suất từng chiến lược bằng mô hình chuỗi thời gian |
| Tập trung điều chỉnh selection bias và dự báo từ thống kê | Đề xuất học thêm hình dạng đường đi của lợi suất |
| Có kết quả trong PDF này | Chưa thấy kết quả FinCast được tái lập trong folder |

**Không lấy kết quả 5,70% của Chen–Dim làm kết quả của mô hình nhóm.** Nếu nhóm muốn chứng minh mô hình mới tốt hơn, phải dùng cùng dữ liệu, mốc thời gian, cách chọn danh mục và baseline phù hợp.

## 11. Tự kiểm tra xem đã hiểu paper chưa

Bạn có thể giải thích lại năm câu hỏi này mà không nhìn tài liệu:

1. Vì sao thử càng nhiều signal thì thành tích tốt nhất trong mẫu càng dễ bị thổi phồng?
2. Một signal khác một danh mục long–short ở điểm nào?
3. EB lấy thông tin từ **riêng chiến lược** và **cả nhóm chiến lược** để làm gì?
4. Vì sao naive có thể chọn gần đúng top 1% mà EB vẫn có giá trị?
5. Vì sao kết quả sau 2004 khiến ta không thể kết luận phương pháp luôn dự báo tốt?

Nếu trả lời được, bạn đã nắm **idea cốt lõi**. Sau đó đọc PDF theo thứ tự: **Abstract → Introduction → Section 2.1–2.4 → Table 2 và Table 3 → Section 4.2 → Conclusion**. Công thức chứng minh trong phụ lục có thể đọc sau.
