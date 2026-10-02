# Skills nghiên cứu và autoresearch

## Đã cài vào Codex

Nguồn: [K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills). Các skill nằm trong [`.agents/skills/`](../../.agents/skills/) của repo Paper, có thể lưu cùng repo trong Git. Đã chuyển đủ 15 skill ra khỏi `~/.codex/skills/`, không còn cài global các skill này. Chúng là hướng dẫn và script hỗ trợ cho agent, không phải MCP. Skill mới có thể được dùng từ lượt tiếp theo; nếu danh sách chưa cập nhật, mở phiên Codex mới.

| Nhóm công việc | Skill đã cài |
|---|---|
| Tổng hợp paper và xác minh tài liệu tham khảo | `literature-review`, `citation-management` |
| Đọc phản biện và viết paper | `scientific-critical-thinking`, `peer-review`, `scientific-writing` |
| Xây dựng câu hỏi và thiết kế nghiên cứu | `hypothesis-generation`, `scientific-brainstorming`, `experimental-design` |
| Khám phá dữ liệu và phân tích thống kê | `exploratory-data-analysis`, `statistical-analysis`, `statsmodels` |
| Tạo biểu đồ khoa học | `scientific-visualization` |
| Làm việc với mô hình machine learning | `scikit-learn`, `pytorch-lightning`, `transformers` |

Ví dụ yêu cầu: “Dùng literature-review tìm các paper về empirical Bayes trong asset pricing”; “Dùng scientific-critical-thinking kiểm tra cách chia train/test của notebook”; “Dùng citation-management kiểm tra DOI của các tài liệu tham khảo”.

Đã kiểm tra đủ 15 thư mục skill có SKILL.md và tên đúng. Cài skill **không đồng nghĩa đã cài mọi thư viện ML, LaTeX hoặc dịch vụ API mà các hướng dẫn có thể sử dụng**. Các phụ thuộc đó cần cài trong môi trường phù hợp khi dùng tính năng tương ứng. MCP tìm paper, Hugging Face và Jupyter đã cài trước đó vẫn giữ nguyên.

## autoresearch: đã clone để tham khảo

Nguồn: [karpathy/autoresearch](https://github.com/karpathy/autoresearch). Bản clone riêng nằm ở `Research/tools/autoresearch/`, có lịch sử Git riêng và được loại khỏi Git của Paper. Commit tham khảo: `228791fb499afffb54b46200aca536f79142f117`.

Idea: agent sửa mô hình → huấn luyện trong ngân sách thời gian cố định → đo kết quả validation → giữ hoặc bỏ thay đổi → ghi nhật ký → thử tiếp.

| File | Vai trò |
|---|---|
| `program.md` | Hướng dẫn cho agent tổ chức và ghi lại thí nghiệm. |
| `train.py` | Mô hình ngôn ngữ và vòng huấn luyện mà agent được sửa. |
| `prepare.py` | Chuẩn bị dữ liệu/tokenizer và phép đánh giá cố định. |
| `pyproject.toml`, `uv.lock` | Mô tả và khóa phụ thuộc của dự án. |

Repo này **không phải MCP**, không phải code tài chính, và không phải một skill chuẩn có SKILL.md. Chưa cài môi trường CUDA/PyTorch riêng, chưa tải dữ liệu huấn luyện và chưa chạy vòng thí nghiệm tự động. Chỉ đã clone code, đọc README/program và kiểm tra cú pháp hai script chính.

Máy hiện có GPU RTX 5060 Ti khoảng 16 GB VRAM. Tác giả thử bản gốc trên H100; cấu hình batch mặc định cần kiểm tra/điều chỉnh trước khi chạy trên máy này. Không thể khẳng định cấu hình mặc định sẽ chạy được chỉ vì có GPU NVIDIA.

## Cách vận dụng vào paper tài chính

Giữ dữ liệu gốc và quy tắc đánh giá cố định. Mỗi lần chỉ thay một phần mô hình LSTM/FinCast, dùng cùng dữ liệu train/validation và cùng ngân sách, rồi lưu cấu hình, seed, metric và quyết định giữ/bỏ.

Vì agent thử nhiều lần, validation cũng có thể bị tối ưu quá mức. **Dữ liệu test cuối cùng phải được giữ riêng**, không dùng kết quả test để chọn thí nghiệm hoặc sửa mô hình. Chia train/validation/test theo thời gian và tránh lấy lợi suất của năm cần dự báo để train, như nhánh OOF theo ID trong notebook hiện tại.

Đây là gợi ý thiết kế workflow; chưa thay đổi notebook, chưa triển khai vòng tự động cho tài chính và chưa tạo kết quả mới.
