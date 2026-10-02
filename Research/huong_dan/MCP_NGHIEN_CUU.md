# MCP phục vụ nghiên cứu

Các server được cấu hình trên máy SSH trong `~/.codex/config.toml`, áp dụng cho Codex của tài khoản này. Chúng không tự chuyển sang máy khác khi clone Git. Sau khi thêm, reload Codex/VS Code hoặc mở phiên mới để nạp công cụ.

| Tên MCP | Công dụng | Nguồn |
|---|---|---|
| `jupyter` | Đọc, sửa và thực thi notebook qua Jupyter chạy tại thư mục Paper. | https://github.com/datalayer/jupyter-mcp-server |
| `huggingface` | Tra cứu model, dataset, tài liệu và paper trên Hugging Face. | https://huggingface.co/docs/hub/agents-mcp |
| `paper-search` | Tìm paper đa nguồn, như arXiv, Semantic Scholar và Crossref; các nguồn khác tùy phiên bản và thông tin truy cập. | https://github.com/openags/paper-search-mcp |
| `arxiv` | Tìm, lưu và đọc paper arXiv. | https://github.com/blazickjp/arxiv-mcp-server |
| `fetch` | Đọc nội dung trang web nguồn qua URL; không phải công cụ tìm kiếm web. | https://github.com/modelcontextprotocol/servers/tree/main/src/fetch |

Ví dụ yêu cầu: “Tìm paper về empirical Bayes và asset pricing, trả tiêu đề, năm, DOI/link, rồi đọc phương pháp của các paper liên quan.” Hoặc: “Dùng Hugging Face tìm model FinCast và đọc model card.”

Môi trường cài đặt nằm ngoài repo: Jupyter ở `~/.local/share/jupyter-mcp/`; các MCP nghiên cứu Python ở `~/.local/share/research-mcp/venv/`. Paper tải bởi server arXiv nằm ở `~/.local/share/research-mcp/papers/`, không trộn vào dữ liệu tài chính gốc.

Tra cứu Hugging Face công khai hoạt động không cần token trong lần kiểm tra. Nếu cần quyền tài khoản, chạy `codex mcp login huggingface` và hoàn tất đăng nhập trên trình duyệt. Không ghi token vào repo.

Một số nguồn học thuật yêu cầu API key/email hoặc giới hạn số lượt gọi; cài MCP không cấp quyền vào paper có paywall. Các bản PDF, dự báo EB hay embedding cũng không được tải tự động chỉ vì đã cài server.

Kiểm tra cấu hình bằng `codex mcp list`. Muốn xem một server: `codex mcp get paper-search`. Muốn gỡ cấu hình: `codex mcp remove paper-search`.

## Kiểm tra sau cài đặt — 02/10/2026

- Hugging Face: kết nối MCP, liệt kê 4 công cụ và tìm được model FinCast công khai.
- Paper Search 0.1.4: liệt kê 57 công cụ; truy vấn Crossref về asset pricing trả kết quả. Không có nghĩa đã thử tất cả 57 công cụ hoặc mọi nguồn.
- arXiv 0.7.3: liệt kê 19 công cụ; truy vấn `ti:FinCast` trả paper `2508.19609`. Chưa tải PDF hoặc kiểm tra mọi chức năng đọc/chuyển đổi.
- Fetch 2026.8.18: gọi công cụ fetch đọc được trang tài liệu Hugging Face.
- Môi trường MCP nghiên cứu: `pip check` không báo xung đột thư viện.

Cấu hình Hugging Face có thể hiển thị “Not logged in” vì chưa hoàn tất OAuth; kiểm tra tra cứu công khai vẫn thành công.
