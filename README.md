# AI_Finance

Repo tìm hiểu **High-Throughput Asset Pricing — Chen & Dim** và thử nghiệm dự báo lợi suất chiến lược bằng FinCast/LSTM.

Hướng nghiên cứu: đọc [idea paper gốc và hướng dùng AI](docs/IDEA_NGHIEN_CUU.md).

Bắt đầu với [luồng dữ liệu và ví dụ từng bước](Research/huong_dan/7_CSV_PAPER_DUNG_NHU_THE_NAO.md). File này giải thích input, tstat, EB, lựa chọn chiến lược và output. Số thật và số minh họa được ghi rõ.

| Thư mục | Nội dung |
|---|---|
| `Research/huong_dan/` | [Hướng dẫn bắt đầu](Research/huong_dan/BAT_DAU_TU_DAY.md), [dữ liệu](Research/huong_dan/README_DATA.md) và [code gốc](Research/huong_dan/CODE_PAPER_GOC_VA_VI_DU_DATA_THAT.md). |
| `Research/paper chính/` | PDF paper chính và giải thích từ cơ bản đến chi tiết. |
| `Research/code_goc_paper/high-throughput-ap/` | Repo công khai của tác giả, có lịch sử Git riêng. |
| `Research/NoteBook_Refer/` | Hai notebook tham khảo: tái hiện phân tích gốc và thử nghiệm AI. |
| `Research/Notebook_test/` | Nơi đặt notebook thử nghiệm; hiện chưa có file. |
| `Research/output_minh_hoa/` | CSV minh họa từng bước, README và script tạo lại ví dụ. |
| `data_raw/` | Bảy CSV dữ liệu gốc đầy đủ; chỉ chứa dữ liệu gốc. |
| `data_sample/` | Mẫu nhỏ để mở và đọc cấu trúc của từng CSV gốc. |
| `data_bo_sung/` | Dữ liệu đối chiếu bổ sung; README ghi nguồn và tác dụng. |
| `docs/` | Tài liệu tham khảo, bản thảo và slide. |
| `docs-add/` | Thư mục tài liệu bổ sung hiện có. |
| `scripts/` | Script tạo mẫu dữ liệu. |
| `archives/` | Ba ZIP tải về ban đầu, giữ nguyên để lưu trữ. |

Tạo lại dữ liệu mẫu: `python scripts/create_data_samples.py`. Tạo lại ví dụ output: `python Research/output_minh_hoa/tao_output_minh_hoa.py`. Chạy từ thư mục chính; lệnh sẽ ghi lại các CSV mẫu tương ứng.

Dữ liệu gốc đầy đủ và ZIP vẫn được giữ **trên máy**, nhưng các file lớn được loại khỏi Git. Khi clone repo ở máy khác, cần bổ sung riêng bốn bảng lợi suất lớn trong `data_raw/`: `DataMinedLongShortReturnsEW.csv`, `DataMinedLongShortReturnsVW.csv`, `PastReturnSignalsLongShort.csv`, `TickerSignalsLongShort.csv`.

Code tác giả cũng được loại khỏi Git của repo này; tải lại bằng `git clone https://github.com/chenandrewy/high-throughput-ap.git Research/code_goc_paper/high-throughput-ap`. Các notebook còn dùng dự báo và embedding tạo sẵn ở môi trường ngoài; không phải tất cả các kết quả này đều có sẵn trong repo.

## Công cụ đã cài

Danh sách đối chiếu trên máy SSH ngày **02/10/2026**. MCP được cấu hình cho tài khoản Codex trên máy SSH. **15 skill nghiên cứu nằm trong repo tại `.agents/skills/`, không cài global**; có thể lưu cùng repo trong Git. Clone repo không tự cài lại MCP hoặc các thư viện phụ thuộc.

### 5 MCP đang bật

| Tên trong Codex | Công dụng | Trạng thái kiểm tra |
|---|---|---|
| `jupyter` | Đọc, sửa, thực thi notebook qua JupyterLab tại thư mục Paper. | Kết nối và liệt kê hai notebook thành công; chưa chạy cell nghiên cứu. |
| `huggingface` | Tra cứu model, dataset, paper và tài liệu Hugging Face. | Tìm model FinCast thành công; chưa đăng nhập OAuth, tra cứu công khai hoạt động. |
| `paper-search` | Tìm paper nhiều nguồn, gồm arXiv, Semantic Scholar, Crossref. | Truy vấn Crossref thành công; chưa kiểm tra mọi nguồn. |
| `arxiv` | Tìm, lưu và đọc paper arXiv. | Tìm được paper FinCast; chưa kiểm tra tải/đọc toàn văn. |
| `fetch` | Đọc nội dung trang web từ URL. | Đọc trang nguồn thành công; đây không phải công cụ tìm kiếm. |

Cấu hình: `~/.codex/config.toml`. Môi trường Python riêng: `~/.local/share/jupyter-mcp/venv/` và `~/.local/share/research-mcp/venv/`. Kiểm tra bằng `codex mcp list`. Chi tiết nguồn và cách dùng: [hướng dẫn MCP](Research/huong_dan/MCP_NGHIEN_CUU.md).

### 15 skill nghiên cứu đã cài

Nguồn: [K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills). Nằm trong [`.agents/skills/`](.agents/skills/); chỉ áp dụng trong repo này và không còn bản global của 15 skill này; mỗi skill là bộ hướng dẫn và tài nguyên hỗ trợ agent, không phải một MCP.

| Skill | Dùng để làm gì? |
|---|---|
| `literature-review` | Tìm, sàng lọc và tổng hợp tài liệu nghiên cứu. |
| `citation-management` | Kiểm tra thông tin trích dẫn, DOI và tạo BibTeX. |
| `scientific-critical-thinking` | Đánh giá lập luận, bằng chứng và hạn chế của nghiên cứu. |
| `peer-review` | Rà soát phương pháp và viết nhận xét phản biện. |
| `scientific-writing` | Soạn và chỉnh sửa nội dung paper. |
| `hypothesis-generation` | Xây dựng giả thuyết có thể kiểm tra. |
| `scientific-brainstorming` | Đề xuất và đánh giá hướng nghiên cứu. |
| `experimental-design` | Thiết kế thí nghiệm và đối chứng. |
| `exploratory-data-analysis` | Khảo sát cấu trúc, chất lượng và đặc điểm dữ liệu. |
| `statistical-analysis` | Chọn phép phân tích và diễn giải kết quả thống kê. |
| `statsmodels` | Hồi quy, suy luận thống kê và mô hình chuỗi thời gian. |
| `scientific-visualization` | Thiết kế biểu đồ và hình khoa học. |
| `scikit-learn` | Xây dựng và đánh giá mô hình machine learning. |
| `pytorch-lightning` | Tổ chức huấn luyện mô hình deep learning. |
| `transformers` | Làm việc với mô hình và công cụ Hugging Face Transformers. |

Skill có thể được dùng từ lượt tiếp theo; mở phiên mới nếu danh sách chưa cập nhật. **Cài skill chưa đồng nghĩa cài mọi thư viện ML hoặc dịch vụ API mà skill hướng dẫn sử dụng.** Xem [chi tiết skills và autoresearch](Research/huong_dan/SKILLS_VA_AUTORESEARCH.md).

### Repo công cụ đã clone để tham khảo

| Repo | Vị trí | Trạng thái |
|---|---|---|
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | `Research/tools/autoresearch/` | Đã clone; chưa cài phụ thuộc huấn luyện, tải dữ liệu hoặc chạy vòng thí nghiệm. Không phải MCP/skill chuẩn. |
| [chenandrewy/high-throughput-ap](https://github.com/chenandrewy/high-throughput-ap) | `Research/code_goc_paper/high-throughput-ap/` | Code paper gốc đã clone; chưa chạy đầy đủ pipeline tái lập. |

Các repo clone có Git riêng và được loại khỏi Git của Paper. Các MCP và skill bổ sung không tự thay đổi notebook hoặc tạo kết quả nghiên cứu.
