# AI_Finance

Dự báo lợi suất năm sau của các chiến lược long–short (Chen & Dim, *High-Throughput Asset Pricing*) bằng mô hình học sâu cải tiến từ baseline (FinCast → Bi-LSTM → attention), học trên họ Accounting EW và so với Empirical Bayes (EB) của paper.

## Bắt đầu ở đâu

1. **Notebook duy nhất:** [code/notebook/final_model.ipynb](code/notebook/final_model.ipynb): từ CSV gốc đến kết quả, từng bước, có checkpoint.
2. **Tài liệu** (`docs/`):
   1. [Baseline ban đầu](docs/1_BASELINE.md)
   2. [Paper gốc làm gì](docs/2_ORIGINAL_PAPER.md)
   3. [Notebook tham khảo làm gì](docs/3_REFERENCE_NOTEBOOK.md)
   4. [Mục tiêu và ý tưởng](docs/4_GOALS_AND_IDEAS.md)
   5. [Mô hình chốt so với baseline, kết quả](docs/5_FINAL_MODEL_VS_BASELINE.md)
   6. [Tổng hợp mọi kết quả và thử nghiệm (đọc đầu tiên)](docs/6_ALL_RESULTS_SUMMARY.md)
   7. [So sánh với EB theo cách chấm của paper](docs/7_COMPARE_WITH_EB.md)
   - Viết paper: [hướng dẫn viết (đọc trước khi viết)](docs/writing_paper/WRITING_GUIDE.md), bài nháp `docs/writing_paper/sn-article.tex`, nội dung mục 3.1 và 3.2 để chép vào Overleaf ở cùng thư mục

## Cài môi trường và chạy tay

```bash
conda create -n paper python=3.13 -y
conda activate paper
pip install -r requirements.txt          # torch CUDA 12.8 + thư viện cần cho code/ và FinCast
cd code
python run.py 0 1 2 3 4 5                # học lại 6 seed, lưu dự báo và checkpoint (~7 phút/seed)
```

Notebook: mở `code/notebook/final_model.ipynb`, chọn kernel **Python (paper)**. Trên máy này env `paper` đã được tạo sẵn.

## Cấu trúc

| Thư mục / file | Nội dung |
|---|---|
| `code/data.py` | CSV gốc → bảng (chiến lược, năm) → cửa sổ 128 tháng → FinCast → PCA16 |
| `code/model.py` | Kiến trúc chốt, từng khối |
| `code/train.py` | Nhãn, nền, trọng số năm; học walk-forward; checkpoint mỗi năm |
| `code/evaluate.py` | Danh mục top 1/5/10%, kiểm định, IC, hình đúng mức |
| `code/run.py` | `python run.py 0 1 2 3 4 5`: học 6 seed; mô hình chốt = seed 1 (`checkpoints/final_s1/`) |
| `code/_cache/` | Cache dữ liệu và dự báo (`forecasts/`), không đưa lên Git |
| `code/checkpoints/` | Trọng số mỗi năm của 6 seed (bản nén seed 1: `docs/zip/`) |
| `code/experiments/` | Mọi thứ chỉ dùng để tìm mô hình chốt (mã, notebook, dự báo các cấu hình, docs cũ) |
| `docs/writing_paper/` | Bài đang viết (`sn-article.tex`), hướng dẫn viết paper, nội dung từng mục để chép vào Overleaf |
| `docs/papers/` | PDF paper dùng trực tiếp (Chen & Dim, HyperAttention, Holen et al.) |
| `docs-add/` | Thư viện paper, bản nháp của nhóm |
| `data_raw/`, `data_sample/`, `data_extra/` | Dữ liệu gốc, mẫu nhỏ, dữ liệu đối chiếu |
| `Research/` | Repo gốc của tác giả, hai notebook tham khảo, mã FinCast |
| `archives/`, `scripts/` | ZIP tải về ban đầu; script tạo mẫu dữ liệu |

Dữ liệu gốc lớn (`data_raw/*.csv`) và mã tác giả không nằm trong Git: tải lại mã bằng `git clone https://github.com/chenandrewy/high-throughput-ap.git Research/original_paper_code/high-throughput-ap`.

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

Cấu hình: `~/.codex/config.toml`. Môi trường Python riêng: `~/.local/share/jupyter-mcp/venv/` và `~/.local/share/research-mcp/venv/`. Kiểm tra bằng `codex mcp list`.

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

Skill có thể được dùng từ lượt tiếp theo; mở phiên mới nếu danh sách chưa cập nhật. **Cài skill chưa đồng nghĩa cài mọi thư viện ML hoặc dịch vụ API mà skill hướng dẫn sử dụng.**

### Repo công cụ đã clone để tham khảo

| Repo | Vị trí | Trạng thái |
|---|---|---|
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | `Research/tools/autoresearch/` | Đã clone; chưa cài phụ thuộc huấn luyện, tải dữ liệu hoặc chạy vòng thí nghiệm. Không phải MCP/skill chuẩn. |
| [chenandrewy/high-throughput-ap](https://github.com/chenandrewy/high-throughput-ap) | `Research/original_paper_code/high-throughput-ap/` | Code paper gốc đã clone; chưa chạy đầy đủ pipeline tái lập. |

Các repo clone có Git riêng và được loại khỏi Git của Paper. Các MCP và skill bổ sung không tự thay đổi notebook hoặc tạo kết quả nghiên cứu.
