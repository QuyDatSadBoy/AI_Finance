# AI_Finance — Chọn chiến lược long–short bằng mô hình học sâu (FinCast + attention)

Mỗi đầu năm Y (1983–2020), mô hình dự báo **lợi suất trung bình tháng của năm Y** cho từng chiến lược long–short (họ **Accounting EW**, khoảng 26.000 chiến lược mỗi năm), chỉ bằng dữ liệu trước năm Y. Đây đúng là đại lượng mà Empirical Bayes (EB) trong *High-Throughput Asset Pricing* (Chen & Dim, [arXiv 2311.10685](https://arxiv.org/abs/2311.10685)) dự báo. Mô hình được so với EB, quy tắc xếp theo \|t\| (không dùng mô hình) và baseline.

**Kết quả chính:** danh mục top 1% của mô hình có Sharpe cao hơn EB ở cả giai đoạn phát triển (1,95 so với 1,55), holdout (1,11 so với 0,82) và cả kỳ (1,62 so với 1,27), mỗi chênh lệch đều có ý nghĩa thống kê. Ở top 5% và 10% hai bên ngang nhau. EB vẫn chính xác hơn về **độ lớn** của dự báo (xem mục 4).

---

## 1. Kết quả

Số là **Sharpe năm hoá** của danh mục top 1% / 5% / 10% theo đúng luật của paper (đảo dấu theo dự báo, xếp theo Sharpe dự báo, chia đều, giữ 12 tháng). **Dev** = 1983–2004 (được dùng để chọn cấu hình), **holdout** = 2005–2020 (chỉ báo cáo, không dùng để chọn).

| Phương pháp | Dev | Holdout | Cả kỳ 1983–2020 |
|---|---|---|---|
| Baseline (FinCast → Bi-LSTM → attention → FC) | 1,34 / 1,23 / 1,16 | 0,61 / 0,51 / 0,54 | 1,07 / 0,97 / 0,94 |
| Quy tắc \|t\| (không mô hình) | 1,95 / 1,99 / 1,85 | 0,84 / 0,75 / 0,83 | 1,52 / 1,51 / 1,47 |
| EB (Chen & Dim, mã dự án chạy lại) | 1,55 / 1,91 / 1,86 | 0,82 / 0,78 / 0,86 | 1,27 / 1,48 / 1,48 |
| **Mô hình chốt (seed 1)** | **1,95** / 1,95 / 1,82 | **1,11 / 0,88 / 0,88** | **1,62 / 1,54** / 1,46 |
| *Trung bình 6 seed cùng cấu hình* | *1,93 / 1,94 / 1,83* | *1,03 / 0,93 / 0,90* | *1,57 / 1,54 / 1,47* |

![Sharpe top 1/5/10%](docs/figures/eb_sharpe.png)

**Kiểm định (so với EB, top 1%)**: p của kiểm định Sharpe HAC (Ledoit–Wolf) là 0,032 (dev), 0,028 (holdout), 0,005 (cả kỳ); bootstrap khối năm cho 0,059 / 0,051 / 0,010. Ở top 5% và 10% không có chênh lệch có ý nghĩa.

**Điều cần đọc kèm, để không hiểu quá mức:**

- Dev top 1% bằng đúng quy tắc \|t\| (1,95). Mô hình cao hơn quy tắc này ở holdout (1,11 so với 0,84), còn ở dev thì ngang.
- Mô hình chỉ thắng EB ở 21 trên 38 năm. Lợi thế nằm ở danh mục gộp 38 năm, không phải mọi năm.
- **IC** (tương quan hạng dự báo – lợi suất thật trên mọi chiến lược): dev 0,151 so với EB 0,188 (p 0,013, EB hơn); holdout 0,043 so với 0,021 (ngang). Mô hình hơn ở **đầu bảng**, không hơn trên toàn bảng.
- **Độ đúng mức** (độ dốc lợi suất thật theo dự báo, 1 là hoàn hảo): dev 0,79 (EB 0,88), holdout 0,19 (EB 0,31). EB đúng mức hơn.
- **Chiến lược nhiễu (Ticker):** chiến lược xếp theo chữ cái tên mã cổ phiếu, vô nghĩa về kinh tế. Mô hình chưa từng thấy Ticker, và chỉ 0,19% top 1% của nó là Ticker (so với 43,5% của tập gộp), tức nhận ra là nhiễu. Ở top 5–10% mô hình lẫn nhiều Ticker hơn EB (3,4% và 9,3% so với 0,2% và 1,3%).
- Seed 1 được chọn bằng dev muộn 1995–2004 trong 6 seed. Trung bình 6 seed (dòng nghiêng) là số công bằng hơn nếu muốn tránh thiên lệch chọn seed; top 1% holdout là 1,03, vẫn hơn EB (0,82).
- Chưa tính chi phí giao dịch. Chỉ làm họ Accounting EW. Họ giá trị theo vốn hoá (VW) đã thử và mô hình **không** hơn EB (ghi ở [docs/6_ALL_RESULTS_SUMMARY.md](docs/6_ALL_RESULTS_SUMMARY.md), mục 3c).

Chi tiết từng cách chấm của paper (Table 2, Figure 1, Figure 4–5, Table 3), có hình: [docs/7_COMPARE_WITH_EB.md](docs/7_COMPARE_WITH_EB.md).

![Lãi cộng dồn top 1%](docs/figures/eb_cumret.png)

---

## 2. Mô hình chốt

![Kiến trúc mô hình chốt](docs/figures/final_model.png)

Từ chuỗi lợi suất đến dự báo, qua 6 khối. Mỗi khối nhận đầu ra của khối trước (`n` = số chiến lược của **cùng một năm**):

| # | Khối | Đầu ra | Học? | Vì sao có khối này |
|---|---|---|---|---|
| 0 | 128 tháng lãi trước năm Y, nhân dấu (chiến lược lỗ thì làm ngược) | `[n, 128]` | – | t-stat gom cả lịch sử thành một con số và bỏ mất hình dạng chuỗi |
| 1 | **FinCast** (pretrained, đóng băng), 4 patch × 32 tháng | `[n, 4, 1280]` | không | đọc hình dạng chuỗi mà không phải học mô hình chuỗi lớn trên lợi suất nhiễu |
| 2 | **PCA trắng hoá 16 chiều**, fit chỉ trên năm ≤ 1982, kẹp [−5, 5] | `[n, 4, 16]` | không | 5.120 số mỗi chiến lược là quá nhiều so với ~26.000 chiến lược mỗi năm |
| 3 | LayerNorm → **Bi-LSTM** (128 mỗi chiều) | `[n, 4, 256]` | có | đọc 4 patch đã nén theo thứ tự thời gian |
| 4 | **Self-attention** 4 đầu giữa 4 patch + attention pooling | `[n, 256]` | có | học đoạn nào của lịch sử quan trọng cho bài toán dự báo |
| 5 | **ISAB** (Set Transformer, 64 điểm tóm tắt) | `[n, 256]` | có | chọn chiến lược là chọn *tương đối* trong năm; chi phí tuyến tính theo `n` |
| 6 | Ghép **t-stat và lãi trung bình** (đã chuẩn hoá) → FC 64 → ReLU → BatchNorm → Dropout 0,3 → 1 | dự báo | có | dựa trên t-stat chứ không thay thế nó |

Phần học được khoảng 1,6 triệu tham số trên nền bộ mã hoá đóng băng.

**Nhãn học:** Sharpe năm sau − Sharpe trung bình của năm − **nền \|t\|** (b × \|t\| chuẩn hoá, b hồi quy trên 15 năm ≤ Y−1), cắt ở phân vị 1% và 99%. Nhãn này bắt mạng học *phần mà t-stat chưa giải thích* và tương quan giữa các chiến lược trong cùng năm. Dự báo %/tháng = (ê + nền \|t\| + mức năm) × độ dao động.

**Huấn luyện:** loss Huber × trọng số năm (kiểu Holen et al.), AdamW (lr 1e-3, weight decay 1e-2), học trên năm ≤ Y−2, kiểm tra trên Y−1, 3 vòng, học lại **mỗi năm** (walk-forward, mỗi bước là một năm). Chuẩn hoá dùng thống kê tập huấn luyện.

**Không nhìn trước (no look-ahead):** mọi thứ của năm Y chỉ dùng tháng trước tháng 1 của Y; PCA fit trên năm ≤ 1982 (trước dự báo đầu tiên); EB chỉ là đối thủ so sánh, **không bao giờ** là đầu vào hay điểm neo. Ticker chỉ dùng ở bước kiểm tra nhiễu, không dùng để học.

**Ablation** (thay một thành phần, học 3 lần mỗi cấu hình; Sharpe trung bình top 1/5/10%; dao động giữa các lần chạy của cấu hình tham chiếu: 0,03 ở dev, 0,13 ở holdout):

| Cấu hình | Dev | Holdout |
|---|---|---|
| **Tham chiếu (cấu hình chốt, học lại)** | 1,85 | 0,84 |
| FinCast khởi tạo ngẫu nhiên (không pretrained) | 1,80 | 0,66 |
| PCA 8 thành phần | 1,95 | 0,67 |
| PCA 32 thành phần | 1,85 | 0,77 |
| PCA 64 thành phần | 1,64 | 0,74 |

Thêm nhãn "Sharpe trừ trung bình năm" cho dev 1,37 → 1,84 và holdout 0,69 → 0,81, là cải thiện lớn nhất (từ quá trình phát triển từng bước, không phải giao thức 3 lần). **ISAB, Huber và trọng số năm khi bỏ từng cái đều không tách được khỏi dao động giữa các lần chạy**, nên không được viết là có đóng góp đã chứng minh. Baseline cũng dùng FinCast, Bi-LSTM và attention pooling, nên khoảng cách với baseline đo hiệu ứng chung của nén, ISAB, đảo dấu và nhãn, không phải của riêng từng khối. Bảng đầy đủ: [docs/writing_paper/ablation_results.csv](docs/writing_paper/ablation_results.csv). Chi tiết mô hình và nhật ký ~60 cấu hình đã thử (kể cả thất bại): [docs/5_FINAL_MODEL_VS_BASELINE.md](docs/5_FINAL_MODEL_VS_BASELINE.md), [docs/6_ALL_RESULTS_SUMMARY.md](docs/6_ALL_RESULTS_SUMMARY.md).

---

## 3. Chạy lại

Cần **GPU NVIDIA** (đã chạy trên RTX 5060 Ti 16 GB, CUDA 12.8). Mỗi seed khoảng 7 phút; dựng cache FinCast + PCA khoảng 9 phút.

```bash
# 1. Môi trường
conda create -n paper python=3.13 -y && conda activate paper
pip install -r requirements.txt

# 2. Mã FinCast (không nằm trong repo này): chạy FinCast cần mã nguồn của tác giả
git clone https://github.com/vincent05r/FinCast-fts.git Research/tools/FinCast-fts
#    trọng số pretrained (Vincent05R/FinCast, v1.pth) tự tải từ Hugging Face lần chạy đầu

# 3. Dữ liệu: đặt 4 CSV của paper Chen & Dim vào data_raw/ (xem mục "Dữ liệu")

# 4. Dựng cache (ma trận lãi, bảng chiến lược × năm, FinCast + PCA16)
cd code
python -c "import data; ids, wide = data.load_returns(); T = data.load_table(); data.extract_tokens(T, ids, wide)"

# 5. Học 6 seed, lưu dự báo (_cache/forecasts/) và checkpoint mỗi năm (checkpoints/final_s<seed>/)
python run.py 0 1 2 3 4 5
```

Notebook duy nhất đi từ CSV gốc đến kết quả, từng bước, có giải thích: [code/notebook/final_model.ipynb](code/notebook/final_model.ipynb) (kernel **Python (paper)**).

**Chỉ muốn kiểm tra kết quả, không học lại:** checkpoint mô hình chốt (seed 1) nằm sẵn ở `code/checkpoints/final_s1/Y1983.pt … Y2020.pt` (38 file); nạp `Y2000.pt` và dự báo lại cho kết quả chênh dự báo đã lưu 7,9e-8. GPU không tất định nên **học lại cùng seed cho số hơi khác**; checkpoint đã lưu thì dự báo cố định.

**Dữ liệu.** Mã dùng 4 CSV của paper (`DataMinedLongShortReturnsEW.csv`, `…VW.csv`, `PastReturnSignalsLongShort.csv`, `TickerSignalsLongShort.csv`, mỗi file 0,6–0,9 GB, tổng khoảng 2,8 GB) cùng các bảng tên chiến lược. Các file này **không nằm trong Git** (vượt giới hạn của GitHub). `data_sample/` có trích mẫu nhỏ để xem cấu trúc; `data_extra/` có dữ liệu đối chiếu. Chỉ họ **Accounting EW** được dùng để huấn luyện và báo cáo.

---

## 4. Cấu trúc thư mục

```
AI_Finance/
├── README.md               ← file này
├── AGENTS.md               quy tắc làm việc của dự án (mục tiêu, nguyên tắc, trạng thái)
├── requirements.txt
├── code/                   MÔ HÌNH CHỐT: chạy mô hình chốt chỉ cần thư mục này
│   ├── data.py             CSV gốc → bảng (chiến lược, năm) → cửa sổ 128 tháng → FinCast → PCA16
│   ├── model.py            kiến trúc chốt, từng khối
│   ├── train.py            nhãn, nền |t|, trọng số năm; học walk-forward; checkpoint mỗi năm
│   ├── evaluate.py         danh mục top 1/5/10%, kiểm định HAC/bootstrap, IC, độ đúng mức
│   ├── run.py              python run.py 0 1 2 3 4 5
│   ├── notebook/final_model.ipynb   từ dữ liệu gốc đến kết quả
│   ├── checkpoints/        trọng số mỗi năm, 6 seed (final_s0 … final_s5); mô hình chốt = final_s1
│   ├── _cache/             cache ma trận và dự báo (không đưa lên Git)
│   └── experiments/        mọi thứ chỉ dùng để TÌM mô hình chốt (không dùng khi chạy mô hình chốt)
├── docs/                   7 tài liệu đánh số (đọc theo thứ tự bên dưới)
│   ├── figures/            hình kiến trúc và hình so với EB
│   ├── papers/             PDF paper dùng trực tiếp
│   ├── writing_paper/      bài đang viết (Overleaf), hướng dẫn viết, danh sách sửa
│   └── zip/                gói nén mã nguồn và checkpoint seed 1, chia nhỏ <100 MB/file
├── data_raw/               4 CSV gốc của paper (không nằm trong Git)
├── data_sample/            trích mẫu nhỏ của từng CSV, để xem cấu trúc
├── data_extra/             dữ liệu đối chiếu (tài liệu tín hiệu, bảng lãi gốc)
├── Research/               hai notebook tham khảo (baseline B, chạy lại paper A)
├── docs-add/               tài liệu tham khảo thêm, bản nháp của nhóm
├── scripts/                create_data_samples.py: tạo lại data_sample/ từ data_raw/
└── .agents/                bộ skill hỗ trợ trợ lý AI (không ảnh hưởng mô hình)
```

**Đọc tài liệu theo thứ tự** (`docs/`): [6. Tổng hợp mọi kết quả (đọc đầu tiên)](docs/6_ALL_RESULTS_SUMMARY.md) → [5. Mô hình chốt so với baseline](docs/5_FINAL_MODEL_VS_BASELINE.md) → [7. So sánh với EB](docs/7_COMPARE_WITH_EB.md) → [1. Baseline](docs/1_BASELINE.md) → [2. Paper gốc](docs/2_ORIGINAL_PAPER.md) → [3. Notebook tham khảo](docs/3_REFERENCE_NOTEBOOK.md) → [4. Mục tiêu và ý tưởng](docs/4_GOALS_AND_IDEAS.md).

**Không nằm trong Git** (có trong `.gitignore`): 4 CSV gốc, `code/_cache/`, dự báo và ma trận trong `code/experiments/`, mã của tác giả (`Research/original_paper_code/`, `Research/tools/`), và các file zip lớn. Mã Chen & Dim gốc: `git clone https://github.com/chenandrewy/high-throughput-ap.git Research/original_paper_code/high-throughput-ap`. Danh sách gói chỉ có trên máy: [docs/zip/README.md](docs/zip/README.md).

---

## 5. Bài báo

Bài đang viết cho *Financial Innovation* (Springer), mẫu `sn-jnl`, nằm trong `docs/writing_paper/overleaf_project/`. Đọc [docs/writing_paper/WRITING_GUIDE.md](docs/writing_paper/WRITING_GUIDE.md) trước khi viết. Nội dung các mục để chép tay vào Overleaf: `docs/writing_paper/*_PASTE.md`; danh sách sửa theo nhận xét của thầy: [update_theo_nhan_xet_PASTE.md](docs/writing_paper/update_theo_nhan_xet_PASTE.md). Không sửa trực tiếp `sn-article.tex`.

## 6. Giới hạn

- Chỉ họ **Accounting EW**, một thị trường (Mỹ), **chưa có chi phí giao dịch**, mô hình chọn trên dev 1983–2004.
- Chưa làm: các họ khác (kế toán VW không hơn EB, họ lợi suất quá khứ chưa thử), hiệu chỉnh nhiều cấu hình (đã thử ~60 cấu hình), kiểm tra ở thị trường khác.
- Lợi thế rõ nhất ở top 1%. EB vẫn đúng mức hơn và xếp toàn bảng tốt hơn ở dev.
- Đóng góp của ISAB, Huber và trọng số năm chưa tách được khỏi dao động giữa các lần chạy.

## 7. Công cụ trợ lý AI trong repo

`.agents/skills/` chứa 15 skill nghiên cứu của [K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills) (tìm tài liệu, kiểm tra trích dẫn, viết, thống kê, trực quan hoá...). `.mcp.json` cấu hình 5 máy chủ MCP (`jupyter`, `huggingface`, `paper-search`, `arxiv`, `fetch`) với đường dẫn riêng của máy tác giả; clone về máy khác thì phải cài lại và sửa đường dẫn. Các công cụ này chỉ hỗ trợ tra cứu và viết, **không** tạo kết quả nghiên cứu. Mã tham khảo clone riêng, có Git riêng và không nằm trong repo này: [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (`Research/tools/autoresearch/`) và [chenandrewy/high-throughput-ap](https://github.com/chenandrewy/high-throughput-ap).
