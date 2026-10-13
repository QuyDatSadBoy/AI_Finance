# AGENTS.md — AI dự báo lợi suất chiến lược, cải tiến từ baseline để vượt EB (Chen & Dim, High-Throughput Asset Pricing)

## 1. Mục tiêu

- **Bài toán:** với mỗi chiến lược long–short và mỗi năm Y (1983–2020), dự báo **lợi suất trung bình tháng của năm Y** (`pred_ret`) chỉ từ dữ liệu trước Y. Đây đúng là đại lượng EB của paper (arXiv 2311.10685) dự báo.
- **Phạm vi hiện tại:** học trên **một họ có tính thống kê, ưu tiên Accounting EW**; sau đó các họ kế toán / lợi suất quá khứ khác. **Không học từ Ticker**; Ticker chỉ dùng ở bước cuối để suy luận (paper: Ticker chiếm 0% top 1%).
- **Thước đo kết quả (theo paper và notebook tham khảo), quan trọng nhất là (1):**
  1. **Danh mục top 1/5/10%** theo luật của paper (đảo dấu theo dự báo, xếp theo Sharpe dự báo, chia đều): lợi suất, t-stat, Sharpe; dev và holdout.
  2. **Dự báo không chệch** (Hình 4–5 paper): lợi suất thật so với dự báo theo nhóm.
  3. **IC theo năm** (tương quan hạng dự báo – lợi suất thật).
  MSE chỉ là loss huấn luyện, không phải thước đo kết quả.

## 2. Bắt buộc đọc trước khi làm

- **`docs/6_ALL_RESULTS_SUMMARY.md`: đọc đầu tiên** — tổng hợp mọi thứ đã làm (mô hình chốt, mọi phiên bản, mọi thử nghiệm, bài học, chỗ lưu checkpoint). Cập nhật file này sau mỗi đợt thử.
- `docs/1_BASELINE.md`: baseline ban đầu (notebook tham khảo: FinCast → Bi-LSTM → attention → concat t-stat, mean_ret → FC → `pred_ret`) và kết quả của nó.
- `docs/2_ORIGINAL_PAPER.md`: paper Chen & Dim làm từng bước, số liệu thật.
- `docs/3_REFERENCE_NOTEBOOK.md`: notebook tham khảo (B huấn luyện, A chạy lại paper) làm từng ô.
- `docs/4_GOALS_AND_IDEAS.md`: mục tiêu, ý tưởng, nguyên tắc.
- `docs/5_FINAL_MODEL_VS_BASELINE.md`: mô hình chốt so với baseline, kết quả, nhật ký mọi cấu hình đã thử.
- `docs/7_COMPARE_WITH_EB.md`: so với EB theo đúng 4 cách chấm của paper (Table 2, Figure 1, Figure 4–5, Table 3), có hình (`docs/figures/`, tạo bằng `code/experiments/eb_compare/make.py`).
- Paper chủ dự án gửi: `docs/papers/` (HyperAttention ICLR 2024, Holen et al. arXiv 2304.12778, Chen & Dim). Tài liệu tham khảo thêm: `docs-add/`.
- Tìm khối / pretrained / paper mới qua MCP `huggingface`, `arxiv`, `paper-search`; ghi rõ nguồn.

## 3. Nguyên tắc (vi phạm là làm sai)

- **A. Từ baseline đi lên:** giữ kiến trúc nền (encoder → khối chuỗi → attention → FC → `pred_ret`), mỗi lần thay **một** khối hoặc kỹ thuật, so theo cặp seed với cấu hình trước, truy vết đầu vào/đầu ra từng khối (`model.FinalNet(trace=...)`) để tìm nguyên nhân trước khi đổi tiếp.
- **B. Không làm láo với EB:** **cấm** dùng đầu ra EB (`thetahat`, `pred_ret` của EB, `eb_*`) làm đầu vào, điểm neo hay "EB × hệ số". EB chỉ là đối thủ so sánh.
- **C. Loss = sai số dự báo hoặc likelihood của lợi suất** (có thể đánh trọng số, thêm thành phần tương quan dự báo–nhãn). Không đưa lợi suất/Sharpe danh mục vào loss; biến thể kiểu R-weighted chỉ là thăm dò, không được chọn.
- **D. Không rò rỉ:** mọi thứ cho năm Y chỉ dùng dữ liệu trước Y; chuẩn hóa bằng thống kê tập train. Mô hình pretrained được dùng thoải mái nhưng phải **ghi rõ là pretrained** và báo kèm bản không pretrained.
- **E. Chọn và báo cáo trung thực:** chọn cấu hình **chỉ bằng dev 1983–2004**; holdout 2005–2020 chỉ báo cáo. Ghi lại mọi cấu hình đã thử (kể cả thất bại). Kiểm định Sharpe: HAC (`evaluate.sharpe_test_hac`) và bootstrap khối năm ≥ 10.000 lần; lợi suất: Newey–West; hiệu chỉnh nhiều cấu hình: `code/experiments/old_code/verify/maxstat.py`. Luôn báo mốc "xếp theo abs(t)" (không mô hình) cạnh EB. Trích dẫn phải xác minh qua MCP.

## 4. Mã nguồn (`code/`)

- `data.py`: CSV gốc → ma trận lãi → bảng (chiến lược, năm) + EB đối thủ → cửa sổ 128 tháng → FinCast (pretrained) → PCA16.
- `model.py`: kiến trúc chốt từng khối (LayerNorm → Bi-LSTM → attention 4 đoạn → ISAB → FC).
- `train.py`: chuẩn bị từng năm (đáp án = Sharpe − TB năm − nền |t|, cắt 1/99%, mức năm, trọng số Holen) và walk-forward (mỗi bước = một năm); checkpoint mỗi năm.
- `evaluate.py`: danh mục top-k, kiểm định HAC / bootstrap / JKM, IC, hình đúng mức.
- `run.py`: chạy đủ seed → `_cache/forecasts/final_s<seed>.npy`, `checkpoints/final_s<seed>/Y<năm>.pt`.
- Notebook duy nhất: `code/notebook/final_model.ipynb` (từ dữ liệu gốc đến kết quả).
- Mọi thứ chỉ dùng để tìm mô hình chốt (mã, notebook, dự báo của ~40 cấu hình, bảng kết quả, docs cũ): `code/experiments/`. Không dùng khi chạy mô hình chốt.

## 5. Môi trường

- Chạy tay / notebook: conda env `paper` (`requirements.txt`, kernel "Python (paper)"). Jupyter MCP dùng kernel "Paper Research (torch GPU)" = `~/.local/share/research-kernel/venv` (cùng phiên bản thư viện).
- GPU RTX 5060 Ti 16 GB: **mỗi lần một job GPU**. Job dài: `nohup` + Monitor; **không** dùng `pkill -f` với mẫu có trong chính lệnh (tự giết shell).
- Đã gỡ (2026-10-07) để dọn máy: `venv_tsfm` (TiRex/TimesFM/Toto), CUDA toolkit và GCC14 tự cài, cache Hugging Face của các mô hình ngoài FinCast. Thử lại các encoder đó phải cài và tải lại.
- Notebook: tạo/sửa qua MCP `jupyter`; thư mục `code/notebook/`. `docs/` chỉ gồm 7 file đánh số (+ `figures/`, `papers/`, `package/` = zip mã nguồn và checkpoint), không thêm file md rời.

## 6. Trạng thái (2026-10-09)

- **Mô hình chốt (seed 1):** FinCast (pretrained) → PCA16 → LayerNorm → Bi-LSTM → attention 4 đoạn → ISAB → ghép t-stat, lãi TB → FC (như baseline); đáp án = Sharpe năm sau − Sharpe TB năm − **nền |t|** (b × |t| chuẩn hoá, b hồi quy trên 15 năm ≤ Y−1); Huber × trọng số năm Holen; AdamW; học ≤ Y−2, kiểm Y−1, học lại mỗi năm, mỗi bước = một năm; học 6 seed, chọn seed 1 bằng dev muộn 1995–2004. Checkpoint `code/checkpoints/final_s1/`. Mô tả: `docs/5_FINAL_MODEL_VS_BASELINE.md`; nhật ký: `code/experiments/results/NHAT_KY_THU_NGHIEM.md`; tổng hợp mọi kết quả: `docs/6_ALL_RESULTS_SUMMARY.md`.
- Sharpe top 1/5/10%: dev 1,95/1,95/1,82 (EB 1,55/1,91/1,86; |t| 1,95/1,99/1,85; baseline 1,34/1,23/1,16); holdout 1,11/0,88/0,88 (EB 0,82/0,78/0,86). Top 1% hơn EB có ý nghĩa (dev HAC p 0,032; holdout 0,028). IC dev 0,151 vs EB 0,188 (p 0,013); holdout 0,043 vs 0,021.
- Đã thử, không chọn (holdout kém): HyperAttention, SwiGLU, thêm 3 feature, trọng số năm gần, loss tương quan, EMA (`code/experiments/old_final3`, `feat_final`, `round4`).
- GPU không tất định → học lại cùng seed cho số khác; checkpoint đã lưu thì dự báo cố định.
- **Chưa làm:** các họ khác; Ticker; hiệu chỉnh nhiều cấu hình; chi phí giao dịch.
