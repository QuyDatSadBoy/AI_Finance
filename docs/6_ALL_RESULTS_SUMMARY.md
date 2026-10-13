# Tổng hợp toàn bộ kết quả và thử nghiệm (đến 2026-10-09)

**Đọc file này đầu tiên** để nắm toàn bộ những gì đã làm: mô hình chốt, mọi phiên bản và thử nghiệm, bài học, chỗ lưu checkpoint. Chi tiết từng bước của mô hình chốt: `docs/5_FINAL_MODEL_VS_BASELINE.md`. Nhật ký gốc từng đợt: `code/experiments/results/NHAT_KY_THU_NGHIEM.md`. Quy tắc làm việc: `AGENTS.md`.

Mọi số là **Sharpe danh mục top 1 / 5 / 10%** theo luật paper (xếp theo Sharpe dự báo, chia đều, giữ 12 tháng). **Dev** = 1983–2004 (được dùng để chọn), **holdout** = 2005–2020 (chỉ báo cáo).

---

## 1. Mô hình chốt

**Bản cũ, seed 1** — checkpoint `code/checkpoints/final_s1/Y1983.pt … Y2020.pt`.

```
128 tháng lãi (nhân dấu) → FinCast (pretrained, đóng băng) → PCA16 → [4 × 16]
→ LayerNorm → Bi-LSTM 128 → attention 4 đoạn → ISAB (64 điểm) → ghép t-stat, lãi TB → FC (như baseline) → ê
dự báo %/tháng = (ê + nền |t| + mức năm) × độ dao động
đáp án học = Sharpe năm sau − Sharpe TB năm − nền |t|   (nền = b × |t| chuẩn hoá, b hồi quy trên 15 năm ≤ Y−1)
loss = Huber × trọng số năm Holen; AdamW lr 1e-3, wd 1e-2; học ≤ Y−2, kiểm Y−1, 3 vòng, mỗi bước = cả một năm
```

| | Dev | Holdout | Cả kỳ |
|---|---|---|---|
| **Mô hình chốt (seed 1)** | **1,95 / 1,95** / 1,82 | **1,11 / 0,88 / 0,88** | **1,62 / 1,54** / 1,46 |
| Trung bình 6 seed cùng cấu hình | 1,93 / 1,94 / 1,83 | 1,03 / 0,93 / 0,90 | 1,57 / 1,54 / 1,47 |
| EB (paper) | 1,55 / 1,91 / 1,86 | 0,82 / 0,78 / 0,86 | 1,27 / 1,48 / 1,48 |
| Quy tắc \|t\| (không mô hình) | 1,95 / 1,99 / 1,85 | 0,84 / 0,75 / 0,83 | 1,52 / 1,51 / 1,47 |
| Baseline (notebook B) | 1,34 / 1,23 / 1,16 | 0,61 / 0,51 / 0,54 | 1,07 / 0,97 / 0,94 |

- So với EB (p HAC / bootstrap): top 1% dev 0,032 / 0,059; holdout 0,028 / 0,051; cả kỳ 0,005 / 0,010. Top 5%, 10%: không có ý nghĩa. Hơn EB 5/6 ô; dev top 10% thấp hơn 0,05 (p 0,32, ngang).
- IC: dev 0,151 vs EB 0,188 (p 0,013, EB hơn); holdout 0,043 vs 0,021.
- Độ đúng mức (cách của paper Figure 4–5, dấu gốc; độ dốc, 1 = hoàn hảo): dev 0,79 (EB 0,88); holdout 0,19 (EB 0,31).
- Chọn seed bằng dev muộn 1995–2004 (TB top 1/5/10%): seed 0…5 = 1,885 / **1,930** / 1,900 / 1,866 / 1,913 / 1,907.
- Kiểm tra checkpoint: nạp `Y2000.pt` và dự báo lại, chênh với dự báo đã lưu 7,9e-8.

---

## 2. Các phiên bản mô hình đầy đủ 38 năm (có checkpoint)

| Phiên bản | Kiến trúc / khác biệt | Seed | Dev (TB 6 seed) | Holdout (TB 6 seed) | Checkpoint + dự báo | Code để nạp |
|---|---|---|---|---|---|---|
| **Bản cũ = CHỐT** | ISAB, FC, nền \|t\|, mỗi bước = 1 năm | 6 | 1,93 / 1,94 / 1,83 | **1,03 / 0,93 / 0,90** | `code/checkpoints/final_s0..5` (dự báo: `_cache/forecasts/final_s*.npy`, bản sao `experiments/old_final/forecasts/`) | `code/model.py`, `code/train.py` |
| Bản cũ, lần chạy đầu | như trên | 6 | 1,95 / 1,97 / 1,85 | 0,96 / 0,88 / 0,86 | chỉ dự báo: `experiments/final_first_run/` | — |
| IH (không SwiGLU) | ISAB → HyperAttention có cổng → FC; Sharpe − TB năm (không nền); lô 4.096 | 6 | 1,85 / 1,91 / 1,80 | 0,82 / 0,87 / 0,87 | `experiments/old_final2/` | `feat_final/code/` với head = FC, 2 số thêm |
| IH + SwiGLU | như IH, đầu ra SwiGLU | 6 | 1,95 / 1,99 / 1,87 | 0,78 / 0,73 / 0,72 | `experiments/old_final3/` | `feat_final/code/` với 2 số thêm |
| IH + SwiGLU + 3 feature (feat) | thêm số tháng, Sharpe in-sample, log độ dao động | 6 | xem dưới | 0,41–0,86 (top 1%, từng seed) | `experiments/feat_final/` | `feat_final/code/` (đúng bản) |
| Baseline (notebook B) | | 3 | 1,34 / 1,23 / 1,16 | 0,61 / 0,51 / 0,54 | dự báo: `_cache/forecasts/baseline_s*.npy` | |

Từng seed của feat (học lại có checkpoint): dev muộn 1,900 / 1,922 / 1,921 / 1,953 / 2,010 / 1,785; holdout top 1% 0,86 / 0,85 / 0,41 / 0,62 / 0,60 / 0,71 → loại (dev cao, holdout thấp: khớp quá khứ).

---

## 3. Thử nghiệm theo đợt

### Đợt 1 (trước 2026-10-07): xây từ baseline lên (trung bình top 1/5/10%)

| Thử | Dev | Holdout | Kết luận |
|---|---|---|---|
| Baseline | 1,24 | 0,56 | mốc |
| + ISAB | 1,37 | 0,69 | |
| + đáp án Sharpe − TB năm | 1,84 | 0,81 | bước lớn nhất |
| + nền \|t\| | 1,87 | 0,92 | |
| + trọng số Holen = bản cũ | 1,90 | 0,95 | **chốt** |
| HyperAttention thay ISAB | 1,81 | 0,88 | ngang |
| Upsampling / mixup | 1,86 / 1,85 | 0,79 / 0,89 | không giúp |
| Đặc trưng tài chính / momentum | 1,47 / 1,47 | 0,84 / 0,89 | kém |
| Encoder Chronos-2, EXAONE, Sundial, FinCast + Sundial | ≤ 1,89 | ≤ 0,80 | kém hơn FinCast |
| Không pretrained | thấp hơn | | |
| Không validation / học một lần đến 2004 | 1,43 / – | 0,57 / 0,77 | cần validation, học lại mỗi năm |
| 2 khối ISAB / ISAB 128 điểm | 1,92 / 1,90 | 0,88 / 0,81 | ngang / kém |
| ListNet / tương quan trong loss / SWA | 1,76 / 1,82 / 1,86 | 0,90 / 0,81 / 0,86 | kém |
| Thêm t-stat 60/24 tháng | 1,50 | 0,86 | kém |
| Loss MSE, likelihood có trọng số | ≈ Huber | | không khác |

### Đợt 2 (2026-10-07/08): thay Bước 6, loss, feature, optimizer (`round2/`, dev, 2 seed)

- Phát hiện: cùng seed chạy lại, dự báo tương quan chỉ 0,70–0,87 (GPU không tất định); Sharpe từng lần lệch ±0,1–0,25.
- Bỏ nền \|t\|, lô 4.096: dev 1,915 (chọn lúc đó). HyperAttention thay thẳng ISAB: 0,40–1,71 (học thuộc). ISAB → HyperAttention có cổng (`IH`): 1,917, IC 0,187.
- Số vòng học (cấu hình chốt): vòng có val loss nhỏ nhất ở 3 seed, đếm trên 38 năm, là vòng 0 (giữ trọng số năm trước) 22–27 năm, vòng 1–2 6–12 năm, vòng 3 4–7 năm. Học 10 vòng (1 seed): dev 1,86 / 1,96 / 1,84, holdout 0,90 / 0,85 / 0,84, nằm trong khoảng 3 vòng (dev TB 1,86–1,94) → giữ 3 vòng.
- Nhãn 24 tháng 1,951 (holdout kém khi chạy đủ); Rank IC khả vi 1,922; CalibRank 1,894; factor momentum 1,510; RAdam / Lion / Schedule-Free / Muon 1,901 / 1,849 / 1,628 / 1,569; thứ hạng làm đáp án 1,58; học thêm 3 họ khác 1,43.

### Đợt 3 (2026-10-08): tinh chỉnh từng khối trên `IH` (`round3/`, dev, 2 seed)

| Thử | Dev TB |
|---|---|
| **Đầu ra SwiGLU** (4 seed) | 1,949 (seed 0+1: 2,012) |
| GRN (TFT) | 1,958 |
| GELU / FC gốc / LayerNorm thay BatchNorm | 1,816 / 1,809 / 0,53 |
| SwiGLU + bỏ LayerNorm / PCA 8 / cửa sổ học toàn bộ | 2,009 / 1,989 / 1,957 |
| SwiGLU + Transformer thay Bi-LSTM / GRU | 1,943 / 1,908 |
| SwiGLU + lr 3e-3 / 3e-4; cửa sổ 300k; PCA 32 / 64 | 1,937 / 1,879; 1,918; 1,917 / 1,799 |
| SwiGLU + bỏ LayerNorm + PCA 8 | 1,921 |

→ Chốt tạm IH + SwiGLU; chạy đủ 6 seed: holdout 0,78 / 0,73 / 0,72 (thua EB) → mở đợt 4.

### Đợt 4 (2026-10-08/09): tìm cấu hình giữ được ở holdout (`round4/`, đủ 38 năm)

Trung bình các seed; từng seed ở `round4/all_seeds.csv`.

| Cấu hình | Seed | Dev | Holdout | Ô hơn EB |
|---|---|---|---|---|
| feat (+3 feature) — lần thử | 2 | 1,97 / 2,01 / 1,88 | 0,90 / 0,88 / 0,87 | 6/6 (học lại 6 seed: không giữ được) |
| tbase_rec5_corr (nền \|t\| + năm gần + loss tương quan) | 4 | 1,93 / 1,98 / 1,85 | 0,84 / 0,90 / 0,91 | 5/6 |
| gate01 (cổng HyperAttention 0,1) | 2 | 1,94 / 1,96 / 1,84 | 0,90 / 0,87 / 0,80 | 4/6 |
| rec5 (năm gần nặng hơn) | 2 | 1,95 / 1,99 / 1,83 | 0,92 / 0,82 / 0,79 | 4/6 |
| corr (Huber + 0,1·(1 − tương quan)) | 2 | 1,95 / 1,98 / 1,84 | 0,95 / 0,82 / 0,78 | 4/6 |
| tbase_rec5 | 6 | 1,93 / 2,01 / 1,86 | 0,82 / 0,78 / 0,83 | 3/6 |
| tbase_nohyper (nền \|t\|, bỏ HyperAttention) | 2 | 1,96 / 2,02 / 1,91 | 0,75 / 0,75 / 0,73 | 3/6 |
| tbase (IH + SwiGLU + nền \|t\|) | 2 | 1,95 / 2,02 / 1,89 | 0,76 / 0,72 / 0,77 | 3/6 |
| tbase_old_rec5 | 2 | 1,63 / 1,70 / 1,60 | 0,84 / 0,84 / 0,83 | 3/6 |
| ema (trọng số trượt 0,99) | 2 | 1,92 / 1,94 / 1,86 | 0,70 / 0,70 / 0,72 | 2/6 |
| tbase_old (dựng lại bản cũ, lô 4.096) | 2 | 1,74 / 1,90 / 1,79 | 0,82 / 0,76 / 0,78 | 2/6 |
| tbase_fc (nền \|t\|, FC thay SwiGLU, có HyperAttention) | 2 | 1,87 / 1,89 / 1,77 | 0,68 / 0,74 / 0,80 | 1/6 |

Hậu xử lý (không học lại):
- Trộn hạng mô hình với \|t\| (α chọn bằng dev = 1, tức không trộn); holdout tốt nhất 0,87 / 0,76 / 0,80.
- Xếp theo Sharpe in-sample, lọc \|t\| > c: holdout ≤ 0,85 / 0,77 / 0,83.
- Ghép bản cũ + IH-SwiGLU 12 seed: dev 1,96 / 1,97 / 1,86, holdout 1,00 / 0,86 / 0,86. Cặp seed 5: dev 1,99 / 1,98 / 1,855, holdout 0,91 / 0,87 / 0,87.

**Từng seed vượt EB 5/6 ô** (không seed nào 6/6 trong 60 seed): bản cũ s1, s3, s4, s5; bản cũ lần đầu s1, s3; corr s0; feat (lần thử) s1; tbase_rec5 s0; IH-SwiGLU s1.

---

## 3b. Kiểm tra trên họ nhiễu Ticker (2026-10-11, `code/experiments/ticker_test/`)

Ticker = chiến lược xếp theo chữ cái tên mã, cố tình vô nghĩa (paper: 0% top 1%). Mô hình chốt seed 1 **không học lại, không học trên Ticker**, dự báo cho kế toán EW + Ticker EW gộp mỗi năm (Ticker = 43,5% số dòng). EB fit trên t-stat gộp.

| | % Ticker top 1% | top 5% | top 10% | Sharpe dự báo TB Ticker / kế toán |
|---|---|---|---|---|
| Mô hình (seed 1) | 0,19 | 3,43 | 9,30 | 0,061 / 0,096 |
| EB | 0,00 | 0,18 | 1,25 | 0,017 / 0,061 |
| Quy tắc \|t\| | 0,00 | 0,52 | 2,88 | |

- Danh mục gộp: mô hình dev 2,02 / 1,83 / 1,70, holdout 1,05 / 0,86 / 0,90; EB dev 1,72 / 1,87 / 1,79, holdout 0,75 / 0,83 / 0,88. Top 1% vẫn hơn EB; top 5/10% dev giảm vì lẫn Ticker.
- IC trên riêng Ticker ≈ 0 (mô hình −0,035, EB −0,044): không đoán được Ticker, đúng như nhiễu. 33 Ticker lọt top 1% có lãi thật TB −0,07 %/tháng.
- Kết luận: mô hình **gần như** loại Ticker ở top 1% như paper, nhưng co Ticker yếu hơn EB (dự báo Ticker ≈ 64% kế toán, EB ≈ 28%) → ở top 5–10% lẫn Ticker nhiều hơn.

## 4. Bài học

1. **Nền \|t\| + kiến trúc đơn giản (ISAB, FC) tổng quát tốt nhất sau 2005.** Cả 6 seed bản cũ đều hơn EB ở holdout top 1% và 5%.
2. HyperAttention, SwiGLU, thêm feature làm **dev cao hơn nhưng holdout thấp hơn** (khớp quá khứ).
3. Mỗi bước học = cả một năm tốt hơn lô 4.096 cho bản cũ (dựng lại với lô 4.096 kém rõ).
4. Nhiễu giữa các seed (±0,1–0,25) lớn hơn hầu hết khác biệt giữa cấu hình → chỉ tin kết quả đủ 38 năm, nhiều seed. Trung bình 2 seed có thể đánh lừa (feat).
5. Đổi đáp án sang "Sharpe − TB năm" là bước tăng lớn nhất từ baseline.
6. Dev top 10% rất khó vượt EB: EB 1,86, quy tắc \|t\| cũng chỉ 1,85.

## 5. Chưa làm

Các họ khác (Accounting VW, Past Return EW/VW); Ticker; fine-tune FinCast; hiệu chỉnh p cho nhiều cấu hình (`old_code/verify/maxstat.py`); chi phí giao dịch; chế độ GPU tất định khi học lại.

## 6. Bản đồ file

| Đường dẫn | Nội dung |
|---|---|
| `code/checkpoints/final_s0..5/` | checkpoint mô hình chốt (seed 1 là bản chốt), 228 file |
| `code/_cache/forecasts/final_s*.npy` | dự báo mô hình chốt; `baseline_s*.npy` dự báo baseline |
| `code/experiments/old_final/forecasts/` | bản sao dự báo mô hình chốt |
| `code/experiments/old_final2/`, `old_final3/`, `feat_final/` | checkpoint (chỉ trọng số mạng, đã bỏ trạng thái optimizer để tiết kiệm ổ) + dự báo IH, IH + SwiGLU, feat |
| `code/experiments/feat_final/code/` | `model.py`, `train.py` của dòng IH/SwiGLU/feat |
| `code/experiments/final_first_run/` | dự báo lần chạy đầu của bản cũ |
| `code/experiments/round2/`, `round3/`, `round4/` | mã + dự báo các đợt thử (`round4/all_seeds.csv`: 60 seed) |
| `code/experiments/config_forecasts/` | dự báo ~1.469 cấu hình cũ (đặt tên theo hash) |
| `code/experiments/results/NHAT_KY_THU_NGHIEM.md` | nhật ký gốc |
| `docs/1_…4_*.md` | baseline, paper gốc, notebook tham khảo, mục tiêu và ý tưởng |
| `code/notebook/final_model.ipynb` | chạy mô hình chốt từ dữ liệu gốc đến kết quả |
| `docs/5_FINAL_MODEL_VS_BASELINE.md` | mô hình chốt so với baseline, từng bước |
| `docs/7_COMPARE_WITH_EB.md`, `docs/figures/` | so với EB theo 4 cách chấm của paper, có hình (`code/experiments/eb_compare/make.py`) |
