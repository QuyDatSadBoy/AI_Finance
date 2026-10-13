# Gói tải về

| File | Nội dung |
|---|---|
| `source_code.zip` | Mã nguồn: `code/` (data, model, train, evaluate, run, notebook, mã thử nghiệm), `requirements.txt`, `README.md`, `AGENTS.md` |
| `checkpoints_final_s1.zip.part00` … `part07` | Checkpoint mô hình chốt (seed 1), 38 file `Y1983.pt` … `Y2020.pt`, chia nhỏ vì GitHub giới hạn 100 MB/file |

Ghép và giải nén checkpoint (ra `code/checkpoints/final_s1/`):

```bash
cat checkpoints_final_s1.zip.part* > checkpoints_final_s1.zip
sha256sum -c SHA256SUMS
unzip checkpoints_final_s1.zip -d ../..
```

Chỉ có trên máy (quá lớn cho GitHub): `checkpoints_all_seeds.zip` (4,3 GB, cả 6 seed). Trên GitHub, checkpoint đủ 6 seed nằm sẵn ở `code/checkpoints/final_s0` … `final_s5`.

Chỉ có trên máy, gồm những gì không đẩy được lên GitHub:

| File | Nội dung |
|---|---|
| `data_raw_csv.zip` (1,1 GB) | 4 CSV dữ liệu gốc của paper (EW, VW, PastReturn, Ticker) |
| `archives.zip` (2,2 GB) | các file zip tải về ban đầu |
| `experiments_data.zip` (9,5 GB) | dự báo, ma trận, checkpoint của mọi thử nghiệm cũ trong `code/experiments/` |
| `code_cache.zip` (490 MB) | cache `code/_cache/` (ma trận lãi, FinCast PCA16, dự báo) |
