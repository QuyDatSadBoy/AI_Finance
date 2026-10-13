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
