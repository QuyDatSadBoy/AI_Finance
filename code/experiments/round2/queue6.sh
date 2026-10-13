cd "$(dirname "$0")"; PY=~/.local/share/research-kernel/venv/bin/python
$PY run_x.py label_rank 0 1 2
$PY run_x.py label_nobase 0 1 2
echo Q6DONE
