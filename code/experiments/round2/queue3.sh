cd "$(dirname "$0")"; PY=~/.local/share/research-kernel/venv/bin/python
$PY run_x.py diag 0 1 2 && $PY run_x.py sub4096 0 1 2 && $PY run_x.py ep10 0 1 2; echo Q3DONE
