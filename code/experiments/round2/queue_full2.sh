cd "$(dirname "$0")"; PY=~/.local/share/research-kernel/venv/bin/python
$PY run_x.py IH_lh24 3 4 5; $PY run_x.py IH 3 4 5
echo FULL2DONE
