cd "$(dirname "$0")"; export QUICK=1; PY=~/.local/share/research-kernel/venv/bin/python
$PY run_x.py IH 0 1; $PY run_x.py Hg 0 1; echo H4DONE
