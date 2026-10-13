cd "$(dirname "$0")"; export QUICK=1; PY=~/.local/share/research-kernel/venv/bin/python
$PY run_x.py H_ev 0 1; $PY run_x.py H_full 0 1; echo H2DONE
