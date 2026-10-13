cd "$(dirname "$0")"; export QUICK=1; PY=~/.local/share/research-kernel/venv/bin/python
$PY run_x.py H_m0 0 1; $PY run_x.py H_exact 0 1; echo H3DONE
