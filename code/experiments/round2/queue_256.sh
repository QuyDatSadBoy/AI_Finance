cd "$(dirname "$0")"; PY=~/.local/share/research-kernel/venv/bin/python
$PY extract_fc256.py > extract_fc256.log 2>&1
QUICK=1 $PY run_x.py IH_256 0 1 2 3
echo D256
