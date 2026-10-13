cd "$(dirname "$0")"; PY=~/.local/share/research-kernel/venv/bin/python
until grep -q Q3DONE queue3.log; do sleep 30; done
$PY families.py > families.log 2>&1
$PY run_x.py fam_all 0 1 2
$PY run_x.py label_sharpe 0 1 2
echo Q4DONE
