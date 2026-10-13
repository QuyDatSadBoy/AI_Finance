cd "$(dirname "$0")"; PY=~/.local/share/research-kernel/venv/bin/python
until grep -q Q4DONE queue4.log; do sleep 30; done
$PY run_x.py sub4096 3 4 5
echo Q5DONE
