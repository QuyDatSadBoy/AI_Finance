cd "$(dirname "$0")"; PY=~/.local/share/research-kernel/venv/bin/python
until grep -q Q6DONE queue6.log; do sleep 30; done
$PY run_x.py pair1 0 1 2
$PY run_x.py pair10 0 1 2
echo Q7DONE
