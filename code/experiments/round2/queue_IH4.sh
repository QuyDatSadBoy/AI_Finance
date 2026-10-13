cd "$(dirname "$0")"; export QUICK=1; PY=~/.local/share/research-kernel/venv/bin/python
until grep -q IH3DONE queue_IH3.log; do sleep 20; done
$PY run_x.py IH_cal 0 1; echo IH4DONE
