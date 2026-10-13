cd "$(dirname "$0")"; export QUICK=1; PY=~/.local/share/research-kernel/venv/bin/python
until grep -q QDONE7 queue_q7.log; do sleep 20; done
$PY run_x.py H_sf 0 1
echo QDONE8
