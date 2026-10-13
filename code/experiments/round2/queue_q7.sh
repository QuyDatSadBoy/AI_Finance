cd "$(dirname "$0")"; export QUICK=1; PY=~/.local/share/research-kernel/venv/bin/python
until grep -q QDONE6 queue_q6.log; do sleep 20; done
for c in H H_lion H_muon H_radam; do $PY run_x.py $c 0 1; done
echo QDONE7
