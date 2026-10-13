cd "$(dirname "$0")"; export QUICK=1; PY=~/.local/share/research-kernel/venv/bin/python
until grep -q QDONE5 queue_q5.log; do sleep 20; done
for c in N_feat_zt N_feat_ydist; do $PY run_x.py $c 0 1; done
echo QDONE6
