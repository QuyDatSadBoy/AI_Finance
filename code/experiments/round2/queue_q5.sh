cd "$(dirname "$0")"; export QUICK=1; PY=~/.local/share/research-kernel/venv/bin/python
until grep -q QDONE queue_q4.log; do sleep 20; done
for c in N_srank01 N_srank05 N_feat_nobs; do $PY run_x.py $c 0 1; done
echo QDONE5
