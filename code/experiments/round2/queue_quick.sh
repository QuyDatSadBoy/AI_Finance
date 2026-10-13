cd "$(dirname "$0")"; export QUICK=1; PY=~/.local/share/research-kernel/venv/bin/python
for c in nobase_fuse sharpe_fuse label_nobase label_sharpe label_rank_pos sharpe_fuse_sub; do $PY run_x.py $c 0 1; done
echo QDONE
