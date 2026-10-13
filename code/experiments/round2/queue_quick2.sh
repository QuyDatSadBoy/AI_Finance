cd "$(dirname "$0")"; export QUICK=1; PY=~/.local/share/research-kernel/venv/bin/python
while pgrep -f "run_x.py sharpe_fuse" >/dev/null; do sleep 10; done
for c in nobase_rank nobase_sub nobase_rank_sub nobase_fuse_sub label_nobase; do $PY run_x.py $c 0 1; done
echo QDONE
