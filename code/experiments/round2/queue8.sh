cd "$(dirname "$0")"; PY=~/.local/share/research-kernel/venv/bin/python
$PY run_x.py sharpe_fuse 0 1 2
$PY run_x.py sharpe_fuse_sub 0 1 2
$PY run_x.py label_rank_pos 0 1 2
$PY run_x.py sharpe_sub 0 1 2
echo Q8DONE
