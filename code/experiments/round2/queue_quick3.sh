cd "$(dirname "$0")"; export QUICK=1; PY=~/.local/share/research-kernel/venv/bin/python
until grep -q QDONE queue_quick2.log; do sleep 15; done
$PY run_x.py nobase_sub 2 3
echo Q3DONE
