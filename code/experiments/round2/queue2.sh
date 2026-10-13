cd "$(dirname "$0")"
PY=~/.local/share/research-kernel/venv/bin/python
until grep -q QUEUEDONE queue.log; do sleep 30; done
$PY extract_sundial.py > extract_sundial.log 2>&1
for c in sundial fc_sundial; do $PY run_x.py $c 0 1 2; done
echo QUEUE2DONE
