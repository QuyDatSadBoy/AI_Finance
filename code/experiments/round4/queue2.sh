until grep -q Q4DONE queue.log; do sleep 30; done
~/.local/share/research-kernel/venv/bin/python q4.py feat 0 1 2>&1 | grep -v Warn
echo Q5DONE
