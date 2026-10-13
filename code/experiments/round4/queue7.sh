until grep -q QALLDONE queue6.log; do sleep 20; done
~/.local/share/research-kernel/venv/bin/python q4.py tbase_rec5 2 3 4 5 2>&1 | grep -v Warn
echo Q7DONE
