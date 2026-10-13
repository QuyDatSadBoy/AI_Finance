until grep -q Q7DONE queue7.log; do sleep 20; done
~/.local/share/research-kernel/venv/bin/python q4.py tbase_rec5_corr 0 1 2 3 2>&1 | grep -v Warn
echo Q8DONE
