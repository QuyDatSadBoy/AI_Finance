until grep -q Q5DONE queue2.log; do sleep 30; done
for c in tbase tbase_rec5; do ~/.local/share/research-kernel/venv/bin/python q4.py $c 0 1 2>&1 | grep -v Warn; done
echo Q6DONE
