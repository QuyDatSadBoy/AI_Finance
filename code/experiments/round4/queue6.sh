PY=~/.local/share/research-kernel/venv/bin/python
while pgrep -f "q4.py tbase 0 1" >/dev/null; do sleep 20; done
for c in tbase_old tbase_rec5 tbase_old_rec5 tbase_fc tbase_nohyper corr gate01 feat; do $PY q4.py $c 0 1 2>&1 | grep -v Warn; done
echo QALLDONE
