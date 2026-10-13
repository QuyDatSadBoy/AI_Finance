PY=~/.local/share/research-kernel/venv/bin/python
while pgrep -f "q4.py ema" >/dev/null; do sleep 20; done
for c in tbase tbase_rec5 corr gate01 feat delta05 rec10; do $PY q4.py $c 0 1 2>&1 | grep -v Warn; done
echo QALLDONE
