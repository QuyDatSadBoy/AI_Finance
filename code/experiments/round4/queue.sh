PY=~/.local/share/research-kernel/venv/bin/python
for c in rec5 ema corr gate01 delta05 rec10 corr05; do $PY q4.py $c 0 1 2>&1 | grep -v Warn; done
echo Q4DONE
