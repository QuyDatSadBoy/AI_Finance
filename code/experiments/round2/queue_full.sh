cd "$(dirname "$0")"; PY=~/.local/share/research-kernel/venv/bin/python
for c in IH_lh24 IH_nobs IH; do $PY run_x.py $c 0 1 2; done
echo FULLDONE
