cd "$(dirname "$0")"; export QUICK=1; PY=~/.local/share/research-kernel/venv/bin/python
until grep -q IH4DONE queue_IH4.log; do sleep 20; done
$PY run_x.py IH_nobs_srank 0 1
$PY run_x.py IH 2 3; $PY run_x.py IH_nobs 2 3; $PY run_x.py IH_srank 2 3
echo IH5DONE
