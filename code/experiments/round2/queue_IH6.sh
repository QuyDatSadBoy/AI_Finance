cd "$(dirname "$0")"; export QUICK=1; PY=~/.local/share/research-kernel/venv/bin/python
until grep -q IH5DONE queue_IH5.log; do sleep 20; done
for c in IH_lh24_nobs IH_lh24_srank IH_lh24_nobs_srank IH_lh36; do $PY run_x.py $c 0 1; done
$PY run_x.py IH_lh24 2 3
echo IH6DONE
