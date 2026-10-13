cd "$(dirname "$0")"; export QUICK=1; PY=~/.local/share/research-kernel/venv/bin/python
until grep -q IH2DONE queue_IH2.log; do sleep 20; done
for c in IH_lh24 IH_lh6 IH_b256 IH_m512; do $PY run_x.py $c 0 1; done; echo IH3DONE
