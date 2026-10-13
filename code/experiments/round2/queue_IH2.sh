cd "$(dirname "$0")"; export QUICK=1; PY=~/.local/share/research-kernel/venv/bin/python
until grep -q IHDONE queue_IH.log; do sleep 20; done
$PY run_x.py IH_fmom 0 1; echo IH2DONE
