cd "$(dirname "$0")"; export QUICK=1; PY=~/.local/share/research-kernel/venv/bin/python
for c in N_tok N_tftinit N_calib01 N_calib1 N_pair1 N_pair10 N_listnet; do $PY run_x.py $c 0 1; done
echo QDONE
