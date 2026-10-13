cd "$(dirname "$0")"; export QUICK=1; PY=~/.local/share/research-kernel/venv/bin/python
while pgrep -f "run_x.py IH 0 1" >/dev/null; do sleep 10; done
for c in IH_calib IH_srank IH_zt IH_ydist IH_nobs IH_lion IH_muon IH_radam IH_sf; do $PY run_x.py $c 0 1; done
echo IHDONE
