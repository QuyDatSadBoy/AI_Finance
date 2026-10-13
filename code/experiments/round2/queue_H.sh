cd "$(dirname "$0")"; export QUICK=1; PY=~/.local/share/research-kernel/venv/bin/python
for c in H H_feat_zt H_feat_ydist H_srank01 H_calib01 H_lion H_muon H_radam H_sf H_feat_nobs; do $PY run_x.py $c 0 1; done
echo HDONE
