cd "$(dirname "$0")"
PY=~/.local/share/research-kernel/venv/bin/python
for c in isab_m128 listnet05 corr05 feat_t60 isab_2lop ema09 feat_nobs hid64 drop05; do $PY run_x.py $c 0 1 2; done
$PY run_x.py final 6 7 8 9 10 11
echo QUEUEDONE
