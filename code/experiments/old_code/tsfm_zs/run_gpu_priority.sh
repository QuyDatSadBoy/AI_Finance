#!/bin/bash
# Priority GPU queue for acct_ew (ONE heavy GPU job at a time; start only when the GPU is free).  Only the configurations that
# screened best in dev (ctx 240, returns input); c128 / --cum / FinCast head are skipped (no dev gain, FinCast head IC ~0.05).
# TiRex is timed on 2000 rows first; the full TiRex run starts only if that took < 10 min.  Then pack_extras.py builds the extras file.
cd "$(dirname "$0")"; FAM=${1:-acct_ew}; LOG=run_gpu_priority_${FAM}.log
PY=~/.local/share/research-kernel/venv/bin/python; T=~/.local/share/research-kernel/venv_tsfm/bin/python
run() { echo "=== $(date +%T) $*" >> $LOG; "$@" >> $LOG 2>&1; echo "=== $(date +%T) exit $?" >> $LOG; }
run $PY fc_chronos2.py $FAM 240 --xl --bs 1024          # best full-sample model in dev (IC 0.213, pIC beyond mean 0.106)
run $PY fc_chronos2.py $FAM 240 --bs 1024               # same model without cross-learning (isolates what --xl adds)
s=$(date +%s); run $T fc_tirex.py $FAM 240 --rows 2000 --bs 2048; dt=$(( $(date +%s) - s ))
echo "=== tirex 2000 rows took ${dt}s" >> $LOG
[ $dt -lt 600 ] && run $T fc_tirex.py $FAM 240 --bs 2048   # best on the 5% sample (pIC beyond mean+m12 0.065 dev)
run $T fc_timesfm.py $FAM 240 --bs 1024
run $T fc_toto2.py $FAM 240 --size 313m --bs 1024
run $PY pack_extras.py $FAM
echo "=== $(date +%T) done" >> $LOG
cp tsfm_x_${FAM}.npy tsfm_x_${FAM}.json ../_cache/ && echo "=== copied to _cache" >> $LOG
