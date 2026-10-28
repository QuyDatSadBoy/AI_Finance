#!/bin/bash
cd /home/phongndh/Documents/Paper/code/experiments/ablation
PY="conda run --no-capture-output -n paper python"
$PY extract.py pca64 && $PY extract.py rand16
for v in noisab mse nowgt pca8 pca32 pca64 nopre; do $PY run_ablation.py $v 0 1 2; done
echo CHAIN_DONE
