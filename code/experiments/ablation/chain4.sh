#!/bin/bash
cd /home/phongndh/Documents/Paper/code/experiments/ablation
PY="conda run --no-capture-output -n paper python"
$PY run_ablation.py full 0 2
$PY run_ablation.py noisab 2
for v in mse nowgt nobase pca8 pca32 pca64 nopre; do $PY run_ablation.py $v 0 2; done
echo CHAIN4_DONE
