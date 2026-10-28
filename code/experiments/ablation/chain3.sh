#!/bin/bash
cd /home/phongndh/Documents/Paper/code/experiments/ablation
PY="conda run --no-capture-output -n paper python"
for v in noisab mse nowgt nobase pca8 pca32 pca64 nopre; do $PY run_ablation.py $v 1; done
echo CHAIN3_DONE
