#!/bin/bash
cd /home/phongndh/Documents/Paper/code/experiments/ablation
while ! grep -q CHAIN_DONE chain.log; do sleep 60; done
conda run --no-capture-output -n paper python run_ablation.py nobase 0 1 2
echo CHAIN2_DONE
