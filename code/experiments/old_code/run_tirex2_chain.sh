#!/bin/sh
# after queue 15: extract TiRex-2 hidden states (venv_tsfm + user-space CUDA 12.8 + GCC 14 host compiler), then queue 16
cd /home/phongndh/Documents/Paper/code
until grep -a -q "QUEUE DONE" _cache/run_q15.log 2>/dev/null; do sleep 30; done
G=$HOME/.local/share/research-kernel/mamba/gcc14/bin
export CUDA_HOME=$HOME/.local/share/research-kernel/cuda-12.8 PATH=$HOME/.local/share/research-kernel/venv_tsfm/bin:$HOME/.local/share/research-kernel/cuda-12.8/bin:$PATH
export NVCC_APPEND_FLAGS="-ccbin $G/x86_64-conda-linux-gnu-g++" CXX=$G/x86_64-conda-linux-gnu-g++ CC=$G/x86_64-conda-linux-gnu-gcc TORCH_CUDA_ARCH_LIST="12.0"
$HOME/.local/share/research-kernel/venv_tsfm/bin/python extract_tirex2.py 240 acct_ew > _cache/tirex2.log 2>&1
$HOME/.local/share/research-kernel/venv_tsfm/bin/python extract_tirex2.py 128 acct_ew >> _cache/tirex2.log 2>&1
$HOME/.local/share/research-kernel/venv/bin/python run_q16.py > _cache/run_q16.log 2>&1
