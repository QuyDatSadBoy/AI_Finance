cd "$(dirname "$0")"; PY=~/.local/share/research-kernel/venv/bin/python
L=/tmp/claude-1000/-home-phongndh-Documents-Paper/386472d0-2c49-461d-bf9b-012496fbe642/scratchpad/run_final2.log
until [ -f NBDONE ]; do sleep 30; done
for k in base gelu ln swiglu grn drop01 lr3e-4 lr3e-3 wd1e-3 wd1e-1 b1024 b2048 b8192 noln gru transformer pool_attn pool_mean win300k winall; do $PY q.py $k 0 1; done
$PY pca64.py > pca64.log 2>&1
for k in pca8 pca32 pca64; do $PY q.py $k 0 1; done
echo QDONE
