cd "$(dirname "$0")"; PY=~/.local/share/research-kernel/venv/bin/python
$PY q.py swiglu 2 3
for k in sw_lr3e-4 sw_lr3e-3 sw_wd1e-3 sw_wd1e-1 sw_b2048 sw_b8192 sw_gru sw_pool_attn sw_win300k sw_winall sw_noln sw_transformer; do $PY q.py $k 0 1; done
echo Q2DONE
$PY pca64.py > pca64.log 2>&1
for k in sw_pool_mean sw_pca8 sw_pca32 sw_pca64; do $PY q.py $k 0 1; done
echo Q3DONE
