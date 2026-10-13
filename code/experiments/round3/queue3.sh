cd "$(dirname "$0")"; PY=~/.local/share/research-kernel/venv/bin/python
for k in sw_gru sw_transformer sw_win300k sw_winall sw_noln; do $PY q.py $k 0 1; done
$PY pca64.py > pca64.log 2>&1
for k in sw_pca8 sw_pca32 sw_pca64; do $PY q.py $k 0 1; done
echo Q3DONE
