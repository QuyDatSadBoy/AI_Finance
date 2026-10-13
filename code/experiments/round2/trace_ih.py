"""Học IH (seed 0) 1983-2000 rồi in dữ liệu thật qua từng khối cho 18144, năm 2000."""
import sys, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import data, train, train_x as X
from run_x import CFGS
torch.set_printoptions(precision=3, sci_mode=False, linewidth=200)
T = data.load_table(); H = data.load_tokens(); ids, wide = data.load_returns(); yr = T.oos_year.values
cfg = CFGS["IH"]
p = X.walk_forward(T, H, cfg, seed=0, years=range(1983, 2001), ids=ids, wide=wide)
net, W = X.NET, X.LASTW; net.eval()
torch.save(net.state_dict(), Path(__file__).parent / "IH_s0_Y2000.pt")
e = int(np.where((T.signalid == 18144) & (yr == 2000))[0][0]); te = W.te; pos = int(np.where(te == e)[0][0])
f = lambda t: np.round(t.detach().cpu().numpy(), 3)
with torch.no_grad():
    x = H[torch.tensor(te, device="cuda")].float(); E = torch.tensor(W.E[te], device="cuda")
    print("VAO tokens", tuple(x.shape), "18144 doan4[:6]", f(x[pos, 3, :6]), "E", f(E[pos]))
    a = net.norm(x); print("LayerNorm", tuple(a.shape), f(a[pos, 3, :6]))
    b, _ = net.seq(a); print("BiLSTM", tuple(b.shape), f(b[pos, 3, :6]))
    P_ = net.pool; z = b + P_.mha(b, b, b, need_weights=False)[0]
    att = torch.softmax(P_.v(torch.tanh(P_.proj(z))).squeeze(-1), 1); c = (att.unsqueeze(-1) * z).sum(1)
    print("att4", f(att[pos]), "pool", tuple(c.shape), f(c[pos, :6]))
    isab, hyp = net.xs[0], net.xs[1]
    d = isab(c[None])[0]; print("ISAB", tuple(d.shape), f(d[pos, :6]))
    hh = hyp.blk(d[None])[0]; g = hyp.a.item(); d2 = d + g * (hh - d)
    print("gate a", round(g, 4), "Hyper", tuple(d2.shape), f(d2[pos, :6]))
    cat = torch.cat([d2, E], 1); print("concat", tuple(cat.shape))
    o = net.head(cat).squeeze(-1); print("FC out", tuple(o.shape), "18144", round(o[pos].item(), 4), "mean", round(o.mean().item(), 4), "rc", round((o - o.mean())[pos].item(), 4))
print("pred 18144 %/thang", p[e], "level", W.level, "sd", T.sd_is.values[e], "target", W.target[e])
# loss vi du nam 1998 (mot lo 4096)
j = np.where(yr == 1998)[0]; rng = np.random.default_rng(0); jj = rng.permutation(j)[:4096]
k = int(np.where(T.signalid.values[jj] == 18144)[0][0]) if (T.signalid.values[jj] == 18144).any() else None
with torch.no_grad():
    o = net(H[torch.tensor(jj, device="cuda")].float(), torch.tensor(W.E[jj], device="cuda")); o = o - o.mean()
    yt = torch.tensor(W.target[jj], device="cuda"); h = torch.nn.functional.huber_loss(o, yt, delta=1.0, reduction="none")
print("loss lo 1998: n", len(jj), "mean huber", round(h.mean().item(), 4), "w1998", W.year_weight[j[0]], "L", round(h.mean().item() * W.year_weight[j[0]], 4))
for i in range(3): print("  vi du", T.signalid.values[jj[i]], "du bao", round(o[i].item(), 3), "dap an", round(yt[i].item(), 3), "h", round(h[i].item(), 4))
