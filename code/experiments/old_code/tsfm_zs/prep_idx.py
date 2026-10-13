"""Save row index + metadata of a family set (for alignment and IC evaluation). CPU only, run in the main venv.
Usage: python prep_idx.py acct_ew            -> idx_acct_ew.npy, meta_acct_ew.npz
       python prep_idx.py all                -> all 655,101 candidate rows (6 families)"""
import sys, numpy as np
sys.path.insert(0, "/home/phongndh/Documents/Paper/code")
import ht_common as h
from ht_sub import make_sub
OUT = "/tmp/claude-1000/-home-phongndh-Documents-Paper/386472d0-2c49-461d-bf9b-012496fbe642/scratchpad/tsfm_zs/"
fam = sys.argv[1] if len(sys.argv) > 1 else "acct_ew"
D = make_sub("+".join(h.FAMILIES) if fam == "all" else fam, with_X=False, with_H=False)
F = D.F
np.save(OUT + f"idx_{fam}.npy", D.idx_full.astype(np.int64))
np.savez(OUT + f"meta_{fam}.npz", idx=D.idx_full, year=D.yr, famcode=D.famcode, y=D.y_or, eb=D.eb_or, sd=D.sd,
         mean_or=F.mean_is.values * D.sg, absT=np.abs(F.tstat.values), nobs=D.nobs,
         cf_mean_small=F.cf_mean_small.values, cf_med_small=F.cf_med_small.values,
         cf_q10_small=F.cf_q10_small.values, cf_q90_small=F.cf_q90_small.values)
print(fam, len(D.idx_full), np.array_equal(np.sort(D.idx_full), D.idx_full))
