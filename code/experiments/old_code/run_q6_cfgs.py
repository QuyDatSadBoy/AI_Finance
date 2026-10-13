"""Named acct_ew configurations (for diag_acct.py and later queues)."""
import ht_base as B
F = "acct_ew"
T = {**B.BASE, "xs": True, "prep": "ydm", "pool": "mha", "fams": F}
A0 = {**T, "hpca": 16, "rc": True, "xset": 4096, "detev": True}
A1 = {**A0, "base": "lin4", "bK": 15}
A2 = {**A1, "lvl": "exp1"}
A3 = {**A2, "ep0": True, "seed2": True, "optkeep": True}
A4 = {**A3, "fx": "dm"}
A5 = {**A4, "hln": False}
CFGS = {"T": T, "Th16": {**T, "hpca": 16}, "T3h16": {**B.BASE, "xs": True, "prep": "ydms", "pool": "mha", "extras": "F2", "fams": F, "hpca": 16},
        "A0": A0, "A1": A1, "A2": A2, "A3": A3, "A4": A4, "A5": A5}
# queue 7 (grids of queue-6 steps + Sharpe-unit label)
CFGS.update({"A1_K10": {**A1, "bK": 10}, "A1_K20": {**A1, "bK": 20}, "A2_r10": {**A1, "lvl": "roll10"},
             "B2": {**A2, "prep": "ysdm"}, "B3": {**A3, "prep": "ysdm"}})
# queue 8: time-series signal (CPU study: own momentum months 2-12 has partial IC ~0.09 in dev and holdout beyond the in-sample mean;
# short pre-1980 windows make pooled fits underweight the mean)
CFGS.update({"C1": {**A1, "extras": "E2"},                                        # + momentum extras
             "C2": {**A1, "extras": "E2", "trmin": 1975, "minrows": 8000},          # + training rows from 1975 only
             "C3": {**A1, "extras": "E2", "enc": "lp7", "hpca": 0, "hln": False},   # non-pretrained log-patch tokens instead of FinCast
             "C4": {**A1, "extras": "E2", "lvl": "roll10"}})
# queue 9: fixed-prior anchor base (mean + own momentum), grid bw on dev; Sharpe-unit variant
A2r = CFGS["A2_r10"]
CFGS.update({f"D_bw{b}": {**A2r, "base": "blend", "bw": b, "bK": 15} for b in (0.25, 0.5, 0.75, 1.0)})
CFGS.update({"Ds_bw0.75": {**A2r, "base": "blend", "bw": 0.75, "prep": "ysdm"}})
CFGS.update({"B2_r10": {**CFGS["B2"], "lvl": "roll10"}})          # level window chosen on dev (roll10 beat exp1 in dev for return units)
# Sharpe-unit line (B2): within one family the level does not change the ranking by forecast/sd, so these target the within-year Sharpe ranking
CFGS.update({"B2_E2": {**CFGS["B2"], "extras": "E2"}, "Ds_bw0.5": {**CFGS["B2"], "base": "blend", "bw": 0.5}, "Ds_bw1.0": {**CFGS["B2"], "base": "blend", "bw": 1.0}})
CFGS["Ds_bw0.75"] = {**CFGS["B2"], "base": "blend", "bw": 0.75}
# queue 10: block-by-block swaps and the user-specified techniques on the candidate B3 (one change at a time, 10 seeds, paired)
B3 = CFGS["B3"]
CFGS.update({
    "B3_gru": {**B3, "seq": "gru"}, "B3_tf": {**B3, "seq": "transformer"}, "B3_noseq": {**B3, "seq": "none"},          # sequence block
    "B3_attn": {**B3, "pool": "attn"}, "B3_query": {**B3, "pool": "query"}, "B3_mean": {**B3, "pool": "mean"},       # pooling block
    "B3_hyper": {**B3, "xk": "hyper"}, "B3_isab": {**B3, "xk": "isab"}, "B3_noxs": {**B3, "xs": False, "rc": False},  # set attention
    "B3_lw4": {**B3, "lw": 4}, "B3_rw4": {**B3, "rw": 4},                                  # Holen et al. L-/R-weighted (rw EXPLORATORY: rule A)
    "B3_up2": {**B3, "aug": "up2"}, "B3_mix2": {**B3, "aug": "mix2"},                       # upsampling / mixup
    "B3_chr2": {**B3, "enc": "chronos2"}, "B3_chr2_240": {**B3, "enc": "chronos2_240"},     # other pretrained encoders (PCA16 kept)
    "B3_zx": {**B3, "zx": "acct_ew", "zcols": "chronos2_xl_dev"},                          # zero-shot forecast as input (PRETRAINED)
})
# queue 11: the loss on B3 (label in Sharpe units, demeaned within family-year; residual of the linear base)
CFGS.update({"B3_mse": {**B3, "loss": "mse"}, "B3_h03": {**B3, "hdelta": 0.3}, "B3_h01": {**B3, "hdelta": 0.1},
             "B3_c05": {**B3, "corrw": 0.5, "corru": "tot"}, "B3_c2": {**B3, "corrw": 2.0, "corru": "tot"}})
# AGENTS.md rule B: non-pretrained counterparts of B3
CFGS.update({"B3_lp7": {**B3, "enc": "lp7", "hpca": 0, "hln": False}, "B3_raw": {**B3, "enc": "raw", "hpca": 0}})
# loss study from the papers + residual data (workflow 'loss-from-papers-and-data'): Holen L-weighting with the loss replaced by its expectation
CFGS.update({"B3_vyH": {**B3, "fyw": "vH"}, "B3_vyP": {**B3, "fyw": "v"}, "B3_vyI": {**B3, "fyw": "inv"},
             "B3_lweq4": {**B3, "lw": 4, "lwm": "eq"}, "B3_nw05": {**B3, "nw": 0.05}})
# the |t| benchmark: ranking by in-sample |t| alone matches B3 on dev top-k Sharpe -> anchor on z|t| and focus the loss on the head
CFGS.update({"Dt0": {**B3, "base": "blend", "bw": 0.0}, "B3_hw3": {**B3, "hw": 3.0}, "Dt0_hw3": {**B3, "base": "blend", "bw": 0.0, "hw": 3.0}})
CFGS.update({"Dt0_nw": {**CFGS["Dt0"], "nw": 0.05}})          # best-MSE loss (heteroscedastic Gaussian likelihood) on the dev-selected Dt0
# queue 12: block-by-block ablations / swaps on the dev-selected Dt0 (one change at a time)
Dt0 = CFGS["Dt0"]
CFGS.update({
    "Dt0_gru": {**Dt0, "seq": "gru"}, "Dt0_tf": {**Dt0, "seq": "transformer"}, "Dt0_noseq": {**Dt0, "seq": "none"},
    "Dt0_attn": {**Dt0, "pool": "attn"}, "Dt0_query": {**Dt0, "pool": "query"}, "Dt0_mean": {**Dt0, "pool": "mean"},
    "Dt0_hyper": {**Dt0, "xk": "hyper"}, "Dt0_isab": {**Dt0, "xk": "isab"}, "Dt0_noxs": {**Dt0, "xs": False, "rc": False},
    "Dt0_up2": {**Dt0, "aug": "up2"}, "Dt0_mix2": {**Dt0, "aug": "mix2"},
    "Dt0_chr2": {**Dt0, "enc": "chronos2"}, "Dt0_chr2_240": {**Dt0, "enc": "chronos2_240"},
    "Dt0_lp7": {**Dt0, "enc": "lp7", "hpca": 0, "hln": False}, "Dt0_raw": {**Dt0, "enc": "raw", "hpca": 0},   # non-pretrained (rule B)
    "Dt0_rw4": {**Dt0, "rw": 4},                                                                              # R-weighted: EXPLORATORY
})
CFGS["Dt0_chr2_240"] = CFGS["Dt0_chr2_240"]   # (dev-selected after queue 12)
CFGS.update({"Dt0_lr20": {**CFGS["Dt0"], "lvl": "ratio20"}})   # Dt0 + ratio level (window chosen on dev among {all, 10, 20, 30})
# queue 14: the user's papers on the dev-selected Dt0 -- HyperAttention sized for the family-year, Holen-style weighting, accuracy (IC) loss,
# newer pretrained encoder (TiRex), and combinations
CFGS.update({
    "Dt0_hyp1k": {**Dt0, "xk": "hyper", "hblock": 1024, "hm": 512},          # HyperAttention: 1024-block exact part + 512 sampled keys
    "Dt0_vyH": {**Dt0, "fyw": "vH"},                                          # Holen et al. L-weighting (loss -> its expectation)
    "Dt0_c2": {**Dt0, "corrw": 2.0, "corru": "tot"},                          # + within-year correlation of the whole forecast (IC-type accuracy)
    "Dt0_tirex": {**Dt0, "enc": "tirex_sub"}, "Dt0_tirex240": {**Dt0, "enc": "tirex240_sub"},   # TiRex (xLSTM, 2025) hidden states, PRETRAINED
    "Dt0_hyp1k_vyH": {**Dt0, "xk": "hyper", "hblock": 1024, "hm": 512, "fyw": "vH"},
    "Dt0_chr2_240_vyH": {**CFGS["Dt0_chr2_240"], "fyw": "vH"}, "Dt0_chr2_240_c2": {**CFGS["Dt0_chr2_240"], "corrw": 2.0, "corru": "tot"},
})
CFGS.update({"Dt0_chr2_240_lr20": {**CFGS["Dt0_chr2_240"], "lvl": "ratio20"}})   # final candidates with the dev-chosen ratio level
# queue 15: a state-of-the-art head (gated residual network, TFT) alone and with the user's papers (HyperAttention, Holen weighting)
CFGS.update({"Dt0_grn": {**Dt0, "head": "grn"}, "Dt0_hyp1k_grn": {**CFGS["Dt0_hyp1k"], "head": "grn"},
             "Dt0_hyp1k_vyH_grn": {**CFGS["Dt0_hyp1k_vyH"], "head": "grn"}, "Dt0_chr2_240_hyp1k": {**CFGS["Dt0_chr2_240"], "xk": "hyper", "hblock": 1024, "hm": 512}})
# queue 16: TiRex-2 (NX-AI, 2026; xLSTM) hidden states as encoder (PRETRAINED), alone and with the user's papers
CFGS.update({"Dt0_tirex2_240": {**Dt0, "enc": "sub:tirex2_240_sub"}, "Dt0_tirex2_128": {**Dt0, "enc": "sub:tirex2_128_sub"},
             "Dt0_tirex2_240_vyH": {**Dt0, "enc": "sub:tirex2_240_sub", "fyw": "vH"}})
# queue 17: both user papers on the Chronos-2 240 encoder; extra seeds 10-19 for the dev leaders
CFGS.update({"Dt0_chr2_240_hyp1k_vyH": {**CFGS["Dt0_chr2_240"], "xk": "hyper", "hblock": 1024, "hm": 512, "fyw": "vH"}})
# final candidates with the dev-chosen ratio level (for the ticker inference test)
for _n in ("Dt0", "Dt0_chr2_240", "Dt0_chr2_240_vyH", "Dt0_chr2_240_hyp1k_vyH", "Dt0_hyp1k_vyH"):
    CFGS[_n + "_lr20"] = {**CFGS[_n], "lvl": "ratio20"}
