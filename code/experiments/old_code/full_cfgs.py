"""Named configurations for the full-data (all Accounting-EW strategies) runs.  The FinCast tokens are pre-projected to PCA16 (hpca=0 here)."""
import ht_base as B
F = "acct_ew"
XSET = 27000                       # >= the largest family-year (26,541): one set-attention batch holds a whole family-year (rc needs it)
# baseline = the reference notebook's model: encoder -> Bi-LSTM -> attention -> concat(t-stat, mean) -> FC, MSE-type loss, no set attention
GOC = {**B.BASE, "fams": F}
T = {**B.BASE, "xs": True, "prep": "ydm", "pool": "mha", "fams": F}
A0 = {**T, "hpca": 0, "rc": True, "xset": XSET, "detev": True}
A1 = {**A0, "base": "lin4", "bK": 15}
A2 = {**A1, "lvl": "exp1"}
A3 = {**A2, "ep0": True, "seed2": True, "optkeep": True}
B3 = {**A3, "prep": "ysdm"}
Dt0 = {**B3, "base": "blend", "bw": 0.0}
FCFGS = {"goc": GOC, "B3": B3, "Dt0": Dt0, "Dt0_vyH": {**Dt0, "fyw": "vH"}}
# --- variants tried on the full-data walk-forward (one change at a time from Dt0)
HY = {"xk": "hyper", "hblock": 512, "hm": 256}                       # HyperAttention (ICLR 2024) set attention: O(n(block+m)) instead of O(n^2) per family-year
FCFGS.update({"Dt0_hyp": {**Dt0, **HY},
              "Dt0_hyp_hw3": {**Dt0, **HY, "hw": 3.0},                                         # head weight: errors on the top-|t| decile count 4x
              "Dt0_hyp_cr": {**Dt0, **HY, "corrw": 1.0, "corru": "sr"},                         # + correlation (Sharpe units) between forecast and label (rule C)
              "Dt0_hyp_vyH": {**Dt0, **HY, "fyw": "vH"},                                       # Holen et al. L-weighting by family-year
              "Dt0_noxs": {**Dt0, "xs": False, "rc": False}})
HW = {**Dt0, **HY, "hw": 3.0}
FCFGS.update({"Dt0_hyp_hw9": {**Dt0, **HY, "hw": 9.0},
              "Dt0_hyp_hw3_xl2": {**HW, "xl": 2},                           # two set-attention blocks
              "Dt0_hyp_hw3_h256": {**HW, "hid": 256},                       # wider Bi-LSTM / attention
              "Dt0_hyp_hw3_ep5": {**HW, "epochs": 5},
              "Dt0_hyp_hw3_vyH": {**HW, "fyw": "vH"}})
FCFGS.update({"Dt0_hyp_nv3": {**Dt0, **HY, "nval": 3}, "Dt0_hyp_hw3_nv3": {**HW, "nval": 3}})      # 3 validation years (Y-3..Y-1), training <= Y-4
FCFGS.update({"Dt0_hyp_P": {**Dt0, **HY, "parts": True}, "Dt0_hyp_hw3_P": {**HW, "parts": True}})       # same models, but the base / level / network-correction parts are saved
# techniques from the project owner's papers / ideas, one at a time on the full-data HyperAttention model
FCFGS.update({"Dt0_hyp_up2": {**Dt0, **HY, "aug": "up2"},          # upsampling: each family-year batch seen twice, the 2nd with input noise
              "Dt0_hyp_mix2": {**Dt0, **HY, "aug": "mix2"},        # mixup inside the family-year
              "Dt0_hyp_nw": {**Dt0, **HY, "nw": 0.05}})            # likelihood loss with the known noise variance of each label (s24^2/12)
# pretrained encoders from the Hugging Face scan (PRETRAINED, reported as such): EXAONE Forecast for Finance (LG-AI-Research, arXiv 2609.04239)
FCFGS.update({"Dt0_hyp_ex": {**Dt0, **HY, "encf": "exaone"}, "Dt0_hyp_both": {**Dt0, **HY, "encf": "both"}})
# attention block (HyperAttention parameters, ISAB) and loss on the full-data model
FCFGS.update({"Dt0_hyp_b256m128": {**Dt0, "xk": "hyper", "hblock": 256, "hm": 128}, "Dt0_hyp_b1024m512": {**Dt0, "xk": "hyper", "hblock": 1024, "hm": 512},
              "Dt0_isab": {**Dt0, "xk": "isab", "xm": 64},
              "Dt0_hyp_mse": {**Dt0, **HY, "loss": "mse"}, "Dt0_hyp_ctot": {**Dt0, **HY, "corrw": 0.5, "corru": "tot"}})
# own time-series signal (momentum months 2-12, 12-month hit rate, drawdown) as network inputs / in the base: information beyond the in-sample t-stat
FCFGS.update({"Dt0_hyp_E2": {**Dt0, **HY, "extras": "E2"}, "Dt0_hyp_E2_bw": {**Dt0, **HY, "extras": "E2", "bw": 0.5}})
FCFGS.update({"Dt0_isab_P": {**Dt0, "xk": "isab", "xm": 64, "parts": True}})       # the selected architecture (ISAB), saving base / level / network-correction parts (one seed for the docs)
# ablation of the label design: same ISAB architecture, but the network predicts the label directly (no base z|t|), two label choices
ISAB = {"xs": True, "pool": "mha", "xk": "isab", "xm": 64, "xset": XSET, "detev": True, "fams": F}
FCFGS.update({"dir_isab": {**B.BASE, **ISAB},                                  # label = oriented return (like the notebook), ISAB attention, no base, no recentring
              "sh_isab_nobase": {**B.BASE, **ISAB, "prep": "ysdm", "rc": True, "lvl": "exp1"}})   # label = Sharpe units minus year mean, ISAB, but NO |t| base
# financial features of each strategy as extra inputs (extras=F1): sd, number of months, recent / multi-horizon mean, sd and t over 6-60 months, hit rate, decay,
# |t| rank in the year, t of the 5 neighbouring strategies; no EB column, no zero-shot forecast
FCFGS.update({"Dt0_isab_F1": {**Dt0, "xk": "isab", "xm": 64, "extras": "F1"}})

# ---- all ideas / methods discussed, one change at a time from the selected architecture (Dt0 + ISAB); 3 seeds each
IS = {**Dt0, "xk": "isab", "xm": 64}
FCFGS.update({
    "I_up2": {**IS, "aug": "up2"}, "I_mix2": {**IS, "aug": "mix2"},                                  # upsampling / mixup (project owner's ideas)
    "I_vyH": {**IS, "fyw": "vH"}, "I_lw4": {**IS, "lw": 4}, "I_lweq4": {**IS, "lw": 4, "lwm": "eq"},   # Holen et al. L-weighting (adapted) / gradient merge (original) / equal-weight merge control
    "I_nw": {**IS, "nw": 0.05}, "I_mse": {**IS, "loss": "mse"}, "I_ctot": {**IS, "corrw": 0.5, "corru": "tot"},   # losses
    "I_nv3": {**IS, "nval": 3}, "I_nv0": {**IS, "nval": 0, "ep0": False},                                # validation window: 3 years / none (train to Y-1, last epoch)
    "I_freeze04": {**IS, "freeze": 2004},                                                                 # train-once control: no learning after 2004
    "I_lp7": {**IS, "encf": "lp7", "hpca": 0, "hln": False}, "I_raw": {**IS, "encf": "raw", "hpca": 0},   # non-pretrained controls
    "I_chr2": {**IS, "encf": "chr2"}, "I_both": {**IS, "encf": "both"}, "I_ex": {**IS, "encf": "exaone"},  # pretrained encoders
})
# direct (raw return) versus intermediate (Sharpe-unit, year-demeaned) label with HyperAttention instead of ISAB
FCFGS.update({"dir_hyp": {**B.BASE, **{**ISAB, "xk": "hyper", "hblock": 512, "hm": 256}},
              "sh_hyp_nobase": {**B.BASE, **{**ISAB, "xk": "hyper", "hblock": 512, "hm": 256}, "prep": "ysdm", "rc": True, "lvl": "exp1"}})
