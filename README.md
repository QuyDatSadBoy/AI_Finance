# AI_Finance

Research materials for **High-Throughput Asset Pricing** (Chen and Dim) and related FinCast/LSTM experiments.

Start with the [plain-language explanation of the original paper](Research/paper%20ch%C3%ADnh/PAPER_GOC_DE_HIEU_INPUT_OUTPUT.md), then read the [detailed paper guide](Research/paper%20ch%C3%ADnh/GIAI_THICH_TU_CO_BAN_DEN_NANG_CAO.md) and the [data guide](Research/README_DATA.md). Reference notebooks are in [`Research/NoteBook_Refer`](Research/NoteBook_Refer/).

For the authors' public code, the reference notebook, and a worked example using an actual strategy in `data_raw/`, read the [code and real-data walkthrough](Research/CODE_PAPER_GOC_VA_VI_DU_DATA_THAT.md).

The authors' complete public repository is cloned locally at `Research/code_goc_paper/high-throughput-ap/` with its own Git history. It is kept separate from this private repository. To recreate the checkout elsewhere, run `git clone https://github.com/chenandrewy/high-throughput-ap.git Research/code_goc_paper/high-throughput-ap`.

The original extracted data are kept in `data_raw/`; [small samples of every raw CSV](data_sample/README.md) are in `data_sample/`. The original `Strategy Returns.zip` archive is kept at the repository root. The large raw data files and downloaded ZIP archives are kept locally and excluded from Git because each exceeds GitHub's regular Git file limit. This repository includes small signal-name tables and sampled return tables. To reconstruct the complete local `data_raw/` folder, add these files separately:

- `DataMinedLongShortReturnsEW.csv`
- `DataMinedLongShortReturnsVW.csv`
- `PastReturnSignalsLongShort.csv`
- `TickerSignalsLongShort.csv`

The notebooks refer to precomputed estimates and embeddings stored in the original authors' or research team's external environments; those artifacts are not included here.
