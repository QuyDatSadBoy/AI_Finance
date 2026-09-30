# AI_Finance

Research materials for **High-Throughput Asset Pricing** (Chen and Dim) and related FinCast/LSTM experiments.

Start with the [plain-language explanation of the original paper](Research/paper%20ch%C3%ADnh/PAPER_GOC_DE_HIEU_INPUT_OUTPUT.md), then read the [detailed paper guide](Research/paper%20ch%C3%ADnh/GIAI_THICH_TU_CO_BAN_DEN_NANG_CAO.md) and the [data guide](Research/README_DATA.md). Reference notebooks are in [`Research/NoteBook_Refer`](Research/NoteBook_Refer/).

The original data are kept in `data/`; the small example file is in `data_sample/`. The large raw data files and downloaded ZIP archives are kept locally and excluded from Git because each exceeds GitHub's regular Git file limit. This repository includes small signal-name tables and a sample return table. To reconstruct the complete local `data/` folder, add these files separately:

- `DataMinedLongShortReturnsEW.csv`
- `DataMinedLongShortReturnsVW.csv`
- `PastReturnSignalsLongShort.csv`
- `TickerSignalsLongShort.csv`

The notebooks refer to precomputed estimates and embeddings stored in the original authors' or research team's external environments; those artifacts are not included here.
