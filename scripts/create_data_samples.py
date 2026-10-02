"""Create small, deterministic CSV excerpts from every file in data_raw/."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data_raw"
SAMPLES = ROOT / "data_sample"

ACCOUNTING_IDS = {"1", "2", "3", "4", "66", "1000", "29315"}
PAST_RETURN_IDS = {"0", "1", "2", "3", "1000", "19401"}
TICKER_IDS = {"0", "1", "2", "3", "1000", "19379"}
EXAMPLE_YEARS = {1983, 2000, 2019, 2020}


def keep_row(filename: str, row: dict[str, str]) -> bool:
    signal_id = row["signalid"]
    if filename.startswith("DataMined"):
        ids = ACCOUNTING_IDS
    elif filename.startswith("PastReturn"):
        ids = PAST_RETURN_IDS
    elif filename.startswith("Ticker"):
        ids = TICKER_IDS
    else:
        raise ValueError(f"Unexpected source file: {filename}")

    if signal_id not in ids:
        return False
    if filename.endswith("SignalList.csv") or filename.endswith("SignalNames.csv"):
        return True

    year = int(row["year"]) if "year" in row else int(row["date"][:4])
    return year in EXAMPLE_YEARS or (
        filename.startswith("DataMinedLongShortReturns")
        and signal_id == "66"
        and 2000 <= year <= 2020
    )


def main() -> None:
    SAMPLES.mkdir(exist_ok=True)
    sources = [
        "DataMinedSignalList.csv",
        "DataMinedLongShortReturnsEW.csv",
        "DataMinedLongShortReturnsVW.csv",
        "PastReturnSignalNames.csv",
        "PastReturnSignalsLongShort.csv",
        "TickerSignalNames.csv",
        "TickerSignalsLongShort.csv",
    ]
    for filename in sources:
        source = RAW / filename
        target = SAMPLES / f"{source.stem}_sample.csv"
        kept = 0
        with source.open("r", newline="", encoding="utf-8") as input_file:
            reader = csv.DictReader(input_file)
            assert reader.fieldnames is not None
            with target.open("w", newline="", encoding="utf-8") as output_file:
                writer = csv.DictWriter(
                    output_file, fieldnames=reader.fieldnames, lineterminator="\n"
                )
                writer.writeheader()
                for row in reader:
                    if keep_row(filename, row):
                        writer.writerow(row)
                        kept += 1
        if kept == 0:
            raise RuntimeError(f"No sample rows selected from {source}")
        print(f"{target.relative_to(ROOT)}: {kept} rows")


if __name__ == "__main__":
    main()
