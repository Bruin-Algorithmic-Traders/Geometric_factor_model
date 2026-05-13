"""Count NaN values in monthly characteristic files.

This script scans all files under the processed monthly characteristics
folder, computes NaN counts by month and column, saves the result to the
results folder, and prints the total number of missing values.
"""

import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Add parent directory to path to import config
sys.path.insert(0, str(ROOT))
import config

# Use processed characteristics from config for analysis
PROCESSED_CHARS_DIR = ROOT / config.CHARS_DIR
OUTPUT_DIR = ROOT / config.OUTPUT_DIR


def _find_characteristic_files() -> list[Path]:
    return sorted(PROCESSED_CHARS_DIR.glob("chars_*.csv"))


def _parse_date_from_filename(path: Path) -> str:
    return path.stem.replace("chars_", "")


def count_nans_by_month_and_column() -> pd.DataFrame:
    files = _find_characteristic_files()
    if not files:
        raise FileNotFoundError(
            f"No monthly characteristic files found in {PROCESSED_CHARS_DIR}"
        )

    rows = []
    for path in files:
        month = _parse_date_from_filename(path)
        df = pd.read_csv(path, low_memory=False)
        nan_counts = df.isna().sum()
        row = {"DATE": month, **nan_counts.to_dict()}
        rows.append(row)

    result = pd.DataFrame(rows)
    result = result.set_index("DATE")
    result.index.name = "DATE"
    result = result.sort_index()
    return result


def save_nan_counts(df: pd.DataFrame, filename: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(output_path)
    return output_path


def main() -> None:
    nan_counts = count_nans_by_month_and_column()
    output_path = save_nan_counts(nan_counts, "nan_counts_monthly.csv")
    total_nans = int(nan_counts.to_numpy().sum())

    print(f"Saved NaN counts by month and column to: {output_path}")
    print(f"Total number of NaN values across all monthly characteristic files: {total_nans}")


if __name__ == "__main__":
    main()
