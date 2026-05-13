"""Clean raw monthly characteristic data.

This script implements the two-stage cleaning patch:

1. Industry Propagator (forward-fill sic2)
   - Group by permno and forward-fill sic2 across months.

2. Tiered Median Fill
   - Group by DATE and sic2: fill numeric NaNs with the industry-month median.
   - Group by DATE only: fill remaining numeric NaNs with the month median.

The cleaned files are written to data/processed/characteristics_by_month and a
summary report is written to results/clean_monthly_summary.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import sys

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


ROOT = Path(__file__).resolve().parent.parent
RAW_CHARS_DIR = ROOT / config.RAW_CHARS_DIR
CLEANED_CHARS_DIR = ROOT / config.CHARS_DIR
RESULTS_DIR = ROOT / config.OUTPUT_DIR


def _get_characteristics_files() -> list[Path]:
    return sorted(RAW_CHARS_DIR.glob("chars_*.csv"))


def _load_all_characteristics(files: list[Path]) -> pd.DataFrame:
    print(f"Loading {len(files)} raw monthly characteristic files...")
    dfs = []
    for path in files:
        print(f"  reading {path.name}")
        dfs.append(pd.read_csv(path, low_memory=False))

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded raw data with {len(df)} rows and {len(df.columns)} columns.")
    return df


def clean_monthly_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    files = _get_characteristics_files()
    if not files:
        raise FileNotFoundError(f"No raw monthly files found in {RAW_CHARS_DIR}")

    df = _load_all_characteristics(files)
    if "sic2" not in df.columns:
        raise KeyError("Expected 'sic2' column in raw characteristics files")

    # Convert DATE to int for sorting and grouping.
    df["DATE"] = df["DATE"].astype(int)
    df = df.sort_values(["permno", "DATE"], ignore_index=True)
    print("Sorted raw data by permno and DATE.")

    # Stage 1: Forward-fill industry codes within each permno.
    df["sic2"] = df.groupby("permno")["sic2"].ffill()
    n_sic2_missing = int(df["sic2"].isna().sum())
    print(f"Forward-filled sic2 by permno. Remaining missing sic2 values before fill: {n_sic2_missing}")

    if n_sic2_missing > 0:
        df["sic2"] = df["sic2"].fillna(99).astype(int)
        print(f"Assigned sic2 = 99 for {n_sic2_missing} remaining missing values.")

    # Collect numeric feature columns excluding identifiers.
    numeric_cols = [
        col
        for col in df.select_dtypes(include=["number"]).columns
        if col not in {"permno", "DATE"}
    ]
    if "sic2" in numeric_cols:
        numeric_cols.remove("sic2")

    print(f"Applying tiered median fill to {len(numeric_cols)} numeric columns.")
    # Stage 2: Tiered median fill for numeric characteristics.
    df[numeric_cols] = df.groupby(["DATE", "sic2"])[numeric_cols].transform(
        lambda x: x.fillna(x.median())
    )
    print("Filled missing numeric values within DATE + sic2 groups.")
    df[numeric_cols] = df.groupby("DATE")[numeric_cols].transform(
        lambda x: x.fillna(x.median())
    )
    print("Filled remaining missing numeric values within DATE groups.")

    return df, df[numeric_cols]


def save_cleaned_files(df: pd.DataFrame) -> None:
    CLEANED_CHARS_DIR.mkdir(parents=True, exist_ok=True)

    for date_value, group in df.groupby("DATE"):
        output_path = CLEANED_CHARS_DIR / f"chars_{date_value}.csv"
        group.sort_values(["permno"], inplace=True)
        group.to_csv(output_path, index=False)


def save_summary(df: pd.DataFrame, before_df: pd.DataFrame | None = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for date_value, group in df.groupby("DATE"):
        n_rows = len(group)
        n_cols = group.shape[1]
        n_missing_after = int(group.isna().sum().sum())
        n_sic2_missing = int(group["sic2"].isna().sum())
        summary_rows.append(
            {
                "DATE": date_value,
                "n_rows": n_rows,
                "n_missing_after": n_missing_after,
                "n_sic2_missing_after": n_sic2_missing,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("DATE")
    output_path = RESULTS_DIR / "clean_monthly_summary.csv"
    summary.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    cleaned_df, _ = clean_monthly_data()
    save_cleaned_files(cleaned_df)
    summary_path = save_summary(cleaned_df)

    total_missing = int(cleaned_df.isna().sum().sum())
    total_sic2_missing = int(cleaned_df["sic2"].isna().sum())
    print(f"Cleaned raw monthly files saved to: {CLEANED_CHARS_DIR}")
    print(f"Summary saved to: {summary_path}")
    print(f"Total remaining missing values after cleaning: {total_missing}")
    print(f"Total remaining missing sic2 values after forward-fill: {total_sic2_missing}")


if __name__ == "__main__":
    main()
