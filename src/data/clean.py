import polars as pl
from pathlib import Path

# -----------------------------------------------------------------------------
# Column Separation By Cleaning Method
# -----------------------------------------------------------------------------

METADATA_COLS = ["DATE", "permno"]

DROPPED_COLS = [
    "sic2",        # Industry id
    "betasq",      # Duplicate of beta
    "pchsaleinv",  # Duplicate of pchsale_pchinvt
    "securedind",  # Duplicate flag for secured
    "herf",        # Industry-level macro constant
    "indmom",      # Industry-level macro constant
    "rd_mve",      # Duplicate R&D ratio
    "rd_sale",     # Duplicate R&D ratio
    "sin",         # >98% sparse flag
    "divi",        # >98% sparse flag
    "divo",        # >98% sparse flag
    "ps",          # Piotroski composite score
    "absacc",      # Absolute accruals (destroys sign of acc)
    "convind",     # Sparse convertible indicator
]

ZERO_FILL_COLS = ["rd"]

# -----------------------------------------------------------------------------
# Modular Processing Functions
# -----------------------------------------------------------------------------
def drop_cols(lf: pl.LazyFrame) -> pl.LazyFrame:
    cols = [col for col in DROPPED_COLS if col in lf.collect_schema().names()]
    return lf.drop(cols)

def filter_cols(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter(
        pl.col("DATE").is_not_null() & pl.col("permno").is_not_null() &
        (pl.col("mvel1") >= pl.col("mvel1").quantile(0.2).over("DATE"))
    )

def fill_missing_vals(lf: pl.LazyFrame, cols : list[str]) -> pl.LazyFrame:
    exprs = []
    for col in cols:
        if col in ZERO_FILL_COLS:
            expr = pl.col(col).fill_null(0.0)
        else:
            expr = pl.col(col).fill_null(pl.col(col).median().over("DATE"))
        exprs.append(expr.alias(col))
    return lf.with_columns(exprs)

def rank_normalise_cols(lf: pl.LazyFrame, cols : list[str]) -> pl.LazyFrame:
    exprs = []
    for col in cols:
        count = pl.col(col).count().over("DATE")
        rank = pl.col(col).rank("average").over("DATE")
        normalised = (pl.when(count > 1)
                        .then((2 * (rank - 1) / (count - 1)) - 1)
                        .otherwise(0.0))
        exprs.append(normalised.alias(col))
    return lf.with_columns(exprs)

def check_nulls(df: pl.DataFrame, cols : list[str]):
    null_df = df.select(cols).null_count()
    null_counts = null_df.to_dicts()[0]
    bad_cols = {col: count for col, count in null_counts.items() if count > 0}
    if bad_cols:
        raise ValueError(f"Null values remain in columns: {bad_cols}")
    else:
        print("No null values remain in the specified columns.")

# -----------------------------------------------------------------------------
# Main cleaning function
# -----------------------------------------------------------------------------

def clean_data():

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1] 

    input_path = project_root / "data" / "processed" / "chars.parquet"
    output_path = project_root / "data" / "processed" / "chars_clean.parquet"

    lf = pl.scan_parquet(input_path)

    lf = drop_cols(lf)
    lf = filter_cols(lf)
    feature_cols = [col for col in lf.collect_schema().names() if col not in METADATA_COLS]
    lf = fill_missing_vals(lf, feature_cols)
    lf = rank_normalise_cols(lf, feature_cols)

    clean_df = lf.collect()
    check_nulls(clean_df, feature_cols)
    clean_df.write_parquet(output_path)

if __name__ == "__main__":
    clean_data()