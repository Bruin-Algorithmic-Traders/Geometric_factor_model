import polars as pl
from pathlib import Path

from src.config import (
    CHARS_CLEAN_PARQUET, CHARS_PARQUET, DROPPED_COLS, MAX_ZEROTRADE, MIN_SIZE_QUANTILE,
    PROCESSED_DIR, RAW_RETURNS_CSV, TARGET_COL, ZERO_FILL_COLS, feature_cols,
)

# -----------------------------------------------------------------------------
# Modular Processing Functions
# -----------------------------------------------------------------------------
def drop_cols(lf: pl.LazyFrame) -> pl.LazyFrame:
    cols = [col for col in DROPPED_COLS if col in lf.collect_schema().names()]
    return lf.drop(cols)

def filter_cols(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter(
        pl.col("DATE").is_not_null() & pl.col("permno").is_not_null() &
        (pl.col("mvel1") >= pl.col("mvel1").quantile(MIN_SIZE_QUANTILE).over("DATE")) &
        (pl.col("zerotrade") < MAX_ZEROTRADE)
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

def join_returns(lf: pl.LazyFrame, returns_path : Path) -> pl.LazyFrame:

    lf = lf.with_columns(
        (pl.col("DATE") // 100).alias("_yyyymm"))

    returns_lf = (
        pl.scan_csv(returns_path)
        .select([
            pl.col("DATE")
            .cast(pl.String)
            .str.to_date("%Y%m%d")
            .dt.offset_by("-1mo")
            .dt.strftime("%Y%m")
            .cast(pl.Int64)
            .alias("_yyyymm"),
            pl.col("permno"),
            pl.col("monthly_ret").cast(pl.Float64, strict = False).alias(TARGET_COL)
        ])
        )
    return lf.join(returns_lf, on=["_yyyymm", "permno"], how="left").drop("_yyyymm")

def filter_returns(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter(
        pl.col(TARGET_COL).is_not_null() &
        ~pl.col(TARGET_COL).is_nan()
        )

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

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    lf = pl.scan_parquet(CHARS_PARQUET)

    lf = drop_cols(lf)
    lf = filter_cols(lf)
    features = feature_cols(lf.collect_schema().names())
    lf = fill_missing_vals(lf, features)
    lf = rank_normalise_cols(lf, features)

    lf = join_returns(lf, RAW_RETURNS_CSV)
    lf = filter_returns(lf)

    clean_df = lf.collect()
    check_nulls(clean_df, features + [TARGET_COL])
    clean_df.write_parquet(CHARS_CLEAN_PARQUET)

if __name__ == "__main__":
    clean_data()