import polars as pl

from src.config import (
    CHARS_CLEAN_PARQUET, CHARS_PARQUET, DROPPED_COLS, MAX_ROW_MISSING_SHARE, MAX_ZEROTRADE,
    MIN_SIZE_QUANTILE, PROCESSED_DIR, RAW_RETURNS_CSV, TARGET_COL, UNKNOWN_SIC2, feature_cols,
)


def fill_sic2(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        pl.col("sic2").forward_fill().over("permno", order_by="DATE")
        .fill_null(UNKNOWN_SIC2).cast(pl.Int64)
    )


def exit_months() -> pl.LazyFrame:
    """(permno, month) of the month after each firm's last return, for firms whose returns stop before the data ends."""
    last = (pl.scan_csv(RAW_RETURNS_CSV).group_by("permno")
            .agg((pl.col("DATE") // 100).max().alias("last")))
    return last.filter(pl.col("last") < pl.col("last").max()).select(
        "permno", pl.when(pl.col("last") % 100 == 12).then(pl.col("last") + 89).otherwise(pl.col("last") + 1).alias("month")
    )


def join_returns(lf: pl.LazyFrame, delisting_return: float | None = None) -> pl.LazyFrame:
    """Chars dated t (built from data up to t-1) are paired with the return dated t. With delisting_return, firms
    that still have chars in the month after their last return are kept for that month with this return."""
    returns = pl.scan_csv(RAW_RETURNS_CSV).select(
        "permno", (pl.col("DATE") // 100).alias("month"), pl.col("monthly_ret").alias(TARGET_COL)
    )
    returns = returns.with_columns(pl.lit(False).alias("is_exit"))
    if delisting_return is not None:
        exits = exit_months().with_columns(pl.lit(delisting_return).alias(TARGET_COL), pl.lit(True).alias("is_exit"))
        returns = pl.concat([returns, exits])
    return (
        lf.with_columns((pl.col("DATE") // 100).alias("month"))
        .join(returns, on=["permno", "month"], how="inner")
        .drop("month")
    )


def filter_size_zerotrade(lf: pl.LazyFrame) -> pl.LazyFrame:
    """The size cutoff is computed on firms with a real return, so added exit rows don't move it."""
    return lf.filter(
        (pl.col("mvel1") >= pl.col("mvel1").filter(~pl.col("is_exit")).quantile(MIN_SIZE_QUANTILE).over("DATE"))
        & (pl.col("zerotrade") < MAX_ZEROTRADE)
    )


def filter_row_missing(lf: pl.LazyFrame, raw_chars: list[str]) -> pl.LazyFrame:
    n_missing = pl.sum_horizontal([pl.col(c).is_null() for c in raw_chars])
    return lf.filter(n_missing <= MAX_ROW_MISSING_SHARE * len(raw_chars))


def rank_normalise(lf: pl.LazyFrame, cols: list[str]) -> pl.LazyFrame:
    """Rank non-missing values to [-1, 1] within each month, then set missing values to 0."""
    exprs = []
    for col in cols:
        count = pl.col(col).count().over("DATE")
        rank = pl.col(col).rank("average").over("DATE")
        normalised = pl.when(count > 1).then(2 * (rank - 1) / (count - 1) - 1).otherwise(0.0)
        exprs.append(normalised.fill_null(0.0).alias(col))
    return lf.with_columns(exprs)


def clean_data(dropped_cols: list[str] = DROPPED_COLS, out_path=CHARS_CLEAN_PARQUET, delisting_return: float | None = None):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    lf = pl.scan_parquet(CHARS_PARQUET)
    raw_chars = feature_cols(lf.collect_schema().names())

    lf = fill_sic2(lf)
    lf = join_returns(lf, delisting_return)
    lf = filter_size_zerotrade(lf)
    lf = filter_row_missing(lf, raw_chars)
    lf = lf.drop(dropped_cols + ["is_exit"])
    lf = rank_normalise(lf, feature_cols(lf.collect_schema().names()))

    df = lf.collect()
    nulls = {c: n for c, n in df.null_count().row(0, named=True).items() if n}
    if nulls:
        raise ValueError(f"Null values remain: {nulls}")
    df.write_parquet(out_path)


if __name__ == "__main__":
    clean_data()
