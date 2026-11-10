import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.colors import LinearSegmentedColormap

from src.config import CHARS_PARQUET, IS_END_DATE, MAX_ZEROTRADE, MIN_SIZE_QUANTILE, RAW_RETURNS_CSV

ID_COLS = ["permno", "DATE"]
META_COLS = ID_COLS + ["sic2"]
BLUE = "#2f6db5"
GREY = "#9aa3ad"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2f6db5", "#e8e8e8", "#c4462f"])

plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.6, "axes.axisbelow": True,
})
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_float_precision(4)


def chars_lazy() -> pl.LazyFrame:
    return pl.scan_parquet(CHARS_PARQUET).with_columns((pl.col("DATE") // 100).alias("ym"))


def char_names() -> list[str]:
    return [c for c in pl.scan_parquet(CHARS_PARQUET).collect_schema().names() if c not in META_COLS]


def returns_lazy() -> pl.LazyFrame:
    return pl.scan_csv(RAW_RETURNS_CSV).select(
        "permno",
        (pl.col("DATE") // 100).alias("ym"),
        pl.col("monthly_ret").cast(pl.Float64, strict=False).alias("ret"),
    )


def size_zerotrade_filter(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter(
        (pl.col("mvel1") >= pl.col("mvel1").quantile(MIN_SIZE_QUANTILE).over("DATE"))
        & (pl.col("zerotrade") < MAX_ZEROTRADE)
    )


def is_universe(with_returns: bool = True) -> pl.DataFrame:
    """In-sample firm-months after the size/zerotrade filter, optionally keeping only rows with a same-month return."""
    lf = size_zerotrade_filter(chars_lazy()).filter(pl.col("DATE") <= IS_END_DATE)
    if with_returns:
        lf = lf.join(returns_lazy(), on=["permno", "ym"], how="inner").filter(pl.col("ret").is_not_nan())
    return lf.collect()


def add_n_missing(df: pl.DataFrame, chars: list[str]) -> pl.DataFrame:
    return df.with_columns(pl.sum_horizontal([pl.col(c).is_null() for c in chars]).alias("n_missing"))


def rank_normalise(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Cross-sectional rank to [-1, 1] within DATE; nulls stay null."""
    return df.with_columns([
        ((pl.col(c).rank("average").over("DATE") - 1) / (pl.col(c).count().over("DATE") - 1) * 2 - 1).alias(c)
        for c in cols
    ])
