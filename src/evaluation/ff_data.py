import polars as pl

from src.config import FF_DIR, MARKET_EXCESS_CSV, RESULTS_DIR

FF_FILES = ["F-F_Research_Data_5_Factors_2x3", "F-F_Momentum_Factor", "F-F_ST_Reversal_Factor"]


def read_monthly_block(path) -> pl.DataFrame:
    """Monthly block of a Ken French CSV as decimals, keyed by YYYYMM."""
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(","))
    columns = [c.strip() for c in lines[start].split(",")[1:]]
    rows = []
    for line in lines[start + 1:]:
        fields = [f.strip() for f in line.split(",")]
        if len(fields[0]) != 6 or not fields[0].isdigit():
            break
        rows.append([int(fields[0])] + [float(f) / 100 for f in fields[1:]])
    df = pl.DataFrame(rows, schema=["YYYYMM", *columns], orient="row")
    if (df.select(pl.exclude("YYYYMM")) <= -0.9999).to_numpy().any():
        raise ValueError(f"missing-value code in {path.name}")
    return df


def load_ff() -> pl.DataFrame:
    """FF5 + RF, momentum and short-term reversal, monthly decimals."""
    frames = [read_monthly_block(FF_DIR / f"{name}.csv") for name in FF_FILES]
    ff = frames[0]
    for frame in frames[1:]:
        ff = ff.join(frame, on="YYYYMM", how="inner")
    return ff.sort("YYYYMM")


def market_excess() -> pl.DataFrame:
    """Market benchmark returns minus the one-month T-bill rate."""
    returns = pl.concat([
        pl.read_csv(RESULTS_DIR / f"benchmark_returns_{s}.csv", columns=["DATE", "market_equal", "market_value"])
        for s in ("is", "oos")
    ]).with_columns(YYYYMM=pl.col("DATE") // 100)
    out = returns.join(load_ff().select("YYYYMM", "RF"), on="YYYYMM", how="left")
    if out["RF"].null_count():
        raise ValueError("months missing from the Fama-French file")
    return out.select(
        "DATE",
        market_equal_excess=pl.col("market_equal") - pl.col("RF"),
        market_value_excess=pl.col("market_value") - pl.col("RF"),
        RF="RF",
    )


def main():
    market_excess().write_csv(MARKET_EXCESS_CSV)
