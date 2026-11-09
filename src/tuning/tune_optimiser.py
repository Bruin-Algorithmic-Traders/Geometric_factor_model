import polars as pl
from typing import List
from src.portfolio.optimiser import SpecBBacktester
from src.config import (
    FACTOR_RETURNS_IS_PARQUET, FACTOR_RETURNS_OOS_PARQUET, GAMMA_GRID, KAPPA_GRID,
    OPTIMISER_GRID_PARQUET,
)

def grid_search_pure_spec_b(
    df_factor_returns: pl.DataFrame,
    gamma_vals: List[float] = GAMMA_GRID,
    kappa_vals: List[float] = KAPPA_GRID
) -> pl.DataFrame:
    records = []

    for gamma in gamma_vals:
        for kappa in kappa_vals:
            backtester = SpecBBacktester(gamma_f=gamma, kappa_f=kappa)

            res = backtester.run_backtest(df_factor_returns)
            m = res["metrics"]

            records.append({
                "Gamma": gamma,
                "Kappa": kappa,
                "Sharpe": round(m["Sharpe Ratio"], 4),
                "Ann_Return": round(m["Annualized Return"], 4),
                "Ann_Vol": round(m["Annualized Volatility"], 4),
                "Avg_Turnover": round(m["Avg Monthly Turnover"], 4)
            })

    return pl.DataFrame(records).sort("Sharpe", descending=True)


def main():
    target_path = FACTOR_RETURNS_IS_PARQUET if FACTOR_RETURNS_IS_PARQUET.exists() else FACTOR_RETURNS_OOS_PARQUET

    if target_path.exists():
        print(f"Loading factor returns from: {target_path}")
        df_factor_returns = pl.read_parquet(target_path)

        print("\nRunning Pure Specification B Grid Search...")
        df_results = grid_search_pure_spec_b(df_factor_returns)

        print("\n==========================================================================")
        print("             PURE SPECIFICATION B GRID SEARCH RESULTS                     ")
        print("==========================================================================")
        print(df_results)
        print("==========================================================================")

        df_results.write_parquet(OPTIMISER_GRID_PARQUET)
        print(f"Grid search summary saved to: {OPTIMISER_GRID_PARQUET}")
    else:
        print(f"Error: Input parquet file not found at {target_path}")


if __name__ == "__main__":
    main()
