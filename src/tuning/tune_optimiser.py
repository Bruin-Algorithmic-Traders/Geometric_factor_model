import polars as pl
from pathlib import Path
from typing import List
from src.portfolio.optimiser import SpecBBacktester

def grid_search_pure_spec_b(
    df_factor_returns: pl.DataFrame,
    gamma_vals: List[float] = [0.5, 1.0, 2.0, 5.0, 10.0],
    kappa_vals: List[float] = [0.0001, 0.0003, 0.0005, 0.0010]
) -> pl.DataFrame:
    records = []

    for gamma in gamma_vals:
        for kappa in kappa_vals:
            backtester = SpecBBacktester(
                gamma_f=gamma,
                kappa_f=kappa,
                gross_factor_leverage=1.0,
                max_factor_weight=0.20,
                mu_window_months=12,
                cov_window_months=36,
                min_history_months=36
            )

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
    project_root = Path(__file__).resolve().parents[2]
    
    insample_path = project_root / "data" / "processed" / "factor_returns_insample.parquet"
    oos_path = project_root / "data" / "processed" / "factor_returns_oos.parquet"
    
    target_path = insample_path if insample_path.exists() else oos_path

    if target_path.exists():
        print(f"Loading factor returns from: {target_path}")
        df_factor_returns = pl.read_parquet(target_path)

        print("\nRunning Pure Specification B Grid Search...")
        df_results = grid_search_pure_spec_b(
            df_factor_returns=df_factor_returns,
            gamma_vals=[0.5, 1.0, 2.0, 5.0, 10.0],
            kappa_vals=[0.0001, 0.0003, 0.0005, 0.0010]
        )

        print("\n==========================================================================")
        print("             PURE SPECIFICATION B GRID SEARCH RESULTS                     ")
        print("==========================================================================")
        print(df_results)
        print("==========================================================================")

        out_path = project_root / "data" / "processed" / "spec_b_pure_grid_results.parquet"
        df_results.write_parquet(out_path)
        print(f"Grid search summary saved to: {out_path}")
    else:
        print(f"Error: Input parquet file not found at {target_path}")


if __name__ == "__main__":
    main()
