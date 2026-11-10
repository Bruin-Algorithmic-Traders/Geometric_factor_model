import polars as pl
from typing import List
from src.factors.build_panel import load_panel
from src.portfolio.optimiser import SpecBBacktester
from src.config import GAMMA_GRID, KAPPA_GRID, OPTIMISER_GRID_PARQUET

def grid_search_pure_spec_b(
    df_factor_returns: pl.DataFrame,
    holdings: list,
    gamma_vals: List[float] = GAMMA_GRID,
    kappa_vals: List[float] = KAPPA_GRID
) -> pl.DataFrame:
    records = []

    for gamma in gamma_vals:
        for kappa in kappa_vals:
            backtester = SpecBBacktester(gamma_f=gamma, kappa_penalty=kappa)

            res = backtester.run_backtest(df_factor_returns, holdings)
            m = res["metrics"]

            records.append({
                "Gamma": gamma,
                "Kappa_penalty": kappa,
                "Sharpe": round(m["Sharpe Ratio"], 4),
                "Gross_Sharpe": round(m["Gross Sharpe Ratio"], 4),
                "Ann_Return": round(m["Annualized Return"], 4),
                "Ann_Vol": round(m["Annualized Volatility"], 4),
                "Avg_Turnover": round(m["Avg Monthly Turnover"], 4)
            })

    return pl.DataFrame(records).sort("Sharpe", descending=True)


def main():
    factor_returns, holdings = load_panel()  # in-sample only by default
    df_results = grid_search_pure_spec_b(factor_returns, holdings)
    print(df_results)
    df_results.write_parquet(OPTIMISER_GRID_PARQUET)


if __name__ == "__main__":
    main()
