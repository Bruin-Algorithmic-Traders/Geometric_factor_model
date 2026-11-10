import itertools

import polars as pl

from src.config import K_GRID, TAU_GRID
from src.factors.build_panel import build_panel
from src.portfolio.optimiser import SpecBBacktester


def evaluate_is_grid(n_clusters_grid=K_GRID, tau_grid=TAU_GRID) -> pl.DataFrame:
    """Score each (K, τ) by the in-sample optimiser Sharpe at default γ and κ."""
    records = []
    for k, tau in itertools.product(n_clusters_grid, tau_grid):
        factor_returns, _, _, holdings = build_panel(n_clusters=k, tau=tau)
        res = SpecBBacktester().run_backtest(factor_returns, holdings)
        m = res["metrics"]
        records.append({"K": k, "tau": tau, "sharpe": m["Sharpe Ratio"], "ann_return": m["Annualized Return"],
                        "ann_vol": m["Annualized Volatility"], "avg_turnover": m["Avg Monthly Turnover"]})
        print(records[-1])
    return pl.DataFrame(records).sort("sharpe", descending=True)


if __name__ == "__main__":
    print(evaluate_is_grid())
