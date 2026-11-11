import itertools
import json
import os
from datetime import date
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl

from src.config import (
    IS_HALF_SPLIT_DATE, K_GRID, KAPPA_GRID, RESULTS_DIR, TAU_GRID, TUNE_WORKERS, TUNING_GRID_CSV,
    CONFIG_LOCKED_JSON,
)
from src.evaluation.metrics import annualised_stats
from src.evaluation.plots import tuning_heatmaps
from src.factors.build_panel import build_panel
from src.portfolio.optimiser import SpecBBacktester


def run_cell(k: int, tau: float, kappa_grid: list) -> list[dict]:
    """Build the in-sample panel for one (K, τ) and run every κ_penalty on it."""
    records = []
    factor_returns, _, _, holdings, rebuilt = build_panel(n_clusters=k, tau=tau)
    for kappa in kappa_grid:
        res = SpecBBacktester(kappa_penalty=kappa).run_backtest(factor_returns, holdings, rebuilt)
        m, r = res["metrics"], res["returns_df"]
        first_half = r.filter(pl.col("DATE") < IS_HALF_SPLIT_DATE)["strategy_return"].to_numpy()
        second_half = r.filter(pl.col("DATE") >= IS_HALF_SPLIT_DATE)["strategy_return"].to_numpy()
        records.append({
            "K": k, "tau": tau, "kappa_penalty": kappa,
            "net_sharpe": m["Sharpe Ratio"], "gross_sharpe": m["Gross Sharpe Ratio"],
            "ann_return": m["Annualized Return"], "ann_vol": m["Annualized Volatility"],
            "ann_cost": m["Annualized Cost"], "stock_turnover": m["Avg Monthly Turnover"],
            "net_sharpe_first_half": annualised_stats(first_half)["sharpe"],
            "net_sharpe_second_half": annualised_stats(second_half)["sharpe"],
        })
    print(f"K={k}, tau={tau} done", flush=True)
    return records


def run_grid(k_grid=K_GRID, tau_grid=TAU_GRID, kappa_grid=KAPPA_GRID, workers=TUNE_WORKERS) -> pl.DataFrame:
    """In-sample results for every (K, τ, κ_penalty); (K, τ) cells run in parallel processes."""
    cells = list(itertools.product(k_grid, tau_grid))
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "POLARS_MAX_THREADS"]:
        os.environ[var] = "1"  # one thread per worker process; the workers themselves are the parallelism
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = pool.map(run_cell, *zip(*cells), [kappa_grid] * len(cells))
    return pl.DataFrame([row for cell in results for row in cell])


PARAMS = {"K": K_GRID, "tau": TAU_GRID, "kappa_penalty": KAPPA_GRID}


def neighbourhood_scores(grid: pl.DataFrame) -> pl.DataFrame:
    """Add `stable_score`: mean net Sharpe of a cell and its immediate neighbours (one grid step in one setting)."""
    index = {name: {v: i for i, v in enumerate(values)} for name, values in PARAMS.items()}
    sharpe = {tuple(index[n][row[n]] for n in PARAMS): row["net_sharpe"] for row in grid.iter_rows(named=True)}
    scores = []
    for cell in (tuple(index[n][row[n]] for n in PARAMS) for row in grid.iter_rows(named=True)):
        neighbours = [cell] + [cell[:d] + (cell[d] + step,) + cell[d + 1:] for d in range(len(cell)) for step in (-1, 1)]
        scores.append(float(np.mean([sharpe[c] for c in neighbours if c in sharpe])))
    return grid.with_columns(pl.Series("stable_score", scores))


def select(grid: pl.DataFrame) -> dict:
    """Highest stable_score among cells with positive net Sharpe in both in-sample halves."""
    eligible = neighbourhood_scores(grid).filter((pl.col("net_sharpe_first_half") > 0) & (pl.col("net_sharpe_second_half") > 0))
    return eligible.sort("stable_score", descending=True).row(0, named=True)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    grid = run_grid()
    grid.write_csv(TUNING_GRID_CSV)
    print(grid.sort("net_sharpe", descending=True).head(10))


def report():
    """Select from the saved grid and save heatmaps around the chosen configuration."""
    grid = pl.read_csv(TUNING_GRID_CSV)
    chosen = select(grid)
    print(neighbourhood_scores(grid).sort("stable_score", descending=True).head(10))
    print("chosen:", chosen)
    tuning_heatmaps(grid, chosen, RESULTS_DIR)


def write_locked_config(trials: int):
    """Record the configuration the OOS run must use (current config values) with its in-sample grid row."""
    import src.config as c
    grid = pl.read_csv(TUNING_GRID_CSV)
    row = grid.filter((pl.col("K") == c.N_CLUSTERS) & (pl.col("tau") == c.TAU) & (pl.col("kappa_penalty") == c.KAPPA_PENALTY)).row(0, named=True)
    names = ["N_CLUSTERS", "TAU", "KAPPA_PENALTY", "TC_BPS", "TARGET_VOL", "MAX_STOCK_GROSS", "MAX_FACTOR_WEIGHT",
             "POOL_MONTHS", "N_INIT", "MAX_ITER", "TOL", "RANDOM_STATE", "HISTORY_MONTHS", "COV_HALF_LIFE_MONTHS",
             "MIN_HISTORY_MONTHS", "MIN_SIZE_QUANTILE", "MAX_ZEROTRADE", "MAX_ROW_MISSING_SHARE", "DROPPED_COLS",
             "IS_START_DATE", "IS_END_DATE", "OOS_START_DATE", "OOS_END_DATE"]
    CONFIG_LOCKED_JSON.write_text(json.dumps({
        "locked_on": date.today().isoformat(), "config": {n: getattr(c, n) for n in names},
        "in_sample": row, "grid_configurations_tried": trials,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
