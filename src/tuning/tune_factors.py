import itertools
import numpy as np
import polars as pl
from src.factors.spec_b import FactorBuilder
from src.models.spherical_kmeans import SphericalKMeans
from src.evaluation.metrics import annualised_stats
from src.config import (
    CHARS_CLEAN_PARQUET, IS_END_DATE, IS_START_DATE, K_GRID, MAX_ITER, RANDOM_STATE,
    TAU_GRID, TARGET_COL, feature_cols,
)

def evaluate_is_grid(n_clusters_grid=K_GRID, temperature_grid=TAU_GRID):
    print(f"Loading data for In-Sample Tuning ({IS_START_DATE} to {IS_END_DATE})...")
    df_all = pl.read_parquet(CHARS_CLEAN_PARQUET).filter(
        (pl.col("DATE") >= IS_START_DATE) & (pl.col("DATE") <= IS_END_DATE)
    )

    features = feature_cols(df_all.columns)
    builder = FactorBuilder()
    grid_results = []
    monthly_data = []
    for date, group in df_all.group_by("DATE", maintain_order=True):
        X = group.select(features).to_numpy()
        returns = group.select(TARGET_COL).to_numpy().flatten()
        monthly_data.append((date, X, returns))

    param_combinations = list(itertools.product(n_clusters_grid, temperature_grid))
    print(f"Testing {len(param_combinations)} hyperparameter combinations across {len(monthly_data)} IS months...\n")

    for k, temp in param_combinations:
        previous_centroids = None
        is_factor_returns = []

        for _, X, returns in monthly_data:
            if len(returns) < k:
                continue

            model = SphericalKMeans(n_clusters=k, max_iter=MAX_ITER, random_state=RANDOM_STATE)
            model.fit(X, init_centroids=previous_centroids)
            previous_centroids = model.centroids.copy()

            Pi = model.transform(X) ** temp
            Pi = Pi / Pi.sum(axis=1, keepdims=True)

            W = builder.compute_weights(Pi)
            f_t = builder.compute_factor_returns(W, returns)
            is_factor_returns.append(f_t)

        F_is = np.array(is_factor_returns)

        mean_monthly = F_is.mean(axis=0)
        std_monthly = F_is.std(axis=0, ddof=1)

        factor_sharpes = np.where(std_monthly > 0, (mean_monthly / std_monthly) * np.sqrt(12), 0.0)
        avg_factor_sharpe = np.mean(factor_sharpes)

        eq_port_ret = F_is.mean(axis=1)
        eq_sharpe = annualised_stats(eq_port_ret)["sharpe"]

        grid_results.append({
            "K": k,
            "Temperature": temp,
            "Avg_Factor_Sharpe": avg_factor_sharpe,
            "EQ_Portfolio_Sharpe": eq_sharpe,
            "Max_Factor_Sharpe": factor_sharpes.max(),
            "Min_Factor_Sharpe": factor_sharpes.min()
        })

        print(f"Config (K={k:<2}, Temp={temp:<3.1f}) | Avg Factor Sharpe: {avg_factor_sharpe:+.3f} | EQ Port Sharpe: {eq_sharpe:+.3f}")

    results_df = pl.DataFrame(grid_results).sort("EQ_Portfolio_Sharpe", descending=True)

    print("\n" + "="*70)
    print("           IN-SAMPLE HYPERPARAMETER GRID SEARCH RESULTS")
    print("="*70)
    print(results_df)

    best_config = results_df.row(0, named=True)
    print(f"\nOptimal In-Sample Config: K = {best_config['K']}, Temperature = {best_config['Temperature']}")
    return best_config

if __name__ == "__main__":
    evaluate_is_grid()
