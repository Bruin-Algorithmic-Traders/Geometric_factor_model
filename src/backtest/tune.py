# src/backtest/tune.py

from pathlib import Path
import itertools
import numpy as np
import polars as pl
from src.factors.spec_b import FactorBuilder
from src.models.spherical_kmeans import SphericalKMeans

IS_START_DATE = 19900101
IS_END_DATE = 20051231

def evaluate_is_grid(n_clusters_grid=[15, 20, 25, 30, 35], temperature_grid=[0.2, 0.35, 0.5, 0.75, 1.0]):
    project_root = Path(__file__).resolve().parents[2]
    clean_path = project_root / "data" / "processed" / "chars_clean.parquet"

    print(f"Loading data for In-Sample Tuning ({IS_START_DATE} to {IS_END_DATE})...")
    df_all = pl.read_parquet(clean_path).filter(
        (pl.col("DATE") >= IS_START_DATE) & (pl.col("DATE") <= IS_END_DATE)
    )

    all_dates = df_all.select("DATE").unique().sort("DATE").to_series().to_list()
    non_feature_cols = ["DATE", "permno", "return"]
    feature_cols = [col for col in df_all.columns if col not in non_feature_cols]
    builder = FactorBuilder()

    param_combinations = list(itertools.product(n_clusters_grid, temperature_grid))
    print(f"Testing {len(param_combinations)} hyperparameter combinations across {len(all_dates)} IS months...\n")

    grid_results = []

    monthly_data = []

    for date, group in df_all.group_by("DATE", maintain_order=True):
        X = group.select(feature_cols).to_numpy()
        returns = group.select("return").to_numpy().flatten()
        monthly_data.append((date, X, returns))

    for k, temp in param_combinations:
        previous_centroids = None
        is_factor_returns = []

        for date, X, returns in monthly_data:
            if len(returns) < k:
                continue

            model = SphericalKMeans(n_clusters=k, max_iter=100, random_state=42)
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
        eq_sharpe = (eq_port_ret.mean() / eq_port_ret.std(ddof=1)) * np.sqrt(12)

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