import numpy as np
import polars as pl
from src.factors.spec_b import FactorBuilder
from src.models.spherical_kmeans import SphericalKMeans
from src.evaluation.metrics import annualised_stats
from src.config import (
    CHARS_CLEAN_PARQUET, FACTOR_RETURNS_OOS_PARQUET, MAX_ITER, N_CLUSTERS,
    OOS_END_DATE, OOS_START_DATE, RANDOM_STATE, TAU, TARGET_COL, feature_cols,
)

def run_backtest(n_clusters: int = N_CLUSTERS, temperature: float = TAU):
    print(f"Loading data for OOS evaluation ({OOS_START_DATE} to {OOS_END_DATE})...")
    df_all = pl.read_parquet(CHARS_CLEAN_PARQUET).filter(
        (pl.col("DATE") >= OOS_START_DATE) & (pl.col("DATE") <= OOS_END_DATE)
    )

    features = feature_cols(df_all.columns)

    monthly_data = []

    for date, group in df_all.group_by("DATE", maintain_order=True):
        X = group.select(features).to_numpy()
        returns = group.select(TARGET_COL).to_numpy().flatten()
        monthly_data.append((date, X, returns))

    print(f"Starting OOS backtest across {len(monthly_data)} months...")
    print(f"Locked Parameters: K = {n_clusters}, Temperature = {temperature}\n")

    builder = FactorBuilder()
    previous_centroids = None
    results = []

    for idx, (date, X, returns) in enumerate(monthly_data):
        if len(returns) < n_clusters:
            continue

        model = SphericalKMeans(n_clusters=n_clusters, max_iter=MAX_ITER, random_state=RANDOM_STATE)
        model.fit(X, init_centroids=previous_centroids)
        previous_centroids = model.centroids.copy()

        X_norm = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        sims = (X_norm @ model.centroids.T) * temperature

        exp_sims = np.exp(sims - np.max(sims, axis=1, keepdims=True))
        Pi = exp_sims / exp_sims.sum(axis=1, keepdims=True)

        W = builder.compute_weights(Pi)
        f_t = builder.compute_factor_returns(W, returns)

        row_dict = {"DATE": date}
        for k in range(n_clusters):
            row_dict[f"Factor_{k+1}"] = f_t[k]
        results.append(row_dict)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(monthly_data):
            print(f"Processed {idx + 1}/{len(monthly_data)} OOS months...")

    factor_returns_df = pl.DataFrame(results)
    factor_returns_df.write_parquet(FACTOR_RETURNS_OOS_PARQUET, compression="zstd")
    print(f"\nSaved OOS factor return time series to: {FACTOR_RETURNS_OOS_PARQUET}")

    print_layer_1_diagnostics(factor_returns_df, n_clusters)

def print_layer_1_diagnostics(df: pl.DataFrame, n_clusters: int):
    print("\n" + "="*70)
    print("    LAYER 1: OUT-OF-SAMPLE STANDALONE FACTOR DIAGNOSTICS (2006 - 2018)")
    print("="*70)
    print(f"{'Factor':<10} | {'Mean Ret (Ann)':<16} | {'Vol (Ann)':<12} | {'Sharpe':<10} | {'t-stat':<10}")
    print("-" * 70)

    for k in range(1, n_clusters + 1):
        col_name = f"Factor_{k}"
        s = annualised_stats(df.select(col_name).to_series().to_numpy())
        ann_mean, ann_vol = s["ann_return"] * 100, s["ann_vol"] * 100
        ann_sharpe, t_stat = s["sharpe"], s["t_stat"]

        print(f"{col_name:<10} | {ann_mean:+15.2f}% | {ann_vol:11.2f}% | {ann_sharpe:10.2f} | {t_stat:10.2f}")

    F_mat = df.select([f"Factor_{k}" for k in range(1, n_clusters + 1)]).to_numpy()
    eq_port_ret = F_mat.mean(axis=1)
    eq_sharpe = annualised_stats(eq_port_ret)["sharpe"]

    print("="*70)
    print(f"  Equal-Weighted Combination OOS Sharpe Ratio (2006-2018): {eq_sharpe:.4f}")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_backtest()
