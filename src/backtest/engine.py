from pathlib import Path
import numpy as np
import polars as pl
from src.factors.spec_b import FactorBuilder
from src.models.spherical_kmeans import SphericalKMeans

OOS_START_DATE = 20060101
OOS_END_DATE = 20181231

def run_backtest(n_clusters: int = 30, temperature: float = 0.35):
    project_root = Path(__file__).resolve().parents[2]

    clean_path = project_root / "data" / "processed" / "chars_clean.parquet"
    output_path = project_root / "data" / "processed" / "factor_returns_oos.parquet"

    print(f"Loading data for OOS evaluation ({OOS_START_DATE} to {OOS_END_DATE})...")
    df_all = pl.read_parquet(clean_path).filter(
        (pl.col("DATE") >= OOS_START_DATE) & (pl.col("DATE") <= OOS_END_DATE)
    )

    non_feature_cols = {"DATE", "permno", "return"}
    feature_cols = [c for c in df_all.columns if c not in non_feature_cols]

    monthly_data = []
    
    for date, group in df_all.group_by("DATE", maintain_order=True):
        X = group.select(feature_cols).to_numpy()
        returns = group.select("return").to_numpy().flatten()
        monthly_data.append((date, X, returns))

    print(f"Starting OOS backtest across {len(monthly_data)} months...")
    print(f"Locked Parameters: K = {n_clusters}, Temperature = {temperature}\n")

    builder = FactorBuilder()
    previous_centroids = None
    results = []

    for idx, (date, X, returns) in enumerate(monthly_data):
        if len(returns) < n_clusters:
            continue

        model = SphericalKMeans(n_clusters=n_clusters, max_iter=100, random_state=42)
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
    factor_returns_df.write_parquet(output_path, compression="zstd")
    print(f"\nSaved OOS factor return time series to: {output_path}")

    print_layer_1_diagnostics(factor_returns_df, n_clusters)

def print_layer_1_diagnostics(df: pl.DataFrame, n_clusters: int):
    T = len(df)
    print("\n" + "="*70)
    print("    LAYER 1: OUT-OF-SAMPLE STANDALONE FACTOR DIAGNOSTICS (2006 - 2018)")
    print("="*70)
    print(f"{'Factor':<10} | {'Mean Ret (Ann)':<16} | {'Vol (Ann)':<12} | {'Sharpe':<10} | {'t-stat':<10}")
    print("-" * 70)

    for k in range(1, n_clusters + 1):
        col_name = f"Factor_{k}"
        returns_seq = df.select(col_name).to_series().to_numpy()
        
        mean_monthly = np.mean(returns_seq)
        std_monthly = np.std(returns_seq, ddof=1)
        
        ann_mean = mean_monthly * 12 * 100
        ann_vol = std_monthly * np.sqrt(12) * 100
        ann_sharpe = (mean_monthly / std_monthly) * np.sqrt(12) if std_monthly > 0 else 0.0
        t_stat = (mean_monthly / (std_monthly / np.sqrt(T))) if std_monthly > 0 else 0.0

        print(f"{col_name:<10} | {ann_mean:+15.2f}% | {ann_vol:11.2f}% | {ann_sharpe:10.2f} | {t_stat:10.2f}")

    F_mat = df.select([f"Factor_{k}" for k in range(1, n_clusters + 1)]).to_numpy()
    eq_port_ret = F_mat.mean(axis=1)
    eq_sharpe = (eq_port_ret.mean() / eq_port_ret.std(ddof=1)) * np.sqrt(12)
    
    print("="*70)
    print(f"  Equal-Weighted Combination OOS Sharpe Ratio (2006-2018): {eq_sharpe:.4f}")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_backtest(n_clusters=30, temperature=0.35)