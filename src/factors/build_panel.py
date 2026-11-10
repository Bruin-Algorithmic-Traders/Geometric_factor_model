import numpy as np
import polars as pl

from src.config import (
    CENTROIDS_NPZ, CHARS_CLEAN_PARQUET, FACTOR_RETURNS_PARQUET, HOLDINGS_PARQUET, IS_END_DATE, N_CLUSTERS,
    PANEL_DIAGNOSTICS_PARQUET, TAU, TARGET_COL, feature_cols,
)
from src.factors.spec_b import FactorBuilder
from src.models.spherical_kmeans import fit_monthly


def build_panel(n_clusters: int = N_CLUSTERS, tau: float = TAU, end_date: int = IS_END_DATE):
    """One continuous run from the first month up to end_date. Returns monthly Spec B factor returns
    (DATE, f_1..f_K), per-month diagnostics, each month's centroids (months x K x features), and each month's
    holdings (permnos, factor weights W of shape firms x K, stock returns) for stock-level costs."""
    df = pl.read_parquet(CHARS_CLEAN_PARQUET).filter(pl.col("DATE") <= end_date)
    features = feature_cols(df.columns)
    groups = [g for _, g in df.group_by("DATE", maintain_order=True) if g.height >= n_clusters]
    months = [g.select(features).to_numpy() for g in groups]

    builder = FactorBuilder()
    rows, diagnostics, centroids, holdings = [], [], [], []
    for g, X, (model, matched) in zip(groups, months, fit_monthly(months, n_clusters)):
        Pi = model.transform(X, tau)
        W = builder.compute_weights(Pi)
        returns = g[TARGET_COL].to_numpy()
        f = builder.compute_factor_returns(W, returns)
        holdings.append((g["permno"].to_numpy(), W, returns))
        date = g["DATE"][0]
        centroids.append(model.centroids)
        rows.append({"DATE": date, **{f"f_{k + 1}": f[k] for k in range(n_clusters)}})
        shares = np.bincount(Pi.argmax(axis=1), minlength=n_clusters) / len(X)
        diagnostics.append({
            "DATE": date, "n_firms": len(X),
            "min_match_cos": None if matched is None else matched.min(),
            "median_match_cos": None if matched is None else float(np.median(matched)),
            "max_stock_weight": np.abs(W).max(),
            "min_cluster_share": shares.min(), "max_cluster_share": shares.max(),
            "avg_max_membership": Pi.max(axis=1).mean(),
        })
    return pl.DataFrame(rows), pl.DataFrame(diagnostics), np.array(centroids), holdings


def save_panel(end_date: int = IS_END_DATE):
    factor_returns, diagnostics, centroids, holdings = build_panel(end_date=end_date)
    factor_returns.write_parquet(FACTOR_RETURNS_PARQUET)
    pl.concat([
        pl.DataFrame({"DATE": np.full(len(permnos), date), "permno": permnos, TARGET_COL: returns})
        .hstack(pl.DataFrame(W, schema=[f"w_{k + 1}" for k in range(W.shape[1])]))
        for date, (permnos, W, returns) in zip(factor_returns["DATE"], holdings)
    ]).write_parquet(HOLDINGS_PARQUET)
    diagnostics.write_parquet(PANEL_DIAGNOSTICS_PARQUET)
    np.savez(CENTROIDS_NPZ, dates=factor_returns["DATE"].to_numpy(), centroids=centroids)
    print(diagnostics.describe())


def load_panel(end_date: int = IS_END_DATE):
    """Read the saved panel up to end_date as (factor_returns, holdings), the same shapes build_panel returns."""
    if not (FACTOR_RETURNS_PARQUET.exists() and HOLDINGS_PARQUET.exists()):
        raise FileNotFoundError("saved panel not found, run `main.py panel` first")
    factor_returns = pl.read_parquet(FACTOR_RETURNS_PARQUET).filter(pl.col("DATE") <= end_date)
    if factor_returns.width - 1 != N_CLUSTERS:
        raise ValueError(f"saved panel has {factor_returns.width - 1} factors but N_CLUSTERS = {N_CLUSTERS}; re-run `main.py panel`")
    df = pl.read_parquet(HOLDINGS_PARQUET).filter(pl.col("DATE") <= end_date)
    w_cols = [c for c in df.columns if c.startswith("w_")]
    holdings = [(g["permno"].to_numpy(), g.select(w_cols).to_numpy(), g[TARGET_COL].to_numpy())
                for _, g in df.group_by("DATE", maintain_order=True)]
    assert len(holdings) == factor_returns.height
    return factor_returns, holdings


if __name__ == "__main__":
    save_panel()
