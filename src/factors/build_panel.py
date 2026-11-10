import polars as pl

from src.config import CHARS_CLEAN_PARQUET, IS_END_DATE, N_CLUSTERS, TAU, TARGET_COL, feature_cols
from src.factors.spec_b import FactorBuilder
from src.models.spherical_kmeans import fit_monthly


def build_panel(n_clusters: int = N_CLUSTERS, tau: float = TAU, end_date: int = IS_END_DATE) -> pl.DataFrame:
    """One continuous run from the first month up to end_date: monthly Spec B factor returns (DATE, f_1..f_K)."""
    df = pl.read_parquet(CHARS_CLEAN_PARQUET).filter(pl.col("DATE") <= end_date)
    features = feature_cols(df.columns)
    groups = [g for _, g in df.group_by("DATE", maintain_order=True) if g.height >= n_clusters]
    months = [g.select(features).to_numpy() for g in groups]

    builder = FactorBuilder()
    rows = []
    for g, X, (model, _) in zip(groups, months, fit_monthly(months, n_clusters)):
        W = builder.compute_weights(model.transform(X, tau))
        f = builder.compute_factor_returns(W, g[TARGET_COL].to_numpy())
        rows.append({"DATE": g["DATE"][0], **{f"f_{k + 1}": f[k] for k in range(n_clusters)}})
    return pl.DataFrame(rows)


if __name__ == "__main__":
    print(build_panel())
