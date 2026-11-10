import polars as pl
import numpy as np
from src.factors.spec_b import FactorBuilder
from src.models.spherical_kmeans import SphericalKMeans
from src.config import CHARS_CLEAN_PARQUET, MAX_ITER, RANDOM_STATE, TARGET_COL, feature_cols

target_date = 19920131
n_test_clusters = 10

print("Loading data...")
df = pl.read_parquet(CHARS_CLEAN_PARQUET).filter(pl.col("DATE") == target_date).drop(["DATE", "permno"])

X = df.select(feature_cols(df.columns)).to_numpy()
returns = df.select(TARGET_COL).to_numpy().flatten()

model = SphericalKMeans(n_clusters=n_test_clusters, max_iter=MAX_ITER, random_state=RANDOM_STATE)
print("Fitting model...")
model.fit(X)
Pi = model.transform(X)

builder = FactorBuilder()
W = builder.compute_weights(Pi)
f_t = builder.compute_factor_returns(W, returns)

assert np.allclose(W.sum(axis=0), 0.0, atol=1e-7), "Dollar neutrality violated!"
assert np.allclose(np.abs(W).sum(axis=0), 1.0, atol=1e-7), "L1 leverage normalization failed!"
print("\n--- Diagnostic Sanity Checks ---")
print(f"1. Centroid Norms == 1.0? {np.allclose(np.linalg.norm(model.centroids, axis=1), 1.0)}")
print(f"2. Membership Rows Sum to 1.0? {np.allclose(Pi.sum(axis=1), 1.0)}")
print(f"3. Any NaNs or Infs? {np.isnan(Pi).any() or np.isinf(Pi).any()}")
print(f"4. Dominant Factor Weight Avg: {Pi.max(axis=1).mean():.3f}")
print(f"5. Min/Max Cluster Sizes (Hard Assignment): {np.bincount(Pi.argmax(axis=1), minlength=n_test_clusters)}")
print(f"\n--- Realized Factor Returns for {target_date} ---")
for k, ret in enumerate(f_t):
    print(f"   Factor {k+1}: {ret * 100:+.2f}%")

