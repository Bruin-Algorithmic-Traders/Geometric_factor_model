import polars as pl
import numpy as np
from src.models.spherical_kmeans import SphericalKMeans

target_date = 19920228

print("Loading data...")
df = pl.read_parquet("data/processed/chars_clean.parquet").filter(pl.col("DATE") == target_date).drop(["DATE", "permno"])
X = df.to_numpy()

model = SphericalKMeans(n_clusters=10, max_iter=100, random_state=42)
print("Fitting model...")
model.fit(X)
Pi = model.transform(X)

print("\n--- Diagnostic Sanity Checks ---")
print(f"1. Centroid Norms == 1.0? {np.allclose(np.linalg.norm(model.centroids, axis=1), 1.0)}")
print(f"2. Membership Rows Sum to 1.0? {np.allclose(Pi.sum(axis=1), 1.0)}")
print(f"3. Any NaNs or Infs? {np.isnan(Pi).any() or np.isinf(Pi).any()}")
print(f"4. Dominant Factor Weight Avg: {Pi.max(axis=1).mean():.3f}")
print(f"5. Min/Max Cluster Sizes (Hard Assignment): {np.bincount(Pi.argmax(axis=1), minlength=10)}")

