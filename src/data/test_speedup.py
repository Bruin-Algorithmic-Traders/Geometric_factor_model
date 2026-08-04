import os
import glob
import time
import polars as pl
from pathlib import Path

# Paths
csv_dir = "data/raw/characteristics_by_month"  # Directory containing your CSV files
parquet_path = "data/processed/chars_clean.parquet"  # Output single compressed Parquet file

# Get all CSV files in the folder
csv_files = glob.glob(f"{csv_dir}/*.csv")
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {csv_dir}")

# --- 1. Measure Total Storage Footprint ---
csv_total_bytes = sum(os.path.getsize(f) for f in csv_files)
csv_size_mb = csv_total_bytes / (1024 * 1024)

# --- 2. Convert Folder of CSVs -> Parquet ---
if not os.path.exists(parquet_path):
    print(f"Converting {len(csv_files)} CSV files into a single Parquet file...")
    # pl.scan_csv + sink_parquet streams data out-of-core without loading everything into RAM at once
    pl.scan_csv(f"{csv_dir}/*.csv").sink_parquet(parquet_path, compression="zstd")

parquet_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
size_reduction = (1 - (parquet_size_mb / csv_size_mb)) * 100

# --- 3. Benchmark Full Data Read ---
# CSV: Multi-file glob scan & collect
t0 = time.perf_counter()
_ = pl.scan_csv(f"{csv_dir}/*.csv").collect()
csv_full_time = time.perf_counter() - t0

# Parquet: Single file read
t0 = time.perf_counter()
_ = pl.read_parquet(parquet_path)
parquet_full_time = time.perf_counter() - t0

speedup_full = csv_full_time / parquet_full_time

# --- 4. Benchmark Column Pruning (Selecting 3 columns out of 80+) ---
select_cols = ["mvel1", "rd", "DATE"]  # Update to match your actual column names

t0 = time.perf_counter()
_ = pl.scan_csv(f"{csv_dir}/*.csv").select(select_cols).collect()
csv_prune_time = time.perf_counter() - t0

t0 = time.perf_counter()
_ = pl.scan_parquet(parquet_path).select(select_cols).collect()
parquet_prune_time = time.perf_counter() - t0

speedup_pruned = csv_prune_time / parquet_prune_time

# --- Summary Output ---
print(f"\n{' Benchmark Results ':═^55}")
print(f"Number of CSV Files:       {len(csv_files):,}")
print(f"Total CSV Directory Size:  {csv_size_mb:.2f} MB")
print(f"Single Parquet File Size:  {parquet_size_mb:.2f} MB")
print(f"Storage Reduction:         {size_reduction:.1f}%")
print(f"─" * 55)
print(f"Full Read (CSV Multi-file): {csv_full_time:.4f} sec")
print(f"Full Read (Parquet):        {parquet_full_time:.4f} sec")
print(f"Full Read Speedup:          {speedup_full:.1f}x")
print(f"─" * 55)
print(f"Selected Cols (CSV):        {csv_prune_time:.4f} sec")
print(f"Selected Cols (Parquet):    {parquet_prune_time:.4f} sec")
print(f"Column Pruning Speedup:     {speedup_pruned:.1f}x")