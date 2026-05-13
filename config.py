"""
config.py
=========
Central configuration for the Geometric Factor Model pipeline.
Edit values here to change model behaviour across all modules.
"""

import os

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
RAW_CHARS_DIR = "data/raw/characteristics_by_month"
CHARS_DIR = "data/processed/characteristics_by_month"  # Processed data used for analysis
RETURNS_FILE = "data/raw/monthly_1990_to_2018.csv"
PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "results"

# Columns that are NOT firm characteristics (excluded from the embedding)
NON_CHAR_COLS = ["permno", "DATE", "sic2"]

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
# Winsorisation limits (cross-sectional, per characteristic)
WINSOR_LOWER = 0.01   # 1st percentile
WINSOR_UPPER = 0.99   # 99th percentile

# Minimum fraction of non-NaN values required to keep a stock in a given month.
# Stocks below this threshold are dropped before clustering / regression.
MIN_VALID_FRAC = 0.50   # stock must have >= 50% of characteristics observed

# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
# Primary clustering method: "spherical_kmeans" or "hdbscan"
# (HDBSCAN is always used for validation regardless of this setting)
CLUSTER_METHOD = "spherical_kmeans"   # change to "hdbscan" for density-based

# --- Spherical K-Means ---
N_CLUSTERS = 10          # M  — number of latent firm types
SKM_MAX_ITER = 300       # maximum EM iterations
SKM_N_INIT = 10          # number of random restarts (best inertia kept)
SKM_TOL = 1e-6           # convergence tolerance

# Soft-weight temperature τ (Equation 6).
# Higher τ → harder (more concentrated) assignments.
# Lower  τ → softer (more uniform) assignments.
TEMPERATURE = 10.0

# --- HDBSCAN ---
HDBSCAN_MIN_CLUSTER_SIZE = 30   # minimum points to form a cluster
HDBSCAN_MIN_SAMPLES = 10        # controls noise sensitivity
HDBSCAN_LEAF_SIZE = 40
# k for the core-distance used in mutual reachability (Equation 5)
HDBSCAN_CORE_K = 5

# ---------------------------------------------------------------------------
# Timing / rolling windows
# ---------------------------------------------------------------------------
# Months of history required before we start trading
WARMUP_MONTHS = 36

# Rolling window length for lambda estimation in Spec A (None = expanding)
ROLLING_WINDOW = 60

# ---------------------------------------------------------------------------
# Specification A  —  Fama–MacBeth
# ---------------------------------------------------------------------------
# Cross-sectional weighting: "equal" or "value" (value-weight by market cap)
SPEC_A_CS_WEIGHT = "equal"

# Universe filter: keep only the top N stocks by market cap each month.
# Set to None to use all stocks.
SPEC_A_UNIVERSE_SIZE = 1500

# Portfolio optimisation (Equation 12)
SPEC_A_GAMMA = 2.0          # risk-aversion γ
SPEC_A_KAPPA = 5e-4         # turnover penalty κ
SPEC_A_MAX_LEVERAGE = 2.0   # gross leverage cap L
SPEC_A_MAX_SINGLE = 0.02    # single-name weight cap w_max (2 %)
SPEC_A_DOLLAR_NEUTRAL = True

# Covariance estimator: "ledoit_wolf", "diagonal", or "factor"
# "factor" uses the M firm-type factor returns as a factor model for Σ.
SPEC_A_COV_ESTIMATOR = "ledoit_wolf"

# Transaction cost (basis points per unit of one-way turnover)
TC_BPS = 10.0

# ---------------------------------------------------------------------------
# Specification B  —  Factor Portfolios
# ---------------------------------------------------------------------------
# Factor leverage / concentration limits (Equation 15)
SPEC_B_MAX_FACTOR_LEVERAGE = 1.0   # Lf  per factor
SPEC_B_MAX_FACTOR_SINGLE = 0.05    # w_max^(f) per factor per stock (5 %)

# Factor-timing optimisation (Equation 18)
SPEC_B_GAMMA_F = 2.0        # risk-aversion for factor combination
SPEC_B_KAPPA_F = 1e-3       # turnover penalty across factors
SPEC_B_MAX_LA = 2.0         # gross exposure cap across factors
SPEC_B_MAX_SINGLE_F = 0.5   # single-factor concentration cap

# Rolling window for factor expected-return estimation (None = expanding)
SPEC_B_ROLLING_WINDOW = 60

# ---------------------------------------------------------------------------
# Output / reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
except (PermissionError, OSError):
    pass  # Will be created at runtime on the user's machine
