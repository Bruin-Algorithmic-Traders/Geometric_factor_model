from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

RAW_CHARS_GLOB = RAW_DIR / "characteristics_by_month" / "*.csv"
RAW_RETURNS_CSV = RAW_DIR / "monthly_1990_to_2018.csv"

CHARS_PARQUET = PROCESSED_DIR / "chars.parquet"
CHARS_CLEAN_PARQUET = PROCESSED_DIR / "chars_clean.parquet"
FACTOR_RETURNS_IS_PARQUET = PROCESSED_DIR / "factor_returns_insample.parquet"
FACTOR_RETURNS_OOS_PARQUET = PROCESSED_DIR / "factor_returns_oos.parquet"
STRATEGY_RETURNS_PARQUET = PROCESSED_DIR / "spec_b_oos_returns.parquet"
STRATEGY_WEIGHTS_PARQUET = PROCESSED_DIR / "spec_b_factor_weights.parquet"
OPTIMISER_GRID_PARQUET = PROCESSED_DIR / "spec_b_pure_grid_results.parquet"

# -----------------------------------------------------------------------------
# Sample split (YYYYMMDD ints, matching the DATE column)
# -----------------------------------------------------------------------------
IS_START_DATE = 19900101
IS_END_DATE = 20051231
OOS_START_DATE = 20060101
OOS_END_DATE = 20181231

# -----------------------------------------------------------------------------
# Column groups
# -----------------------------------------------------------------------------
METADATA_COLS = ["DATE", "permno"]
TARGET_COL = "return"

DROPPED_COLS = [
    "sic2",        # Industry id
    "betasq",      # Duplicate of beta
    "pchsaleinv",  # Duplicate of pchsale_pchinvt
    "securedind",  # Duplicate flag for secured
    "herf",        # Industry-level macro constant
    "indmom",      # Industry-level macro constant
    "rd_mve",      # Duplicate R&D ratio
    "rd_sale",     # Duplicate R&D ratio
    "sin",         # >98% sparse flag
    "divi",        # >98% sparse flag
    "divo",        # >98% sparse flag
    "ps",          # Piotroski composite score
    "absacc",      # Absolute accruals (destroys sign of acc)
    "convind",     # Sparse convertible indicator
]

ZERO_FILL_COLS = ["rd"]


def feature_cols(columns: list[str]) -> list[str]:
    return [c for c in columns if c not in METADATA_COLS and c != TARGET_COL]


# -----------------------------------------------------------------------------
# Universe filters (clean.py)
# -----------------------------------------------------------------------------
MIN_SIZE_QUANTILE = 0.2   # drop firms below this mvel1 quantile each month
MAX_ZEROTRADE = 10        # drop firms with zerotrade >= this

# -----------------------------------------------------------------------------
# Spherical k-means + memberships
# -----------------------------------------------------------------------------
N_CLUSTERS = 30
TAU = 0.35
MAX_ITER = 100
TOL = 1e-4
RANDOM_STATE = 42

# Factor construction
FACTOR_W_MAX = None       # per-stock cap in Spec B factors (None = off)

# -----------------------------------------------------------------------------
# Factor-timing optimiser (paper eq. 18)
# -----------------------------------------------------------------------------
GAMMA_F = 1.0             # factor risk aversion
KAPPA_F = 0.0003          # turnover penalty and realised cost per unit turnover
GROSS_FACTOR_LEVERAGE = 1.0   # L_a
MAX_FACTOR_WEIGHT = 0.20      # a_max
MU_WINDOW_MONTHS = 12
COV_WINDOW_MONTHS = 36
MIN_HISTORY_MONTHS = 36

# -----------------------------------------------------------------------------
# Tuning grids
# -----------------------------------------------------------------------------
K_GRID = [15, 20, 25, 30, 35]
TAU_GRID = [0.2, 0.35, 0.5, 0.75, 1.0]
GAMMA_GRID = [0.5, 1.0, 2.0, 5.0, 10.0]
KAPPA_GRID = [0.0001, 0.0003, 0.0005, 0.0010]
