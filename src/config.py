from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_CHARS_GLOB = RAW_DIR / "characteristics_by_month" / "*.csv"
RAW_RETURNS_CSV = RAW_DIR / "monthly_1990_to_2018.csv"

CHARS_PARQUET = PROCESSED_DIR / "chars.parquet"
CHARS_CLEAN_PARQUET = PROCESSED_DIR / "chars_clean.parquet"
FACTOR_RETURNS_PARQUET = PROCESSED_DIR / "factor_returns.parquet"
CENTROIDS_NPZ = PROCESSED_DIR / "centroids.npz"
PANEL_DIAGNOSTICS_PARQUET = PROCESSED_DIR / "panel_diagnostics.parquet"
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
METADATA_COLS = ["DATE", "permno", "sic2"]
TARGET_COL = "return"

DROPPED_COLS = [
    # over 30% missing in-sample after the row filter
    "realestate", "rd_sale", "rd_mve", "stdacc", "stdcf", "orgcap", "secured", "grltnoa",
    # binary flags with under 10% ones
    "sin", "divi", "divo", "rd",
    # fewest drops so no pair has |avg rank corr| > 0.85; ties keep the higher in-sample |IC t|
    "betasq", "pchsale_pchinvt", "dolvol", "pchquick", "mvel1", "turn", "retvol", "quick",
]


def feature_cols(columns: list[str]) -> list[str]:
    return [c for c in columns if c not in METADATA_COLS and c != TARGET_COL]


# -----------------------------------------------------------------------------
# Universe filters (clean.py)
# -----------------------------------------------------------------------------
MIN_SIZE_QUANTILE = 0.2       # drop firms below this mvel1 quantile each month
MAX_ZEROTRADE = 10            # drop firms with zerotrade >= this
MAX_ROW_MISSING_SHARE = 0.5   # drop firm-months missing more than this share of raw characteristics
UNKNOWN_SIC2 = -1

# -----------------------------------------------------------------------------
# Spherical k-means + memberships
# -----------------------------------------------------------------------------
N_CLUSTERS = 30
TAU = 10
MAX_ITER = 100
TOL = 1e-4
RANDOM_STATE = 42
POOL_MONTHS = 3             # fit clusters on the last 3 months of firms stacked together
N_INIT = 10                 # k-means++ restarts for a cold-start fit (first month only)

# Factor construction
MAX_STOCK_WEIGHT = 0.10   # error if one stock exceeds this share of a factor's gross (in-sample worst is 5.2% at K=35, τ=100)

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
TAU_GRID = [1, 3, 10, 30, 100]
GAMMA_GRID = [0.5, 1.0, 2.0, 5.0, 10.0]
KAPPA_GRID = [0.0001, 0.0003, 0.0005, 0.0010]
