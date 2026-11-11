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
CHARS_CLEAN_RETURNS_FREE_PARQUET = PROCESSED_DIR / "chars_clean_returns_free.parquet"
FACTOR_RETURNS_PARQUET = PROCESSED_DIR / "factor_returns.parquet"
CENTROIDS_NPZ = PROCESSED_DIR / "centroids.npz"
REBUILT_HISTORY_NPY = PROCESSED_DIR / "rebuilt_history.npy"
HOLDINGS_PARQUET = PROCESSED_DIR / "holdings.parquet"
PANEL_DIAGNOSTICS_PARQUET = PROCESSED_DIR / "panel_diagnostics.parquet"
FACTOR_RETURNS_OOS_PARQUET = PROCESSED_DIR / "factor_returns_oos.parquet"
STRATEGY_RETURNS_PARQUET = PROCESSED_DIR / "spec_b_oos_returns.parquet"
STRATEGY_WEIGHTS_PARQUET = PROCESSED_DIR / "spec_b_factor_weights.parquet"
TUNING_GRID_CSV = RESULTS_DIR / "tuning_grid.csv"
CONFIG_LOCKED_JSON = RESULTS_DIR / "config_locked.json"

# -----------------------------------------------------------------------------
# Sample split (YYYYMMDD ints, matching the DATE column)
# -----------------------------------------------------------------------------
IS_START_DATE = 19900101
IS_END_DATE = 20051231
OOS_START_DATE = 20060101
IS_HALF_SPLIT_DATE = 20000101   # in-sample robustness check: 1993-1999 vs 2000-2005
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

# Robustness variant that never looks at returns: redundancy ties broken by lower missingness, then the more
# primitive variable (keeps pchsale_pchinvt over pchsaleinv and mvel1 over ill); flags and scores kept as above.
RETURNS_FREE_DROPPED_COLS = [c for c in DROPPED_COLS if c not in ("pchsale_pchinvt", "mvel1")] + ["pchsaleinv", "ill"]


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
N_CLUSTERS = 40            # locked after in-sample tuning (results/config_locked.json)
TAU = 1.0                  # locked
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
KAPPA_PENALTY = 0.0001    # turnover penalty in the objective (locked)
TC_BPS = 10               # realised one-way trading cost in bp per $ traded (fixed, not tuned)
TARGET_VOL = 0.10             # ex-ante annual vol budget: aᵀΣ̂a ≤ TARGET_VOL² / 12 (replaces the γ risk term)
MAX_STOCK_GROSS = 3.0         # Σ|stock weights| ≤ 3 × capital
MAX_FACTOR_WEIGHT = 0.60      # |a_k| ≤ 0.6, i.e. no factor above 20% of the 3x gross budget
COV_HALF_LIFE_MONTHS = 12    # EWMA half-life for Σ̂ (μ̂ is the plain mean of the HISTORY_MONTHS rebuilt months)
MIN_HISTORY_MONTHS = 36
HISTORY_MONTHS = 60          # months of rebuilt history kept per month (μ̂ window and EWMA cut-off)

# -----------------------------------------------------------------------------
# Tuning grids
# -----------------------------------------------------------------------------
K_GRID = [15, 20, 25, 30, 35, 40, 50]
TAU_GRID = [0.01, 0.3, 1, 3, 10, 30, 100]   # 0.01 ≈ linear-in-cos baseline
KAPPA_GRID = [0.0001, 0.0003, 0.0010, 0.0030]   # κ_penalty: 1, 3, 10, 30 bp
TUNE_WORKERS = 4    # parallel (K, τ) cells; each holds ~1-1.5 GB
