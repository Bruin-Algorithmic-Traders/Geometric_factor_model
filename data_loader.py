"""
data_loader.py
==============
Handles all I/O for the Geometric Factor Model pipeline.

Key convention (from the paper):
    Characteristics observed at the END of month t-1  →  inform portfolio
    held DURING month t  →  realise returns at end of month t.

So for each pair (chars_date, ret_date):
    chars_date  = the characteristic snapshot date   (t-1)
    ret_date    = the subsequent month-end date       (t)
"""

import os
import glob
import re
from datetime import datetime

import numpy as np
import pandas as pd

from config import (
    CHARS_DIR,
    RETURNS_FILE,
    NON_CHAR_COLS,
    MIN_VALID_FRAC,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_date_int(d: int) -> datetime:
    """Convert an integer date like 19900228 to a datetime object."""
    return datetime.strptime(str(d), "%Y%m%d")


def _date_to_int(dt: datetime) -> int:
    return int(dt.strftime("%Y%m%d"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_chars_dates() -> list[int]:
    """
    Return sorted list of integer dates for which characteristic files exist.
    Files are expected to be named  chars_YYYYMMDD.csv  in CHARS_DIR.
    """
    pattern = os.path.join(CHARS_DIR, "chars_*.csv")
    files = sorted(glob.glob(pattern))
    dates = []
    for f in files:
        m = re.search(r"chars_(\d{8})\.csv$", f)
        if m:
            dates.append(int(m.group(1)))
    return sorted(dates)


def get_returns_dates() -> list[int]:
    """Return sorted list of integer dates present in the returns file."""
    ret = pd.read_csv(RETURNS_FILE, usecols=["DATE"])
    return sorted(ret["DATE"].unique().tolist())


def get_date_pairs() -> list[tuple[int, int]]:
    """
    Build a list of (chars_date, ret_date) pairs such that:
        chars at chars_date  →  predict returns at ret_date  (next period).

    Both dates must be present in their respective datasets.
    Returns pairs in chronological order.
    """
    char_dates = set(get_chars_dates())
    ret_dates  = sorted(get_returns_dates())

    pairs = []
    for i, ret_date in enumerate(ret_dates[1:], start=1):          # ret_date = t
        prev_ret_date = ret_dates[i - 1]                            # end of t-1
        # Characteristic file dated at same month-end as previous return date
        if prev_ret_date in char_dates:
            pairs.append((prev_ret_date, ret_date))

    return pairs


def load_characteristics(date: int) -> pd.DataFrame:
    """
    Load the characteristic file for a given integer date.

    Parameters
    ----------
    date : int  e.g. 19900228

    Returns
    -------
    DataFrame with index=permno, columns = characteristic names.
    Non-characteristic columns (permno, DATE, sic2) are dropped.
    Stocks with fewer than MIN_VALID_FRAC of characteristics are removed.
    """
    fpath = os.path.join(CHARS_DIR, f"chars_{date}.csv")
    df = pd.read_csv(fpath, low_memory=False)

    # Set index
    df = df.set_index("permno")

    # Drop non-characteristic columns
    drop_cols = [c for c in NON_CHAR_COLS if c in df.columns and c != "permno"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])

    # Drop stocks that are almost entirely NaN
    valid_frac = df.notna().mean(axis=1)
    df = df[valid_frac >= MIN_VALID_FRAC]

    return df


def load_returns(date: int) -> pd.Series:
    """
    Load cross-sectional returns for a single month-end date.

    Parameters
    ----------
    date : int  e.g. 19900330

    Returns
    -------
    pd.Series  indexed by permno with monthly excess return values.
    Rows with NaN returns are dropped.
    """
    # Load lazily to avoid reading the whole file repeatedly
    ret = pd.read_csv(RETURNS_FILE, usecols=["permno", "DATE", "monthly_ret"])
    ret = ret[ret["DATE"] == date].set_index("permno")["monthly_ret"]
    ret = ret.dropna()
    return ret


def load_returns_panel() -> pd.DataFrame:
    """
    Load the FULL returns panel into memory as a DataFrame.

    Returns
    -------
    DataFrame with MultiIndex (DATE, permno) and column 'monthly_ret'.
    Kept in memory because it is accessed many times during backtesting.
    """
    ret = pd.read_csv(RETURNS_FILE, usecols=["permno", "DATE", "monthly_ret"])
    ret = ret.dropna(subset=["monthly_ret"])
    ret = ret.set_index(["DATE", "permno"])
    return ret


def load_market_cap(date: int) -> pd.Series:
    """
    Load market-cap proxy (mvel1 = price * shares) from the characteristics
    file for a given date. Used for value-weighting in Spec A.

    Returns
    -------
    pd.Series  indexed by permno.
    """
    fpath = os.path.join(CHARS_DIR, f"chars_{date}.csv")
    df = pd.read_csv(fpath, usecols=["permno", "mvel1"], low_memory=False)
    df = df.set_index("permno")["mvel1"]
    return df.dropna()


def get_char_columns() -> list[str]:
    """
    Return the list of characteristic column names by inspecting the first
    available file.  Non-numeric and metadata columns are excluded.
    """
    dates = get_chars_dates()
    df = load_characteristics(dates[0])
    return df.columns.tolist()
