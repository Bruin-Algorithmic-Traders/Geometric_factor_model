"""
normalization.py
================
Implements the preprocessing pipeline described in Section 2.2 of the paper.

Pipeline (applied cross-sectionally each month):
    1. Winsorise each characteristic at the [lower, upper] percentiles
    2. Cross-sectional z-score standardisation
    3. Fill remaining NaN with 0  (treats missing as "average" firm)
    4. L2-normalise each firm's vector to the unit hypersphere  →  u_{i,t-1}
    5. Build the N×N cosine-distance matrix  D_{t-1}  (Equation 3)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mstats

from config import MASKED_COSINE_SHRINKAGE, WINSOR_LOWER, WINSOR_UPPER


# ---------------------------------------------------------------------------
# Step 1 — Winsorisation
# ---------------------------------------------------------------------------

def winsorize(df: pd.DataFrame,
              lower: float = WINSOR_LOWER,
              upper: float = WINSOR_UPPER) -> pd.DataFrame:
    """
    Clip each characteristic column at the cross-sectional [lower, upper]
    percentiles to remove outlier influence on the embedding.

    Parameters
    ----------
    df    : DataFrame  (N firms × P characteristics)
    lower : lower percentile (e.g. 0.01 for 1st percentile)
    upper : upper percentile (e.g. 0.99 for 99th percentile)

    Returns
    -------
    DataFrame with same shape, winsorised values.
    """
    out = df.copy()
    for col in out.columns:
        col_data = out[col].dropna()
        if len(col_data) < 5:
            continue
        lo = col_data.quantile(lower)
        hi = col_data.quantile(upper)
        out[col] = out[col].clip(lo, hi)
    return out


# ---------------------------------------------------------------------------
# Step 2 — Cross-sectional z-score standardisation
# ---------------------------------------------------------------------------

def cross_sectional_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Subtract cross-sectional mean and divide by cross-sectional std for each
    characteristic column.  Columns with zero variance are zeroed out.

    Parameters
    ----------
    df : DataFrame  (N firms × P characteristics)

    Returns
    -------
    DataFrame of same shape with z-scored values.
    """
    means = df.mean(skipna=True)
    stds  = df.std(skipna=True)

    # Avoid division by zero
    stds = stds.replace(0.0, np.nan)

    standardised = (df - means) / stds

    # Columns with zero variance become NaN → fill with 0
    return standardised


# ---------------------------------------------------------------------------
# Step 3 — Fill NaN
# ---------------------------------------------------------------------------

def fill_missing(df: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
    """
    Fill any remaining NaN with fill_value.  After z-scoring, 0 corresponds
    to the cross-sectional mean, which is the natural imputation.
    """
    return df.fillna(fill_value)


# ---------------------------------------------------------------------------
# Step 4 — L2 normalisation to unit sphere  (Equation 1)
# ---------------------------------------------------------------------------

def to_unit_sphere(df: pd.DataFrame) -> pd.DataFrame:
    """
    Project each firm's characteristic vector onto the (P-1)-sphere by
    dividing by its L2 norm:

        u_{i,t-1} = z_{i,t-1} / ||z_{i,t-1}||_2          (Equation 1)

    Firms with zero-norm vectors (all characteristics are 0 after imputation)
    are left as zero vectors (they will have undefined cosine distance; they
    are typically removed upstream via MIN_VALID_FRAC).

    Parameters
    ----------
    df : DataFrame  (N firms × P characteristics), all-numeric, no NaN

    Returns
    -------
    DataFrame of same shape with unit-norm rows.
    """
    X = df.values.astype(np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1.0, norms)
    U = X / norms
    return pd.DataFrame(U, index=df.index, columns=df.columns)


def masked_cosine_similarity(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    shrink: bool = MASKED_COSINE_SHRINKAGE,
) -> np.ndarray:
    """
    Compute cosine similarity using only mutually observed coordinates.

    NaNs mark missing characteristics. For each pair, the dot product and both
    norms are computed on the overlapping observed characteristics only. When
    shrink=True, similarities are multiplied by overlap_fraction so low-overlap
    comparisons are pulled toward neutral similarity rather than looking falsely
    precise.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = X if Y is None else np.asarray(Y, dtype=np.float64)
    X_valid = np.isfinite(X)
    Y_valid = np.isfinite(Y)
    X_filled = np.where(X_valid, X, 0.0)
    Y_filled = np.where(Y_valid, Y, 0.0)

    sims = np.zeros((X.shape[0], Y.shape[0]), dtype=np.float64)
    n_features = max(X.shape[1], 1)

    for j in range(Y.shape[0]):
        overlap = X_valid & Y_valid[j]
        overlap_count = overlap.sum(axis=1)
        dot = (X_filled * Y_filled[j] * overlap).sum(axis=1)
        x_norm = np.sqrt(((X_filled ** 2) * overlap).sum(axis=1))
        y_norm = np.sqrt(((Y_filled[j] ** 2) * overlap).sum(axis=1))
        denom = x_norm * y_norm
        sim = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)
        if shrink:
            sim *= overlap_count / n_features
        sims[:, j] = np.clip(sim, -1.0, 1.0)

    return sims


# ---------------------------------------------------------------------------
# Step 5 — Cosine distance matrix  (Equations 2 & 3)
# ---------------------------------------------------------------------------

def cosine_distance_matrix(U: np.ndarray) -> np.ndarray:
    """
    Compute the N×N cosine distance matrix from unit-normed embeddings:

        D[i,j] = 1 - u_i^T u_j                            (Equation 2 / 3)

    Because U is already on the unit sphere, the dot product matrix is the
    cosine similarity matrix, and d(i,j) = 1 - sim(i,j).

    Parameters
    ----------
    U : ndarray  (N × P), rows are unit vectors

    Returns
    -------
    D : ndarray  (N × N), values in [0, 2]
    """
    # Clamp similarities to [-1, 1] for numerical safety
    if np.isnan(U).any():
        sim = masked_cosine_similarity(U)
    else:
        sim = np.clip(U @ U.T, -1.0, 1.0)
    D = 1.0 - sim
    # Force diagonal to 0 and symmetrise
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    return D


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame,
               return_array: bool = False
               ) -> tuple[pd.DataFrame, np.ndarray | None]:
    """
    Run the complete preprocessing pipeline on a raw characteristics DataFrame:

        winsorize → standardise

    Missing values are kept as NaN so downstream clustering can use masked
    cosine similarity instead of treating missing characteristics as average.

    Parameters
    ----------
    df           : raw DataFrame (N × P), index=permno
    return_array : if True, also return the raw numpy array U

    Returns
    -------
    U_df  : DataFrame (N × P), unit-normed embeddings, index=permno
    U_arr : numpy array (N × P) if return_array=True, else None
    """
    df = winsorize(df)
    df = cross_sectional_standardize(df)
    U_df = df

    if return_array:
        return U_df, U_df.values.astype(np.float64)
    return U_df, None
