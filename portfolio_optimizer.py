"""
portfolio_optimizer.py
======================
Mean-variance portfolio optimisation with the constraints from Equations (12)
and (18) of the paper, implemented via CVXPY.

Two entry points:
    optimize_portfolio_a   —  Spec A single-name MV optimisation (Eq. 12)
    optimize_portfolio_b   —  Spec B factor-timing MV optimisation (Eq. 18)

A fast fallback (rank-based long-short) is provided for cases where CVXPY
is unavailable or for quick smoke-tests.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from config import (
    SPEC_A_GAMMA,
    SPEC_A_KAPPA,
    SPEC_A_MAX_LEVERAGE,
    SPEC_A_MAX_SINGLE,
    SPEC_A_DOLLAR_NEUTRAL,
    SPEC_B_GAMMA_F,
    SPEC_B_KAPPA_F,
    SPEC_B_MAX_LA,
    SPEC_B_MAX_SINGLE_F,
)


# ---------------------------------------------------------------------------
# Covariance estimation helpers
# ---------------------------------------------------------------------------

def _impute_partial_history(R: pd.DataFrame) -> pd.DataFrame:
    """
    Fill partial return-history gaps using a simple market-model regression.

    For each stock with some observed history, estimate:

        r_i,t = alpha_i + beta_i * r_m,t + eps_i,t

    on the months where r_i,t is observed, where r_m,t is the equal-weighted
    cross-sectional mean return in the current lookback window. Missing months
    are then filled with the fitted value. If a stock has too few observations
    for a stable regression, fall back to its own sample mean.
    """
    if R.empty:
        return R

    market = R.mean(axis=1, skipna=True)
    R_filled = R.copy()

    for col in R.columns:
        s = R[col]
        missing = s.isna()
        if not missing.any():
            continue

        observed = s.notna() & market.notna()
        if observed.sum() >= 6 and market.loc[observed].std() > 0:
            x = market.loc[observed].values
            y = s.loc[observed].values
            beta, alpha = np.polyfit(x, y, deg=1)
            preds = alpha + beta * market.loc[missing].values
            fill_vals = pd.Series(preds, index=s.index[missing])
        else:
            stock_mean = s.loc[observed].mean()
            if not np.isfinite(stock_mean):
                stock_mean = 0.0
            fill_vals = pd.Series(stock_mean, index=s.index[missing])

        R_filled.loc[missing, col] = fill_vals

    return R_filled.fillna(0.0)

def estimate_covariance(
    returns_panel: pd.DataFrame,
    permnos: list,
    method: str = "ledoit_wolf",
    min_obs: int = 24,
) -> tuple[np.ndarray, list]:
    """
    Estimate a covariance matrix from a panel of historical returns.

    Parameters
    ----------
    returns_panel : DataFrame (T × N), each column is a stock's return series
    permnos       : list of permno ids to extract (columns must match)
    method        : "ledoit_wolf" | "sample" | "diagonal"
    min_obs       : minimum T required; falls back to diagonal if not met

    Returns
    -------
    Sigma : ndarray (K × K), where K is the number of names with at least one
        return observation in the lookback window
    active_permnos : list of permnos corresponding to Sigma
    """
    R = returns_panel.reindex(columns=permnos).dropna(how="all")

    # Some names can appear in the current characteristic universe before they
    # have any usable return history in the covariance lookback window.
    has_history = R.notna().any(axis=0)
    active_permnos = has_history[has_history].index.tolist()
    R_active = R[active_permnos]

    fallback_scale = R_active.stack().std()
    if not np.isfinite(fallback_scale) or fallback_scale <= 0:
        fallback_scale = R.stack().std()
    if not np.isfinite(fallback_scale) or fallback_scale <= 0:
        fallback_scale = 0.05

    if len(R) < min_obs or len(active_permnos) < 2:
        # Fall back to diagonal with sample variances
        vols = R_active.std().reindex(active_permnos).fillna(fallback_scale)
        return np.diag(vols.values ** 2), active_permnos

    R_active = _impute_partial_history(R_active)

    X = R_active.values
    if method == "ledoit_wolf":
        lw = LedoitWolf(assume_centered=False)
        lw.fit(X)
        Sigma_active = lw.covariance_
    elif method == "sample":
        Sigma_active = np.cov(X.T)
    elif method == "diagonal":
        Sigma_active = np.diag(np.var(X, axis=0))
    else:
        raise ValueError(f"Unknown covariance method: {method}")

    # Ensure positive definiteness with a small ridge
    min_eig = np.linalg.eigvalsh(Sigma_active).min()
    if min_eig < 1e-8:
        Sigma_active += (1e-6 - min_eig) * np.eye(len(active_permnos))

    return Sigma_active, active_permnos


# ---------------------------------------------------------------------------
# Spec A  —  single-name portfolio optimisation  (Equation 12)
# ---------------------------------------------------------------------------

def optimize_portfolio_a(
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_prev: np.ndarray,
    gamma: float  = SPEC_A_GAMMA,
    kappa: float  = SPEC_A_KAPPA,
    max_lev: float = SPEC_A_MAX_LEVERAGE,
    w_max: float  = SPEC_A_MAX_SINGLE,
    dollar_neutral: bool = SPEC_A_DOLLAR_NEUTRAL,
    neutrality_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """
    Solve the constrained mean-variance problem (Equation 12):

        max_w  w^T μ - (γ/2) w^T Σ w - κ ||w - w_prev||_1
        s.t.   1^T w = 0            (dollar-neutral, if enabled)
               ||w||_1 ≤ L          (gross leverage cap)
               ||w||_∞ ≤ w_max      (single-name cap)
               A w = 0              (optional neutrality constraints)

    Parameters
    ----------
    mu      : ndarray (N,) expected returns
    Sigma   : ndarray (N × N) covariance matrix
    w_prev  : ndarray (N,) portfolio weights from previous period (for turnover)
    gamma   : risk-aversion coefficient
    kappa   : turnover penalty (ℓ1 proxy)
    max_lev : gross leverage cap L
    w_max   : single-name weight cap
    dollar_neutral     : enforce 1^T w = 0
    neutrality_matrix  : ndarray (K × N) for A w = 0 constraints

    Returns
    -------
    w : ndarray (N,) optimal portfolio weights
    """
    import cvxpy as cp

    N = len(mu)

    w     = cp.Variable(N)
    delta = cp.Variable(N)   # |w - w_prev| auxiliary variable for turnover

    # --- Objective ---
    ret_term      = mu @ w
    risk_term     = (gamma / 2) * cp.quad_form(w, Sigma)
    turnover_term = kappa * cp.sum(delta)

    objective = cp.Maximize(ret_term - risk_term - turnover_term)

    # --- Constraints ---
    constraints = [
        delta >= w - w_prev,
        delta >= w_prev - w,
    ]

    if dollar_neutral:
        constraints.append(cp.sum(w) == 0)

    constraints += [
        cp.norm1(w) <= max_lev,
        w <=  w_max,
        w >= -w_max,
    ]

    if neutrality_matrix is not None:
        constraints.append(neutrality_matrix @ w == 0)

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except Exception as e:
            warnings.warn(f"Portfolio optimisation failed: {e}. Returning zeros.")
            return np.zeros(N)

    if w.value is None:
        warnings.warn("Optimisation returned None. Returning zeros.")
        return np.zeros(N)

    return np.array(w.value)


# ---------------------------------------------------------------------------
# Spec B  —  factor-timing optimisation  (Equation 18)
# ---------------------------------------------------------------------------

def optimize_portfolio_b(
    mu_f: np.ndarray,
    Sigma_f: np.ndarray,
    a_prev: np.ndarray,
    gamma_f: float  = SPEC_B_GAMMA_F,
    kappa_f: float  = SPEC_B_KAPPA_F,
    max_la: float   = SPEC_B_MAX_LA,
    a_max: float    = SPEC_B_MAX_SINGLE_F,
    self_financing: bool = True,
) -> np.ndarray:
    """
    Solve the factor-timing optimisation (Equation 18):

        max_a  a^T μ_f - (γ_f/2) a^T Σ_f a - κ_f ||a - a_prev||_1
        s.t.   1^T a = 0            (optional self-financing across factors)
               ||a||_1 ≤ La
               ||a||_∞ ≤ a_max

    Parameters
    ----------
    mu_f    : ndarray (M,) factor expected returns
    Sigma_f : ndarray (M × M) factor covariance matrix
    a_prev  : ndarray (M,) previous factor weights
    gamma_f : risk-aversion for factor combination
    kappa_f : turnover penalty
    max_la  : gross exposure cap across factors
    a_max   : single-factor concentration cap
    self_financing : enforce 1^T a = 0

    Returns
    -------
    a : ndarray (M,) optimal factor weights
    """
    import cvxpy as cp

    M = len(mu_f)
    a     = cp.Variable(M)
    delta = cp.Variable(M)

    ret_term      = mu_f @ a
    risk_term     = (gamma_f / 2) * cp.quad_form(a, Sigma_f)
    turnover_term = kappa_f * cp.sum(delta)

    objective = cp.Maximize(ret_term - risk_term - turnover_term)

    constraints = [
        delta >= a - a_prev,
        delta >= a_prev - a,
        cp.norm1(a) <= max_la,
        a <=  a_max,
        a >= -a_max,
    ]

    if self_financing:
        constraints.append(cp.sum(a) == 0)

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except Exception as e:
            warnings.warn(f"Factor-timing optimisation failed: {e}. Returning zeros.")
            return np.zeros(M)

    if a.value is None:
        warnings.warn("Factor optimisation returned None. Returning zeros.")
        return np.zeros(M)

    return np.array(a.value)


# ---------------------------------------------------------------------------
# Fallback: simple rank-based long-short (no CVXPY required)
# ---------------------------------------------------------------------------

def rank_longshort(
    mu: np.ndarray,
    n_long: int  = 50,
    n_short: int = 50,
) -> np.ndarray:
    """
    Simple dollar-neutral long-short portfolio: equal-weight top n_long stocks,
    equal-weight short bottom n_short stocks.  No covariance or optimisation.

    Used as a quick diagnostic or fallback when CVXPY is unavailable.
    """
    N = len(mu)
    n_long  = min(n_long,  N // 2)
    n_short = min(n_short, N // 2)

    w = np.zeros(N)
    ranked = np.argsort(mu)
    w[ranked[-n_long:]]  =  1.0 / n_long    # long
    w[ranked[:n_short]]  = -1.0 / n_short   # short
    return w


# ---------------------------------------------------------------------------
# Covariance utility for factor returns  (used in Spec B factor timing)
# ---------------------------------------------------------------------------

def factor_covariance(
    factor_returns: np.ndarray,
    method: str = "ledoit_wolf",
) -> np.ndarray:
    """
    Estimate M×M factor covariance from T×M factor return panel.

    Parameters
    ----------
    factor_returns : ndarray (T × M)
    method         : "ledoit_wolf" | "sample" | "diagonal"

    Returns
    -------
    Sigma_f : ndarray (M × M)
    """
    T, M = factor_returns.shape
    if T < 4:
        return np.eye(M) * np.var(factor_returns)

    if method == "ledoit_wolf":
        lw = LedoitWolf()
        lw.fit(factor_returns)
        Sigma_f = lw.covariance_
    elif method == "sample":
        Sigma_f = np.cov(factor_returns.T)
    elif method == "diagonal":
        Sigma_f = np.diag(np.var(factor_returns, axis=0))
    else:
        raise ValueError(f"Unknown covariance method: {method}")

    # Small ridge for numerical safety
    min_eig = np.linalg.eigvalsh(Sigma_f).min()
    if min_eig < 1e-8:
        Sigma_f += (1e-6 - min_eig) * np.eye(M)

    return Sigma_f
