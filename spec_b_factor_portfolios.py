"""
spec_b_factor_portfolios.py
============================
Specification B  —  Firm-type portfolios as factor returns
Section 2.4.2 of "A Geometric Approach to Firm-Type Factor Models"

Timing convention (no look-ahead):
───────────────────────────────────
Each iteration processes pair  (chars_date, ret_date)  = (t−1, t):

  Step 1  Load chars at t−1  →  Pi_{t−1}  →  W_{t−1}  (factor weights for period t)
  Step 2  Compute f_t = W_{t−1}ᵀ rₜ           (Eq. 16, factor return during period t)
  Step 3  BEFORE appending f_t: estimate a_{t−1} from {f₁,…,f_{t−1}}
  Step 4  REALISE  R_t^(B) = a_{t−1}ᵀ f_t      (no look-ahead: a uses past f only)
  Step 5  Append f_t to buffer; now estimate a_t for NEXT period's portfolio

Additional output:
  Time-series regression  r_{i,t} = αᵢ + βᵢᵀ fₜ + εᵢₜ  (Eq. 17)
  Tests αᵢ = 0 and reports mean R².

Usage
─────
    python spec_b_factor_portfolios.py [--method spherical_kmeans|hdbscan]
                                        [--n_clusters 10]
                                        [--tau 10.0]
                                        [--universe 1500]
                                        [--start 19950101]
                                        [--no_timing]

Saves:
    results/spec_b_factor_returns.csv
    results/spec_b_returns.csv
    results/spec_b_performance.png
    results/spec_b_factor_heatmap.png
    results/spec_b_clustering_results.pkl.gz  ← all soft membership matrices Π_{t-1}
    results/spec_b_betas.csv
    results/spec_b_alphas.csv
    results/spec_b_metrics.csv
"""
from __future__ import annotations

import argparse
import os
import warnings
import numpy as np
import pandas as pd
from collections import deque

import config
from data_loader   import (get_date_pairs, load_characteristics,
                            load_market_cap, load_returns_panel)
from normalization import preprocess
from clustering    import cluster_and_weight, align_cluster_labels
from portfolio_optimizer import (optimize_portfolio_b, factor_covariance)
from backtesting   import (BacktestResult, compute_tc, plot_performance,
                            plot_factor_returns, plot_cluster_weights_over_time,
                            print_metrics_table, save_metrics_csv)


# ===========================================================================
# Factor portfolio construction  (Equations 14 & 15)
# ===========================================================================

def build_factor_portfolios(
    Pi_df: pd.DataFrame,
    max_factor_lev: float    = config.SPEC_B_MAX_FACTOR_LEVERAGE,
    max_factor_single: float = config.SPEC_B_MAX_FACTOR_SINGLE,
) -> pd.DataFrame:
    """
    Convert the soft membership matrix Π_{t−1} (N × M) into M dollar-neutral
    factor-mimicking portfolio weight vectors  w^(m)  (Equations 14–15).

    For each firm type m:
        w̃^(m)_i = π_{i,m} − mean_j(π_{j,m})          (demean  →  dollar-neutral)
        w^(m)_i  = w̃^(m)_i / Σ_j|w̃^(m)_j|             (normalise ‖w‖₁ = 1)

    Constraints (Eq. 15):
        ‖w^(m)‖₁ ≤ max_factor_lev   (re-scaled if needed)
        |w^(m)_i| ≤ max_factor_single (single-name cap, then re-normalise)

    Returns
    -------
    W_df : DataFrame (N × M), index=permno
    """
    Pi = Pi_df.values.astype(np.float64)   # N × M
    N, M = Pi.shape

    # Demean each factor column across firms (ensures dollar-neutrality)
    Pi_dm = Pi - Pi.mean(axis=0, keepdims=True)

    # L1-normalise each factor column
    l1 = np.abs(Pi_dm).sum(axis=0, keepdims=True)
    l1 = np.where(l1 == 0.0, 1.0, l1)
    W = Pi_dm / l1                         # each col sums to 0

    # Apply single-name cap; re-normalise to preserve ‖w‖₁ = 1
    W = np.clip(W, -max_factor_single, max_factor_single)
    l1r = np.abs(W).sum(axis=0, keepdims=True)
    l1r = np.where(l1r == 0.0, 1.0, l1r)
    W = W / l1r

    # Scale to max_factor_lev
    W = W * max_factor_lev

    return pd.DataFrame(W, index=Pi_df.index, columns=Pi_df.columns)


# ===========================================================================
# Time-series regression  (Equation 17)
# ===========================================================================

def time_series_regression(
    returns_panel: pd.DataFrame,
    factor_returns: pd.DataFrame,
    min_obs: int = 24,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """
    OLS regression of each asset's return on the M factor returns:

        r_{i,t} = αᵢ + βᵢᵀ fₜ + εᵢₜ          (Equation 17)

    Parameters
    ----------
    returns_panel  : DataFrame (T × N)  asset returns (DATE index, permno cols)
    factor_returns : DataFrame (T × M)  factor returns  (DATE index)
    min_obs        : minimum valid observations per asset

    Returns
    -------
    alphas  : Series (N,)  intercepts αᵢ
    betas   : DataFrame (N × M)  factor loadings βᵢₘ
    r2      : Series (N,)  OLS R²
    """
    common = returns_panel.index.intersection(factor_returns.index)
    R = returns_panel.loc[common]
    F = factor_returns.loc[common].fillna(factor_returns.mean())

    # Keep only assets with enough data
    valid = R.columns[R.notna().sum() >= min_obs]
    R = R[valid]

    alphas_dict, betas_dict, r2_dict = {}, {}, {}
    for perm in R.columns:
        y = R[perm]
        mask = y.notna() & F.notna().all(axis=1)
        if mask.sum() < min_obs:
            continue
        X = np.column_stack([np.ones(mask.sum()), F.loc[mask].values])
        yv = y.loc[mask].values
        try:
            theta, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
        except np.linalg.LinAlgError:
            continue
        y_hat = X @ theta
        ss_res = np.sum((yv - y_hat) ** 2)
        ss_tot = np.sum((yv - yv.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        alphas_dict[perm] = theta[0]
        betas_dict[perm]  = theta[1:]
        r2_dict[perm]     = r2

    alpha_s  = pd.Series(alphas_dict, name="alpha")
    betas_df = pd.DataFrame(betas_dict, index=factor_returns.columns).T
    betas_df.index.name = "permno"
    r2_s     = pd.Series(r2_dict, name="r2")

    return alpha_s, betas_df, r2_s


# ===========================================================================
# Main backtest loop
# ===========================================================================

def run_spec_b(
    method: str             = None,
    n_clusters: int         = None,
    tau: float              = None,
    universe_size: int      = None,
    start_date: int         = None,
    use_factor_timing: bool = True,
    rolling_window: int     = None,
    verbose: bool           = True,
) -> BacktestResult:
    """
    Run the full Specification B backtest and return a BacktestResult.

    Parameters
    ----------
    method            : "spherical_kmeans" or "hdbscan"
    n_clusters        : M (SKM only; HDBSCAN chooses M automatically)
    tau               : temperature for soft membership weights
    universe_size     : keep top-N stocks by market cap each month;
                        use None or 0 for full universe
    start_date        : integer date (YYYYMMDD) to begin sampling
    use_factor_timing : True → optimise factor weights a_t (Eq. 18);
                        False → equal-weight all factors
    rolling_window    : months in rolling window for μ̂^(f), Σ̂^(f)
    verbose           : print period-by-period progress
    """
    # Defaults from config
    method        = method        or config.CLUSTER_METHOD
    n_clusters    = n_clusters    or config.N_CLUSTERS
    tau           = tau           or config.TEMPERATURE
    if universe_size is None:
        universe_size = config.SPEC_A_UNIVERSE_SIZE
    if universe_size is not None and universe_size <= 0:
        universe_size = None
    rolling_window = rolling_window or config.SPEC_B_ROLLING_WINDOW
    warmup        = config.WARMUP_MONTHS

    if verbose:
        print("Loading returns panel …")
    ret_panel = load_returns_panel()
    ret_wide  = ret_panel["monthly_ret"].unstack(level="permno")

    pairs = get_date_pairs()
    if start_date:
        pairs = [(cd, rd) for cd, rd in pairs if cd >= start_date]

    if len(pairs) < warmup + 2:
        raise ValueError("Not enough date pairs for the requested warmup period.")

    M = n_clusters

    # ── Rolling buffers ──────────────────────────────────────────────────
    # f_buffer holds past factor returns {f_1,...,f_{k-1}} (before current f_k)
    # a_prev is a_{t-1}: factor timing weights estimated in the PREVIOUS iteration
    f_buffer: deque = deque(maxlen=rolling_window)
    a_prev: pd.Series | None = None
    prev_prototypes: pd.DataFrame | None = None
    next_type_id = 0

    # ── Storage ──────────────────────────────────────────────────────────
    strategy_returns = []
    dates_out        = []
    weights_hist     = []
    turnover_hist    = []
    leverage_hist    = []
    factor_returns_all: dict = {}     # DATE → {type_0: f0, type_1: f1, …}
    avg_Pi_ts: list          = []
    clustering_results = {}           # store all Pi_df for each date

    if verbose:
        print(f"\nSpec B  |  method={method}  M={n_clusters}  τ={tau}  timing={use_factor_timing}")
        print(f"Warmup {warmup} months; rolling window {rolling_window} months\n")

    for idx, (chars_date, ret_date) in enumerate(pairs):

        # ── Step 1: Load and preprocess characteristics at t−1 ───────────
        try:
            chars_df = load_characteristics(chars_date)
        except FileNotFoundError:
            if verbose:
                print(f"  [{chars_date}] chars file missing – skipping")
            continue

        # Universe filter: top-N by market cap
        if universe_size:
            try:
                mktcap = load_market_cap(chars_date)
                top = mktcap.nlargest(universe_size).index
                chars_df = chars_df.loc[chars_df.index.intersection(top)]
            except Exception:
                pass

        if len(chars_df) < 50:
            continue

        U_df, U_arr = preprocess(chars_df, return_array=True)
        permnos_t   = U_df.index.values

        # ── Step 2: Cluster → Π_{t−1} ────────────────────────────────────
        try:
            Pi_df, model = cluster_and_weight(
                U_arr,
                permnos   = permnos_t,
                method    = method,
                n_clusters = n_clusters,
                tau       = tau,
            )
        except Exception as e:
            warnings.warn(f"  [{chars_date}] clustering failed: {e} – skipping")
            continue

        Pi_df, curr_prototypes, next_type_id, _ = align_cluster_labels(
            Pi_df,
            U_arr,
            prev_prototypes=prev_prototypes,
            next_type_id=next_type_id,
        )
        prev_prototypes = curr_prototypes

        M = Pi_df.shape[1]

        avg_Pi_ts.append(Pi_df.mean(axis=0).rename(chars_date))

        # Store full clustering results for later analysis
        clustering_results[chars_date] = Pi_df

        # ── Step 3: Build factor portfolios W_{t−1}  (Equations 14–15) ───
        W_df = build_factor_portfolios(Pi_df)

        # ── Step 4: Compute f_t = W_{t−1}ᵀ rₜ  (Equation 16) ────────────
        try:
            r_t = ret_wide.loc[ret_date].dropna()
        except KeyError:
            continue

        common = W_df.index.intersection(r_t.index)
        if len(common) == 0:
            continue

        W_realized = W_df.loc[common]
        r_aligned = r_t.loc[common].values
        f_t = pd.Series(W_realized.values.T @ r_aligned, index=W_realized.columns)
        factor_returns_all[ret_date] = f_t.to_dict()

        # ── Step 5: Estimate a_{t−1} from past {f₁,…,f_{t−1}} ───────────
        #    f_buffer currently holds only PAST factor returns (before f_t).
        #    This ensures a_{t−1} has NO access to f_t  →  no look-ahead.
        if len(f_buffer) >= warmup:
            F_hist_df = pd.DataFrame(list(f_buffer)).fillna(0.0)
            curr_cols = Pi_df.columns.tolist()
            F_hist = F_hist_df.reindex(columns=curr_cols, fill_value=0.0).values

            if use_factor_timing:
                mu_f    = F_hist.mean(axis=0)
                Sigma_f = factor_covariance(F_hist, method="ledoit_wolf")

                if a_prev is None:
                    a_prev = pd.Series(0.0, index=curr_cols)
                else:
                    a_prev = a_prev.reindex(curr_cols).fillna(0.0)

                try:
                    a_curr_arr = optimize_portfolio_b(
                        mu_f    = mu_f,
                        Sigma_f = Sigma_f,
                        a_prev  = a_prev.values,
                        gamma_f = config.SPEC_B_GAMMA_F,
                        kappa_f = config.SPEC_B_KAPPA_F,
                        max_la  = config.SPEC_B_MAX_LA,
                        a_max   = config.SPEC_B_MAX_SINGLE_F,
                    )
                    a_curr = pd.Series(a_curr_arr, index=curr_cols)
                except Exception as e:
                    warnings.warn(f"[{chars_date}] factor timing failed: {e}. Equal-weight.")
                    a_curr = pd.Series(np.ones(M) / M, index=curr_cols)
            else:
                # Equal-weight long across all factors (no timing optimisation)
                a_curr = pd.Series(np.ones(M) / M, index=Pi_df.columns)

            # ── Step 6: REALISE R_t^(B) = a_{t−1}ᵀ f_t ──────────────────
            if a_prev is not None:
                a_prev_curr = a_prev.reindex(f_t.index).fillna(0.0)
                gross_ret = float(a_prev_curr.values @ f_t.values)

                # Transaction cost proxy: turnover in factor weights
                tc = compute_tc(
                    a_curr.values,
                    a_prev_curr.reindex(a_curr.index).fillna(0.0).values,
                    tc_bps=config.TC_BPS / 10.0,
                )
                net_ret = gross_ret - tc

                strategy_returns.append(net_ret)
                dates_out.append(ret_date)

                to  = float(np.sum(np.abs(a_curr.values - a_prev_curr.reindex(a_curr.index).fillna(0.0).values))) / 2.0
                lev = float(np.sum(np.abs(a_curr.values)))
                weights_hist.append((ret_date, a_curr.values.copy()))
                turnover_hist.append((ret_date, to))
                leverage_hist.append((ret_date, lev))

                if verbose and idx % 12 == 0:
                    print(f"  t={ret_date}  ret={net_ret:+.4f}  "
                          f"a_lev={lev:.2f}  to={to:.4f}")

            a_prev = a_curr.copy()

        else:
            # Still in warmup: track but don't trade
            if a_prev is None:
                a_prev = pd.Series(0.0, index=Pi_df.columns)
            else:
                a_prev = a_prev.reindex(Pi_df.columns).fillna(0.0)

        # ── Step 7: Append f_t to buffer (for next period's estimation) ──
        f_buffer.append(f_t)

    # ────────────────────────────────────────────────────────────────────────
    # Assemble result
    # ────────────────────────────────────────────────────────────────────────
    returns_series = pd.Series(
        strategy_returns, index=dates_out, name="Spec_B_Return"
    )

    result = BacktestResult(
        returns       = returns_series,
        weights_hist  = weights_hist,
        turnover_hist = turnover_hist,
        leverage_hist = leverage_hist,
        metadata      = {
            "label":      f"Spec B ({method})",
            "method":     method,
            "n_clusters": M,
            "tau":        tau,
            "timing":     use_factor_timing,
        },
    )

    # ── Factor return panel ────────────────────────────────────────────────
    factor_ret_df = (
        pd.DataFrame(factor_returns_all).T
        .sort_index()
        .rename_axis("DATE")
        .astype(float)
    )

    # ── Time-series regression  (Equation 17) ─────────────────────────────
    alpha_s = betas_df = r2_s = None
    if len(factor_ret_df) > 24 and len(ret_wide) > 0:
        if verbose:
            print("\nRunning time-series regressions …")
        try:
            alpha_s, betas_df, r2_s = time_series_regression(ret_wide, factor_ret_df)
        except Exception as e:
            warnings.warn(f"Time-series regression failed: {e}")

    # ── Save outputs ───────────────────────────────────────────────────────
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Factor return panel
    fr_path = os.path.join(config.OUTPUT_DIR, "spec_b_factor_returns.csv")
    factor_ret_df.to_csv(fr_path)

    # Strategy returns
    ret_path = os.path.join(config.OUTPUT_DIR, "spec_b_returns.csv")
    returns_series.to_csv(ret_path, header=True)

    # Regression outputs
    if alpha_s is not None:
        alpha_s.to_csv(os.path.join(config.OUTPUT_DIR, "spec_b_alphas.csv"), header=True)
        betas_df.to_csv(os.path.join(config.OUTPUT_DIR, "spec_b_betas.csv"))
        r2_s.to_csv(os.path.join(config.OUTPUT_DIR, "spec_b_r2.csv"), header=True)
        if verbose:
            print(f"\nTime-series regression:")
            print(f"  Mean |α|: {alpha_s.abs().mean():.4f}")
            print(f"  Mean R²:  {r2_s.mean():.3f}")

    # Save all clustering results to compressed pickle
    if clustering_results:
        import pickle
        import gzip
        cluster_path = os.path.join(config.OUTPUT_DIR, "spec_b_clustering_results.pkl.gz")
        with gzip.open(cluster_path, 'wb') as f:
            pickle.dump(clustering_results, f)
        if verbose:
            print(f"  Clustering results → {cluster_path}")

    # Cluster weight chart
    if avg_Pi_ts:
        pi_df = pd.DataFrame([{"DATE": s.name, **s.to_dict()} for s in avg_Pi_ts]).set_index("DATE")
        plot_cluster_weights_over_time(
            pi_df,
            title=f"Avg Firm-Type Membership ({method})",
            save_path=os.path.join(config.OUTPUT_DIR, "spec_b_cluster_weights.png"),
        )

    # Factor return heatmap
    if len(factor_ret_df) > 12:
        plot_factor_returns(
            factor_ret_df,
            title=f"Factor Returns Heatmap — {method}  M={M}",
            save_path=os.path.join(config.OUTPUT_DIR, "spec_b_factor_heatmap.png"),
        )

    # Performance chart
    chart_path = plot_performance(
        result,
        title=f"Spec B — {method}  M={M}  timing={use_factor_timing}",
        save_path=os.path.join(config.OUTPUT_DIR, "spec_b_performance.png"),
    )

    # Metrics CSV
    save_metrics_csv(
        [result],
        save_path=os.path.join(config.OUTPUT_DIR, "spec_b_metrics.csv"),
    )

    print_metrics_table(result, label=f"Spec B ({method}, M={M})")
    print(f"  Factor returns → {fr_path}")
    print(f"  Returns        → {ret_path}")
    print(f"  Chart          → {chart_path}")

    return result


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args():
    p = argparse.ArgumentParser(
        description="Run Specification B: Geometric Factor Portfolio Model"
    )
    p.add_argument("--method", choices=["spherical_kmeans", "hdbscan"],
                   default=config.CLUSTER_METHOD)
    p.add_argument("--n_clusters", type=int, default=config.N_CLUSTERS)
    p.add_argument("--tau",        type=float, default=config.TEMPERATURE)
    p.add_argument(
        "--universe",
        type=int,
        default=config.SPEC_A_UNIVERSE_SIZE,
        help="Max stocks by market cap per month; use 0 for full universe",
    )
    p.add_argument("--start",      type=int, default=None,
                   help="Start chars_date as YYYYMMDD integer")
    p.add_argument("--no_timing",  action="store_true",
                   help="Equal-weight factors instead of MV timing")
    p.add_argument("--rolling_window", type=int, default=config.SPEC_B_ROLLING_WINDOW)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_spec_b(
        method            = args.method,
        n_clusters        = args.n_clusters,
        tau               = args.tau,
        universe_size     = args.universe,
        start_date        = args.start,
        use_factor_timing = not args.no_timing,
        rolling_window    = args.rolling_window,
        verbose           = True,
    )
