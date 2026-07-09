"""
spec_a_fama_macbeth.py
======================
Specification A  —  Firm types as priced characteristics (Fama–MacBeth)
Section 2.4.1 of "A Geometric Approach to Firm-Type Factor Models"

Pipeline (each month t, after WARMUP_MONTHS of history):
    1. Load & preprocess characteristics at t-1  →  unit sphere embeddings U
    2. Cluster  →  soft membership matrix Π_{t-1}  (N × M)
    3. Cross-sectional OLS:
           r_{i,t} = α_t + Σ_m π_{i,m,t-1} λ_{m,t} + ε_{i,t}
       Store λ_{m,t} for the current month.
    4. Form expected-return forecasts using rolling average λ̂:
           μ̂_{i,t+1|t} = α̂ + Σ_m π_{i,m,t} λ̂_m
    5. Estimate covariance Σ̂_{t} from recent return history.
    6. Solve constrained MV portfolio (Equation 12).
    7. Realise return and record.

Usage
-----
    python spec_a_fama_macbeth.py [--method spherical_kmeans|hdbscan]
                                   [--n_clusters 10]
                                   [--tau 10.0]
                                   [--universe 1500]
                                   [--start 19950101]

The script prints a performance summary and saves:
    results/spec_a_returns.csv
    results/spec_a_performance.png
    results/spec_a_lambda_ts.csv    ← time-series of λ̂_{m,t} per firm type
    results/spec_a_clustering_results.pkl.gz  ← all soft membership matrices Π_{t-1}
    results/spec_a_metrics.csv
"""
from __future__ import annotations

import argparse
import sys
import os
import warnings
import numpy as np
import pandas as pd
from collections import deque

# --- Project modules ---
import config
from data_loader   import (get_date_pairs, load_characteristics,
                            load_market_cap, load_returns_panel)
from normalization import preprocess
from clustering    import cluster_and_weight, align_cluster_labels
from portfolio_optimizer import (optimize_portfolio_a, estimate_covariance,
                                  rank_longshort)
from backtesting   import (BacktestResult, compute_tc, plot_performance,
                            print_metrics_table, save_metrics_csv,
                            plot_cluster_weights_over_time)


# ===========================================================================
# Cross-sectional OLS (single period)
# ===========================================================================

def _cs_regression(
    r_t: pd.Series,
    Pi: pd.DataFrame,
    weights: pd.Series = None,
) -> tuple[float, np.ndarray]:
    """
    Estimate α_t and λ_{m,t} by WLS cross-sectional regression:

        r_{i,t} = α_t + Σ_m π_{i,m,t-1} λ_{m,t} + ε_{i,t}

    Parameters
    ----------
    r_t     : Series (N,) excess returns indexed by permno
    Pi      : DataFrame (N × M) membership weights, index=permno
    weights : optional Series (N,) cross-sectional weights (value-weight)

    Returns
    -------
    alpha     : float intercept α_t
    lambdas   : ndarray (M,) prices of risk λ_{m,t}
    """
    # Align
    common = Pi.index.intersection(r_t.index)
    if len(common) < Pi.shape[1] + 5:
        return np.nan, np.full(Pi.shape[1], np.nan)

    Pi_ = Pi.loc[common].values            # (N × M)
    r_  = r_t.loc[common].values           # (N,)
    M   = Pi_.shape[1]

    if weights is not None:
        w_ = weights.reindex(common).fillna(1.0).values
        w_ = w_ / w_.sum()
    else:
        w_ = np.ones(len(common)) / len(common)

    # Π rows sum to 1, so [1 | Π] is exactly collinear. Drop one membership
    # column and treat it as the baseline type absorbed by the intercept.
    if M > 1:
        X = np.column_stack([np.ones(len(r_)), Pi_[:, :-1]])
    else:
        X = np.ones((len(r_), 1))

    # WLS: θ̂ = (X^T W X)^{-1} X^T W r
    W = np.diag(w_)
    XtW  = X.T @ W
    XtWX = XtW @ X
    XtWr = XtW @ r_

    try:
        theta = np.linalg.solve(XtWX, XtWr)
    except np.linalg.LinAlgError:
        try:
            theta = np.linalg.lstsq(XtWX, XtWr, rcond=None)[0]
        except Exception:
            return np.nan, np.full(M, np.nan)

    alpha = theta[0]
    lambdas = np.zeros(M)
    if M > 1:
        lambdas[:-1] = theta[1:]
    return alpha, lambdas


# ===========================================================================
# Main backtest loop
# ===========================================================================

def run_spec_a(
    method: str         = None,
    n_clusters: int     = None,
    tau: float          = None,
    universe_size: int  = None,
    start_date: int     = None,
    rolling_window: int = None,
    cov_method: str     = None,
    use_optimizer: bool = True,
    verbose: bool       = True,
) -> BacktestResult:
    """
    Run the full Specification A backtest and return a BacktestResult.

    Parameters
    ----------
    method       : "spherical_kmeans" or "hdbscan"
    n_clusters   : M (ignored for HDBSCAN)
    tau          : soft-weight temperature
    universe_size: keep top-N stocks by market cap each month;
                   use None or 0 for full universe
    start_date   : integer date to start trading (WARMUP_MONTHS of burn-in before)
    rolling_window : number of past months for rolling λ̂ estimation (None=expanding)
    cov_method   : "ledoit_wolf" | "diagonal" | "sample"
    use_optimizer: if False, use rank_longshort fallback (much faster)
    verbose      : print progress
    """
    # --- defaults from config ---
    method        = method        or config.CLUSTER_METHOD
    n_clusters    = n_clusters    or config.N_CLUSTERS
    tau           = tau           or config.TEMPERATURE
    if universe_size is None:
        universe_size = config.SPEC_A_UNIVERSE_SIZE
    if universe_size is not None and universe_size <= 0:
        universe_size = None
    if rolling_window is None:
        rolling_window = config.ROLLING_WINDOW
    cov_method    = cov_method    or config.SPEC_A_COV_ESTIMATOR

    warmup        = config.WARMUP_MONTHS
    min_lambda_history = rolling_window if rolling_window is not None else warmup

    # --- load returns panel once ---
    if verbose:
        print("Loading returns panel …")
    ret_panel = load_returns_panel()
    # Pivot to (DATE × permno)
    ret_wide = ret_panel["monthly_ret"].unstack(level="permno")

    # --- date pairs ---
    pairs = get_date_pairs()
    if start_date:
        pairs = [(cd, rd) for cd, rd in pairs if cd >= start_date]

    if len(pairs) < min_lambda_history + 2:
        raise ValueError("Not enough date pairs for the requested lambda window.")

    M = n_clusters  # may be updated by HDBSCAN

    # Storage
    lambda_history: deque = deque(maxlen=rolling_window)    # valid λ̂ buffer
    alpha_history:  deque = deque(maxlen=rolling_window)

    strategy_returns  = []
    dates_out         = []
    weights_hist      = []
    turnover_hist     = []
    leverage_hist     = []
    lambda_ts         = []          # full λ time series for analysis
    avg_Pi_ts         = []          # average Π weights over time
    clustering_results = {}         # store all Pi_df for each date

    w_prev: np.ndarray | None = None
    prev_prototypes: pd.DataFrame | None = None
    next_type_id = 0

    if verbose:
        print(f"\nSpec A  |  method={method}  M={n_clusters}  τ={tau}")
        print(
            f"Trading after {min_lambda_history} valid lambda months "
            f"({len(pairs)} candidate periods)\n"
        )

    for idx, (chars_date, ret_date) in enumerate(pairs):

        # ----------------------------------------------------------------
        # Step 1: Load & preprocess characteristics at t-1
        # ----------------------------------------------------------------
        try:
            chars_df = load_characteristics(chars_date)
        except FileNotFoundError:
            if verbose:
                print(f"  [{chars_date}] chars file missing – skipping")
            continue

        # Universe filter: top N by market cap
        if universe_size:
            try:
                mktcap = load_market_cap(chars_date)
                top_permnos = mktcap.nlargest(universe_size).index
                chars_df = chars_df.loc[chars_df.index.intersection(top_permnos)]
            except Exception:
                pass   # if market cap unavailable, use full universe

        if len(chars_df) < 50:
            if verbose:
                print(f"  [{chars_date}] too few firms after filter – skipping")
            continue

        U_df, U_arr = preprocess(chars_df, return_array=True)
        permnos_t = U_df.index.values

        # ----------------------------------------------------------------
        # Step 2: Cluster  →  Π_{t-1}
        # ----------------------------------------------------------------
        try:
            Pi_df, model = cluster_and_weight(
                U_arr,
                permnos=permnos_t,
                method=method,
                n_clusters=n_clusters,
                tau=tau,
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

        if verbose and method == "hdbscan":
            n_found = getattr(model, "n_clusters_", Pi_df.shape[1])
            n_noise = getattr(model, "n_noise_", 0)
            noise_frac = getattr(model, "noise_fraction_", 0.0)
            cluster_sizes = getattr(model, "cluster_sizes_", [])
            largest = max(cluster_sizes) if cluster_sizes else 0
            median_size = int(np.median(cluster_sizes)) if cluster_sizes else 0
            print(
                f"  [{chars_date}] HDBSCAN clusters={n_found} "
                f"noise={n_noise} ({noise_frac:.1%}) "
                f"largest={largest} median={median_size}"
            )

        M = Pi_df.shape[1]

        # Track average Π (for the cluster-weights-over-time plot)
        avg_Pi_ts.append(Pi_df.mean(axis=0).rename(chars_date))

        # Store full clustering results for later analysis
        clustering_results[chars_date] = Pi_df

        # ----------------------------------------------------------------
        # Step 3: Form portfolio for ret_date using only PAST λ estimates.
        # At month t we can use characteristics from t-1 and λ-history
        # estimated through t-1, then realise on returns at t.
        # ----------------------------------------------------------------
        if len(lambda_history) >= min_lambda_history:
            lambda_hat = pd.concat(list(lambda_history), axis=1).T.mean(axis=0, skipna=True)
            alpha_hat  = np.nanmean(list(alpha_history))

            Pi_aligned = Pi_df.reindex(columns=lambda_hat.index, fill_value=0.0)
            mu_hat = alpha_hat + Pi_aligned.values @ lambda_hat.values
            mu_hat_series = pd.Series(mu_hat, index=Pi_df.index)

            # ------------------------------------------------------------
            # Step 4: Portfolio optimisation
            # ------------------------------------------------------------
        # Align returns history for covariance estimation
            N = len(permnos_t)

            if use_optimizer:
                # Use only returns known before the current holding period.
                if rolling_window:
                    hist_dates = [p[1] for p in pairs[max(0, idx - rolling_window): idx]]
                else:
                    hist_dates = [p[1] for p in pairs[:idx]]

                available_hist = [d for d in hist_dates if d in ret_wide.index]
                ret_hist = ret_wide.loc[available_hist, :]

                Sigma, active_permnos = estimate_covariance(
                    ret_hist,
                    permnos=permnos_t.tolist(),
                    method=cov_method,
                )

                if w_prev is None or len(w_prev) != N:
                    w_prev = np.zeros(N)

                mu_vec = mu_hat_series.reindex(permnos_t).fillna(0.0).values
                active_mask = np.isin(permnos_t, np.array(active_permnos))
                active_idx = np.flatnonzero(active_mask)

                try:
                    if len(active_idx) < 2:
                        w_new = np.zeros(N)
                    else:
                        w_active = optimize_portfolio_a(
                            mu    = mu_vec[active_idx],
                            Sigma = Sigma,
                            w_prev = w_prev[active_idx],
                            gamma  = config.SPEC_A_GAMMA,
                            kappa  = config.SPEC_A_KAPPA,
                            max_lev = config.SPEC_A_MAX_LEVERAGE,
                            w_max  = config.SPEC_A_MAX_SINGLE,
                            dollar_neutral = config.SPEC_A_DOLLAR_NEUTRAL,
                        )
                        w_new = np.zeros(N)
                        w_new[active_idx] = w_active
                except Exception as e:
                    warnings.warn(f"[{chars_date}] optimizer failed: {e}. Using rank LS.")
                    w_new = rank_longshort(mu_vec)
            else:
                mu_vec = mu_hat_series.reindex(permnos_t).fillna(0.0).values
                w_new  = rank_longshort(mu_vec)
                if w_prev is None or len(w_prev) != N:
                    w_prev = np.zeros(N)

            # ------------------------------------------------------------
            # Step 5: Realise return at the current ret_date
            # ------------------------------------------------------------
            try:
                r_t = ret_wide.loc[ret_date].dropna()
            except KeyError:
                w_prev = w_new
                continue

            common = np.intersect1d(permnos_t, r_t.index)
            if len(common) == 0:
                w_prev = w_new
                continue

            w_aligned = pd.Series(w_new, index=permnos_t).reindex(common).fillna(0.0).values
            r_aligned = r_t.reindex(common).values

            gross_ret = float(w_aligned @ r_aligned)
            tc        = compute_tc(w_new, w_prev, tc_bps=config.TC_BPS)
            net_ret   = gross_ret - tc

            strategy_returns.append(net_ret)
            dates_out.append(ret_date)

            turnover  = float(np.sum(np.abs(w_new - w_prev))) / 2.0
            leverage  = float(np.sum(np.abs(w_new)))
            weights_hist.append((ret_date, w_new))
            turnover_hist.append((ret_date, turnover))
            leverage_hist.append((ret_date, leverage))

            w_prev = w_new

            if verbose and idx % 12 == 0:
                print(f"  t={ret_date}  ret={net_ret:+.4f}  "
                      f"lev={leverage:.2f}  to={turnover:.4f}")

        # ----------------------------------------------------------------
        # Step 6: After the return is observed, estimate λ_{m,t}.
        # ----------------------------------------------------------------
        try:
            r_t = ret_wide.loc[ret_date].dropna()
        except KeyError:
            if verbose:
                print(f"  [{ret_date}] returns missing – skipping")
            continue

        common_n = len(Pi_df.index.intersection(r_t.index))
        if common_n < config.SPEC_A_MIN_VALID_LAMBDA_OBS:
            if verbose:
                print(
                    f"  [{ret_date}] only {common_n} firms for lambda "
                    f"(<{config.SPEC_A_MIN_VALID_LAMBDA_OBS}) – skipping"
                )
            continue

        if config.SPEC_A_CS_WEIGHT == "value":
            try:
                mktcap_w = load_market_cap(chars_date)
                cs_weights = mktcap_w
            except Exception:
                cs_weights = None
        else:
            cs_weights = None

        alpha_t, lambda_t = _cs_regression(r_t, Pi_df, weights=cs_weights)

        if np.isnan(alpha_t):
            continue

        alpha_history.append(alpha_t)
        lambda_series = pd.Series(lambda_t, index=Pi_df.columns, name=ret_date)
        lambda_history.append(lambda_series)
        lambda_ts.append(lambda_series)

    # ----------------------------------------------------------------
    # Assemble results
    # ----------------------------------------------------------------
    returns_series = pd.Series(strategy_returns, index=dates_out, name="Spec_A_Return")

    result = BacktestResult(
        returns       = returns_series,
        weights_hist  = weights_hist,
        turnover_hist = turnover_hist,
        leverage_hist = leverage_hist,
        metadata      = {
            "label":     f"Spec A ({method})",
            "method":    method,
            "n_clusters": M,
            "tau":        tau,
        },
    )

    # ----------------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------------
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Returns CSV
    ret_path = os.path.join(config.OUTPUT_DIR, "spec_a_returns.csv")
    returns_series.to_csv(ret_path, header=True)

    # Lambda time-series CSV
    if lambda_ts:
        lts = pd.DataFrame([{"DATE": s.name, **s.to_dict()} for s in lambda_ts]).set_index("DATE")
        lts.to_csv(os.path.join(config.OUTPUT_DIR, "spec_a_lambda_ts.csv"))

    # Save all clustering results to compressed pickle
    if clustering_results:
        import pickle
        import gzip
        cluster_path = os.path.join(config.OUTPUT_DIR, "spec_a_clustering_results.pkl.gz")
        with gzip.open(cluster_path, 'wb') as f:
            pickle.dump(clustering_results, f)
        print(f"  Clustering results → {cluster_path}")

    # Cluster weights over time
    if avg_Pi_ts:
        pi_df = pd.DataFrame([{"DATE": s.name, **s.to_dict()} for s in avg_Pi_ts]).set_index("DATE")
        plot_cluster_weights_over_time(
            pi_df,
            title=f"Avg Firm-Type Membership Over Time ({method})",
            save_path=os.path.join(config.OUTPUT_DIR, "spec_a_cluster_weights.png"),
        )

    # Performance chart
    chart_path = plot_performance(
        result,
        title=f"Spec A Performance — {method}  M={M}",
        save_path=os.path.join(config.OUTPUT_DIR, "spec_a_performance.png"),
    )

    # Metrics CSV
    save_metrics_csv(
        [result],
        save_path=os.path.join(config.OUTPUT_DIR, "spec_a_metrics.csv"),
    )

    print_metrics_table(result, label=f"Spec A ({method}, M={M})")
    print(f"  Returns  → {ret_path}")
    print(f"  Chart    → {chart_path}")

    return result


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run Specification A: Fama-MacBeth Geometric Factor Model"
    )
    parser.add_argument(
        "--method", choices=["spherical_kmeans", "hdbscan"],
        default=config.CLUSTER_METHOD,
        help="Clustering method (default: %(default)s)",
    )
    parser.add_argument(
        "--n_clusters", type=int, default=config.N_CLUSTERS,
        help="Number of firm types M (ignored for HDBSCAN, default: %(default)s)",
    )
    parser.add_argument(
        "--tau", type=float, default=config.TEMPERATURE,
        help="Soft-weight temperature τ (default: %(default)s)",
    )
    parser.add_argument(
        "--universe", type=int, default=config.SPEC_A_UNIVERSE_SIZE,
        help="Max stocks by market cap per month; use 0 for full universe (default: %(default)s)",
    )
    parser.add_argument(
        "--start", type=int, default=None,
        help="Start chars_date as YYYYMMDD integer (default: first available)",
    )
    parser.add_argument(
        "--rolling_window", type=int, default=config.ROLLING_WINDOW,
        help="Rolling window for λ̂ estimation (default: %(default)s)",
    )
    parser.add_argument(
        "--cov_method", choices=["ledoit_wolf", "diagonal", "sample"],
        default=config.SPEC_A_COV_ESTIMATOR,
        help="Covariance estimator (default: %(default)s)",
    )
    parser.add_argument(
        "--no_optimizer", action="store_true",
        help="Use simple rank long-short instead of CVXPY optimizer (faster)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_spec_a(
        method        = args.method,
        n_clusters    = args.n_clusters,
        tau           = args.tau,
        universe_size = args.universe,
        start_date    = args.start,
        rolling_window = args.rolling_window,
        cov_method    = args.cov_method,
        use_optimizer = not args.no_optimizer,
        verbose       = True,
    )
