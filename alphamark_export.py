"""
alphamark_export.py
===================
Export a monthly, AlphaMark-compatible panel from the geometric factor model.

AlphaMark expects one PKL per date containing:
  - a `ticker` column
  - one or more numeric signal columns
  - one or more numeric target-return columns
  - one or more numeric bet-size columns

This exporter writes combined per-date PKLs so AlphaMark can read signals,
targets, and bet sizes from the same directory.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from collections import deque

import numpy as np
import pandas as pd

import config
from clustering import align_cluster_labels, cluster_and_weight
from data_loader import (
    get_date_pairs,
    load_characteristics,
    load_market_cap,
    load_returns_panel,
)
from normalization import preprocess


def _cs_regression(
    r_t: pd.Series,
    pi_df: pd.DataFrame,
) -> tuple[float, np.ndarray]:
    """
    Cross-sectional regression used to update Spec A lambda estimates.

    This mirrors the regression logic used in the main pricing pipeline but is
    kept local so the AlphaMark export remains self-contained.
    """
    common = pi_df.index.intersection(r_t.index)
    if len(common) < pi_df.shape[1] + 5:
        return np.nan, np.full(pi_df.shape[1], np.nan)

    pi_vals = pi_df.loc[common].values
    r_vals = r_t.loc[common].values
    n_clusters = pi_vals.shape[1]

    if n_clusters > 1:
        X = np.column_stack([np.ones(len(r_vals)), pi_vals[:, :-1]])
    else:
        X = np.ones((len(r_vals), 1))

    try:
        theta, _, _, _ = np.linalg.lstsq(X, r_vals, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, np.full(n_clusters, np.nan)

    alpha = float(theta[0])
    lambdas = np.zeros(n_clusters)
    if n_clusters > 1:
        lambdas[:-1] = theta[1:]
    return alpha, lambdas


def export_alphamark_panel(
    method: str | None = None,
    n_clusters: int | None = None,
    tau: float | None = None,
    universe_size: int | None = None,
    start_date: int | None = None,
    rolling_window: int | None = None,
    output_dir: str | None = None,
    verbose: bool = True,
) -> dict[str, str]:
    """
    Export monthly combined PKLs compatible with AlphaMark.

    Signals exported:
      - `pret_type_*`        aligned soft cluster memberships
      - `pret_speca_muhat`   rolling Spec A expected-return forecast

    Targets exported:
      - `fret_monthly_ret`

    Bet sizes exported:
      - `bs_equal`
      - `bs_mvel1`
    """
    method = method or config.CLUSTER_METHOD
    n_clusters = n_clusters or config.N_CLUSTERS
    tau = tau or config.TEMPERATURE
    if universe_size is None:
        universe_size = config.SPEC_A_UNIVERSE_SIZE
    if universe_size is not None and universe_size <= 0:
        universe_size = None
    if rolling_window is None:
        rolling_window = config.ROLLING_WINDOW
    output_dir = output_dir or config.ALPHAMARK_EXPORT_DIR
    warmup = config.WARMUP_MONTHS
    min_lambda_history = rolling_window if rolling_window is not None else warmup

    combined_dir = os.path.join(output_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)

    if verbose:
        print("Loading returns panel for AlphaMark export …")
    ret_panel = load_returns_panel()
    ret_wide = ret_panel["monthly_ret"].unstack(level="permno")

    pairs = get_date_pairs()
    if start_date:
        pairs = [(cd, rd) for cd, rd in pairs if cd >= start_date]
    if not pairs:
        raise ValueError("No date pairs available for export.")

    lambda_history: deque = deque(maxlen=rolling_window)
    alpha_history: deque = deque(maxlen=rolling_window)
    prev_prototypes: pd.DataFrame | None = None
    next_type_id = 0

    manifest_rows: list[dict] = []
    all_signal_cols: set[str] = set()
    first_ret_date: int | None = None
    last_ret_date: int | None = None

    for idx, (chars_date, ret_date) in enumerate(pairs):
        try:
            chars_df = load_characteristics(chars_date)
        except FileNotFoundError:
            if verbose:
                print(f"[{chars_date}] characteristics missing; skipping export date")
            continue

        if universe_size:
            try:
                mktcap = load_market_cap(chars_date)
                top_permnos = mktcap.nlargest(universe_size).index
                chars_df = chars_df.loc[chars_df.index.intersection(top_permnos)]
            except Exception:
                mktcap = pd.Series(dtype=float)
        else:
            try:
                mktcap = load_market_cap(chars_date)
            except Exception:
                mktcap = pd.Series(dtype=float)

        if len(chars_df) < 50:
            continue

        U_df, U_arr = preprocess(chars_df, return_array=True)
        permnos_t = U_df.index.values

        try:
            pi_df, _ = cluster_and_weight(
                U_arr,
                permnos=permnos_t,
                method=method,
                n_clusters=n_clusters,
                tau=tau,
            )
        except Exception as exc:
            warnings.warn(f"[{chars_date}] clustering failed during AlphaMark export: {exc}")
            continue

        pi_df, curr_prototypes, next_type_id, _ = align_cluster_labels(
            pi_df,
            U_arr,
            prev_prototypes=prev_prototypes,
            next_type_id=next_type_id,
        )
        prev_prototypes = curr_prototypes

        try:
            r_t = ret_wide.loc[ret_date].dropna()
        except KeyError:
            continue

        if len(lambda_history) >= min_lambda_history:
            lambda_hat = pd.concat(list(lambda_history), axis=1).T.mean(axis=0, skipna=True)
            alpha_hat = np.nanmean(list(alpha_history))
            pi_aligned = pi_df.reindex(columns=lambda_hat.index, fill_value=0.0)
            mu_hat = alpha_hat + pi_aligned.values @ lambda_hat.values
            mu_hat_series = pd.Series(mu_hat, index=pi_df.index)
        else:
            mu_hat_series = pd.Series(np.nan, index=pi_df.index)

        export_df = pd.DataFrame(index=pi_df.index)
        export_df["ticker"] = export_df.index.astype(str)
        export_df["date"] = pd.to_datetime(str(ret_date), format="%Y%m%d")

        cluster_signal_df = pi_df.add_prefix("pret_")
        export_df = export_df.join(cluster_signal_df)
        export_df["pret_speca_muhat"] = mu_hat_series.reindex(export_df.index)
        export_df["fret_monthly_ret"] = r_t.reindex(export_df.index)
        export_df["bs_equal"] = 1.0
        export_df["bs_mvel1"] = pd.to_numeric(mktcap.reindex(export_df.index), errors="coerce")

        # AlphaMark needs a usable target; keep rows with realized returns.
        export_df = export_df.loc[export_df["fret_monthly_ret"].notna()].copy()

        if export_df.empty:
            continue

        all_signal_cols.update([c for c in export_df.columns if c.startswith("pret_")])
        out_path = os.path.join(combined_dir, f"alpha_{ret_date}.pkl")
        export_df.reset_index(drop=True).to_pickle(out_path)

        manifest_rows.append(
            {
                "chars_date": chars_date,
                "ret_date": ret_date,
                "n_rows": len(export_df),
                "n_signal_cols": len([c for c in export_df.columns if c.startswith("pret_")]),
                "path": out_path,
            }
        )
        if first_ret_date is None:
            first_ret_date = ret_date
        last_ret_date = ret_date

        common_n = len(pi_df.index.intersection(r_t.index))
        if common_n < config.SPEC_A_MIN_VALID_LAMBDA_OBS:
            continue

        alpha_t, lambda_t = _cs_regression(r_t, pi_df)
        if not np.isnan(alpha_t):
            alpha_history.append(alpha_t)
            lambda_history.append(pd.Series(lambda_t, index=pi_df.columns, name=ret_date))

        if verbose and idx % 24 == 0:
            print(
                f"[{ret_date}] exported {len(export_df)} rows "
                f"with {len([c for c in export_df.columns if c.startswith('pret_')])} signals"
            )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(output_dir, "manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)

    alphamark_cfg = {
        "signals_input": {"dir": combined_dir, "glob": "alpha_*.pkl"},
        "targets_input": {"dir": combined_dir, "glob": "alpha_*.pkl"},
        "betsizes_input": {"dir": combined_dir, "glob": "alpha_*.pkl"},
        "signal_prefix": "pret_",
        "signal_regex": None,
        "signal_list": sorted(all_signal_cols) if all_signal_cols else None,
        "target_prefix": "fret_",
        "target_regex": None,
        "target_list": ["fret_monthly_ret"],
        "bet_prefix": "bs_",
        "bet_regex": None,
        "bet_list": ["bs_equal", "bs_mvel1"],
        "spy_ticker": None,
        "spy_single_name": None,
        "quantiles": [1.0, 0.75, 0.5, 0.25],
        "type_quantile": "cumulative",
        "interval_start": str(first_ret_date) if first_ret_date is not None else None,
        "interval_end": str(last_ret_date) if last_ret_date is not None else None,
        "notes": {
            "frequency": "monthly",
            "ticker_column": "ticker",
            "date_column": "date",
            "generated_by": "alphamark_export.py",
        },
    }
    cfg_path = os.path.join(output_dir, "alphamark_runner_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(alphamark_cfg, f, indent=2)

    readme_path = os.path.join(output_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            "AlphaMark-compatible export\n"
            "===========================\n\n"
            f"Combined PKLs: {combined_dir}\n"
            f"Manifest CSV: {manifest_path}\n"
            f"Config template: {cfg_path}\n\n"
            "Columns exported per file:\n"
            "  - ticker\n"
            "  - date\n"
            "  - pret_type_* (aligned cluster memberships)\n"
            "  - pret_speca_muhat (rolling Spec A expected return)\n"
            "  - fret_monthly_ret\n"
            "  - bs_equal\n"
            "  - bs_mvel1\n"
        )

    if verbose:
        print(f"AlphaMark export written to {output_dir}")
        print(f"  Combined PKLs  → {combined_dir}")
        print(f"  Manifest       → {manifest_path}")
        print(f"  Config template→ {cfg_path}")

    return {
        "output_dir": output_dir,
        "combined_dir": combined_dir,
        "manifest_path": manifest_path,
        "config_path": cfg_path,
        "readme_path": readme_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export AlphaMark-compatible monthly PKLs from the geometric factor model"
    )
    parser.add_argument(
        "--method",
        choices=["spherical_kmeans", "hdbscan"],
        default=config.CLUSTER_METHOD,
    )
    parser.add_argument("--n_clusters", type=int, default=config.N_CLUSTERS)
    parser.add_argument("--tau", type=float, default=config.TEMPERATURE)
    parser.add_argument(
        "--universe",
        type=int,
        default=config.SPEC_A_UNIVERSE_SIZE,
        help="Max stocks by market cap per month; use 0 for full universe",
    )
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--rolling_window", type=int, default=config.ROLLING_WINDOW)
    parser.add_argument("--output_dir", type=str, default=config.ALPHAMARK_EXPORT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export_alphamark_panel(
        method=args.method,
        n_clusters=args.n_clusters,
        tau=args.tau,
        universe_size=args.universe,
        start_date=args.start,
        rolling_window=args.rolling_window,
        output_dir=args.output_dir,
        verbose=True,
    )
