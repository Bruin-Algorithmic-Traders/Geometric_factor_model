"""
backtesting.py
==============
Backtest engine and performance analytics for the Geometric Factor Model.

Metrics computed
----------------
  Annualised Return     — geometric mean of monthly returns × 12
  Annualised Volatility — std of monthly returns × √12
  Sharpe Ratio          — Ann. Return / Ann. Vol  (rf = 0 for excess returns)
  Sortino Ratio         — Ann. Return / downside deviation × √12
  Max Drawdown          — peak-to-trough decline on cumulative wealth curve
  Calmar Ratio          — Ann. Return / |Max Drawdown|
  Hit Rate              — fraction of months with positive return
  Avg Monthly Turnover  — average ∑|Δw| per month (one-way)
  Avg Gross Leverage    — average ∑|w| per month
  Information Ratio     — (Ann. Return − Benchmark) / tracking error (if bmk)
  Skewness / Kurtosis   — distributional moments of monthly returns
  Monthly Win/Loss      — best and worst monthly return

Plots
-----
  Cumulative wealth curve (log scale optional)
  Monthly return bar chart
  Drawdown chart
  Factor return heatmap (for Spec B)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns

from config import OUTPUT_DIR, TC_BPS


# ===========================================================================
# Core backtest result container
# ===========================================================================

class BacktestResult:
    """
    Holds the output of a single backtest run.

    Attributes
    ----------
    returns        : pd.Series  monthly strategy returns (after TC)
    weights_hist   : list of (date, ndarray) weight snapshots
    turnover_hist  : list of (date, float) one-way turnover each period
    leverage_hist  : list of (date, float) gross leverage each period
    metadata       : dict  arbitrary tags (spec, method, n_clusters, …)
    """

    def __init__(
        self,
        returns: pd.Series,
        weights_hist: list   = None,
        turnover_hist: list  = None,
        leverage_hist: list  = None,
        metadata: dict       = None,
    ):
        self.returns       = returns.dropna()
        self.weights_hist  = weights_hist  or []
        self.turnover_hist = turnover_hist or []
        self.leverage_hist = leverage_hist or []
        self.metadata      = metadata      or {}

    # -------------------------------------------------------------------
    def metrics(self, periods_per_year: int = 12) -> dict:
        """Compute and return all performance metrics as a dictionary."""
        r = self.returns
        if len(r) == 0:
            return {"error": "No returns data"}

        ann = periods_per_year

        # --- Core return / vol ---
        mean_ret  = r.mean()
        std_ret   = r.std(ddof=1)
        ann_ret   = (1 + mean_ret) ** ann - 1
        ann_vol   = std_ret * np.sqrt(ann)

        # --- Sharpe (rf = 0, excess returns assumed) ---
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

        # --- Sortino ---
        downside = r[r < 0].std(ddof=1) * np.sqrt(ann)
        sortino  = ann_ret / downside if downside > 0 else np.nan

        # --- Drawdown ---
        cum  = (1 + r).cumprod()
        roll = cum.cummax()
        dd   = (cum - roll) / roll
        max_dd = dd.min()

        # --- Calmar ---
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

        # --- Hit rate ---
        hit_rate = (r > 0).mean()

        # --- Turnover / leverage ---
        if self.turnover_hist:
            avg_turnover = np.mean([t for _, t in self.turnover_hist])
        else:
            avg_turnover = np.nan

        if self.leverage_hist:
            avg_leverage = np.mean([l for _, l in self.leverage_hist])
        else:
            avg_leverage = np.nan

        # --- Higher moments ---
        skew  = float(r.skew())
        kurt  = float(r.kurtosis())   # excess kurtosis

        # --- Best / worst months ---
        best_month  = r.max()
        worst_month = r.min()

        return {
            "Ann. Return":     ann_ret,
            "Ann. Volatility": ann_vol,
            "Sharpe Ratio":    sharpe,
            "Sortino Ratio":   sortino,
            "Max Drawdown":    max_dd,
            "Calmar Ratio":    calmar,
            "Hit Rate":        hit_rate,
            "Avg Turnover":    avg_turnover,
            "Avg Leverage":    avg_leverage,
            "Skewness":        skew,
            "Excess Kurtosis": kurt,
            "Best Month":      best_month,
            "Worst Month":     worst_month,
            "N Months":        len(r),
        }

    def summary(self) -> pd.Series:
        """Return metrics as a formatted pd.Series."""
        m = self.metrics()
        pct_keys = {
            "Ann. Return", "Ann. Volatility", "Max Drawdown",
            "Hit Rate", "Best Month", "Worst Month", "Avg Turnover", "Avg Leverage"
        }
        out = {}
        for k, v in m.items():
            if isinstance(v, float):
                if k in pct_keys:
                    out[k] = f"{v:.2%}"
                else:
                    out[k] = f"{v:.3f}"
            else:
                out[k] = v
        return pd.Series(out, name=self.metadata.get("label", "Strategy"))

    # -------------------------------------------------------------------
    def compare(self, other: "BacktestResult") -> pd.DataFrame:
        """Side-by-side metric comparison of two BacktestResult objects."""
        s1 = self.metrics()
        s2 = other.metrics()
        keys = list(s1.keys())
        return pd.DataFrame(
            {
                self.metadata.get("label", "Strategy A"): s1,
                other.metadata.get("label", "Strategy B"): s2,
            },
            index=keys,
        )


# ===========================================================================
# Transaction cost calculation
# ===========================================================================

def compute_tc(w_new: np.ndarray, w_old: np.ndarray, tc_bps: float = TC_BPS) -> float:
    """
    Proportional transaction cost based on one-way turnover.

        TC = tc_bps / 10_000  ×  Σ|w_new - w_old|

    Note: the sum is the two-way turnover; dividing by 2 gives one-way.
    We charge on one-way convention (tc_bps per unit of one-way turnover).
    """
    return (tc_bps / 10_000) * np.sum(np.abs(w_new - w_old)) / 2.0


# ===========================================================================
# Plotting utilities
# ===========================================================================

def _set_style():
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
    })


def plot_performance(
    result: "BacktestResult",
    benchmark: pd.Series   = None,
    title: str             = "Strategy Performance",
    save_path: str         = None,
    log_scale: bool        = False,
) -> str:
    """
    Four-panel performance dashboard:
      1. Cumulative wealth (with optional benchmark)
      2. Drawdown
      3. Monthly returns bar chart
      4. Rolling 12-month Sharpe

    Parameters
    ----------
    result    : BacktestResult
    benchmark : pd.Series of benchmark monthly returns (optional)
    title     : figure title
    save_path : file path to save PNG (auto-generated if None)
    log_scale : use log-scale for cumulative wealth

    Returns
    -------
    save_path : str path where the figure was saved
    """
    _set_style()
    r = result.returns

    dates = r.index
    if isinstance(dates[0], (int, np.integer)):
        # Convert integer dates to datetime
        dates = pd.to_datetime(dates.astype(str), format="%Y%m%d")

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    ax_cum  = fig.add_subplot(gs[0, :])
    ax_dd   = fig.add_subplot(gs[1, :])
    ax_bar  = fig.add_subplot(gs[2, 0])
    ax_roll = fig.add_subplot(gs[2, 1])

    # ---- 1. Cumulative wealth ----
    cum = (1 + r).cumprod()
    ax_cum.plot(dates, cum.values, color="#2166ac", linewidth=1.8, label="Strategy")
    if benchmark is not None:
        bm_dates = benchmark.index
        if isinstance(bm_dates[0], (int, np.integer)):
            bm_dates = pd.to_datetime(bm_dates.astype(str), format="%Y%m%d")
        bm_cum = (1 + benchmark).cumprod()
        ax_cum.plot(bm_dates, bm_cum.values, color="#d73027", linewidth=1.2,
                    linestyle="--", label="Benchmark")
    ax_cum.set_ylabel("Wealth (starting $1)" if not log_scale else "Log Wealth")
    ax_cum.set_title("Cumulative Return")
    if log_scale:
        ax_cum.set_yscale("log")
    ax_cum.legend()
    ax_cum.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_cum.xaxis.set_major_locator(mdates.YearLocator(3))

    # ---- 2. Drawdown ----
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    ax_dd.fill_between(dates, dd.values, 0, color="#fc8d59", alpha=0.7)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_title("Drawdown")
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_dd.xaxis.set_major_locator(mdates.YearLocator(3))

    # ---- 3. Monthly returns bar ----
    colors = ["#2166ac" if x >= 0 else "#d73027" for x in r.values]
    ax_bar.bar(dates, r.values * 100, color=colors, width=20)
    ax_bar.axhline(0, color="black", linewidth=0.7)
    ax_bar.set_ylabel("Return (%)")
    ax_bar.set_title("Monthly Returns")
    ax_bar.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_bar.xaxis.set_major_locator(mdates.YearLocator(5))

    # ---- 4. Rolling 12-month Sharpe ----
    roll_sharpe = (r.rolling(12).mean() / r.rolling(12).std(ddof=1)) * np.sqrt(12)
    ax_roll.plot(dates, roll_sharpe.values, color="#4dac26", linewidth=1.5)
    ax_roll.axhline(0, color="black", linewidth=0.7)
    ax_roll.axhline(1, color="#b8e186", linewidth=0.9, linestyle="--")
    ax_roll.set_ylabel("Sharpe (12m rolling)")
    ax_roll.set_title("Rolling 12-Month Sharpe")
    ax_roll.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_roll.xaxis.set_major_locator(mdates.YearLocator(5))

    # Save
    if save_path is None:
        label = result.metadata.get("label", "strategy").replace(" ", "_").lower()
        save_path = os.path.join(OUTPUT_DIR, f"{label}_performance.png")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_factor_returns(
    factor_returns: pd.DataFrame,
    title: str   = "Factor Return Heatmap (Monthly %)",
    save_path: str = None,
) -> str:
    """
    Heatmap of monthly factor returns.  Each row = year, each column = month.
    One subplot per factor type.

    Parameters
    ----------
    factor_returns : DataFrame (T × M) with integer DATE index or datetime
    title          : figure title
    save_path      : output path

    Returns
    -------
    save_path : str
    """
    _set_style()
    M = factor_returns.shape[1]
    fig, axes = plt.subplots(M, 1, figsize=(14, 2.5 * M))
    if M == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight="bold")

    idx = factor_returns.index
    if isinstance(idx[0], (int, np.integer)):
        idx = pd.to_datetime(idx.astype(str), format="%Y%m%d")

    fr = factor_returns.copy()
    fr.index = idx

    for ax, col in zip(axes, factor_returns.columns):
        s = fr[col] * 100   # to percent
        pivot = s.groupby([s.index.year, s.index.month]).mean().unstack()
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="RdYlGn",
            center=0,
            annot=False,
            fmt=".1f",
            linewidths=0.3,
            cbar_kws={"shrink": 0.6},
        )
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Year")

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "factor_returns_heatmap.png")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_cluster_weights_over_time(
    weights_df: pd.DataFrame,
    title: str   = "Average Cluster Membership Over Time",
    save_path: str = None,
) -> str:
    """
    Stacked area chart showing how the average firm-type membership weights
    evolve over the backtest.

    Parameters
    ----------
    weights_df : DataFrame (T × M) average Π weight each month per firm type
    """
    _set_style()
    idx = weights_df.index
    if isinstance(idx[0], (int, np.integer)):
        idx = pd.to_datetime(idx.astype(str), format="%Y%m%d")

    fig, ax = plt.subplots(figsize=(14, 5))
    weights_df.index = idx
    weights_df.plot.area(ax=ax, colormap="tab10", alpha=0.75)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Avg Membership Weight")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(3))

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "cluster_weights_over_time.png")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ===========================================================================
# Metrics table printer
# ===========================================================================

def print_metrics_table(result: "BacktestResult", label: str = None):
    """Pretty-print a performance metrics table to stdout."""
    m = result.metrics()
    label = label or result.metadata.get("label", "Strategy")
    print(f"\n{'='*52}")
    print(f"  Performance Summary: {label}")
    print(f"{'='*52}")
    for k, v in m.items():
        if isinstance(v, float):
            if k in {"Ann. Return", "Ann. Volatility", "Max Drawdown",
                     "Hit Rate", "Best Month", "Worst Month",
                     "Avg Turnover", "Avg Leverage"}:
                print(f"  {k:<25}  {v:>9.2%}")
            else:
                print(f"  {k:<25}  {v:>9.3f}")
        else:
            print(f"  {k:<25}  {v!s:>9}")
    print(f"{'='*52}\n")


def save_metrics_csv(results: list["BacktestResult"], save_path: str = None) -> str:
    """
    Save a side-by-side metrics CSV for multiple BacktestResult objects.

    Parameters
    ----------
    results   : list of BacktestResult objects
    save_path : output path (auto-generated if None)
    """
    rows = {}
    for r in results:
        label = r.metadata.get("label", f"Strategy_{id(r)}")
        rows[label] = r.metrics()

    df = pd.DataFrame(rows)

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "metrics_comparison.csv")

    df.to_csv(save_path)
    return save_path
