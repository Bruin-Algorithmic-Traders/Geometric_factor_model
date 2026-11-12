import numpy as np
import polars as pl
from scipy import stats

from src.config import CONFIG_LOCKED_JSON, MARKET_EXCESS_CSV, RESULTS_DIR, TARGET_VOL, TUNING_GRID_CSV

NW_LAGS = 6
EULER_GAMMA = 0.5772156649


def annualised_stats(monthly_returns: np.ndarray) -> dict:
    """Annualised mean, vol, Sharpe and t-stat of a monthly return series."""
    r = np.asarray(monthly_returns, dtype=float)
    mean = r.mean()
    std = r.std(ddof=1)
    return {
        "ann_return": mean * 12,
        "ann_vol": std * np.sqrt(12),
        "sharpe": (mean / std) * np.sqrt(12) if std > 0 else 0.0,
        "t_stat": mean / (std / np.sqrt(len(r))) if std > 0 else 0.0,
    }


def newey_west_t(r: np.ndarray, lags: int = NW_LAGS) -> float:
    """t-stat of the mean with Bartlett-kernel Newey-West variance."""
    e = r - r.mean()
    T = len(r)
    var = e @ e / T + 2 * sum((1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / T for l in range(1, lags + 1))
    return r.mean() / np.sqrt(var / T)


def probabilistic_sharpe(r: np.ndarray, benchmark_sr: float = 0.0) -> float:
    """P(true monthly Sharpe > benchmark_sr), Bailey & Lopez de Prado; uses raw (non-excess) kurtosis."""
    sr = r.mean() / r.std(ddof=1)
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)
    return stats.norm.cdf((sr - benchmark_sr) * np.sqrt(len(r) - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2))


def drawdown(r: np.ndarray) -> tuple[float, int]:
    """Max drawdown and longest peak-to-recovery spell (months) of the cumulative sum of returns."""
    cum = np.concatenate([[0.0], np.cumsum(r)])
    under = cum < np.maximum.accumulate(cum)
    longest = run = 0
    for u in under:
        run = run + 1 if u else 0
        longest = max(longest, run)
    return float((cum - np.maximum.accumulate(cum)).min()), longest


def series_metrics(r: np.ndarray, mkt: np.ndarray) -> dict:
    T = len(r)
    s = annualised_stats(r)
    sr_m = r.mean() / r.std(ddof=1)
    se = np.sqrt((1 + sr_m**2 / 2) / T) * np.sqrt(12)
    max_dd, dd_months = drawdown(r)
    return {
        "months": T,
        "ann_return": s["ann_return"],
        "ann_vol": s["ann_vol"],
        "net_sharpe": s["sharpe"],
        "sharpe_se": se,
        "sharpe_ci_low": s["sharpe"] - 1.96 * se,
        "sharpe_ci_high": s["sharpe"] + 1.96 * se,
        "iid_t_stat": s["t_stat"],
        "nw_t_stat": newey_west_t(r),
        "psr_vs_0": probabilistic_sharpe(r),
        "max_drawdown": max_dd,
        "max_dd_months": dd_months,
        "calmar": s["ann_return"] / abs(max_dd),
        "hit_rate": (r > 0).mean(),
        "skew": stats.skew(r),
        "excess_kurtosis": stats.kurtosis(r),
        "worst_month": r.min(),
        "beta_mkt_rf": np.cov(r, mkt)[0, 1] / mkt.var(ddof=1),
        "corr_mkt_rf": np.corrcoef(r, mkt)[0, 1],
        "realised_to_target_vol": s["ann_vol"] / TARGET_VOL,
    }


def deflated_sharpe(r: np.ndarray, n_trials: int, trial_sharpe_sd_annual: float) -> dict:
    """Deflated Sharpe (Bailey & Lopez de Prado 2014): PSR against the expected max Sharpe of n_trials
    independent trials; the trial sd is converted to monthly to match T in months."""
    sd = trial_sharpe_sd_annual / np.sqrt(12)
    z = stats.norm.ppf
    sr0 = sd * ((1 - EULER_GAMMA) * z(1 - 1 / n_trials) + EULER_GAMMA * z(1 - 1 / (n_trials * np.e)))
    return {"n_trials": n_trials, "trial_sharpe_sd_monthly": sd, "sr0_annual": sr0 * np.sqrt(12),
            "deflated_sharpe": probabilistic_sharpe(r, sr0)}


def metrics_table(suffix: str) -> pl.DataFrame:
    returns = pl.read_csv(RESULTS_DIR / f"benchmark_returns_{suffix}.csv")
    seeds = [c for c in returns.columns if c.startswith("random_directions_")]
    mean = pl.mean_horizontal(seeds)
    returns = returns.with_columns(random_directions_mean_10vol=mean * TARGET_VOL / (mean.std() * np.sqrt(12)))
    excess = pl.read_csv(MARKET_EXCESS_CSV).select("DATE", "market_equal_excess", "market_value_excess")
    from src.evaluation.ff_data import load_ff
    ff = load_ff().select("YYYYMM", "Mkt-RF")
    data = (returns.drop("market_equal", "market_value").join(excess, on="DATE", how="inner")
            .with_columns(YYYYMM=pl.col("DATE") // 100).join(ff, on="YYYYMM", how="inner"))
    if data.height != returns.height:
        raise ValueError("months lost when joining market data")

    summary = pl.read_csv(RESULTS_DIR / f"benchmarks_{suffix}.csv").select(
        "benchmark", pl.col("gross_sharpe").alias("gross_sharpe_no_ci"), "stock_turnover", pl.col("ann_cost").alias("cost_drag_ann"))
    mkt = data["Mkt-RF"].to_numpy()
    names = [c for c in data.columns if c not in ("DATE", "YYYYMM", "Mkt-RF")]
    table = pl.DataFrame([{"run": n, **series_metrics(data[n].to_numpy(), mkt)} for n in names])
    table = table.join(summary, left_on="run", right_on="benchmark", how="left")

    seed_sr = table.filter(pl.col("run").is_in(seeds))["net_sharpe"]
    spread = pl.DataFrame({"run": ["random_seeds_median", "random_seeds_min", "random_seeds_max"],
                           "net_sharpe": [seed_sr.median(), seed_sr.min(), seed_sr.max()]})
    return pl.concat([table, spread], how="diagonal_relaxed")


def main():
    import json
    for suffix in ("is", "oos"):
        metrics_table(suffix).write_csv(RESULTS_DIR / f"metrics_{suffix}.csv")
    r = pl.read_csv(RESULTS_DIR / "benchmark_returns_is.csv")["strategy"].to_numpy()
    sd = pl.read_csv(TUNING_GRID_CSV)["net_sharpe"].std()
    n_all = json.loads(CONFIG_LOCKED_JSON.read_text())["grid_configurations_tried"]
    rows = [{"bound": "conservative (all trials independent)", **deflated_sharpe(r, n_all, sd)},
            {"bound": "final risk-budget round only", **deflated_sharpe(r, 196, sd)}]
    pl.DataFrame(rows).write_csv(RESULTS_DIR / "deflated_sharpe_is.csv")
