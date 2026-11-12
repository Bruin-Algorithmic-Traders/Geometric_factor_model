import json
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment

from src.config import (
    CHARS_CLEAN_DELIST_100_PARQUET, CHARS_CLEAN_DELIST_30_PARQUET, CHARS_CLEAN_PARQUET,
    CHARS_CLEAN_RETURNS_FREE_PARQUET, CHARS_PARQUET, CONFIG_LOCKED_JSON, IS_END_DATE,
    IS_START_DATE, MAX_FACTOR_WEIGHT, N_RANDOM_DIRECTIONS, OOS_END_DATE, OOS_START_DATE, POOL_MONTHS, PROJECT_ROOT, RESULTS_DIR,
    TARGET_COL, TUNE_WORKERS,
)
from src.data.clean import exit_months
from src.evaluation.metrics import annualised_stats
from src.evaluation.plots import benchmark_curves
from src.factors.build_panel import build_panel
from src.models.spherical_kmeans import SphericalKMeans
from src.portfolio.optimiser import SpecBBacktester


class RuleBacktester(SpecBBacktester):
    """Same μ̂, Σ̂, costs and limits as the optimiser, but factor weights from a simple rule instead of eq. 18,
    scaled up until the vol budget, the stock gross cap or the per-factor cap binds."""

    def __init__(self, rule: str, **kwargs):
        super().__init__(**kwargs)
        self.rule = rule

    def _solve(self, mu_f, sigma_f, a_prev, W):
        vol = np.sqrt(np.diag(sigma_f))
        d = np.sign(mu_f) / vol if self.rule == "inverse_vol" else mu_f / vol ** 2
        scale = min(np.sqrt(self.max_monthly_var / (d @ sigma_f @ d)),
                    self.max_stock_gross / np.abs(W @ d).sum(),
                    self.a_max / np.abs(d).max())
        return scale * d


def fit_monthly_pca(months: list[np.ndarray], n_clusters: int, pool_months: int = POOL_MONTHS):
    """Drop-in for fit_monthly: directions are ± the top n_clusters / 2 principal axes of the same pooled,
    unit-length firm vectors, Hungarian-matched to last month's in the same way."""
    prev = None
    for t in range(len(months)):
        pooled = np.vstack(months[max(0, t - pool_months + 1): t + 1])
        pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        axes = np.linalg.svd(pooled, full_matrices=False)[2][: n_clusters // 2]
        model = SphericalKMeans(n_clusters)
        model.centroids = np.vstack([axes, -axes])
        matched = None
        if prev is not None:
            _, order = linear_sum_assignment(-(prev @ model.centroids.T))
            model.centroids = model.centroids[order]
            matched = np.einsum("kp,kp->k", model.centroids, prev)
        prev = model.centroids
        yield model, matched


def fit_monthly_random(months: list[np.ndarray], n_clusters: int, seed: int = 0):
    """Placebo drop-in for fit_monthly: n_clusters random unit directions, drawn once and kept for every month."""
    directions = np.random.default_rng(seed).normal(size=(n_clusters, months[0].shape[1]))
    model = SphericalKMeans(n_clusters)
    model.centroids = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    for _ in months:
        yield model, None


def market_returns(end_date: int = IS_END_DATE) -> pl.DataFrame:
    """Equal- and value-weighted (lagged market cap `mvel1`) return of the cleaned universe each month."""
    universe = pl.read_parquet(CHARS_CLEAN_PARQUET, columns=["DATE", "permno", TARGET_COL]).filter(pl.col("DATE") <= end_date)
    size = pl.read_parquet(CHARS_PARQUET, columns=["DATE", "permno", "mvel1"])
    return (
        universe.join(size, on=["DATE", "permno"], how="left")
        .group_by("DATE").agg(
            pl.col(TARGET_COL).mean().alias("market_equal"),
            ((pl.col(TARGET_COL) * pl.col("mvel1")).sum() / pl.col("mvel1").sum()).alias("market_value"),
        ).sort("DATE")
    )


def benchmark_specs(n_random: int = N_RANDOM_DIRECTIONS) -> dict:
    """name -> how it differs from the locked strategy: `panel` (build_panel arguments), `rule` (static mix
    instead of the optimiser), `cap` (per-factor cap) and `exits` (report the strategy's weight in exiting firms)."""
    return {
        "strategy": {},
        "inverse_vol_mix": {"rule": "inverse_vol"},
        "diagonal_mv_mix": {"rule": "diagonal_mv"},
        "linear_tau_0.01": {"panel": {"tau": 0.01}},
        # f(+v) ≈ −f(−v) for each PCA axis, so a pair at ±0.6 is one bet of 1.2; halving the cap keeps each axis ≤ 0.6
        "pca_directions": {"panel": {"fit": fit_monthly_pca}, "cap": MAX_FACTOR_WEIGHT / 2},
        "returns_free_features": {"panel": {"clean_path": CHARS_CLEAN_RETURNS_FREE_PARQUET}},
        # firms that vanish are kept for their exit month at −30% (typical bad-performance delisting) or −100%
        "delisting_minus30": {"panel": {"clean_path": CHARS_CLEAN_DELIST_30_PARQUET}, "exits": True},
        "delisting_minus100": {"panel": {"clean_path": CHARS_CLEAN_DELIST_100_PARQUET}, "exits": True},
        **{f"random_directions_{s}": {"panel": {"fit": partial(fit_monthly_random, seed=s)}} for s in range(n_random)},
    }


def run_spec(name: str, spec: dict, end_date: int) -> pl.DataFrame:
    """Build the panel up to end_date and run one benchmark over every month; returns its monthly returns_df."""
    fr, _, _, holdings, rebuilt = build_panel(end_date=end_date, **spec.get("panel", {}))
    kwargs = {"max_factor_weight": spec["cap"]} if "cap" in spec else {}
    backtester = RuleBacktester(spec["rule"], **kwargs) if "rule" in spec else SpecBBacktester(**kwargs)
    res = backtester.run_backtest(fr, holdings, rebuilt)
    returns_df = res["returns_df"]
    if spec.get("exits"):
        returns_df = returns_df.with_columns(exit_weights(fr, holdings, res["weights_df"]))
    print(f"{name} done", flush=True)
    return returns_df


def exit_weights(fr: pl.DataFrame, holdings: list, weights_df: pl.DataFrame) -> list[pl.Series]:
    """Per month, the strategy's total long and total short stock weight (share of capital) in firms in their exit month."""
    exits = exit_months().collect().group_by("month").agg("permno")
    exits = {m: set(p) for m, p in exits.iter_rows()}
    A = weights_df.drop("DATE").to_numpy()
    start = fr.height - len(A)
    long, short = [], []
    for i, a in enumerate(A):
        permnos, W, _ = holdings[start + i]
        w = W @ a
        mask = np.isin(permnos, list(exits.get(fr["DATE"][start + i] // 100, ())))
        long.append(float(w[mask & (w > 0)].sum()))
        short.append(float(-w[mask & (w < 0)].sum()))
    return [pl.Series("exit_long_weight", long), pl.Series("exit_short_weight", short)]


def run_benchmarks(start_date: int, end_date: int, workers: int = TUNE_WORKERS):
    """Run every benchmark continuously from the first month to end_date (so OOS positions carry over from
    in-sample), then score only the months in [start_date, end_date]. Returns the wide table of net monthly
    returns, a summary table and a cost table."""
    specs = benchmark_specs()
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "POLARS_MAX_THREADS"]:
        os.environ[var] = "1"
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = pool.map(run_spec, specs.keys(), specs.values(), [end_date] * len(specs))
        runs = {name: r.filter(pl.col("DATE") >= start_date) for name, r in zip(specs, results)}

    returns = runs["strategy"].select("DATE")
    for name, r in runs.items():
        returns = returns.join(r.select("DATE", pl.col("strategy_return").alias(name)), on="DATE")
    returns = returns.join(market_returns(end_date), on="DATE")

    rows = []
    for name in returns.columns[1:]:
        r = returns[name].to_numpy()
        stats = annualised_stats(r)
        run = runs.get(name)
        rows.append({
            "benchmark": name, "net_sharpe": stats["sharpe"],
            "gross_sharpe": None if run is None else annualised_stats(run["gross_return"].to_numpy())["sharpe"],
            "ann_return": stats["ann_return"], "ann_vol": stats["ann_vol"],
            "ann_cost": None if run is None else 12 * run["cost"].mean(),
            "stock_turnover": None if run is None else run["turnover"].mean(),
            "corr_with_strategy": float(np.corrcoef(r, returns["strategy"].to_numpy())[0, 1]),
            "exit_long_weight": run["exit_long_weight"].mean() if run is not None and "exit_long_weight" in run.columns else None,
            "exit_short_weight": run["exit_short_weight"].mean() if run is not None and "exit_short_weight" in run.columns else None,
        })
    return returns, pl.DataFrame(rows), cost_sensitivity(runs)


def cost_sensitivity(runs: dict, cost_bps=(5, 10, 20, 50)) -> pl.DataFrame:
    """Net Sharpe of each run if every $ of stock traded cost cost_bps instead of TC_BPS (same trades;
    the optimiser's turnover penalty stays at its locked value)."""
    rows = []
    for name, r in runs.items():
        rows.append({"benchmark": name, **{
            f"net_sharpe_{bps}bp": annualised_stats((r["gross_return"] - r["turnover"] * bps / 1e4).to_numpy())["sharpe"]
            for bps in cost_bps}})
    return pl.DataFrame(rows)


def shuffled_returns_run(seed: int, end_date: int = IS_END_DATE) -> dict:
    """Leak test: shuffle stock returns across firms within each month, then build the locked panel and run the
    locked strategy on it. Characteristics no longer say anything about returns, so gross Sharpe should be ~0."""
    df = pl.read_parquet(CHARS_CLEAN_PARQUET).filter(pl.col("DATE") <= end_date)
    df = df.with_columns(pl.col(TARGET_COL).shuffle(seed).over("DATE"))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shuffled.parquet"
        df.write_parquet(path)
        fr, _, _, holdings, rebuilt = build_panel(end_date=end_date, clean_path=path)
    m = SpecBBacktester().run_backtest(fr, holdings, rebuilt)["metrics"]
    print(f"seed {seed} done", flush=True)
    return {"seed": seed, "net_sharpe": m["Sharpe Ratio"], "gross_sharpe": m["Gross Sharpe Ratio"],
            "ann_cost": m["Annualized Cost"], "stock_turnover": m["Avg Monthly Turnover"]}


def leak_test(seeds=range(8), workers=TUNE_WORKERS) -> pl.DataFrame:
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "POLARS_MAX_THREADS"]:
        os.environ[var] = "1"
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(shuffled_returns_run, seeds))
    result = pl.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    result.write_csv(RESULTS_DIR / "leak_test_is.csv")
    with pl.Config(tbl_cols=-1, float_precision=3):
        print(result)
        print(result.select(pl.exclude("seed")).mean())
    return result


def main(start_date: int = IS_START_DATE, end_date: int = IS_END_DATE, suffix: str = "is", workers: int = TUNE_WORKERS):
    returns, summary, costs = run_benchmarks(start_date, end_date, workers)
    RESULTS_DIR.mkdir(exist_ok=True)
    returns.write_csv(RESULTS_DIR / f"benchmark_returns_{suffix}.csv")
    summary.write_csv(RESULTS_DIR / f"benchmarks_{suffix}.csv")
    costs.write_csv(RESULTS_DIR / f"benchmarks_costs_{suffix}.csv")
    benchmark_curves(returns, RESULTS_DIR / f"benchmarks_{suffix}.png")
    with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=200, float_precision=3):
        print(summary)
        print(costs)


def run_oos():
    """The single out-of-sample run: refuses to start if the config differs from the locked one or if OOS
    results already exist."""
    import src.config as c
    locked = json.loads(CONFIG_LOCKED_JSON.read_text())["config"]
    changed = [n for n, v in locked.items() if json.loads(json.dumps(getattr(c, n), default=str)) != v]
    if changed:
        raise RuntimeError(f"config differs from {CONFIG_LOCKED_JSON.name}: {changed}")
    if (RESULTS_DIR / "benchmarks_oos.csv").exists():
        raise RuntimeError("OOS results already exist; the OOS run happens once")
    git = lambda *args: subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if git("status", "--porcelain", "--", "src", "main.py", "pyproject.toml", "uv.lock"):
        raise RuntimeError("uncommitted code changes; commit before the OOS run so results trace to one commit")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "oos_commit.txt").write_text(f"{git('rev-parse', 'HEAD')}\n{git('describe', '--tags', '--always')}\n")
    main(OOS_START_DATE, OOS_END_DATE, "oos", workers=2)  # OOS panels are ~2x larger; 2 workers fit in 8 GB


if __name__ == "__main__":
    main()
