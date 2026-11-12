import numpy as np
import polars as pl
from scipy import stats

from src.config import IS_END_DATE, N_RANDOM_DIRECTIONS, OOS_START_DATE, RESULTS_DIR
from src.evaluation.replay import REPLAY_MONTHLY_CSV

N_BOOT = 10_000
MEAN_BLOCK = 6
SEED = 42
COST_BPS = (5, 10, 20, 50)
ALPHA = 0.025  # one-sided, the same rule as "the 95% CI excludes 0"
SEEDS = [f"random_directions_{s}" for s in range(N_RANDOM_DIRECTIONS)]
RANDOM_PORTFOLIO = "portfolio of 8 random strategies (includes a diversification benefit; costs overstated because trades are not netted)"

# (benchmark, cost in bp, claim, pre-registered out of sample)
COMPARISONS = [
    ("inverse_vol_mix", 10, "better", True),
    ("diagonal_mv_mix", 10, "better", False),
    ("linear_tau_0.01", 10, "equal", True),
    ("pca_directions", 10, "better", True),
    ("pca_directions", 20, "better", True),
    ("returns_free_features", 10, "equal", True),
    *[(s, bps, "equal" if bps <= 10 else "better", True) for bps in COST_BPS for s in SEEDS],
    (RANDOM_PORTFOLIO, 10, "better", False),
]


def sharpe(r: np.ndarray) -> np.ndarray:
    """Annualised Sharpe along the last axis."""
    return r.mean(-1) / r.std(-1, ddof=1) * np.sqrt(12)


def stationary_bootstrap(T: int, n_boot: int = N_BOOT, mean_block: int = MEAN_BLOCK, seed: int = SEED) -> np.ndarray:
    """Politis-Romano stationary bootstrap month indices (n_boot × T): geometric blocks, wrapping around."""
    rng = np.random.default_rng(seed)
    idx = np.empty((n_boot, T), dtype=int)
    idx[:, 0] = rng.integers(T, size=n_boot)
    new_block = rng.random((n_boot, T)) < 1 / mean_block
    starts = rng.integers(T, size=(n_boot, T))
    for t in range(1, T):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def memmel_p(a: np.ndarray, b: np.ndarray) -> float:
    """One-sided p of Sharpe(a) > Sharpe(b), Jobson-Korkie test with Memmel's correction (monthly units)."""
    s1, s2 = sharpe(a) / np.sqrt(12), sharpe(b) / np.sqrt(12)
    rho = np.corrcoef(a, b)[0, 1]
    var = (2 - 2 * rho + 0.5 * (s1**2 + s2**2 - 2 * s1 * s2 * rho**2)) / len(a)
    return float(1 - stats.norm.cdf((s1 - s2) / np.sqrt(var)))


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]).clip(max=1)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def verdict(claim: str, low: float, high: float, point: float) -> str:
    if claim == "equal":
        return "no difference" if low <= 0 <= high else "differs"
    if low > 0:
        return "significant"
    return "directional, not significant" if point > 0 else "no evidence"


def net_returns(monthly: pl.DataFrame, bps: int) -> pl.DataFrame:
    """Wide table of net monthly returns per run at a given cost per $ traded, plus the random portfolio."""
    wide = (monthly.with_columns(net=pl.col("gross_return") - pl.col("turnover") * bps / 1e4)
            .pivot(on="run", index="DATE", values="net").sort("DATE"))
    return wide.with_columns(pl.mean_horizontal(SEEDS).alias(RANDOM_PORTFOLIO))


def compare_sample(monthly: pl.DataFrame, sample: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    in_sample = pl.col("DATE") <= IS_END_DATE if sample == "is" else pl.col("DATE") >= OOS_START_DATE
    tables = {bps: net_returns(monthly, bps).filter(in_sample) for bps in COST_BPS}
    idx = stationary_bootstrap(tables[10].height)
    rows = []
    for bench, bps, claim, prereg in COMPARISONS:
        a, b = tables[bps]["strategy"].to_numpy(), tables[bps][bench].to_numpy()
        diffs = sharpe(a[idx]) - sharpe(b[idx])
        point = float(sharpe(a) - sharpe(b))
        low, high = np.percentile(diffs, [2.5, 97.5])
        rows.append({"benchmark": bench, "cost_bps": bps, "claim": claim,
                     "preregistered": prereg and sample == "oos",
                     "sharpe_strategy": float(sharpe(a)), "sharpe_benchmark": float(sharpe(b)),
                     "sharpe_diff": point, "ci_low": low, "ci_high": high,
                     "p_boot": float((diffs <= 0).mean()), "p_memmel": memmel_p(a, b),
                     "verdict": verdict(claim, low, high, point)})
    table = pl.DataFrame(rows)
    holm_col = np.full(table.height, np.nan)
    for bps in COST_BPS:
        mask = (table["benchmark"].is_in(SEEDS) & (table["cost_bps"] == bps)).to_numpy()
        holm_col[mask] = holm(table["p_boot"].to_numpy()[mask])
    table = table.with_columns(p_holm_seeds=pl.Series(holm_col).fill_nan(None))
    table = table.with_columns(verdict_holm=pl.when(pl.col("p_holm_seeds").is_null()).then(None)
                               .when(pl.col("claim") == "equal")
                               .then(pl.when(pl.col("p_holm_seeds") < ALPHA).then(pl.lit("differs")).otherwise(pl.lit("no difference")))
                               .when(pl.col("p_holm_seeds") < ALPHA).then(pl.lit("significant"))
                               .when(pl.col("sharpe_diff") > 0).then(pl.lit("directional, not significant"))
                               .otherwise(pl.lit("no evidence")))

    ranks = []
    for bps in COST_BPS:
        sr = {run: float(sharpe(tables[bps][run].to_numpy())) for run in ["strategy", *SEEDS]}
        ranks.append({"sample": sample, "cost_bps": bps, "strategy_sharpe": sr["strategy"],
                      "seed_median": float(np.median([sr[s] for s in SEEDS])),
                      "rank_among_9": 1 + sum(v > sr["strategy"] for v in sr.values())})
    return table, pl.DataFrame(ranks)


def main():
    monthly = pl.read_csv(REPLAY_MONTHLY_CSV)
    ranks = []
    for sample in ("is", "oos"):
        table, rank = compare_sample(monthly, sample)
        table.write_csv(RESULTS_DIR / f"compare_{sample}.csv")
        ranks.append(rank)
    pl.concat(ranks).write_csv(RESULTS_DIR / "compare_rank.csv")
