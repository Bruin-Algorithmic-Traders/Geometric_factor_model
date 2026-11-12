import numpy as np
import polars as pl
from scipy.stats import spearmanr

from src.config import (
    FACTOR_RETURNS_PARQUET, IS_END_DATE, MARKET_EXCESS_CSV, MIN_HISTORY_MONTHS, OOS_END_DATE, OOS_START_DATE,
    REBUILT_HISTORY_NPY, RESULTS_DIR, TARGET_VOL,
)
from src.evaluation.metrics import newey_west_t, series_metrics
from src.evaluation.replay import REPLAY_MONTHLY_CSV

EXPLORATORY_DIR = RESULTS_DIR / "exploratory"
# an annual accounting variable that fully refreshes once a year has month-to-month autocorrelation ~11/12 ≈ 0.92;
# below 0.9 a feature changes faster than that
FAST_AUTOCORR = 0.9


def mu_hat_ic(factor_returns: pl.DataFrame, rebuilt: np.ndarray, min_history: int = MIN_HISTORY_MONTHS) -> pl.DataFrame:
    """Each traded month t: Spearman corr across factors of μ̂_t (mean of the rebuilt history the optimiser
    uses) with the factor returns realised in month t."""
    F = factor_returns.drop("DATE").to_numpy()
    rows = []
    for t in range(min_history, len(F)):
        history = rebuilt[t][~np.isnan(rebuilt[t]).any(axis=1)]
        rows.append({"DATE": factor_returns["DATE"][t], "ic": spearmanr(history.mean(axis=0), F[t]).statistic})
    return pl.DataFrame(rows)


def ic_summary(ic: pl.DataFrame) -> pl.DataFrame:
    """Mean IC and its Newey-West t by sample and by year."""
    labelled = ic.with_columns(
        sample=pl.when(pl.col("DATE") <= IS_END_DATE).then(pl.lit("is")).otherwise(pl.lit("oos")),
        year=(pl.col("DATE") // 10000).cast(pl.String),
    )
    rows = []
    for key in ("sample", "year"):
        for (group,), g in labelled.group_by(key, maintain_order=True):
            x = g["ic"].to_numpy()
            rows.append({"group": key, "label": group, "months": len(x), "ic_mean": x.mean(),
                         "nw_t": newey_west_t(x) if len(x) > 6 else None})
    return pl.DataFrame(rows)


def saved_in_sample_panel() -> tuple[pl.DataFrame, np.ndarray]:
    """Factor returns and rebuilt history from the saved in-sample panel (`main.py panel`), without the holdings."""
    factor_returns = pl.read_parquet(FACTOR_RETURNS_PARQUET)
    return factor_returns, np.load(REBUILT_HISTORY_NPY)[: factor_returns.height]


def d1_mu_hat_ic():
    from src.factors.build_panel import build_panel
    factor_returns, _, _, _, rebuilt = build_panel(end_date=OOS_END_DATE)
    ic = mu_hat_ic(factor_returns, rebuilt)
    ic.write_csv(EXPLORATORY_DIR / "d1_mu_hat_ic_monthly.csv")
    ic_summary(ic).write_csv(EXPLORATORY_DIR / "d1_mu_hat_ic_summary.csv")


SUBPERIODS = {
    "2006-08": (20060101, 20081231),
    "2009-11": (20090101, 20111231),
    "2012-15": (20120101, 20151231),
    "2016-18": (20160101, 20181231),
    "oos_excl_2008": (OOS_START_DATE, OOS_END_DATE),
}


def d4_subperiods():
    """Step-3 metrics at 10 bp for each OOS sub-period, with gross Sharpe, turnover and cost from the replay."""
    from src.evaluation.ff_data import load_ff
    monthly = pl.read_csv(REPLAY_MONTHLY_CSV)
    net = monthly.pivot(on="run", index="DATE", values="net_return").sort("DATE")
    seeds = [c for c in net.columns if c.startswith("random_directions_")]
    market = pl.read_csv(MARKET_EXCESS_CSV).select("DATE", "market_equal_excess", "market_value_excess")
    ff = load_ff().select("YYYYMM", "Mkt-RF")
    data = (net.join(market, on="DATE").with_columns(YYYYMM=pl.col("DATE") // 100).join(ff, on="YYYYMM"))
    tables = []
    for label, (start, end) in SUBPERIODS.items():
        period = data.filter(pl.col("DATE").is_between(start, end))
        if label == "oos_excl_2008":
            period = period.filter(pl.col("DATE") // 10000 != 2008)
        mean = period.select(pl.mean_horizontal(seeds)).to_series()
        period = period.with_columns(random_directions_mean_10vol=mean * TARGET_VOL / (mean.std() * np.sqrt(12)))
        mkt = period["Mkt-RF"].to_numpy()
        runs = [c for c in period.columns if c not in ("DATE", "YYYYMM", "Mkt-RF")]
        table = pl.DataFrame([{"period": label, "run": run, **series_metrics(period[run].to_numpy(), mkt)} for run in runs])
        costs = (monthly.filter(pl.col("DATE").is_in(period["DATE"].implode()))
                 .group_by("run").agg(gross=pl.col("gross_return"), stock_turnover=pl.col("turnover").mean(),
                                      cost_drag_ann=12 * pl.col("cost").mean()))
        gross = pl.DataFrame([{"run": r, **{f"gross_{k}": v for k, v in series_metrics(np.array(g), np.array(g)).items()
                                            if k in ("net_sharpe", "sharpe_ci_low", "sharpe_ci_high")}}
                              for r, g in costs.select("run", "gross").iter_rows()])
        gross = gross.rename({"gross_net_sharpe": "gross_sharpe", "gross_sharpe_ci_low": "gross_ci_low",
                              "gross_sharpe_ci_high": "gross_ci_high"})
        table = table.join(gross, on="run", how="left").join(costs.drop("gross"), on="run", how="left")
        seed_sr = table.filter(pl.col("run").is_in(seeds))["net_sharpe"]
        spread = pl.DataFrame({"period": label, "run": ["random_seeds_median", "random_seeds_min", "random_seeds_max"],
                               "net_sharpe": [seed_sr.median(), seed_sr.min(), seed_sr.max()]})
        if label == "oos_excl_2008":
            table = table.with_columns(pl.lit(None).alias(c) for c in ("max_drawdown", "max_dd_months", "calmar"))
        tables += [table, spread]
    pl.concat(tables, how="diagonal_relaxed").write_csv(EXPLORATORY_DIR / "d4_subperiods.csv")


def d8_char_autocorr() -> pl.DataFrame:
    """Each feature's month-to-month cross-sectional Spearman correlation (firms present in both months), averaged
    over in-sample months; features below FAST_AUTOCORR are labelled fast."""
    from src.config import CHARS_CLEAN_PARQUET, feature_cols
    df = pl.read_parquet(CHARS_CLEAN_PARQUET).filter(pl.col("DATE") <= IS_END_DATE)
    features = feature_cols(df.columns)
    months = df.select(pl.col("DATE").unique().sort()).to_series()
    prev = df.join(pl.DataFrame({"DATE": months[1:], "PREV": months[:-1]}), on="DATE")
    pairs = prev.join(df, left_on=["PREV", "permno"], right_on=["DATE", "permno"], suffix="_prev")
    monthly = pairs.group_by("DATE").agg(pl.corr(f, f"{f}_prev", method="spearman").alias(f) for f in features)
    out = (monthly.select(features).mean().transpose(include_header=True, header_name="feature", column_names=["autocorr"])
           .sort("autocorr"))
    return out.with_columns(fast=pl.col("autocorr") < FAST_AUTOCORR)


def fast_weight_share(directions: np.ndarray, fast: np.ndarray) -> float:
    """Share of the squared weight of unit directions (K x features) that sits on fast features."""
    return float((directions[:, fast] ** 2).sum() / (directions ** 2).sum())


def d8_fast_weight(autocorr: pl.DataFrame) -> pl.DataFrame:
    """Fast-feature weight share of the k-means centroids (each traded in-sample month) vs each random seed."""
    from src.config import CENTROIDS_NPZ, CHARS_CLEAN_PARQUET, N_CLUSTERS, N_RANDOM_DIRECTIONS, feature_cols
    from src.evaluation.benchmarks import fit_monthly_random
    features = feature_cols(pl.read_parquet_schema(CHARS_CLEAN_PARQUET).keys())
    fast = np.isin(features, autocorr.filter("fast")["feature"].to_list())
    z = np.load(CENTROIDS_NPZ)
    kmeans = [fast_weight_share(C, fast) for C in z["centroids"][MIN_HISTORY_MONTHS:]]
    rows = [{"directions": "kmeans", "months": len(kmeans), "fast_share": np.mean(kmeans),
             "fast_share_min": np.min(kmeans), "fast_share_max": np.max(kmeans)}]
    for seed in range(N_RANDOM_DIRECTIONS):
        model, _ = next(fit_monthly_random([np.empty((1, len(features)))], N_CLUSTERS, seed=seed))
        share = fast_weight_share(model.centroids, fast)
        rows.append({"directions": f"random_directions_{seed}", "months": 1, "fast_share": share,
                     "fast_share_min": share, "fast_share_max": share})
    rows.append({"directions": "share_of_features_that_are_fast", "fast_share": fast.mean()})
    return pl.DataFrame(rows)


def turnover_split(name: str, spec: dict) -> pl.DataFrame:
    """Re-run one benchmark in-sample and split each month's stock trades into within-factor
    |W_t a_{t-1} - drifted w_{t-1}| and from changing a |W_t (a_t - a_{t-1})|; the first traded month is dropped."""
    from src.factors.build_panel import build_panel
    from src.portfolio.optimiser import SpecBBacktester, stock_turnover
    fr, _, _, holdings, rebuilt = build_panel(end_date=IS_END_DATE, **spec.get("panel", {}))
    res = SpecBBacktester().run_backtest(fr, holdings, rebuilt)
    returns, A = res["returns_df"], res["weights_df"].drop("DATE").to_numpy()
    start = fr.height - len(A)
    rows = []
    for i in range(1, len(A)):
        prev_permnos, prev_W, prev_r = holdings[start + i - 1]
        permnos, W, _ = holdings[start + i]
        drifted = prev_W @ A[i - 1] * (1 + prev_r) / (1 + returns["gross_return"][i - 1])
        rows.append({"DATE": returns["DATE"][i], "run": name, "net_return": returns["strategy_return"][i],
                     "turnover": returns["turnover"][i],
                     "turnover_check": stock_turnover(prev_permnos, drifted, permnos, W @ A[i]),
                     "within_factor": stock_turnover(prev_permnos, drifted, permnos, W @ A[i - 1]),
                     "from_a": float(np.abs(W @ (A[i] - A[i - 1])).sum())})
    return pl.DataFrame(rows)


def d8_turnover(workers: int = 2):
    import os
    from concurrent.futures import ProcessPoolExecutor
    from src.evaluation.benchmarks import benchmark_specs
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "POLARS_MAX_THREADS"]:
        os.environ[var] = "1"
    specs = {k: v for k, v in benchmark_specs().items() if k == "strategy" or k.startswith("random_directions_")}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        monthly = pl.concat(pool.map(turnover_split, specs.keys(), specs.values()))
    frozen = pl.read_csv(RESULTS_DIR / "benchmark_returns_is.csv").unpivot(index="DATE", variable_name="run", value_name="frozen")
    check = monthly.join(frozen, on=["DATE", "run"], how="left")
    max_diff = max((check["net_return"] - check["frozen"]).abs().max(), (check["turnover"] - check["turnover_check"]).abs().max())
    if check["frozen"].null_count() or max_diff > 1e-10:
        raise RuntimeError(f"in-sample replay does not reproduce the frozen returns (max diff {max_diff})")
    monthly.drop("turnover_check").write_csv(EXPLORATORY_DIR / "d8_turnover_monthly.csv")
    (monthly.group_by("run", maintain_order=True)
     .agg(pl.len().alias("months"), pl.col("turnover").mean(), pl.col("within_factor").mean(), pl.col("from_a").mean())
     .with_columns(within_share=pl.col("within_factor") / (pl.col("within_factor") + pl.col("from_a")))
     .write_csv(EXPLORATORY_DIR / "d8_turnover_split.csv"))
    print(f"replay check passed, max diff {max_diff}")


def d8_score_autocorr() -> pl.DataFrame:
    """Each traded in-sample month: Spearman corr, across firms present in both months, of each firm's score on a
    direction (unit firm vector · direction) this month vs last month, averaged over the K directions. K-means
    scores use each month's own centroids, so centroid drift counts; random directions are fixed."""
    from scipy.stats import rankdata
    from src.config import CENTROIDS_NPZ, CHARS_CLEAN_PARQUET, N_CLUSTERS, N_RANDOM_DIRECTIONS, feature_cols
    from src.evaluation.benchmarks import fit_monthly_random
    df = pl.read_parquet(CHARS_CLEAN_PARQUET).filter(pl.col("DATE") <= IS_END_DATE)
    features = feature_cols(df.columns)
    groups = [g for _, g in df.group_by("DATE", maintain_order=True)]
    z = np.load(CENTROIDS_NPZ)
    assert len(groups) == len(z["dates"])
    sets = {"kmeans": z["centroids"]}
    for seed in range(N_RANDOM_DIRECTIONS):
        model, _ = next(fit_monthly_random([np.empty((1, len(features)))], N_CLUSTERS, seed=seed))
        sets[f"random_directions_{seed}"] = np.repeat(model.centroids[None], len(groups), axis=0)
    rows = []
    for t in range(MIN_HISTORY_MONTHS, len(groups)):
        prev, cur = groups[t - 1], groups[t]
        _, i_prev, i_cur = np.intersect1d(prev["permno"].to_numpy(), cur["permno"].to_numpy(), return_indices=True)
        X_prev, X_cur = (g.select(features).to_numpy()[i] for g, i in ((prev, i_prev), (cur, i_cur)))
        X_prev, X_cur = (X / np.linalg.norm(X, axis=1, keepdims=True) for X in (X_prev, X_cur))
        for name, C in sets.items():
            a, b = rankdata(X_prev @ C[t - 1].T, axis=0), rankdata(X_cur @ C[t].T, axis=0)
            a, b = a - a.mean(axis=0), b - b.mean(axis=0)
            corr = (a * b).sum(axis=0) / np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
            rows.append({"DATE": cur["DATE"][0], "directions": name, "score_autocorr": corr.mean()})
    return pl.DataFrame(rows)


def d8():
    autocorr = d8_char_autocorr()
    autocorr.write_csv(EXPLORATORY_DIR / "d8_char_autocorr.csv")
    d8_fast_weight(autocorr).write_csv(EXPLORATORY_DIR / "d8_fast_weight.csv")
    d8_score_autocorr_summary()
    d8_turnover()


def d8_score_autocorr_summary():
    scores = d8_score_autocorr()
    scores.write_csv(EXPLORATORY_DIR / "d8_score_autocorr_monthly.csv")
    (scores.group_by("directions", maintain_order=True)
     .agg(pl.len().alias("months"), pl.col("score_autocorr").mean().alias("mean"),
          pl.col("score_autocorr").min().alias("min"), pl.col("score_autocorr").max().alias("max"))
     .write_csv(EXPLORATORY_DIR / "d8_score_autocorr.csv"))


def d3_by_year():
    """Per run and calendar year (10 bp): gross and net return, cost, stock turnover and gross/net Sharpe."""
    monthly = pl.read_csv(REPLAY_MONTHLY_CSV).with_columns(year=pl.col("DATE") // 10000)
    sharpe = lambda c: pl.col(c).mean() / pl.col(c).std() * np.sqrt(12)
    (monthly.group_by("run", "year", maintain_order=True)
     .agg(pl.len().alias("months"), (12 * pl.col("gross_return").mean()).alias("gross_ann"),
          (12 * pl.col("net_return").mean()).alias("net_ann"), (12 * pl.col("cost").mean()).alias("cost_ann"),
          pl.col("turnover").mean().alias("stock_turnover"), sharpe("gross_return").alias("gross_sharpe"),
          sharpe("net_return").alias("net_sharpe"))
     .sort("run", "year")
     .write_csv(EXPLORATORY_DIR / "d3_by_year.csv"))


ROLLING_MONTHS = 36


def d7_rolling_sharpe():
    """Annualised net Sharpe (10 bp) over trailing 36-month windows, labelled by the window's last month."""
    returns = pl.concat([pl.read_csv(RESULTS_DIR / f"benchmark_returns_{s}.csv") for s in ("is", "oos")])
    returns = (returns.drop("market_equal", "market_value")
               .join(pl.read_csv(MARKET_EXCESS_CSV).drop("RF"), on="DATE").sort("DATE"))
    runs = [c for c in returns.columns if c != "DATE"]
    first = returns["DATE"].shift(ROLLING_MONTHS - 1)
    rolling = returns.select(
        "DATE",
        window_start=first,
        window=pl.when(pl.col("DATE") <= IS_END_DATE).then(pl.lit("is"))
        .when(first >= OOS_START_DATE).then(pl.lit("oos")).otherwise(pl.lit("mixed")),
        **{r: pl.col(r).rolling_mean(ROLLING_MONTHS) / pl.col(r).rolling_std(ROLLING_MONTHS) * np.sqrt(12) for r in runs},
    ).drop_nulls(runs[0])
    rolling.write_csv(EXPLORATORY_DIR / "d7_rolling_sharpe.csv")
