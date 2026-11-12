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
