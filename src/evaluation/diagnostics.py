import numpy as np
import polars as pl
from scipy.stats import spearmanr

from src.config import (
    FACTOR_RETURNS_PARQUET, IS_END_DATE, MIN_HISTORY_MONTHS, OOS_END_DATE, OOS_START_DATE, REBUILT_HISTORY_NPY,
    RESULTS_DIR,
)
from src.evaluation.metrics import newey_west_t

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
