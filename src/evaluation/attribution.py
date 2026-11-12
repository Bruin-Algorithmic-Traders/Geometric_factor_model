import numpy as np
import polars as pl

from src.config import RESULTS_DIR
from src.evaluation.ff_data import load_ff
from src.evaluation.metrics import NW_LAGS

MODELS = {
    "market": ["Mkt-RF"],
    "ff5_umd": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
    "ff5_umd_strev": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "ST_Rev"],
}
MARKET_COLS = ["market_equal", "market_value"]


def ols_newey_west(y: np.ndarray, X: np.ndarray, lags: int = NW_LAGS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS coefficients, Bartlett-kernel Newey-West t-stats and residuals; X includes the constant."""
    T = len(y)
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    g = X * u[:, None]
    S = g.T @ g / T
    for l in range(1, lags + 1):
        G = g[l:].T @ g[:-l] / T
        S += (1 - l / (lags + 1)) * (G + G.T)
    cov = T * XtX_inv @ S @ XtX_inv
    return beta, beta / np.sqrt(np.diag(cov)), u


def attribution_table(suffix: str) -> pl.DataFrame:
    returns = pl.read_csv(RESULTS_DIR / f"benchmark_returns_{suffix}.csv").drop(MARKET_COLS)
    data = returns.with_columns(YYYYMM=pl.col("DATE") // 100).join(load_ff(), on="YYYYMM", how="inner")
    if data.height != returns.height:
        raise ValueError("months lost when joining Fama-French data")
    rows = []
    for run in returns.columns[1:]:
        y = data[run].to_numpy()
        for model, factors in MODELS.items():
            X = np.column_stack([np.ones(len(y)), data.select(factors).to_numpy()])
            beta, t, u = ols_newey_west(y, X)
            alpha_ann = 12 * beta[0]
            resid_vol_ann = u.std(ddof=X.shape[1]) * np.sqrt(12)
            r2 = 1 - (u @ u) / ((y - y.mean()) @ (y - y.mean()))
            terms = [("alpha_ann", alpha_ann, t[0])] + [(f, b, tt) for f, b, tt in zip(factors, beta[1:], t[1:])]
            terms += [("r2", r2, None), ("info_ratio", alpha_ann / resid_vol_ann, None), ("months", len(y), None)]
            rows += [{"run": run, "model": model, "term": term, "estimate": float(est),
                      "nw_t": None if tt is None else float(tt)} for term, est, tt in terms]
    return pl.DataFrame(rows)


def main():
    for suffix in ("is", "oos"):
        attribution_table(suffix).write_csv(RESULTS_DIR / f"attribution_{suffix}.csv")
