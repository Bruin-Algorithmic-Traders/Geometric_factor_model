import numpy as np
import polars as pl
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from typing import Dict
from src.evaluation.metrics import annualised_stats
from src.config import (
    COV_HALF_LIFE_MONTHS, FACTOR_RETURNS_OOS_PARQUET, GAMMA_F, GROSS_FACTOR_LEVERAGE, KAPPA_PENALTY,
    MAX_FACTOR_WEIGHT, MIN_HISTORY_MONTHS, PROCESSED_DIR,
    STRATEGY_RETURNS_PARQUET, STRATEGY_WEIGHTS_PARQUET, TC_BPS,
)

def stock_turnover(prev_permnos: np.ndarray, prev_w: np.ndarray, permnos: np.ndarray, w: np.ndarray) -> float:
    """Σ|w - prev_w| over the union of stocks; a stock missing on one side has weight 0 there."""
    all_permnos = np.union1d(prev_permnos, permnos)
    before, after = np.zeros(len(all_permnos)), np.zeros(len(all_permnos))
    before[np.searchsorted(all_permnos, prev_permnos)] = prev_w
    after[np.searchsorted(all_permnos, permnos)] = w
    return float(np.abs(after - before).sum())


def estimate_moments(history: np.ndarray, half_life: float = COV_HALF_LIFE_MONTHS) -> tuple[np.ndarray, np.ndarray]:
    """μ̂ = plain mean of the (rebuilt) history; Σ̂ = exponentially weighted covariance with Ledoit-Wolf shrinkage.
    history: months x K, oldest first; rows with NaN (before the sample) are ignored."""
    history = history[~np.isnan(history).any(axis=1)]
    mu_hat = history.mean(axis=0)
    weights = 0.5 ** (np.arange(len(history))[::-1] / half_life)
    weights /= weights.sum()
    centred = history - weights @ history
    sigma_hat = LedoitWolf(assume_centered=True).fit(centred * np.sqrt(weights * len(history))[:, None]).covariance_
    return mu_hat, sigma_hat


class SpecBBacktester:
    def __init__(
        self,
        gamma_f: float = GAMMA_F,                              # Factor risk aversion
        kappa_penalty: float = KAPPA_PENALTY,                  # Turnover penalty in the objective
        tc_bps: float = TC_BPS,                                # Realised cost, bp per $ traded
        gross_factor_leverage: float = GROSS_FACTOR_LEVERAGE,  # Max gross exposure cap (L_a)
        max_factor_weight: float = MAX_FACTOR_WEIGHT,          # Single-factor concentration cap (a_max)
        min_history_months: int = MIN_HISTORY_MONTHS           # Initial burn-in history
    ):
        self.gamma_f = gamma_f
        self.kappa_penalty = kappa_penalty
        self.tc = tc_bps / 1e4
        self.L_a = gross_factor_leverage
        self.a_max = max_factor_weight
        self.min_history = min_history_months

    def _solve_equation_18(
        self,
        mu_f: np.ndarray,
        sigma_f: np.ndarray,
        a_prev: np.ndarray
    ) -> np.ndarray:
        M = len(mu_f)
        a = cp.Variable(M)

        # Objective Function: Expected Return - Risk Penalty - L1 Turnover Penalty (Eq. 18)
        expected_ret = a @ mu_f
        factor_risk = (self.gamma_f / 2.0) * cp.quad_form(a, cp.psd_wrap(sigma_f))
        turnover_penalty = self.kappa_penalty * cp.norm(a - a_prev, 1)

        objective = cp.Maximize(expected_ret - factor_risk - turnover_penalty)

        constraints = [
            cp.norm(a, 1) <= self.L_a,        # Gross leverage constraint
            cp.norm(a, "inf") <= self.a_max   # Max single-factor exposure
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CLARABEL, verbose=False)

        if prob.status not in ["optimal", "optimal_inaccurate"] or a.value is None:
            return a_prev

        return a.value

    def run_backtest(self, df_factor_returns: pl.DataFrame, holdings: list, rebuilt: np.ndarray, date_col: str = "DATE") -> Dict:
        """holdings[t] = (permnos, W, stock returns) and rebuilt[t] = past months' factor returns re-formed with month t's
        centroids, both for the month in row t of df_factor_returns (from build_panel). μ̂ and Σ̂ come from rebuilt[t].
        Realised cost = tc × stock turnover, where last month's holdings are first drifted by their returns."""
        dates = df_factor_returns[date_col].to_list()
        factor_cols = [col for col in df_factor_returns.columns if col != date_col]
        F_matrix = df_factor_returns.select(factor_cols).to_numpy()

        T, M = F_matrix.shape
        assert len(holdings) == T and len(rebuilt) == T, "holdings and rebuilt need one entry per factor-return row"
        gross_returns, costs, net_returns, turnovers, factor_turnovers, weights_history, oos_dates = [], [], [], [], [], [], []
        a_prev = np.zeros(M)
        prev_permnos, prev_w = np.array([], dtype=np.int64), np.array([])

        for t in range(self.min_history, T):
            mu_hat, sigma_hat = estimate_moments(rebuilt[t])

            # Solve Equation (18)
            a_opt = self._solve_equation_18(mu_hat, sigma_hat, a_prev)

            # Realised performance & net return accounting (Eq. 20)
            realized_f_t = F_matrix[t, :]
            gross_return = float(np.dot(a_opt, realized_f_t))
            permnos, W, stock_returns = holdings[t]
            w = W @ a_opt
            turnover = stock_turnover(prev_permnos, prev_w, permnos, w)
            cost = turnover * self.tc
            factor_turnovers.append(float(np.sum(np.abs(a_opt - a_prev))))
            prev_permnos, prev_w = permnos, w * (1 + stock_returns) / (1 + gross_return)

            gross_returns.append(gross_return)
            costs.append(cost)
            net_returns.append(gross_return - cost)
            turnovers.append(turnover)
            weights_history.append(a_opt)
            oos_dates.append(dates[t])

            a_prev = a_opt.copy()

        df_returns = pl.DataFrame({
            date_col: oos_dates,
            "gross_return": gross_returns,
            "cost": costs,
            "strategy_return": net_returns,
            "turnover": turnovers,
            "factor_turnover": factor_turnovers
        })

        df_weights = pl.DataFrame(
            np.array(weights_history),
            schema=factor_cols
        ).with_columns(pl.Series(date_col, oos_dates))

        stats = annualised_stats(np.array(net_returns))
        ann_ret, ann_vol, sharpe = float(stats["ann_return"]), float(stats["ann_vol"]), float(stats["sharpe"])
        gross_sharpe = float(annualised_stats(np.array(gross_returns))["sharpe"])

        return {
            "metrics": {
                "Annualized Return": ann_ret,
                "Annualized Volatility": ann_vol,
                "Sharpe Ratio": sharpe,
                "Gross Sharpe Ratio": gross_sharpe,
                "Annualized Cost": float(np.mean(costs) * 12),
                "Avg Monthly Turnover": float(np.mean(turnovers)),
                "Avg Factor Turnover": float(np.mean(factor_turnovers))
            },
            "returns_df": df_returns,
            "weights_df": df_weights
        }


def main():
    if FACTOR_RETURNS_OOS_PARQUET.exists():
        df_factor_returns = pl.read_parquet(FACTOR_RETURNS_OOS_PARQUET)

        backtester = SpecBBacktester()

        results = backtester.run_backtest(df_factor_returns)

        print("\n==========================================")
        print(" PURE SPECIFICATION B: OUT-OF-SAMPLE      ")
        print("==========================================")
        for metric, val in results["metrics"].items():
            print(f"{metric:25s}: {val:8.4f}")
        print("==========================================")

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        results["returns_df"].write_parquet(STRATEGY_RETURNS_PARQUET)
        results["weights_df"].write_parquet(STRATEGY_WEIGHTS_PARQUET)


if __name__ == "__main__":
    main()
