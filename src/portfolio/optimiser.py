import numpy as np
import polars as pl
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from typing import Dict
from src.evaluation.metrics import annualised_stats
from src.config import (
    COV_WINDOW_MONTHS, FACTOR_RETURNS_OOS_PARQUET, GAMMA_F, GROSS_FACTOR_LEVERAGE, KAPPA_PENALTY,
    MAX_FACTOR_WEIGHT, MIN_HISTORY_MONTHS, MU_WINDOW_MONTHS, PROCESSED_DIR,
    STRATEGY_RETURNS_PARQUET, STRATEGY_WEIGHTS_PARQUET, TC_BPS,
)

class SpecBBacktester:
    def __init__(
        self,
        gamma_f: float = GAMMA_F,                              # Factor risk aversion
        kappa_penalty: float = KAPPA_PENALTY,                  # Turnover penalty in the objective
        tc_bps: float = TC_BPS,                                # Realised cost, bp per $ traded
        gross_factor_leverage: float = GROSS_FACTOR_LEVERAGE,  # Max gross exposure cap (L_a)
        max_factor_weight: float = MAX_FACTOR_WEIGHT,          # Single-factor concentration cap (a_max)
        mu_window_months: int = MU_WINDOW_MONTHS,              # Rolling window for mu_hat
        cov_window_months: int = COV_WINDOW_MONTHS,            # Rolling window for Sigma_hat
        min_history_months: int = MIN_HISTORY_MONTHS           # Initial burn-in history
    ):
        self.gamma_f = gamma_f
        self.kappa_penalty = kappa_penalty
        self.tc = tc_bps / 1e4
        self.L_a = gross_factor_leverage
        self.a_max = max_factor_weight
        self.mu_window = mu_window_months
        self.cov_window = cov_window_months
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

    def run_backtest(self, df_factor_returns: pl.DataFrame, date_col: str = "DATE") -> Dict:
        dates = df_factor_returns[date_col].to_list()
        factor_cols = [col for col in df_factor_returns.columns if col != date_col]
        F_matrix = df_factor_returns.select(factor_cols).to_numpy()

        T, M = F_matrix.shape
        gross_returns, costs, net_returns, turnovers, weights_history, oos_dates = [], [], [], [], [], []
        a_prev = np.zeros(M)

        for t in range(self.min_history, T):
            hist_F = F_matrix[:t, :]

            # 1. Rolling window expected factor returns (raw sample mean)
            mu_slice = hist_F[-self.mu_window:, :] if t >= self.mu_window else hist_F
            mu_hat = np.mean(mu_slice, axis=0)

            # 2. Rolling window factor covariance (Ledoit-Wolf shrinkage)
            cov_slice = hist_F[-self.cov_window:, :] if t >= self.cov_window else hist_F
            lw = LedoitWolf(assume_centered=False)
            sigma_hat = lw.fit(cov_slice).covariance_

            # 3. Solve exact Equation (18)
            a_opt = self._solve_equation_18(mu_hat, sigma_hat, a_prev)

            # 4. Realized performance & net return accounting (Eq. 20)
            realized_f_t = F_matrix[t, :]
            gross_return = float(np.dot(a_opt, realized_f_t))
            turnover = float(np.sum(np.abs(a_opt - a_prev)))
            cost = turnover * self.tc

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
            "turnover": turnovers
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
                "Avg Monthly Turnover": float(np.mean(turnovers))
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
