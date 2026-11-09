import numpy as np


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
