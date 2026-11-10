import numpy as np

from src.config import MAX_STOCK_WEIGHT

class FactorBuilder:
    def compute_weights(self, Pi: np.ndarray) -> np.ndarray:
        """Paper eq. 14: demean memberships across firms (dollar-neutral), then scale each factor to unit gross."""
        W_tilde = Pi - np.mean(Pi, axis=0, keepdims=True)
        l1_norms = np.maximum(np.sum(np.abs(W_tilde), axis=0, keepdims=True), 1e-12)
        W = W_tilde / l1_norms
        if np.abs(W).max() > MAX_STOCK_WEIGHT:
            raise ValueError(f"A single stock has {np.abs(W).max():.2%} of a factor (limit {MAX_STOCK_WEIGHT:.0%})")
        return W

    def compute_factor_returns(self, W : np.ndarray, returns : np.ndarray) -> np.ndarray:
        return W.T @ returns
