import numpy as np

from src.config import FACTOR_W_MAX

class FactorBuilder:
    def __init__(self, w_max : float | None = FACTOR_W_MAX):
        self.w_max = w_max

    def compute_weights(self, Pi: np.ndarray) -> np.ndarray:
        col_means = np.mean(Pi, axis=0, keepdims=True)
        W_tilde = Pi - col_means

        if self.w_max is not None:
            W_tilde = np.clip(W_tilde, -self.w_max, self.w_max)
            W_tilde = W_tilde - np.mean(W_tilde, axis=0, keepdims=True)

        l1_norms = np.sum(np.abs(W_tilde), axis=0, keepdims=True)
        l1_norms = np.maximum(l1_norms, 1e-12)
        W = W_tilde / l1_norms
        return W

    def compute_factor_returns(self, W : np.ndarray, returns : np.ndarray) -> np.ndarray:
        return W.T @ returns