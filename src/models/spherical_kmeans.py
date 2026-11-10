import numpy as np
from scipy.optimize import linear_sum_assignment

from src.config import MAX_ITER, N_CLUSTERS, N_INIT, POOL_MONTHS, RANDOM_STATE, TAU, TOL

class SphericalKMeans:
    def __init__(self, n_clusters=N_CLUSTERS, max_iter=MAX_ITER, tol=TOL, random_state=RANDOM_STATE, n_init=N_INIT):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        self.centroids = None

    def _normalise(self, X : np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norms == 0, 1, norms)

    def _kmeans_plus_plus_init(self, X_norm : np.ndarray, rng : np.random.Generator) -> np.ndarray:
        N, P = X_norm.shape
        centroids = np.empty((self.n_clusters, P))
        centroids[0] = X_norm[rng.integers(N)]
        for k in range(1, self.n_clusters):
            max_sims = np.max(X_norm @ centroids[:k].T, axis=1)
            # on the unit sphere squared Euclidean distance = 2(1 - cos), so k-means++ samples ∝ (1 - cos)
            weights = np.maximum(0.0, 1 - max_sims)
            total = weights.sum()
            probabilities = weights / total if total > 0 else np.ones(N) / N
            centroids[k] = X_norm[rng.choice(N, p=probabilities)]
        return centroids

    def _lloyd(self, X_norm : np.ndarray, centroids : np.ndarray) -> tuple[np.ndarray, float]:
        for _ in range(self.max_iter):
            sim = X_norm @ centroids.T
            labels = np.argmax(sim, axis=1)
            counts = np.bincount(labels, minlength=self.n_clusters)
            new_centroids = np.zeros_like(centroids)
            for k in np.flatnonzero(counts):
                new_centroids[k] = X_norm[labels == k].mean(axis=0)
            # reseed empty clusters with distinct worst-fit points (lowest best-match cosine)
            empty = np.flatnonzero(counts == 0)
            if empty.size:
                worst = np.argsort(sim.max(axis=1))[:empty.size]
                new_centroids[empty] = X_norm[worst]
            new_centroids = self._normalise(new_centroids)

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < self.tol:
                break
        objective = float(np.max(X_norm @ centroids.T, axis=1).sum())
        return centroids, objective

    def fit(self, X : np.ndarray, init_centroids : np.ndarray | None = None) -> "SphericalKMeans":
        """Warm start from init_centroids if given, else best of n_init k-means++ runs by total cosine (eq. 4)."""
        X_norm = self._normalise(X)
        if init_centroids is not None:
            self.centroids, _ = self._lloyd(X_norm, self._normalise(init_centroids))
            return self
        rng = np.random.default_rng(self.random_state)
        runs = [self._lloyd(X_norm, self._kmeans_plus_plus_init(X_norm, rng)) for _ in range(self.n_init)]
        self.centroids = max(runs, key=lambda run: run[1])[0]
        return self

    def transform(self, X : np.ndarray, tau : float = TAU) -> np.ndarray:
        """Soft memberships, paper eq. 6: softmax over clusters of tau * cos(x, centroid)."""
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet.")
        X_norm = self._normalise(X)
        sim = X_norm @ self.centroids.T
        exp_sim = np.exp(tau * (sim - np.max(sim, axis=1, keepdims=True)))
        return exp_sim / np.sum(exp_sim, axis=1, keepdims=True)


def fit_monthly(months: list[np.ndarray], n_clusters: int = N_CLUSTERS, pool_months: int = POOL_MONTHS):
    """For each month t, fit on the stacked firms of months t-pool_months+1..t, warm-started from month t-1,
    then reorder centroids (Hungarian) to best match month t-1. Yields (model, matched cosines or None)."""
    prev = None
    for t in range(len(months)):
        pooled = np.vstack(months[max(0, t - pool_months + 1): t + 1])
        model = SphericalKMeans(n_clusters).fit(pooled, prev)
        matched = None
        if prev is not None:
            _, order = linear_sum_assignment(-(prev @ model.centroids.T))
            model.centroids = model.centroids[order]
            matched = np.einsum("kp,kp->k", model.centroids, prev)
        prev = model.centroids
        yield model, matched
