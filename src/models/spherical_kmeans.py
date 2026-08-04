import numpy as np

from config import MAX_ITER, TOL, N_CLUSTERS, RANDOM_STATE, TAU

class SphericalKMeans:
    def __init__(self, n_clusters=N_CLUSTERS, max_iter=MAX_ITER, tol=TOL, random_state = RANDOM_STATE):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def _normalise(self, X : np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norms == 0, 1, norms)

    def _kmeans_plus_plus_init(self, X_norm : np.ndarray, rng : np.random.Generator) -> np.ndarray:
        N, P = X_norm.shape
        centroids = np.empty((self.n_clusters, P))
        first_idx = rng.integers(N)
        centroids[0] = X_norm[first_idx]
        for k in range(1, self.n_clusters):
            sims = X_norm @ centroids[:k].T
            max_sims = np.max(sims, axis=1)
            distances = np.maximum(0.0, 1 - max_sims)
            probabilities = distances ** 2
            prob_sum = probabilities.sum()
            if prob_sum > 0:
                probabilities /= prob_sum
            else:
                probabilities = np.ones(N) / N
            next_idx = rng.choice(N, p=probabilities)
            centroids[k] = X_norm[next_idx]
        return centroids

    def fit(self, X : np.ndarray) -> "SphericalKMeans":
        rng = np.random.default_rng(self.random_state)
        X_norm = self._normalise(X)
        N, P = X_norm.shape
        centroids = self._kmeans_plus_plus_init(X_norm, rng)

        for _ in range(self.max_iter):

            sim = X_norm @ centroids.T
            labels = np.argmax(sim, axis=1)
            new_centroids = np.zeros_like(centroids)
            max_sims = np.max(sim, axis=1)

            for k in range(self.n_clusters):
                mask = (labels == k)
                if np.any(mask):
                    new_centroids[k] = X_norm[mask].mean(axis=0)
                else:
                    new_centroids[k] = np.argmin(max_sims)

            new_centroids = self._normalise(new_centroids)

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < self.tol:
                break

        self.centroids = centroids
        return self

    def transform(self, X : np.ndarray, tau : float = TAU) -> np.ndarray:
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet.")
        X_norm = self._normalise(X)
        sim = X_norm @ self.centroids.T
        exp_sim = np.exp((sim - np.max(sim, axis=1, keepdims=True)) / tau)
        return exp_sim / np.sum(exp_sim, axis=1, keepdims=True)
        
    