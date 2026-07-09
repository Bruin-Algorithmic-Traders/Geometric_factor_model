"""
clustering.py
=============
Implements both clustering methods described in Section 2.3 of the paper.

Methods
-------
SphericalKMeans  —  Section 2.3.1  (primary, used for pricing)
run_hdbscan      —  Section 2.3.2  (used for validation)

Both methods produce an N×M soft membership matrix  Π_{t-1}  (Equation 7)
via temperature-scaled softmax over cosine distances (Equation 6).

For HDBSCAN, soft weights use the algorithm's built-in membership
probabilities, broadcast across discovered clusters.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from normalization import masked_cosine_similarity

from config import (
    N_CLUSTERS,
    SKM_MAX_ITER,
    SKM_N_INIT,
    SKM_TOL,
    TEMPERATURE,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    HDBSCAN_LEAF_SIZE,
    RANDOM_SEED,
)


def _normalise_nan_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64).copy()
    finite = np.isfinite(X)
    norms = np.sqrt(np.where(finite, X ** 2, 0.0).sum(axis=1, keepdims=True))
    X = np.divide(X, norms, out=X, where=norms > 0)
    X[~finite] = np.nan
    return X


# ===========================================================================
# Spherical K-Means  (Section 2.3.1)
# ===========================================================================

class SphericalKMeans:
    """
    Spherical K-Means clustering on data that already lives on the unit sphere.

    Unlike standard K-Means (which minimises squared Euclidean distance),
    Spherical K-Means maximises intra-cluster cosine similarity, which is
    equivalent to minimising the objective in Equation (4):

        J_sph = Σ_m Σ_{i∈C_m} (1 - u_i^T μ_m)

    where μ_m is the normalised centroid of cluster m.

    Multiple random restarts are run (n_init) and the solution with the
    lowest objective value is retained.

    Parameters
    ----------
    n_clusters   : M — number of firm types
    max_iter     : maximum EM iterations per restart
    n_init       : number of random restarts
    tol          : centroid-shift convergence threshold
    random_state : RNG seed for reproducibility
    """

    def __init__(
        self,
        n_clusters: int  = N_CLUSTERS,
        max_iter: int    = SKM_MAX_ITER,
        n_init: int      = SKM_N_INIT,
        tol: float       = SKM_TOL,
        random_state: int = RANDOM_SEED,
    ):
        self.n_clusters   = n_clusters
        self.max_iter     = max_iter
        self.n_init       = n_init
        self.tol          = tol
        self.random_state = random_state

        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None    = None
        self.inertia_: float               = np.inf

    # -------------------------------------------------------------------
    def _run_once(self, X: np.ndarray, rng: np.random.RandomState) -> tuple:
        """Single run of Spherical K-Means. Returns (centroids, labels, cost)."""
        N = X.shape[0]

        # ---------- k-means++ initialisation on the sphere ----------
        centroids = self._kmeanspp_init(X, rng)

        for _ in range(self.max_iter):
            # --- Assignment (maximise cosine similarity) ---
            sims   = masked_cosine_similarity(X, centroids)        # N × M
            labels = np.argmax(sims, axis=1)

            # --- Update centroids ---
            new_centroids = np.zeros_like(centroids)
            for m in range(self.n_clusters):
                mask = labels == m
                if mask.sum() > 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        new_centroids[m] = np.nanmean(X[mask], axis=0)
                else:
                    # Empty cluster: reinitialise randomly
                    new_centroids[m] = X[rng.randint(N)]

            new_centroids = _normalise_nan_rows(new_centroids)

            # --- Convergence check ---
            diff = np.abs(new_centroids - centroids)
            shift = np.nanmax(diff) if np.isfinite(diff).any() else 0.0
            centroids = new_centroids
            if shift < self.tol:
                break

        # Objective: Σ_i (1 - u_i^T μ_{label_i})
        final_sims = masked_cosine_similarity(X, centroids)
        cost = float(np.sum(1.0 - final_sims[np.arange(N), labels]))
        return centroids, labels, cost

    def _kmeanspp_init(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """
        K-Means++ initialisation adapted for cosine distance.
        Selects initial centroids with probability proportional to the
        cosine distance to the nearest already-chosen centroid.
        """
        N = X.shape[0]
        centroids = []

        # First centroid: random
        idx = rng.randint(N)
        centroids.append(X[idx])

        for _ in range(1, self.n_clusters):
            C = np.stack(centroids)                # (k × P)
            sims = masked_cosine_similarity(X, C)  # (N × k)
            # Cosine distance to nearest centroid
            max_sims = sims.max(axis=1)
            dists = np.clip(1.0 - max_sims, 0.0, 2.0)
            if dists.sum() > 0:
                probs = dists / dists.sum()
            else:
                probs = np.ones(N) / N
            idx = rng.choice(N, p=probs)
            centroids.append(X[idx])

        C = np.stack(centroids)
        return _normalise_nan_rows(C)

    # -------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "SphericalKMeans":
        """
        Fit Spherical K-Means on unit-normed embedding matrix X (N × P).

        Parameters
        ----------
        X : ndarray (N × P), rows must be unit vectors.

        Returns
        -------
        self
        """
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) > n_samples ({X.shape[0]})"
            )

        base_rng = np.random.RandomState(self.random_state)

        best_centroids, best_labels, best_cost = None, None, np.inf
        for _ in range(self.n_init):
            seed = base_rng.randint(0, 2**31)
            rng  = np.random.RandomState(seed)
            centroids, labels, cost = self._run_once(X, rng)
            if cost < best_cost:
                best_centroids, best_labels, best_cost = centroids, labels, cost

        self.centroids_ = best_centroids
        self.labels_    = best_labels
        self.inertia_   = best_cost
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard cluster assignments for new data X (N × P)."""
        if self.centroids_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return np.argmax(masked_cosine_similarity(X, self.centroids_), axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_


# ===========================================================================
# Soft membership weights  (Equations 6 & 7)
# ===========================================================================

def compute_soft_weights_skm(
    U: np.ndarray,
    centroids: np.ndarray,
    tau: float = TEMPERATURE,
) -> np.ndarray:
    """
    Convert hard cluster centroids to soft membership weights via
    temperature-scaled softmax over cosine distances (Equation 6):

        π_{i,m} = exp(-τ · d(i, μ_m)) / Σ_j exp(-τ · d(i, μ_j))

    where d(i, μ_m) = 1 - u_i^T μ_m  (cosine distance).

    Parameters
    ----------
    U         : ndarray (N × P), unit-normed firm embeddings
    centroids : ndarray (M × P), unit-normed cluster centroids
    tau       : temperature  τ  (higher → harder assignments)

    Returns
    -------
    Pi : ndarray (N × M), rows sum to 1
    """
    # Cosine distances to centroids  (N × M)
    sims   = masked_cosine_similarity(U, centroids)
    dists  = 1.0 - sims

    # Softmax with temperature (subtract row-max for numerical stability)
    logits = -tau * dists
    logits -= logits.max(axis=1, keepdims=True)
    exp_l  = np.exp(logits)
    Pi     = exp_l / exp_l.sum(axis=1, keepdims=True)
    return Pi


def compute_soft_weights_hdbscan(
    labels: np.ndarray,
    soft_probs: np.ndarray,
) -> np.ndarray:
    """
    Convert HDBSCAN output to a soft membership matrix.

    HDBSCAN assigns:
      - labels[i] ∈ {0, 1, ..., M-1} for cluster members
      - labels[i] = -1  for noise points
      - soft_probs[i] ∈ [0, 1] = membership strength

    We treat noise points as belonging equally to all clusters (uniform row),
    and cluster members get their HDBSCAN probability concentrated on their
    assigned cluster.

    Parameters
    ----------
    labels     : ndarray (N,) integer cluster labels (-1 = noise)
    soft_probs : ndarray (N,) HDBSCAN membership probability per point

    Returns
    -------
    Pi         : ndarray (N × M) where M = number of unique non-noise clusters
    """
    unique_clusters = sorted(set(labels) - {-1})
    M = len(unique_clusters)
    if M == 0:
        raise ValueError("HDBSCAN found no clusters. Try reducing min_cluster_size.")

    cluster_idx = {c: i for i, c in enumerate(unique_clusters)}
    N  = len(labels)
    Pi = np.full((N, M), 1.0 / M)   # default: uniform (noise points)

    for i, (lbl, prob) in enumerate(zip(labels, soft_probs)):
        if lbl != -1:
            row = np.zeros(M)
            col = cluster_idx[lbl]
            row[col] = prob
            # Distribute remaining probability uniformly
            remainder = 1.0 - prob
            row += remainder / M
            Pi[i] = row

    return Pi


# ===========================================================================
# HDBSCAN wrapper  (Section 2.3.2)
# ===========================================================================

def run_hdbscan(
    D: np.ndarray,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int      = HDBSCAN_MIN_SAMPLES,
    leaf_size: int        = HDBSCAN_LEAF_SIZE,
) -> tuple[np.ndarray, np.ndarray, object]:
    """
    Run HDBSCAN on the precomputed cosine distance matrix (Equation 5).

    The mutual reachability distance is computed internally by HDBSCAN:

        d_mreach(i,j) = max{ core_k(i), core_k(j), d(i,j) }

    where core_k(i) is the cosine distance to the k-th nearest neighbour.

    Parameters
    ----------
    D                : ndarray (N × N) cosine distance matrix
    min_cluster_size : minimum cluster size (controls granularity)
    min_samples      : number of samples in neighbourhood for core-point test
    leaf_size        : leaf size for ball-tree (performance tuning)

    Returns
    -------
    labels     : ndarray (N,) integer labels (-1 = noise)
    soft_probs : ndarray (N,) membership probabilities
    clusterer  : fitted hdbscan.HDBSCAN object (for inspection / validation)
    """
    try:
        import hdbscan as hdbscan_lib
    except ImportError:
        raise ImportError("Install hdbscan:  pip install hdbscan --break-system-packages")

    clusterer = hdbscan_lib.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        leaf_size=leaf_size,
        cluster_selection_method="eom",
        prediction_data=False,
    )
    labels = clusterer.fit_predict(D)
    soft_probs = clusterer.probabilities_

    n_clusters = len(set(labels) - {-1})
    n_noise    = (labels == -1).sum()
    cluster_sizes = []
    if n_clusters > 0:
        cluster_sizes = sorted(
            [int((labels == c).sum()) for c in sorted(set(labels) - {-1})],
            reverse=True,
        )

    # Attach summary diagnostics so callers can inspect HDBSCAN similarly to
    # how SphericalKMeans exposes labels_ and inertia_.
    clusterer.labels_ = labels
    clusterer.soft_probs_ = soft_probs
    clusterer.n_clusters_ = n_clusters
    clusterer.n_noise_ = int(n_noise)
    clusterer.noise_fraction_ = float(n_noise / len(labels)) if len(labels) else 0.0
    clusterer.cluster_sizes_ = cluster_sizes

    if n_clusters == 0:
        warnings.warn(
            "HDBSCAN found 0 clusters. All points labelled noise. "
            "Try reducing min_cluster_size or min_samples.",
            RuntimeWarning,
        )
    else:
        pass  # caller can log n_clusters / n_noise if desired

    return labels, soft_probs, clusterer


# ===========================================================================
# Master clustering entry point
# ===========================================================================

def cluster_and_weight(
    U: np.ndarray,
    permnos: np.ndarray,
    method: str       = "spherical_kmeans",
    n_clusters: int   = N_CLUSTERS,
    tau: float        = TEMPERATURE,
    skm_kwargs: dict  = None,
    hdbscan_kwargs: dict = None,
) -> tuple[pd.DataFrame, object]:
    """
    Run clustering and return a soft membership DataFrame.

    Parameters
    ----------
    U          : ndarray (N × P), unit-normed firm embeddings
    permnos    : ndarray (N,), firm identifiers (index for output)
    method     : "spherical_kmeans" or "hdbscan"
    n_clusters : M (only used for spherical_kmeans)
    tau        : temperature for soft weights (spherical_kmeans)
    skm_kwargs : extra kwargs passed to SphericalKMeans
    hdbscan_kwargs : extra kwargs passed to run_hdbscan

    Returns
    -------
    Pi_df  : DataFrame (N × M), soft membership weights, index=permno
    model  : the fitted clustering object (SphericalKMeans or HDBSCAN clusterer)
    """
    skm_kwargs     = skm_kwargs     or {}
    hdbscan_kwargs = hdbscan_kwargs or {}

    if method == "spherical_kmeans":
        model = SphericalKMeans(n_clusters=n_clusters, **skm_kwargs)
        model.fit(U)
        Pi = compute_soft_weights_skm(U, model.centroids_, tau=tau)

    elif method == "hdbscan":
        from normalization import cosine_distance_matrix
        D = cosine_distance_matrix(U)
        labels, soft_probs, model = run_hdbscan(D, **hdbscan_kwargs)
        Pi = compute_soft_weights_hdbscan(labels, soft_probs)
        # Update n_clusters to what HDBSCAN found
        n_clusters = Pi.shape[1]

    else:
        raise ValueError(f"Unknown clustering method: '{method}'. "
                         "Choose 'spherical_kmeans' or 'hdbscan'.")

    col_names = [f"type_{m}" for m in range(n_clusters)]
    Pi_df = pd.DataFrame(Pi, index=permnos, columns=col_names)
    return Pi_df, model


def compute_cluster_prototypes(U: np.ndarray, Pi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build unit-norm cluster prototypes from soft memberships.

    Each prototype is the membership-weighted average embedding for that
    cluster, projected back onto the unit sphere.
    """
    if Pi_df.empty:
        return pd.DataFrame(index=[], columns=np.arange(U.shape[1]), dtype=float)

    weights = Pi_df.values.astype(np.float64)            # N x M
    valid = np.isfinite(U)
    U_filled = np.where(valid, U, 0.0)
    weight_sums = weights.T @ valid.astype(np.float64)
    proto = weights.T @ U_filled
    proto = np.divide(proto, weight_sums, out=np.full_like(proto, np.nan), where=weight_sums > 0)
    proto = _normalise_nan_rows(proto)
    return pd.DataFrame(proto, index=Pi_df.columns)


def align_cluster_labels(
    Pi_df: pd.DataFrame,
    U: np.ndarray,
    prev_prototypes: pd.DataFrame | None,
    next_type_id: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, int, dict[str, str]]:
    """
    Align current month cluster labels to the previous month's labels.

    Matching is done with cosine similarity between cluster prototypes and a
    Hungarian assignment. Unmatched current clusters receive new persistent
    labels `type_k`.
    """
    curr_prototypes = compute_cluster_prototypes(U, Pi_df)
    if prev_prototypes is None or prev_prototypes.empty or curr_prototypes.empty:
        rename_map = {
            col: f"type_{next_type_id + i}"
            for i, col in enumerate(Pi_df.columns)
        }
        Pi_aligned = Pi_df.rename(columns=rename_map)
        proto_aligned = curr_prototypes.rename(index=rename_map)
        return Pi_aligned, proto_aligned, next_type_id + len(rename_map), rename_map

    curr_cols = list(curr_prototypes.index)
    prev_cols = list(prev_prototypes.index)
    sims = masked_cosine_similarity(curr_prototypes.values, prev_prototypes.values)
    cost = 1.0 - sims
    row_ind, col_ind = linear_sum_assignment(cost)

    rename_map: dict[str, str] = {}
    used_prev: set[str] = set()
    for r, c in zip(row_ind, col_ind):
        curr = curr_cols[r]
        prev = prev_cols[c]
        rename_map[curr] = prev
        used_prev.add(prev)

    for curr in curr_cols:
        if curr not in rename_map:
            rename_map[curr] = f"type_{next_type_id}"
            next_type_id += 1

    Pi_aligned = Pi_df.rename(columns=rename_map)
    proto_aligned = curr_prototypes.rename(index=rename_map)

    # If multiple current clusters collapse onto the same persistent label,
    # aggregate them before downstream use.
    Pi_aligned = Pi_aligned.T.groupby(level=0).sum().T
    proto_aligned = proto_aligned.groupby(level=0).mean()
    proto_aligned.loc[:, :] = _normalise_nan_rows(proto_aligned.values)

    return Pi_aligned, proto_aligned, next_type_id, rename_map
