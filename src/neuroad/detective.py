"""
The Detective — unsupervised phenotype discovery entry point.

The same frozen embedding the probe reads can be clustered with no labels at
all. `discover()` returns cluster assignments plus 2-D PCA coordinates the UI
can scatter, so every gauntlet test can subsequently be re-run *per cluster*
(does a putative subtype survive its own leakage / brain-age challenge?).

Grounded methodology (REDUCE-THEN-CLUSTER):
  * Euclidean distance degenerates in high dimension, so we PCA-whiten the
    embedding to ``min(20, D-1, n//5)`` dimensions BEFORE any clustering.
  * Methods: 'kmeans' (k by silhouette), 'gmm' (GaussianMixture, k by min BIC),
    and the scikit-learn 'hdbscan' density fallback. All ship with scikit-learn
    — no UMAP / external HDBSCAN dependency that could jeopardize the offline
    build.
  * STABILITY is the primary quality gate: `bootstrap_stability` computes each
    cluster's mean Jaccard over 80%-resample refits (greedy overlap match). A
    cluster with mean Jaccard < ~0.6 is unstable / noise, not a phenotype.
  * `trustworthiness` of the 2-D projection accompanies the coords so the UI can
    say how faithful the scatter is.

`discover()` returns:
  {labels, coords_2d, k, method, silhouette, stability, trustworthiness}.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness as _trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from . import contract

RANDOM_STATE = 0

#: Mean-Jaccard floor below which a cluster is treated as unstable / noise.
STABILITY_FLOOR = 0.60


# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------
def _n_reduced_dims(n: int, D: int) -> int:
    """min(20, D-1, n//5), floored at 2 — the reduce-then-cluster budget."""
    return max(2, min(20, D - 1, n // 5))


def _reduce(Xs: np.ndarray, n_dims: int) -> np.ndarray:
    """PCA-whiten to ``n_dims`` so Euclidean distance is well-conditioned."""
    n_dims = min(n_dims, Xs.shape[1], max(1, Xs.shape[0] - 1))
    return PCA(n_components=n_dims, whiten=True,
               random_state=RANDOM_STATE).fit_transform(Xs)


# ---------------------------------------------------------------------------
# Cluster fitters (operate on the REDUCED space)
# ---------------------------------------------------------------------------
def _fit_labels(X_red: np.ndarray, k: int, method: str = "kmeans") -> np.ndarray:
    if method == "gmm":
        # Diagonal covariance: far fewer parameters than 'full', which keeps BIC
        # honest in the whitened (noise-amplified) space where full covariance
        # overfits the isotropic directions.
        gm = GaussianMixture(n_components=k, covariance_type="diag",
                             n_init=1, random_state=RANDOM_STATE)
        return gm.fit_predict(X_red)
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    return km.fit_predict(X_red)


def _select_k_kmeans(X_red: np.ndarray, lo: int, hi: int):
    best_k, best_sil, best_labels = 1, -1.0, np.zeros(len(X_red), dtype=int)
    for k in range(max(lo, 2), hi + 1):
        labels = _fit_labels(X_red, k, "kmeans")
        if len(np.unique(labels)) < 2:
            continue
        sil = float(silhouette_score(X_red, labels))
        if sil > best_sil:
            best_k, best_sil, best_labels = k, sil, labels
    return best_k, best_sil, best_labels


def _select_k_gmm(X_red: np.ndarray, lo: int, hi: int):
    """k by MINIMUM BIC; silhouette reported for the chosen fit."""
    best_k, best_bic, best_labels = 1, np.inf, np.zeros(len(X_red), dtype=int)
    for k in range(max(lo, 2), hi + 1):
        gm = GaussianMixture(n_components=k, covariance_type="diag",
                             n_init=1, random_state=RANDOM_STATE).fit(X_red)
        bic = float(gm.bic(X_red))
        if bic < best_bic:
            best_k, best_bic = k, bic
            best_labels = gm.predict(X_red)
    if len(np.unique(best_labels)) < 2:
        return 1, None, best_labels
    sil = float(silhouette_score(X_red, best_labels))
    return best_k, sil, best_labels


# ---------------------------------------------------------------------------
# Stability — the primary quality gate
# ---------------------------------------------------------------------------
def _greedy_jaccard(ref_labels: np.ndarray, sub_labels: np.ndarray,
                    sub_idx: np.ndarray, ref_clusters: np.ndarray) -> dict:
    """Greedy one-to-one overlap match between the reference clustering and a
    bootstrap clustering, returning per-reference-cluster Jaccard on this fold."""
    ref_on_sub = ref_labels[sub_idx]
    sub_clusters = np.unique(sub_labels)
    # Jaccard matrix (ref x sub) restricted to the bootstrap sample.
    J = np.zeros((len(ref_clusters), len(sub_clusters)))
    ref_sets = [set(np.where(ref_on_sub == c)[0]) for c in ref_clusters]
    sub_sets = [set(np.where(sub_labels == c)[0]) for c in sub_clusters]
    for i, a in enumerate(ref_sets):
        for j, b in enumerate(sub_sets):
            union = len(a | b)
            J[i, j] = (len(a & b) / union) if union else 0.0
    # Greedy: repeatedly take the globally-best remaining (ref, sub) pair.
    out = {int(c): 0.0 for c in ref_clusters}
    used_ref, used_sub = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(-J, axis=None), J.shape))[0]
    for i, j in order:
        if i in used_ref or j in used_sub:
            continue
        out[int(ref_clusters[i])] = float(J[i, j])
        used_ref.add(i)
        used_sub.add(j)
        if len(used_ref) == len(ref_clusters) or len(used_sub) == len(sub_clusters):
            break
    return out


def bootstrap_stability(X_red: np.ndarray, k: int, B: int = 50,
                        method: str = "kmeans",
                        frac: float = 0.8, random_state: int = RANDOM_STATE) -> dict:
    """Per-cluster mean Jaccard over ``B`` frac-resample refits.

    For each bootstrap we cluster an 80% subsample into ``k`` clusters, greedily
    match those clusters to the full-data reference clustering by overlap, and
    record each reference cluster's Jaccard with its best match. The mean over
    bootstraps is that cluster's stability; < ~0.6 flags it unstable / noise.

    Returns {cluster_id: mean_jaccard}.
    """
    n = len(X_red)
    if n < 8 or k < 2:
        return {}
    ref_labels = _fit_labels(X_red, k, method)
    ref_clusters = np.unique(ref_labels)
    acc = {int(c): [] for c in ref_clusters}
    rng = np.random.default_rng(random_state)
    size = max(k + 1, int(round(frac * n)))
    for _ in range(B):
        idx = rng.choice(n, size=size, replace=False)
        sub = X_red[idx]
        if len(np.unique(_fit_labels(sub, min(k, len(idx) - 1), method))) < 2:
            continue
        sub_labels = _fit_labels(sub, k, method)
        fold = _greedy_jaccard(ref_labels, sub_labels, idx, ref_clusters)
        for c, v in fold.items():
            acc[c].append(v)
    return {c: (round(float(np.mean(v)), 3) if v else None) for c, v in acc.items()}


# ---------------------------------------------------------------------------
# Density fallback
# ---------------------------------------------------------------------------
def _hdbscan(X_red: np.ndarray, coords: np.ndarray) -> dict:
    from sklearn.cluster import HDBSCAN
    min_size = max(3, len(X_red) // 20)
    labels = HDBSCAN(min_cluster_size=min_size).fit_predict(X_red)
    real = labels[labels >= 0]
    k = int(len(np.unique(real))) if len(real) else 0
    sil = None
    if k >= 2:
        try:
            sil = round(float(silhouette_score(X_red[labels >= 0], real)), 3)
        except ValueError:
            sil = None
    stability = {}
    return {"labels": labels, "coords_2d": coords, "k": k,
            "method": "hdbscan", "silhouette": sil,
            "stability": stability, "trustworthiness": None}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def discover(df: pd.DataFrame, k_range: tuple[int, int] = (2, 6),
             method: str = "kmeans", B: int = 50) -> dict:
    """Reduce-then-cluster the embedding matrix and return UI-ready structure.

    Returns {'labels','coords_2d','k','method','silhouette','stability',
    'trustworthiness'}:
      * labels          -> (n,) int cluster ids (HDBSCAN noise = -1),
      * coords_2d       -> (n, 2) PCA coordinates for scatter plots,
      * k               -> number of clusters found,
      * method          -> 'kmeans' | 'gmm' | 'hdbscan' | 'none',
      * silhouette      -> quality score on the reduced space (None if undefined),
      * stability       -> {cluster_id: mean bootstrap Jaccard} (primary gate),
      * trustworthiness -> faithfulness of the 2-D projection (None if undefined).
    """
    X = contract.embedding_matrix(df)
    n = len(X)
    Xs = StandardScaler().fit_transform(X) if n else X

    # 2-D coordinates for the UI (unwhitened — preserves relative spread).
    if n >= 2 and X.shape[1] >= 2:
        coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xs)
    else:
        coords = np.zeros((n, 2))

    if n < 4:
        return {"labels": np.zeros(n, dtype=int), "coords_2d": coords,
                "k": 1, "method": "none", "silhouette": None,
                "stability": {}, "trustworthiness": None}

    # REDUCE (PCA-whiten) before clustering.
    X_red = _reduce(Xs, _n_reduced_dims(n, X.shape[1]))

    if method == "hdbscan":
        return _hdbscan(X_red, coords)

    lo, hi = k_range
    hi = min(hi, n - 1)
    if hi < 2:
        return _hdbscan(X_red, coords)

    if method == "gmm":
        best_k, best_sil, best_labels = _select_k_gmm(X_red, lo, hi)
    else:
        method = "kmeans"
        best_k, best_sil, best_labels = _select_k_kmeans(X_red, lo, hi)

    if best_k < 2:  # never resolved -> density fallback
        return _hdbscan(X_red, coords)

    stability = bootstrap_stability(X_red, best_k, B=B, method=method)

    trust = None
    try:
        n_neighbors = int(min(10, max(2, (n - 1) // 3)))
        trust = round(float(_trustworthiness(X_red, coords,
                                             n_neighbors=n_neighbors)), 3)
    except (ValueError, IndexError):
        trust = None

    return {"labels": best_labels, "coords_2d": coords, "k": int(best_k),
            "method": method,
            "silhouette": None if best_sil is None else round(best_sil, 3),
            "stability": stability, "trustworthiness": trust}
