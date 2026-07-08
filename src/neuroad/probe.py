"""
The ONE reused head — a standardized linear probe on frozen embeddings.

Point the same head at any `LABEL_TARGETS` column and it becomes a different
tool:
  * conversion / dx_binary -> the candidate signal,
  * site / scanner        -> the STAR leakage test (same code, different label).

Nothing here ever uses a label-defining column as a feature: the features are
always the embedding matrix (`contract.embedding_matrix`). Metadata columns
(age, sex, site, scanner, biomarkers) are targets or covariates, never inputs
to the probe.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import contract

RANDOM_STATE = 0


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------
class LinearProbe:
    """Standardized logistic-regression head over an embedding matrix.

    A thin wrapper so every module speaks the same `.fit / .predict_proba /
    .decision_scores` contract regardless of how many classes the target has.
    """

    def __init__(self, C: float = 1.0, random_state: int = RANDOM_STATE):
        self.C = C
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                C=self.C, max_iter=2000, random_state=self.random_state,
                class_weight="balanced")),
        ])
        self.pipeline.fit(X, y)
        self.classes_ = self.pipeline.named_steps["clf"].classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.pipeline is not None, "call .fit() first"
        return self.pipeline.predict_proba(np.asarray(X, dtype=float))

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        """A 1-D score usable for ROC / correlation.

        Binary  -> P(positive class).  Multiclass -> max class probability.
        """
        proba = self.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba.max(axis=1)


# ---------------------------------------------------------------------------
# Pointing the head at a label
# ---------------------------------------------------------------------------
def _encode(y: np.ndarray) -> np.ndarray:
    """Map arbitrary label values to contiguous 0..k-1 integer codes."""
    classes = np.unique(y)
    return np.searchsorted(classes, y)


def point_head(df: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Extract (X, y, groups) for a `LABEL_TARGETS` key.

    * X       -> the embedding matrix (never a label-defining column).
    * y       -> integer class codes for the requested target.
    * groups  -> site codes for subject/site-disjoint CV on OUTCOME targets;
                 None for site/scanner targets (grouping by the thing you are
                 trying to predict would be degenerate).

    Rows with a missing target (and MCI for dx_binary) are dropped, and X / y /
    groups are filtered in lockstep so callers get aligned arrays.
    """
    if target not in contract.LABEL_TARGETS:
        raise ValueError(
            f"target {target!r} not in LABEL_TARGETS {list(contract.LABEL_TARGETS)}")

    X_all = contract.embedding_matrix(df)

    if target == "conversion":
        raw = pd.to_numeric(df["conversion"], errors="coerce")
        mask = raw.notna().to_numpy()
        y = raw[mask].to_numpy(dtype=int)
    elif target == "dx_binary":
        dx = df["dx"].astype("string")
        mapping = {"AD": 1, "CN": 0}
        mapped = dx.map(mapping)
        mask = mapped.notna().to_numpy()
        y = mapped[mask].to_numpy(dtype=int)
    elif target in ("site", "scanner"):
        col = df[target].astype("string")
        mask = col.notna().to_numpy()
        y = _encode(col[mask].to_numpy())
    else:  # pragma: no cover - guarded above
        raise ValueError(target)

    X = X_all[mask]

    if target in ("site", "scanner"):
        groups: Optional[np.ndarray] = None
    else:
        site = df["site"].astype("string")
        # Rows with a missing site fall back to their own singleton group.
        site_filled = site.fillna("__na__").to_numpy()[mask]
        groups = _encode(site_filled)

    return X, y, groups


# ---------------------------------------------------------------------------
# Cross-validated ROC-AUC (subject/site-disjoint when groups are given)
# ---------------------------------------------------------------------------
def _n_splits(y_codes: np.ndarray, groups: Optional[np.ndarray], cap: int = 5) -> int:
    counts = np.bincount(y_codes)
    counts = counts[counts > 0]
    n = int(min(cap, counts.min()))
    if groups is not None:
        n = int(min(n, len(np.unique(groups))))
    return max(n, 2)


def cross_val_auc(X: np.ndarray, y: np.ndarray,
                  groups: Optional[np.ndarray] = None) -> float:
    """Cross-validated ROC-AUC using out-of-fold probabilities.

    * groups given  -> StratifiedGroupKFold (subject/site-disjoint folds).
    * groups None   -> StratifiedKFold.
    Binary -> standard AUC; multiclass -> macro one-vs-rest AUC.
    Returns 0.5 (chance) when the data is too thin to evaluate honestly.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) < 2 or len(y) < 4:
        return 0.5
    y_codes = np.searchsorted(classes, y)

    n_splits = _n_splits(y_codes, groups)
    if n_splits < 2:
        return 0.5

    # If group-aware CV is infeasible (too few groups), fall back gracefully.
    use_groups = groups is not None and len(np.unique(groups)) >= n_splits
    if use_groups:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=RANDOM_STATE)
        split_iter = splitter.split(X, y_codes, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=RANDOM_STATE)
        split_iter = splitter.split(X, y_codes)

    oof = np.full((len(y), len(classes)), np.nan)
    for tr, te in split_iter:
        if len(np.unique(y_codes[tr])) < 2:
            continue
        probe = LinearProbe().fit(X[tr], y_codes[tr])
        proba = probe.predict_proba(X[te])
        # Align model's (possibly partial) class set into the global columns.
        for j, cls in enumerate(probe.classes_):
            oof[te, cls] = proba[:, j]

    evaluated = ~np.isnan(oof).any(axis=1)
    if evaluated.sum() < 2 or len(np.unique(y_codes[evaluated])) < 2:
        return 0.5
    yv = y_codes[evaluated]
    pv = oof[evaluated]
    try:
        if len(classes) == 2:
            return float(roc_auc_score(yv, pv[:, 1]))
        # renormalize rows so ovr macro AUC is well-defined
        pv = pv / pv.sum(axis=1, keepdims=True)
        return float(roc_auc_score(yv, pv, multi_class="ovr",
                                   average="macro", labels=np.arange(len(classes))))
    except ValueError:
        return 0.5
