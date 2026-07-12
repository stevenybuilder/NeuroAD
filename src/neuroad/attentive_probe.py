"""
attentive_probe — the NeuroVFM-style probe, adapted honestly to our substrate.

NeuroVFM (Nat Med 2026, the generalist neuroimaging JEPA model) freezes its
Vol-JEPA encoder and trains **attentive MLP probes** for downstream diagnosis —
an MLP class head plus a learned attention over image instances. We chose this
over a U-Net decoder because it matches our frozen-NeuroJEPA substrate, upgrades
the probe we already have, adds interpretable grounding, and needs no new data.

HONEST ADAPTATION (paramount): our NeuroJEPA embedding is a single POOLED 768-d
vector per subject, NOT per-patch tokens, so we cannot reproduce literal spatial
attention MAPS (that would need per-patch embeddings we do not extract). What
this module delivers instead:

  1. ``MLPProbe`` — a nonlinear MLP classification head over the frozen embedding,
     run through the IDENTICAL leakage-honest machinery as the linear probe
     (StandardScaler + automatic PCA front-end fit inside each fold, site-disjoint
     CV, bootstrap CI, within-site permutation null, repeated-CV ensembling). It is
     a drop-in ``probe_factory`` for ``probe.auc_ci_perm`` / ``probe.cross_val_auc``.
  2. ``feature_grounding`` — leave-one-group-out ATTRIBUTION over the embedding
     block and each NAMED clinical feature (p_tau217, gfap, nfl, apoe4, age): the
     drop in cross-validated AUC when a group is removed. This is the interpretable
     "what drives the signal" story — grounded in named features, honestly labelled
     as attribution (not a learned spatial attention map).

Pure sklearn/numpy: no torch, no network, deterministic — same offline contract
as ``probe``. At our n (hundreds, not millions) the MLP is deliberately small and
strongly regularized, and the PCA front-end keeps its input low-dimensional, so it
does not overfit relative to the linear head — we REPORT the comparison rather than
assume the nonlinear head wins.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from . import contract, probe

#: Small, strongly-regularized head — sized for n in the hundreds, not millions.
DEFAULT_HIDDEN = (32,)
DEFAULT_ALPHA = 1.0            # L2 penalty; high on purpose (thin data)
DEFAULT_MAX_ITER = 800

#: Named clinical features used for interpretable grounding (when present).
GROUNDING_FEATURES = ("p_tau217", "gfap", "nfl", "apoe4", "age")


class MLPProbe:
    """Nonlinear MLP head with the SAME pipeline contract as ``probe.LinearProbe``.

    StandardScaler + automatic PCA front-end (reused from ``probe`` so p>>n
    embeddings are reduced exactly as the linear probe reduces them) + a small
    regularized ``MLPClassifier``. Exposes ``.fit`` / ``.predict_proba`` /
    ``.decision_scores`` / ``.classes_`` so it plugs into every leakage-honest CV
    helper in ``probe`` via the ``probe_factory`` hook.
    """

    def __init__(self, *, hidden_layer_sizes=DEFAULT_HIDDEN,
                 alpha: float = DEFAULT_ALPHA, max_iter: int = DEFAULT_MAX_ITER,
                 random_state: int = probe.RANDOM_STATE,
                 reduce: str | None = "auto", n_components: Optional[int] = None,
                 whiten: bool = True):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.random_state = random_state
        self.reduce = reduce
        self.n_components = n_components
        self.whiten = whiten
        self.pipeline: Optional[Pipeline] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_components_: Optional[int] = None

    def _build(self, n_samples: int, n_features: int) -> Pipeline:
        if self.n_components is not None:
            n_comp: Optional[int] = int(min(self.n_components, n_features,
                                            max(2, n_samples - 1)))
        elif self.reduce == "auto":
            n_comp = probe.auto_n_components(n_samples, n_features)
        else:
            n_comp = None
        self.n_components_ = n_comp
        steps = [("scale", StandardScaler())]
        if n_comp is not None:
            steps.append(("pca", PCA(n_components=n_comp, whiten=self.whiten,
                                     random_state=self.random_state)))
        steps.append(("clf", MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
            max_iter=self.max_iter, random_state=self.random_state,
            early_stopping=False)))
        return Pipeline(steps)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPProbe":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.pipeline = self._build(X.shape[0], X.shape[1])
        self.pipeline.fit(X, y)
        self.classes_ = self.pipeline.named_steps["clf"].classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.pipeline is not None, "call .fit() first"
        return self.pipeline.predict_proba(np.asarray(X, dtype=float))

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)


def _mlp_factory():
    """Zero-arg factory so ``probe.*`` CV helpers can build fresh MLP heads."""
    return MLPProbe()


# ---------------------------------------------------------------------------
# Evaluation: MLP head vs linear, through the identical honest machinery
# ---------------------------------------------------------------------------


def evaluate(df: pd.DataFrame, target: str = "dx_binary", *,
             n_repeats: int = probe.N_REPEATS_ENSEMBLE,
             n_boot: int = probe.N_BOOT, n_perm: int = probe.N_PERM) -> dict:
    """Site-disjoint, ensembled AUC/CI/p for the MLP head, next to the linear one.

    Points the reused head at ``target`` (``point_head`` gives the frozen
    embedding matrix + site groups) and runs BOTH the linear probe and the MLP
    probe through ``probe.auc_ci_perm`` so the comparison is apples-to-apples
    (same folds, same bootstrap/permutation). Returns a JSON-safe dict; the
    verdict states honestly whether the nonlinear head actually helps at this n.
    """
    X, y, groups = probe.point_head(df, target)
    linear = probe.auc_ci_perm(X, y, groups, n_boot=n_boot, n_perm=n_perm,
                               n_repeats=n_repeats)
    mlp = probe.auc_ci_perm(X, y, groups, n_boot=n_boot, n_perm=n_perm,
                            n_repeats=n_repeats, probe_factory=_mlp_factory)
    la, ma = linear.get("auc"), mlp.get("auc")
    delta = None if (la is None or ma is None) else round(float(ma) - float(la), 4)
    if not mlp.get("ci_excludes_chance"):
        verdict = "MLP head: AUC not distinguishable from chance (95% CI includes 0.5)"
    elif delta is None:
        verdict = "MLP vs linear: comparison unavailable (thin data)"
    elif delta > 0 and (mlp.get("ci_lo") or 0) > (linear.get("auc") or 1):
        verdict = (f"MLP head improves on linear (delta=+{delta:.4f}; MLP CI lower "
                   f"bound clears the linear point estimate)")
    elif abs(delta) <= 0.01:
        verdict = (f"MLP head matches the linear probe (delta={delta:+.4f}) — the "
                   f"frozen embedding is already near-linearly separable at this n")
    elif delta < 0:
        verdict = (f"MLP head does not beat linear (delta={delta:+.4f}) — nonlinearity "
                   f"does not pay off at this n; prefer the simpler linear head")
    else:
        verdict = (f"MLP head slightly ahead of linear (delta=+{delta:.4f}) but CIs "
                   f"overlap — treat as suggestive")
    return {
        "target": target,
        "n": int(mlp.get("n", len(y))),
        "linear": {k: linear.get(k) for k in
                   ("auc", "ci_lo", "ci_hi", "p_perm", "ci_excludes_chance")},
        "mlp": {k: mlp.get(k) for k in
                ("auc", "ci_lo", "ci_hi", "p_perm", "ci_excludes_chance")},
        "delta_auc_mlp_minus_linear": delta,
        "verdict": verdict,
        "head": {"hidden_layer_sizes": list(DEFAULT_HIDDEN), "alpha": DEFAULT_ALPHA},
    }


# ---------------------------------------------------------------------------
# Interpretable grounding: leave-one-group-out attribution over named features
# ---------------------------------------------------------------------------


def feature_grounding(df: pd.DataFrame, target: str = "dx_binary", *,
                      n_repeats: int = probe.N_REPEATS_ENSEMBLE,
                      use_mlp: bool = False) -> dict:
    """What drives the signal: LOO-AUC drop for the embedding + each named feature.

    Builds a fused matrix of the frozen embedding block plus each available named
    clinical feature, gets the full-model site-disjoint ensembled AUC, then removes
    each group in turn and reports the AUC drop (attribution). Larger drop == the
    group carries more of the AD-vs-CN signal. This is the interpretable
    "grounding" — over NAMED features, honestly attribution (not a learned spatial
    attention map, which our pooled embedding cannot provide).

    Uses the fast LINEAR head by default (attribution — which group carries signal
    — does not need the MLP nonlinearity, and the linear head is ~30x cheaper);
    set ``use_mlp=True`` to attribute through the nonlinear head. Degrades to
    ``{}`` when the slice is too thin or the target is unusable.
    """
    factory = _mlp_factory if use_mlp else None
    if target != "dx_binary":
        # grounding is defined for the AD/CN contrast (named biomarkers apply there)
        target = "dx_binary"
    dx = df["dx"].astype("string").map({"AD": 1, "CN": 0})
    mask = dx.notna().to_numpy()
    if mask.sum() < 8 or dx[mask].nunique() < 2:
        return {}
    emb = contract.embedding_matrix(df)[mask]
    y = dx[mask].to_numpy(dtype=int)
    site = df["site"].astype("string").fillna("__na__").to_numpy()[mask]
    groups = np.unique(site, return_inverse=True)[1]

    # Named feature columns actually present + non-degenerate on this slice.
    named: dict[str, np.ndarray] = {}
    for col in GROUNDING_FEATURES:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)[mask]
            if np.isfinite(v).sum() >= 0.5 * len(v) and np.nanstd(v) > 0:
                named[col] = np.nan_to_num(v, nan=float(np.nanmean(v)))

    blocks: dict[str, np.ndarray] = {"embedding": emb}
    for k, v in named.items():
        blocks[k] = v.reshape(-1, 1)

    def _auc(block_names: list[str]) -> float:
        Xf = np.hstack([blocks[b] for b in block_names])
        return float(probe.cross_val_auc(Xf, y, groups=groups, n_repeats=n_repeats,
                                         probe_factory=factory))

    all_names = list(blocks.keys())
    full = _auc(all_names)
    attribution = []
    for b in all_names:
        rest = [n for n in all_names if n != b]
        drop = round(full - _auc(rest), 4) if rest else None
        attribution.append({"group": b, "loo_auc_drop": drop})
    attribution.sort(key=lambda r: (r["loo_auc_drop"] is not None,
                                    r["loo_auc_drop"] or 0.0), reverse=True)
    top = attribution[0]["group"] if attribution else ""
    return {
        "target": "dx_binary",
        "n": int(mask.sum()),
        "full_auc": round(full, 4),
        "groups": all_names,
        "attribution": attribution,
        "top_driver": top,
        "note": ("leave-one-group-out AUC attribution over the frozen embedding + "
                 "named clinical features (interpretable grounding). Attribution, "
                 "NOT a learned spatial attention map — the pooled NeuroJEPA "
                 "embedding has no per-region tokens."),
    }
