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

from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata as sstats_rankdata

from . import contract

RANDOM_STATE = 0

# ---------------------------------------------------------------------------
# Automatic PCA/whitening front-end for p >> n cohorts.
#
# A logistic head on a full 768-d frozen-FM embedding at n = 60-100 subjects is
# in the p >> n regime and its cross-validated AUC is optimistically inflated
# (the standalone Neuro-JEPA leakage report ships raw-768d AUC ~0.998 and flags
# it as inflated). The DEFENSIBLE number the researcher-track scripts already
# trust is the PCA-reduced one (PCA-10 whitened -> AUC ~0.93). To make the
# gauntlet + leakage_margin report that SAME defensible number instead of the
# inflated one, `LinearProbe` / `cross_val_auc` transparently prepend a
# PCA(whiten=True) step whenever the training matrix is p >> n (n < k * D and
# D > cap). Low-dimensional cohorts (the tabular 3-4 feature blocks, the D=16
# synthetic test cohorts at n >> D) are left untouched, so nothing changes for
# them. The PCA is fit INSIDE each CV fold (on the training rows only), so the
# reduction never leaks test-fold variance.
# ---------------------------------------------------------------------------
#: Cap on retained components — matches the PCA-10 the standalone scripts trust.
PCA_MAX_COMPONENTS = 10
#: Reduce only when n_samples < PCA_TRIGGER_K * n_features (clear p >> n). At
#: k=2 the D=16 synthetic test cohorts (n >= ~60) and every tabular block are
#: left at full dimensionality; only the 768-d frozen-FM embeddings reduce.
PCA_TRIGGER_K = 2


def auto_n_components(n_samples: int, n_features: int,
                      cap: int = PCA_MAX_COMPONENTS,
                      k: int = PCA_TRIGGER_K) -> Optional[int]:
    """Number of PCA components to retain for a p >> n probe, or None to skip.

    Returns None (no reduction) when the cohort is already low-dimensional
    (``n_features <= cap``) or has enough samples per feature
    (``n_samples >= k * n_features``). Otherwise returns
    ``min(cap, n_features, n_samples - 1)`` so the reduced space is always
    well-posed inside a CV fold.
    """
    if n_features <= cap:
        return None
    if n_samples >= k * n_features:
        return None
    return int(min(cap, n_features, max(2, n_samples - 1)))


def build_probe_pipeline(n_samples: int, n_features: int, C: float = 1.0,
                         random_state: int = RANDOM_STATE,
                         reduce: str | None = "auto",
                         n_components: Optional[int] = None,
                         whiten: bool = True) -> tuple[Pipeline, Optional[int]]:
    """Assemble the standardized probe pipeline, with an optional PCA front-end.

    ``reduce="auto"`` selects PCA automatically for p >> n cohorts (see
    :func:`auto_n_components`); ``n_components`` forces a fixed reduction;
    ``reduce=None`` disables it. Returns ``(pipeline, n_components_used)`` where
    ``n_components_used`` is None when no PCA step was added.
    """
    if n_components is not None:
        n_comp: Optional[int] = int(min(n_components, n_features, max(2, n_samples - 1)))
    elif reduce == "auto":
        n_comp = auto_n_components(n_samples, n_features)
    else:
        n_comp = None
    steps = [("scale", StandardScaler())]
    if n_comp is not None:
        steps.append(("pca", PCA(n_components=n_comp, whiten=whiten,
                                 random_state=random_state)))
    steps.append(("clf", LogisticRegression(
        C=C, max_iter=2000, random_state=random_state, class_weight="balanced")))
    return Pipeline(steps), n_comp


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------
class LinearProbe:
    """Standardized logistic-regression head over an embedding matrix.

    A thin wrapper so every module speaks the same `.fit / .predict_proba /
    .decision_scores` contract regardless of how many classes the target has.
    """

    def __init__(self, C: float = 1.0, random_state: int = RANDOM_STATE,
                 reduce: str | None = "auto", n_components: Optional[int] = None,
                 whiten: bool = True):
        self.C = C
        self.random_state = random_state
        self.reduce = reduce
        self.n_components = n_components
        self.whiten = whiten
        self.pipeline: Optional[Pipeline] = None
        self.classes_: Optional[np.ndarray] = None
        #: Number of PCA components actually used (None -> no reduction).
        self.n_components_: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.pipeline, self.n_components_ = build_probe_pipeline(
            X.shape[0], X.shape[1], C=self.C, random_state=self.random_state,
            reduce=self.reduce, n_components=self.n_components, whiten=self.whiten)
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


#: Default number of repeated CV splits to average over. 1 preserves the
#: historical single-split behaviour byte-for-byte; the referee-facing entry
#: points can opt into a small ensemble (see ``N_REPEATS_ENSEMBLE``) to damp the
#: split-seed variance that a single fold assignment carries at small n.
N_REPEATS = 1
#: The ensemble size the fusion/referee path uses when it wants the stabilized
#: number — the small-n analog of NeuroVFM's "ensemble the trained probes".
N_REPEATS_ENSEMBLE = 8


def _oof_proba_once(X: np.ndarray, y_codes: np.ndarray, classes: np.ndarray,
                    groups: Optional[np.ndarray], seed: int,
                    probe_factory: Optional[Callable] = None) -> Optional[np.ndarray]:
    """One stratified (site-disjoint when grouped) CV pass.

    Returns a full-length ``(n_rows, n_classes)`` OOF probability matrix (NaN on
    rows that never received a prediction), or ``None`` if the split is infeasible.
    The PCA front-end is fit inside each training fold only, so no test-fold
    variance leaks. Factored out of :func:`cross_val_oof` so repeated splits with
    different ``seed`` values can be averaged.

    ``probe_factory`` is a zero-arg callable returning a fresh probe with the
    ``.fit`` / ``.predict_proba`` / ``.classes_`` contract (default
    :class:`LinearProbe`); pass :class:`~neuroad.attentive_probe.MLPProbe` to run
    the nonlinear attentive head through the identical leakage-honest machinery.
    """
    make_probe = probe_factory or LinearProbe
    n_splits = _n_splits(y_codes, groups)
    if n_splits < 2:
        return None
    use_groups = groups is not None and len(np.unique(groups)) >= n_splits
    if use_groups:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=seed)
        split_iter = splitter.split(X, y_codes, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=seed)
        split_iter = splitter.split(X, y_codes)

    oof = np.full((len(y_codes), len(classes)), np.nan)
    for tr, te in split_iter:
        if len(np.unique(y_codes[tr])) < 2:
            continue
        probe = make_probe().fit(X[tr], y_codes[tr])
        proba = probe.predict_proba(X[te])
        # Align model's (possibly partial) class set into the global columns.
        for j, cls in enumerate(probe.classes_):
            oof[te, cls] = proba[:, j]
    return oof


def cross_val_oof(X: np.ndarray, y: np.ndarray,
                  groups: Optional[np.ndarray] = None,
                  n_repeats: int = N_REPEATS,
                  probe_factory: Optional[Callable] = None):
    """Out-of-fold probabilities for a probe over ``X``.

    Returns ``(y_codes_eval, proba_eval, classes, groups_eval)`` where the
    ``*_eval`` arrays are restricted to the rows that received a full OOF
    prediction, or ``None`` when the data is too thin to evaluate honestly.
    The probe uses the SAME automatic PCA front-end as :class:`LinearProbe`,
    fit inside each training fold only.

    ``n_repeats`` runs the whole site-disjoint CV that many times with distinct
    split seeds (``RANDOM_STATE + r``) and averages each subject's OOF
    probability across the repeats it was evaluated in. ``n_repeats=1`` (the
    default) reproduces the historical single-split result exactly; a small
    ensemble (e.g. :data:`N_REPEATS_ENSEMBLE`) damps the split-seed variance a
    single fold assignment carries at small n — the small-cohort analog of
    NeuroVFM's probe ensembling, without any calibration machinery that would
    overfit at n≈60.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) < 2 or len(y) < 4:
        return None
    y_codes = np.searchsorted(classes, y)

    n_reps = max(1, int(n_repeats))
    proba_sum = np.zeros((len(y), len(classes)), dtype=float)
    proba_cnt = np.zeros((len(y), len(classes)), dtype=float)
    for r in range(n_reps):
        oof_r = _oof_proba_once(X, y_codes, classes, groups, seed=RANDOM_STATE + r,
                                probe_factory=probe_factory)
        if oof_r is None:
            if r == 0:
                return None
            continue
        seen = ~np.isnan(oof_r)
        proba_sum[seen] += oof_r[seen]
        proba_cnt[seen] += 1.0

    with np.errstate(invalid="ignore"):
        oof = np.where(proba_cnt > 0, proba_sum / proba_cnt, np.nan)

    evaluated = ~np.isnan(oof).any(axis=1)
    if evaluated.sum() < 2 or len(np.unique(y_codes[evaluated])) < 2:
        return None
    groups_eval = None if groups is None else np.asarray(groups)[evaluated]
    return y_codes[evaluated], oof[evaluated], classes, groups_eval


def _fast_binary_auc(y: np.ndarray, score: np.ndarray) -> Optional[float]:
    """Rank-based binary ROC-AUC (Mann-Whitney U), tie-corrected.

    Equivalent to ``roc_auc_score`` for the binary case but ~10-50x cheaper per
    call, which matters inside the bootstrap/permutation loops (thousands of
    evaluations). Uses average ranks so ties are handled exactly as sklearn does.
    """
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = sstats_rankdata(score)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _auc_from_oof(y_codes: np.ndarray, proba: np.ndarray,
                  classes: np.ndarray) -> Optional[float]:
    """ROC-AUC from an OOF (y_codes, proba) pair; None if undefined."""
    if len(np.unique(y_codes)) < 2:
        return None
    if len(classes) == 2:
        return _fast_binary_auc(y_codes, proba[:, 1])
    # Macro one-vs-rest == mean of per-class (class c vs rest) binary AUCs, which
    # matches sklearn's roc_auc_score(multi_class="ovr", average="macro") but is
    # far cheaper per call (no per-call validation) — critical in the boot loops.
    p = proba / proba.sum(axis=1, keepdims=True)
    aucs = []
    for c in range(len(classes)):
        a = _fast_binary_auc((y_codes == c).astype(int), p[:, c])
        if a is None:
            return None
        aucs.append(a)
    return float(np.mean(aucs)) if aucs else None


def cross_val_auc(X: np.ndarray, y: np.ndarray,
                  groups: Optional[np.ndarray] = None,
                  n_repeats: int = N_REPEATS,
                  probe_factory: Optional[Callable] = None) -> float:
    """Cross-validated ROC-AUC using out-of-fold probabilities.

    * groups given  -> StratifiedGroupKFold (subject/site-disjoint folds).
    * groups None   -> StratifiedKFold.
    Binary -> standard AUC; multiclass -> macro one-vs-rest AUC.
    ``n_repeats`` averages OOF scores over that many split seeds (see
    :func:`cross_val_oof`); the default 1 is the historical single split.
    ``probe_factory`` selects the head (default :class:`LinearProbe`).
    Returns 0.5 (chance) when the data is too thin to evaluate honestly.
    """
    out = cross_val_oof(X, y, groups, n_repeats=n_repeats,
                        probe_factory=probe_factory)
    if out is None:
        return 0.5
    y_codes, proba, classes, _ = out
    auc = _auc_from_oof(y_codes, proba, classes)
    return 0.5 if auc is None else auc


# ---------------------------------------------------------------------------
# Uncertainty: bootstrap CI + stratified label-permutation null on the AUC
# ---------------------------------------------------------------------------
#: Default resample counts. Deliberately modest so the headline metrics stay
#: fast in the demo/test path; callers (e.g. build_demo_data) can crank them up.
N_BOOT = 1000
N_PERM = 1000


def _shuffle_within_groups(y_codes: np.ndarray, groups: Optional[np.ndarray],
                           rng: np.random.Generator) -> np.ndarray:
    """Permute labels globally, or WITHIN each site group when groups is given.

    Permuting within site groups keeps each site's label composition fixed, so
    the null preserves the site structure and tests only the label<->score
    association that is NOT explained by which site a subject came from.
    """
    yp = np.array(y_codes, copy=True)
    if groups is None:
        rng.shuffle(yp)
        return yp
    groups = np.asarray(groups)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) > 1:
            yp[idx] = yp[rng.permutation(idx)]
    return yp


def auc_ci_perm(X: np.ndarray, y: np.ndarray,
                groups: Optional[np.ndarray] = None,
                n_boot: int = N_BOOT, n_perm: int = N_PERM,
                random_state: int = RANDOM_STATE,
                return_arrays: bool = False,
                n_repeats: int = N_REPEATS,
                probe_factory: Optional[Callable] = None) -> dict:
    """Cross-validated AUC with a bootstrap 95% CI and a permutation-null p.

    Computes OOF scores ONCE, then (a) bootstraps subjects to get a percentile
    95% CI on the AUC and (b) permutes the labels (within site groups when
    ``groups`` is given) to get ``p_perm`` = P(permuted AUC >= observed) — the
    label-permutation null for "AUC no better than chance". This lets a verdict
    be stated as "CI excludes chance" rather than a bare point-estimate cutoff.

    Limitation (statistical-honesty disclosure): the null FIXES the fitted OOF
    scores — the probe is never refit under permutation and the bootstrap
    resamples the frozen (y, proba) pair — so model-selection variance is
    under-propagated and the reported permutation ``p_perm`` is a LOWER BOUND on
    the true p-value (anticonservative). A defensible speed tradeoff, disclosed
    here and in reports/methods.md.

    ``n_repeats`` averages the OOF scores over that many split seeds before the
    bootstrap/permutation run (see :func:`cross_val_oof`); the default 1 keeps
    the historical single-split number. The CI/permutation null still fix the
    (averaged) OOF scores, so the same lower-bound disclosure applies.

    Returns a JSON-safe dict:
        {auc, ci_lo, ci_hi, p_perm, n, n_boot_ok, ci_excludes_chance}
    with ci/p_perm None when the data is too thin. When ``return_arrays`` is
    True the raw ``boot`` and ``perm`` numpy arrays are also included (for a
    caller that needs to combine distributions, e.g. the leakage margin) — those
    keys are NOT JSON-safe and must be stripped before serialization.
    """
    out = cross_val_oof(X, y, groups, n_repeats=n_repeats,
                        probe_factory=probe_factory)
    base = {"auc": 0.5, "ci_lo": None, "ci_hi": None, "p_perm": None,
            "n": int(len(y)), "n_boot_ok": 0, "ci_excludes_chance": False}
    if out is None:
        if return_arrays:
            base["boot"] = np.array([]); base["perm"] = np.array([])
        return base
    y_codes, proba, classes, groups_eval = out
    auc = _auc_from_oof(y_codes, proba, classes)
    if auc is None:
        if return_arrays:
            base["boot"] = np.array([]); base["perm"] = np.array([])
        return base

    rng = np.random.default_rng(random_state)
    n = len(y_codes)
    idx = np.arange(n)
    binary = len(classes) == 2

    boot = np.empty(0)
    if n_boot and n_boot > 0:
        vals = []
        for _ in range(n_boot):
            b = rng.choice(idx, size=n, replace=True)
            a = _auc_from_oof(y_codes[b], proba[b], classes)
            if a is not None:
                vals.append(a)
        boot = np.asarray(vals, dtype=float)

    perm = np.empty(0)
    if n_perm and n_perm > 0:
        if binary:
            # Under a LABEL permutation the scores are fixed, so the tie-corrected
            # rank vector is fixed too: AUC = (ranks over positives - k(k+1)/2) /
            # (k*(n-k)). Precompute ranks once and vectorize over permutations.
            ranks = sstats_rankdata(proba[:, 1])
            k = int((y_codes == 1).sum())
            vals = []
            if 0 < k < n:
                denom = k * (n - k)
                base_sub = k * (k + 1) / 2.0
                for _ in range(n_perm):
                    yp = _shuffle_within_groups(y_codes, groups_eval, rng)
                    vals.append((ranks[yp == 1].sum() - base_sub) / denom)
            perm = np.asarray(vals, dtype=float)
        else:
            vals = []
            for _ in range(n_perm):
                yp = _shuffle_within_groups(y_codes, groups_eval, rng)
                a = _auc_from_oof(yp, proba, classes)
                if a is not None:
                    vals.append(a)
            perm = np.asarray(vals, dtype=float)

    ci_lo = ci_hi = None
    if boot.size:
        ci_lo, ci_hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    p_perm = None
    if perm.size:
        p_perm = float((1 + int(np.sum(perm >= auc))) / (1 + perm.size))

    result = {
        "auc": round(float(auc), 4),
        "ci_lo": None if ci_lo is None else round(ci_lo, 4),
        "ci_hi": None if ci_hi is None else round(ci_hi, 4),
        "p_perm": None if p_perm is None else round(p_perm, 4),
        "n": int(n),
        "n_boot_ok": int(boot.size),
        # "CI excludes chance" == the lower CI bound sits above 0.5.
        "ci_excludes_chance": bool(ci_lo is not None and ci_lo > 0.5),
    }
    if return_arrays:
        result["boot"] = boot
        result["perm"] = perm
        result["_auc_full"] = float(auc)
    return result
