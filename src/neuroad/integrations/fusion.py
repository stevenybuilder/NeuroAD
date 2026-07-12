"""
fusion — a FITTED, leakage-honest late/feature-level fusion head for AD-vs-CN.

Where ``multimodal_transformer`` ships a hand-set logistic SURROGATE (coefficients
encode directionality, are not fitted), this module is the opposite: it actually
FITS and VALIDATES a fusion head on the real ADNI contract table and reports the
number the gauntlet would report — out-of-fold, site-disjoint cross-validated AUC
with a bootstrap 95% CI and a permutation-null p (``probe.auc_ci_perm``). It never
invents data; it simply runs the ONE reused head over three aligned views of the
AD/CN + plasma-present slice and compares them honestly.

Three views (identical rows, so the comparison is fair):
  * ``emb_only``       — the frozen embedding matrix (``contract.embedding_matrix``).
  * ``plasma_tabular`` — the standardized plasma/tabular block
                         (p_tau217, gfap, nfl, apoe4, age, sex).
  * ``fusion``         — the embedding concatenated with the plasma/tabular block,
                         fed to a single logistic head (feature-level fusion).

Standardization is performed INSIDE each CV fold by the probe pipeline's
``StandardScaler`` (fit on the training rows only), so mixing z-scored embeddings
with pg/mL plasma markers is leakage-free by construction — nothing is scaled on
data the fold has not seen.

Honesty stamps: every result is stamped ``source="fitted_fusion"``,
``model="adni_late_fusion"`` — deliberately DISTINCT from the surrogate's
``source="offline_surrogate"`` / ``model="surrogate_logistic"``. The verdict never
claims fusion superiority the confidence intervals do not support. This is
ADNI-only decision support and depends on the gated ADNI export; it is NOT
outcome-validated against known AD drugs.

------------------------------------------------------------------------------
Attention-weighted late fusion (``attention_fusion``)
------------------------------------------------------------------------------
On top of the concat feature-level head above, this module ships a genuinely
ATTENTION-WEIGHTED LATE fusion. It is NOT a transformer and makes no such claim:
it is a principled numpy/sklearn late-fusion gate. Each modality gets its OWN
leakage-free, site-disjoint out-of-fold P(AD) score (the identical CV machinery
as ``probe.cross_val_oof``); a per-modality gating weight is then formed as a
SOFTMAX over each modality's above-chance contribution (data-driven attention),
and the fused score is the gate-weighted sum of the standardized per-modality
scores. The gate lets the model up/down-weight imaging vs plasma per the data,
and the learned weights are reported for interpretability.

The gate learner degrades honestly: a torch-based gate optimizer is LAZY-imported
behind the ``fusion`` extra and, when torch is absent (the default here), the
principled numpy softmax gate is used instead — the result stamps which learner
actually ran (``gate_learner`` = ``"numpy"`` / ``"torch"``), never faking torch.

Alongside the fused number this reports (a) a per-modality ABLATION + ATTRIBUTION
table — each modality's standalone AUC, its gate weight, and the leave-one-out
drop in fused AUC (which modality drives the gain), each with a bootstrap 95% CI;
and (b) CALIBRATION (Brier score + expected/maximum calibration error) of the
leakage-free out-of-fold fused P(AD). A clearly-marked SEAM accepts a THIRD
imaging-embedding frame (the 768-d NeuroJEPA embedding) as an extra modality once
available — pass ``imaging_embedding=`` and it is aligned by ``subject_id`` and
gated in exactly like the other blocks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .. import contract, probe

# ---------------------------------------------------------------------------
# Provenance stamps (single source of truth) + the fusion feature block.
# ---------------------------------------------------------------------------

#: Provenance for a genuinely fitted-and-validated fusion head. Distinct from the
#: surrogate stamps in ``multimodal_transformer`` so the two can never be confused.
SOURCE_FITTED = "fitted_fusion"
MODEL_LATE_FUSION = "adni_late_fusion"

#: The plasma/tabular block fused with the frozen embedding. These are exactly the
#: contract columns the ADNI plasma ensemble carries; any absent column makes the
#: plasma/fusion views unavailable (the emb-only view still runs — see below).
FUSION_FEATURES: list[str] = ["p_tau217", "gfap", "nfl", "apoe4", "age", "sex"]

#: The three aligned modality views this head compares.
VIEW_NAMES = ("emb_only", "plasma_tabular", "fusion")

_SINGLE_VIEWS = ("emb_only", "plasma_tabular")

_DISCLAIMER = (
    "FITTED feature-level fusion: a single logistic head over the frozen "
    "embedding concatenated with the standardized plasma/tabular block "
    "(p_tau217, gfap, nfl, apoe4, age, sex). AUC is out-of-fold, site-disjoint "
    "cross-validated with a bootstrap 95% CI and permutation-null p (identical "
    "machinery to the gauntlet). ADNI-only decision support; depends on the "
    "gated ADNI export and is NOT outcome-validated against known AD drugs."
)


# ---------------------------------------------------------------------------
# View construction (mirrors probe.point_head's dx_binary/site-group logic)
# ---------------------------------------------------------------------------


def _encode_groups(values: np.ndarray) -> np.ndarray:
    """Contiguous 0..k-1 codes (same convention as probe._encode)."""
    classes = np.unique(values)
    return np.searchsorted(classes, values)


def _plasma_block(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the plasma/tabular feature block, numeric and sex-encoded.

    Missing columns are materialized as all-NaN so the row mask can drop them;
    ``sex`` is encoded F->1.0 / M->0.0. Nullable pandas dtypes (Int8) are coerced
    to float via ``pd.to_numeric``.
    """
    block = pd.DataFrame(index=df.index)
    for col in FUSION_FEATURES:
        if col == "sex":
            block[col] = (df["sex"].astype("string").map({"F": 1.0, "M": 0.0})
                          if "sex" in df.columns else np.nan)
        elif col in df.columns:
            block[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            block[col] = np.nan
    return block


@dataclass
class FusionViews:
    """Three aligned (X, y, groups) views over the AD/CN + plasma-present slice.

    ``y`` and ``groups`` are shared across views (identical rows). ``plasma_available``
    is False when the plasma/tabular block cannot be completed for any AD/CN row —
    then ``plasma_tabular`` / ``fusion`` are None and only ``emb_only`` is fittable.
    """
    y: np.ndarray
    groups: np.ndarray
    emb_only: np.ndarray
    plasma_tabular: Optional[np.ndarray]
    fusion: Optional[np.ndarray]
    plasma_available: bool
    n: int
    n_ad: int
    n_cn: int
    n_sites: int
    emb_dim: int
    note: str = ""
    #: Boolean mask (length == len(df)) selecting the rows kept in this slice, so
    #: a caller can align a THIRD modality (e.g. a NeuroJEPA embedding frame) to
    #: exactly the same subjects. None only on legacy hand-constructed instances.
    row_mask: Optional[np.ndarray] = None


def build_fusion_views(df: pd.DataFrame) -> FusionViews:
    """Build the emb-only / plasma+tabular / fusion views for dx_binary (AD vs CN).

    Reuses ``point_head``'s dx_binary mapping (AD->1, CN->0) and site-group logic
    so grouping is site-disjoint and every view's rows align. The slice is
    COMPLETE-CASE over the plasma/tabular block (all of p_tau217, gfap, nfl,
    apoe4, age, sex present) intersected with AD/CN rows, so the three views share
    one identical row set with no imputation — the only leakage-free way to make
    "does fusion beat either single modality" a fair, like-for-like comparison.
    """
    dx = df["dx"].astype("string")
    dx_binary = dx.map({"AD": 1, "CN": 0})

    block = _plasma_block(df)
    plasma_complete = block.notna().all(axis=1)

    emb = contract.embedding_matrix(df)
    site = df["site"].astype("string").fillna("__na__").to_numpy()

    plasma_available = bool((dx_binary.notna() & plasma_complete).any())

    if plasma_available:
        mask = (dx_binary.notna() & plasma_complete).to_numpy()
        note = ""
    else:
        # No AD/CN row has a complete plasma block: emb-only still runs on all
        # AD/CN rows so the head degrades gracefully instead of returning nothing.
        mask = dx_binary.notna().to_numpy()
        note = ("plasma/tabular block unavailable (missing or all-NaN plasma "
                "columns); only the emb-only view is fittable")

    y = dx_binary[mask].to_numpy(dtype=int)
    groups = _encode_groups(site[mask])
    X_emb = emb[mask]

    if plasma_available:
        X_plasma = block.to_numpy(dtype=float)[mask]
        X_fusion = np.hstack([X_emb, X_plasma])
    else:
        X_plasma = None
        X_fusion = None

    return FusionViews(
        y=y, groups=groups, emb_only=X_emb,
        plasma_tabular=X_plasma, fusion=X_fusion,
        plasma_available=plasma_available,
        n=int(mask.sum()), n_ad=int((y == 1).sum()), n_cn=int((y == 0).sum()),
        n_sites=int(len(np.unique(groups))), emb_dim=int(X_emb.shape[1]),
        note=note, row_mask=mask,
    )


# ---------------------------------------------------------------------------
# Fitted-and-validated result
# ---------------------------------------------------------------------------


@dataclass
class FusionResult:
    """A FITTED late-fusion comparison, stamped with honest provenance.

    ``views`` maps each view name to a ``probe.auc_ci_perm`` dict (auc, ci_lo,
    ci_hi, p_perm, n, ci_excludes_chance). ``delta_auc`` is fusion minus the best
    single modality; ``verdict`` states only what the CIs support.
    """
    source: str = SOURCE_FITTED
    model: str = MODEL_LATE_FUSION
    target: str = "dx_binary"
    n: int = 0
    n_ad: int = 0
    n_cn: int = 0
    n_sites: int = 0
    views: dict = field(default_factory=dict)      # view -> auc_ci_perm dict
    best_single: str = ""
    delta_auc: Optional[float] = None
    ci_overlap: Optional[bool] = None
    plasma_available: bool = True
    verdict: str = ""
    features_used: dict = field(default_factory=dict)
    disclaimer: str = _DISCLAIMER
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "model": self.model,
            "target": self.target,
            "n": int(self.n),
            "n_ad": int(self.n_ad),
            "n_cn": int(self.n_cn),
            "n_sites": int(self.n_sites),
            "views": {k: dict(v) for k, v in self.views.items()},
            "best_single": self.best_single,
            "delta_auc": self.delta_auc,
            "ci_overlap": self.ci_overlap,
            "plasma_available": bool(self.plasma_available),
            "verdict": self.verdict,
            "features_used": dict(self.features_used),
            "disclaimer": self.disclaimer,
            "error": self.error,
        }


def _cis_overlap(a: dict, b: dict) -> Optional[bool]:
    """True if two auc_ci_perm CIs overlap; None if either CI is missing."""
    if None in (a.get("ci_lo"), a.get("ci_hi"), b.get("ci_lo"), b.get("ci_hi")):
        return None
    return not (a["ci_lo"] > b["ci_hi"] or b["ci_lo"] > a["ci_hi"])


def _honest_verdict(fusion: dict, single: dict, single_name: str,
                    delta: Optional[float], overlap: Optional[bool]) -> str:
    """State only what the confidence intervals support — never overclaim.

    Superiority is asserted ONLY when fusion's 95% CI does not overlap the best
    single modality's CI (a genuinely separated interval). A positive delta with
    overlapping CIs is reported as, at most, suggestive.
    """
    if not fusion.get("ci_excludes_chance"):
        return ("fusion AUC not distinguishable from chance (95% CI includes "
                "0.5) — no usable signal")
    d = 0.0 if delta is None else delta
    tag = f"{single_name} (AUC {single.get('auc')})"
    if d > 0 and overlap is False:
        return (f"fusion superior to best single modality {tag}: "
                f"delta=+{d:.4f}, non-overlapping 95% CIs")
    if d > 0 and fusion.get("ci_lo") is not None and \
            single.get("auc") is not None and fusion["ci_lo"] > single["auc"]:
        return (f"fusion likely improves on {tag}: delta=+{d:.4f} and fusion "
                f"95% CI lower bound clears the single-modality point estimate, "
                f"but CIs overlap — treat as suggestive")
    if overlap:
        sign = "+" if d >= 0 else ""
        return (f"no CI-supported difference between fusion and best single "
                f"modality {tag}: delta={sign}{d:.4f} within overlapping 95% CIs")
    if d < 0:
        return (f"fusion does not improve on best single modality {tag}: "
                f"delta={d:.4f}")
    return (f"fusion vs best single modality {tag}: delta=+{d:.4f} "
            f"(CI relationship indeterminate)")


def compare_fusion_vs_single(df: pd.DataFrame, *,
                             n_boot: int = probe.N_BOOT,
                             n_perm: int = probe.N_PERM,
                             random_state: int = probe.RANDOM_STATE,
                             n_repeats: int = probe.N_REPEATS) -> FusionResult:
    """Fit + validate the three views and return an honest fusion comparison.

    Referee-consumable entry point: give it a contract table (AD/CN rows with the
    plasma block) and it returns a :class:`FusionResult` with per-view
    ``auc_ci_perm`` dicts, delta-AUC, CI-overlap, and a CI-honest verdict. Each
    view is validated through the SAME out-of-fold, site-disjoint machinery the
    gauntlet uses, so the head is trained-and-validated and leakage-free by
    construction. Degrades gracefully (no raise) when the slice is too thin or
    the plasma block is absent.

    ``n_repeats`` averages each view's OOF scores over that many split seeds
    (``probe.N_REPEATS_ENSEMBLE`` is the small-n ensemble size); the default 1
    keeps the single-split number, so shipped reports don't move unless a caller
    asks for the stabilized estimate.
    """
    v = build_fusion_views(df)

    features_used = {
        "emb_only": {"kind": "frozen embedding", "n_features": v.emb_dim},
        "plasma_tabular": list(FUSION_FEATURES),
        "fusion": {"kind": "embedding + plasma_tabular concat",
                   "n_features": v.emb_dim + len(FUSION_FEATURES)},
    }

    def _fit(X: Optional[np.ndarray]) -> dict:
        if X is None:
            return {"auc": None, "ci_lo": None, "ci_hi": None, "p_perm": None,
                    "n": v.n, "n_boot_ok": 0, "ci_excludes_chance": False,
                    "unavailable": True}
        return probe.auc_ci_perm(X, v.y, v.groups, n_boot=n_boot, n_perm=n_perm,
                                 random_state=random_state, n_repeats=n_repeats)

    views = {
        "emb_only": _fit(v.emb_only),
        "plasma_tabular": _fit(v.plasma_tabular),
        "fusion": _fit(v.fusion),
    }

    result = FusionResult(
        n=v.n, n_ad=v.n_ad, n_cn=v.n_cn, n_sites=v.n_sites,
        views=views, plasma_available=v.plasma_available,
        features_used=features_used,
        error=v.note,
    )

    if not v.plasma_available:
        emb = views["emb_only"]
        result.best_single = "emb_only"
        result.verdict = (
            "plasma/tabular block unavailable — only the emb-only view was "
            f"fitted (AUC {emb.get('auc')}); no fusion comparison possible")
        return result

    # Pick the stronger single modality by point-estimate AUC (fair: identical
    # rows), then judge fusion against it using CI overlap — not the point gap.
    singles = {k: views[k] for k in _SINGLE_VIEWS}
    best_name = max(_SINGLE_VIEWS, key=lambda k: (singles[k].get("auc") or 0.0))
    best = singles[best_name]
    fusion = views["fusion"]

    delta = None
    if fusion.get("auc") is not None and best.get("auc") is not None:
        delta = round(float(fusion["auc"]) - float(best["auc"]), 4)
    overlap = _cis_overlap(fusion, best)

    result.best_single = best_name
    result.delta_auc = delta
    result.ci_overlap = overlap
    result.verdict = _honest_verdict(fusion, best, best_name, delta, overlap)
    return result


# ===========================================================================
# ATTENTION-WEIGHTED LATE FUSION
#
# A principled numpy/sklearn attention-weighted LATE fusion (NOT a transformer).
# Each modality contributes a leakage-free, site-disjoint out-of-fold P(AD)
# score; a softmax gate over each modality's above-chance contribution forms the
# per-modality attention weight; the fused score is the gate-weighted sum of the
# standardized per-modality scores. Reports gates, an ablation/attribution table,
# and calibration of the fused probability. Seam included for a 3rd modality.
# ===========================================================================

#: Provenance for the attention-weighted late-fusion head (distinct model tag so
#: it is never confused with the concat feature-level head or the surrogate).
MODEL_ATTENTION_FUSION = "adni_attention_late_fusion"

#: Canonical modality-block names. "imaging" == the frozen contract embedding,
#: "plasma" == the plasma/tabular block, "neurojepa" == the (optional) external
#: 768-d NeuroJEPA imaging embedding wired through the seam below.
MODALITY_IMAGING = "imaging"
MODALITY_PLASMA = "plasma"
MODALITY_NEUROJEPA = "neurojepa"

#: Expected dimensionality of the NeuroJEPA imaging embedding (seam contract).
NEUROJEPA_EMBED_DIM = 768

#: Default softmax temperature for the attention gate. Contributions live in
#: [0, 0.5] (AUC-0.5, clipped at 0); T=0.1 gives a meaningful but non-degenerate
#: spread (a ~0.1 AUC edge roughly triples a modality's gate share).
GATE_TEMPERATURE = 0.1

_ATTENTION_DISCLAIMER = (
    "ATTENTION-WEIGHTED LATE fusion (NOT a transformer): a principled "
    "numpy/sklearn softmax gate over per-modality out-of-fold P(AD) scores. Each "
    "modality's score is leakage-free and site-disjoint (identical CV machinery "
    "to the gauntlet); the gate weight is a softmax over each modality's "
    "above-chance contribution. Gate weights and the leave-one-out attribution "
    "are computed from labelled out-of-fold scores over the full slice, so they "
    "carry a small in-sample optimism (disclosed); the per-view AUC/CI/p_perm "
    "and the fused AUC use the same frozen-score bootstrap + within-site "
    "permutation null as probe.auc_ci_perm. ADNI-only decision support; depends "
    "on the gated ADNI export and is NOT outcome-validated against known AD drugs."
)


# ---------------------------------------------------------------------------
# Leakage-free per-modality out-of-fold scores (aligned to the input rows)
# ---------------------------------------------------------------------------


def _oof_binary_score(X: np.ndarray, y: np.ndarray,
                      groups: Optional[np.ndarray],
                      random_state: int = probe.RANDOM_STATE,
                      n_repeats: int = probe.N_REPEATS):
    """Full-length out-of-fold P(class==1) for a probe over ``X``.

    Mirrors ``probe.cross_val_oof`` fold-for-fold (same ``_n_splits``, same
    Stratified[Group]KFold, same ``LinearProbe`` incl. the automatic PCA
    front-end) but returns a length-``len(y)`` score vector (NaN on rows that
    never received an OOF prediction) so multiple modalities can be aligned
    row-for-row. Because the split depends only on ``y``/``groups`` (identical
    across modality blocks), the evaluated-row mask matches across modalities.

    ``n_repeats`` runs the SAME site-disjoint OOF pass that many times with
    distinct split seeds (``probe.RANDOM_STATE + r``) and averages each
    subject's P(class==1) over the repeats it was evaluated in — the attention
    path's analog of :func:`probe.cross_val_oof`'s repeated-CV ensembling.
    ``n_repeats=1`` (the default) reproduces the historical single-split score
    exactly (same seed ``probe.RANDOM_STATE``, same folds, same ``LinearProbe``).

    Returns ``(score_full, evaluated_mask, y_codes)`` or ``None`` when too thin,
    with ``evaluated_mask`` flagging the rows scored in >= 1 repeat.
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) != 2 or len(y) < 4:
        return None
    y_codes = np.searchsorted(classes, y)

    n_splits = probe._n_splits(y_codes, groups)
    if n_splits < 2:
        return None

    use_groups = groups is not None and len(np.unique(groups)) >= n_splits

    def _score_once(split_seed: int) -> np.ndarray:
        """One stratified (site-disjoint when grouped) OOF pass at ``split_seed``."""
        if use_groups:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                            random_state=split_seed)
            split_iter = splitter.split(X, y_codes, groups)
        else:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                       random_state=split_seed)
            split_iter = splitter.split(X, y_codes)

        s = np.full(len(y), np.nan)
        for tr, te in split_iter:
            if len(np.unique(y_codes[tr])) < 2:
                continue
            p = probe.LinearProbe(random_state=random_state).fit(X[tr], y_codes[tr])
            # decision_scores -> P(positive) for a binary head; positive == code 1.
            s[te] = p.decision_scores(X[te])
        return s

    n_reps = max(1, int(n_repeats))
    score_sum = np.zeros(len(y), dtype=float)
    score_cnt = np.zeros(len(y), dtype=float)
    for r in range(n_reps):
        s_r = _score_once(probe.RANDOM_STATE + r)
        seen = ~np.isnan(s_r)
        score_sum[seen] += s_r[seen]
        score_cnt[seen] += 1.0

    with np.errstate(invalid="ignore"):
        score = np.where(score_cnt > 0, score_sum / score_cnt, np.nan)

    evaluated = ~np.isnan(score)
    if evaluated.sum() < 4 or len(np.unique(y_codes[evaluated])) < 2:
        return None
    return score, evaluated, y_codes


# ---------------------------------------------------------------------------
# Frozen-score CI + within-site permutation null (reuses probe internals)
# ---------------------------------------------------------------------------


def _frozen_score_ci_perm(y_codes: np.ndarray, score: np.ndarray,
                          groups: Optional[np.ndarray],
                          n_boot: int, n_perm: int,
                          random_state: int = probe.RANDOM_STATE) -> dict:
    """Bootstrap 95% CI + within-group permutation p for a FROZEN 1-D score.

    Identical statistics to ``probe.auc_ci_perm``'s binary path (frozen-score
    bootstrap over subjects; label permutation WITHIN site groups holding the
    scores fixed) but applied to an already-computed late-fusion score, so the
    output dict has the SAME shape/keys as ``auc_ci_perm``. Reuses
    ``probe._fast_binary_auc`` and ``probe._shuffle_within_groups`` verbatim.
    """
    y_codes = np.asarray(y_codes)
    score = np.asarray(score, dtype=float)
    n = len(y_codes)
    base = {"auc": None, "ci_lo": None, "ci_hi": None, "p_perm": None,
            "n": int(n), "n_boot_ok": 0, "ci_excludes_chance": False}
    auc = probe._fast_binary_auc(y_codes, score)
    if auc is None:
        return base

    rng = np.random.default_rng(random_state)
    idx = np.arange(n)

    boot = np.empty(0)
    if n_boot and n_boot > 0:
        vals = []
        for _ in range(n_boot):
            b = rng.choice(idx, size=n, replace=True)
            a = probe._fast_binary_auc(y_codes[b], score[b])
            if a is not None:
                vals.append(a)
        boot = np.asarray(vals, dtype=float)

    perm = np.empty(0)
    if n_perm and n_perm > 0:
        ranks = probe.sstats_rankdata(score)
        k = int((y_codes == 1).sum())
        if 0 < k < n:
            denom = k * (n - k)
            base_sub = k * (k + 1) / 2.0
            vals = []
            for _ in range(n_perm):
                yp = probe._shuffle_within_groups(y_codes, groups, rng)
                vals.append((ranks[yp == 1].sum() - base_sub) / denom)
            perm = np.asarray(vals, dtype=float)

    ci_lo = ci_hi = None
    if boot.size:
        ci_lo, ci_hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    p_perm = None
    if perm.size:
        p_perm = float((1 + int(np.sum(perm >= auc))) / (1 + perm.size))

    return {
        "auc": round(float(auc), 4),
        "ci_lo": None if ci_lo is None else round(ci_lo, 4),
        "ci_hi": None if ci_hi is None else round(ci_hi, 4),
        "p_perm": None if p_perm is None else round(p_perm, 4),
        "n": int(n),
        "n_boot_ok": int(boot.size),
        "ci_excludes_chance": bool(ci_lo is not None and ci_lo > 0.5),
    }


# ---------------------------------------------------------------------------
# The attention gate (numpy default; lazy-torch seam that degrades honestly)
# ---------------------------------------------------------------------------


def _softmax(v: np.ndarray, temperature: float) -> np.ndarray:
    """Numerically-stable softmax of ``v / temperature`` (returns a prob vector)."""
    v = np.asarray(v, dtype=float) / max(float(temperature), 1e-8)
    v = v - v.max()
    e = np.exp(v)
    s = e.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full(len(v), 1.0 / len(v))
    return e / s


def _numpy_attention_gate(contributions: np.ndarray,
                          temperature: float) -> np.ndarray:
    """Softmax attention gate over per-modality above-chance contributions.

    ``contributions[m] = max(AUC_m - 0.5, 0)`` — a modality below chance gets no
    positive contribution (but softmax still assigns it a small floor share; it
    is never up-weighted for being anti-correlated). Data-driven, no extra label
    fit beyond the per-fold base models.
    """
    return _softmax(np.clip(contributions, 0.0, None), temperature)


def _torch_attention_gate(scores: np.ndarray, y_codes: np.ndarray,
                          temperature: float,
                          steps: int = 300) -> Optional[np.ndarray]:
    """Optional torch gate optimizer — lazy-imported; returns None if torch absent.

    Learns per-modality gate logits by minimizing BCE of
    ``sigmoid(sum_m softmax(g)_m * z_m)`` against the labels. This is the seam
    for a learned (rather than AUC-heuristic) gate; it is only exercised when the
    optional ``fusion`` torch extra is installed. When torch is unavailable it
    returns ``None`` so the caller falls back to the numpy gate — the result then
    honestly records ``gate_learner="numpy"`` and NEVER pretends torch ran.

    ``scores`` is (n_rows, n_modalities) of standardized per-modality OOF scores.
    """
    try:  # lazy: keep module import torch-free (torch is a heavy optional dep)
        import torch  # noqa: F401
    except Exception:
        return None
    try:
        Z = torch.as_tensor(np.asarray(scores, dtype=float), dtype=torch.float64)
        yt = torch.as_tensor(np.asarray(y_codes, dtype=float), dtype=torch.float64)
        g = torch.zeros(Z.shape[1], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([g], lr=0.05)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for _ in range(int(steps)):
            opt.zero_grad()
            alpha = torch.softmax(g / max(float(temperature), 1e-8), dim=0)
            logits = Z @ alpha
            loss = loss_fn(logits, yt)
            loss.backward()
            opt.step()
        with torch.no_grad():
            alpha = torch.softmax(g / max(float(temperature), 1e-8), dim=0)
        out = alpha.detach().cpu().numpy().astype(float)
        if not np.all(np.isfinite(out)) or out.sum() <= 0:
            return None
        return out / out.sum()
    except Exception:
        return None


def learn_attention_gates(contributions: np.ndarray, scores: np.ndarray,
                          y_codes: np.ndarray, *, temperature: float,
                          learner: str = "numpy") -> tuple[np.ndarray, str]:
    """Dispatch to the requested gate learner, degrading to numpy honestly.

    * ``learner="numpy"`` (default) -> AUC-contribution softmax gate.
    * ``learner="torch"`` / ``"auto"`` -> try the lazy torch optimizer; if torch
      is absent (or it fails) fall back to the numpy gate.

    Returns ``(gate_vector, learner_actually_used)`` where the second element is
    the ground truth of which path ran — so provenance can never overstate torch.
    """
    if learner in ("torch", "auto"):
        g = _torch_attention_gate(scores, y_codes, temperature)
        if g is not None:
            return g, "torch"
    return _numpy_attention_gate(contributions, temperature), "numpy"


# ---------------------------------------------------------------------------
# Calibration of a probability (Brier + expected/maximum calibration error)
# ---------------------------------------------------------------------------


def calibration_metrics(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """Brier score + ECE/MCE with a reliability table for a probability vector.

    ``y`` is 0/1, ``p`` in [0, 1]. Brier is the mean squared error of ``p``. ECE
    is the sample-weighted mean gap between confidence and empirical accuracy over
    ``n_bins`` equal-width probability bins; MCE is the max such gap. All JSON-safe.
    """
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    n = len(y)
    if n == 0:
        return {"brier": None, "ece": None, "mce": None, "n": 0,
                "n_bins": int(n_bins), "reliability": []}
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    reliability = []
    ece = 0.0
    mce = 0.0
    for i in range(int(n_bins)):
        lo, hi = edges[i], edges[i + 1]
        # last bin is closed on the right so p==1.0 lands somewhere.
        in_bin = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        cnt = int(in_bin.sum())
        if cnt == 0:
            reliability.append({"bin_lo": round(float(lo), 4),
                                "bin_hi": round(float(hi), 4), "count": 0,
                                "confidence": None, "accuracy": None})
            continue
        conf = float(p[in_bin].mean())
        acc = float(y[in_bin].mean())
        gap = abs(conf - acc)
        ece += (cnt / n) * gap
        mce = max(mce, gap)
        reliability.append({"bin_lo": round(float(lo), 4),
                            "bin_hi": round(float(hi), 4), "count": cnt,
                            "confidence": round(conf, 4),
                            "accuracy": round(acc, 4)})
    return {"brier": round(brier, 4), "ece": round(float(ece), 4),
            "mce": round(float(mce), 4), "n": int(n), "n_bins": int(n_bins),
            "reliability": reliability}


# ---------------------------------------------------------------------------
# Modality-block assembly (imaging + plasma [+ optional NeuroJEPA seam])
# ---------------------------------------------------------------------------


def _align_external_embedding(df: pd.DataFrame,
                              frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Align an external embedding ``frame`` to ``df`` rows by ``subject_id``.

    ``frame`` must carry ``subject_id`` plus ``emb_*`` columns (the seam contract
    for the 768-d NeuroJEPA embedding). Returns ``(X_full, present_mask)`` where
    ``X_full`` is (len(df), D) reindexed onto ``df.subject_id`` (NaN for absent
    subjects) and ``present_mask`` flags rows with a complete embedding.
    """
    if "subject_id" not in frame.columns:
        raise ValueError("imaging_embedding frame must have a 'subject_id' column")
    emb_cols = contract.embedding_columns(frame)
    if not emb_cols:
        raise ValueError("imaging_embedding frame has no emb_* embedding columns")
    indexed = frame.set_index(frame["subject_id"].astype("string"))[emb_cols]
    # drop duplicate subject_ids (keep first) so reindex is well-defined.
    indexed = indexed[~indexed.index.duplicated(keep="first")]
    reindexed = indexed.reindex(df["subject_id"].astype("string"))
    X_full = reindexed.to_numpy(dtype=float)
    present = ~np.isnan(X_full).any(axis=1)
    return X_full, present


@dataclass
class ModalityBlocks:
    """Aligned modality blocks + shared (y, groups) for attention late fusion."""
    y_codes: np.ndarray
    groups: np.ndarray
    blocks: dict           # name -> X (n_rows, d_m)
    n: int
    n_ad: int
    n_cn: int
    n_sites: int
    seam_open: bool        # True when the NeuroJEPA modality is NOT yet wired
    note: str = ""


def _build_modality_blocks(df: pd.DataFrame,
                           imaging_embedding: Optional[pd.DataFrame]
                           ) -> Optional[ModalityBlocks]:
    """Assemble imaging + plasma [+ NeuroJEPA] blocks over one aligned row set.

    Complete-case over the plasma block (as in ``build_fusion_views``) AND, when
    ``imaging_embedding`` is supplied, over the aligned NeuroJEPA embedding — so
    every modality shares one identical, imputation-free row set. Returns None
    when fewer than two modalities are usable (attention needs >= 2 to gate).
    """
    v = build_fusion_views(df)
    if not v.plasma_available or v.row_mask is None:
        return None  # need the plasma block (>= 2 modalities) for attention

    mask = np.asarray(v.row_mask, dtype=bool)
    blocks: dict = {
        MODALITY_IMAGING: v.emb_only,
        MODALITY_PLASMA: v.plasma_tabular,
    }
    seam_open = True
    note = ""

    if imaging_embedding is not None:
        X_full, present = _align_external_embedding(df, imaging_embedding)
        # tighten the shared row set to subjects that also have a NeuroJEPA vec.
        new_mask = mask & present
        if int(new_mask.sum()) < 4 or \
                len(np.unique(df["dx"].astype("string")[new_mask].map(
                    {"AD": 1, "CN": 0}).dropna())) < 2:
            note = ("NeuroJEPA embedding provided but too few subjects overlap the "
                    "AD/CN + plasma slice; third modality not wired this run")
        else:
            # re-slice imaging/plasma onto the tightened mask.
            keep = new_mask[mask]        # position of new rows within old slice
            blocks[MODALITY_IMAGING] = v.emb_only[keep]
            blocks[MODALITY_PLASMA] = v.plasma_tabular[keep]
            blocks[MODALITY_NEUROJEPA] = X_full[new_mask]
            mask = new_mask
            seam_open = False

    dx = df["dx"].astype("string").map({"AD": 1, "CN": 0})
    y_codes = dx[mask].to_numpy(dtype=int)
    site = df["site"].astype("string").fillna("__na__").to_numpy()[mask]
    groups = _encode_groups(site)

    return ModalityBlocks(
        y_codes=y_codes, groups=groups, blocks=blocks,
        n=int(mask.sum()), n_ad=int((y_codes == 1).sum()),
        n_cn=int((y_codes == 0).sum()), n_sites=int(len(np.unique(groups))),
        seam_open=seam_open, note=note,
    )


# ---------------------------------------------------------------------------
# Attention-weighted late-fusion result
# ---------------------------------------------------------------------------


@dataclass
class AttentionFusionResult:
    """A FITTED attention-weighted late-fusion comparison, honestly stamped.

    ``gates`` maps modality -> softmax attention weight; ``modalities`` maps
    modality -> its standalone frozen-score ``auc_ci_perm`` dict; ``fused`` is the
    gate-weighted late-fusion score's ``auc_ci_perm`` dict; ``attribution`` is the
    per-modality ablation/attribution table; ``calibration`` is Brier/ECE of the
    leakage-free out-of-fold fused P(AD). ``gate_learner`` is the learner that
    ACTUALLY ran ("numpy" or "torch"), never the one merely requested.
    """
    source: str = SOURCE_FITTED
    model: str = MODEL_ATTENTION_FUSION
    target: str = "dx_binary"
    n: int = 0
    n_ad: int = 0
    n_cn: int = 0
    n_sites: int = 0
    modality_names: list = field(default_factory=list)
    gates: dict = field(default_factory=dict)
    gate_learner: str = "numpy"
    gate_temperature: float = GATE_TEMPERATURE
    modalities: dict = field(default_factory=dict)   # name -> auc_ci_perm dict
    fused: dict = field(default_factory=dict)         # fused score auc_ci_perm
    attribution: list = field(default_factory=list)   # ablation/attribution rows
    calibration: dict = field(default_factory=dict)
    top_modality: str = ""
    neurojepa_wired: bool = False
    seam_open: bool = True
    verdict: str = ""
    disclaimer: str = _ATTENTION_DISCLAIMER
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "model": self.model,
            "target": self.target,
            "n": int(self.n),
            "n_ad": int(self.n_ad),
            "n_cn": int(self.n_cn),
            "n_sites": int(self.n_sites),
            "modality_names": list(self.modality_names),
            "gates": {k: (None if v is None else round(float(v), 4))
                      for k, v in self.gates.items()},
            "gate_learner": self.gate_learner,
            "gate_temperature": float(self.gate_temperature),
            "modalities": {k: dict(v) for k, v in self.modalities.items()},
            "fused": dict(self.fused),
            "attribution": [dict(r) for r in self.attribution],
            "calibration": dict(self.calibration),
            "top_modality": self.top_modality,
            "neurojepa_wired": bool(self.neurojepa_wired),
            "seam_open": bool(self.seam_open),
            "verdict": self.verdict,
            "disclaimer": self.disclaimer,
            "error": self.error,
        }


def _fuse(scores_std: dict, gate: dict, names: list) -> np.ndarray:
    """Gate-weighted sum of standardized per-modality scores."""
    return sum(gate[m] * scores_std[m] for m in names)


def attention_fusion(df: pd.DataFrame, *,
                     imaging_embedding: Optional[pd.DataFrame] = None,
                     temperature: float = GATE_TEMPERATURE,
                     learner: str = "numpy",
                     n_boot: int = probe.N_BOOT,
                     n_perm: int = probe.N_PERM,
                     random_state: int = probe.RANDOM_STATE,
                     n_repeats: int = probe.N_REPEATS) -> AttentionFusionResult:
    """Attention-weighted LATE fusion over imaging + plasma [+ NeuroJEPA].

    Referee-consumable entry point. Computes a leakage-free, site-disjoint
    out-of-fold P(AD) score per modality, forms a softmax attention gate over each
    modality's above-chance contribution, fuses the gate-weighted standardized
    scores, and reports: per-modality and fused AUC/CI/p_perm (frozen-score
    bootstrap + within-site permutation, same statistics as
    ``probe.auc_ci_perm``), a per-modality ablation/attribution table with CIs,
    and Brier/ECE calibration of the out-of-fold fused probability.

    Seam: pass ``imaging_embedding`` (a frame with ``subject_id`` + ``emb_*``
    columns — the 768-d NeuroJEPA embedding) to add it as a THIRD modality; it is
    aligned by ``subject_id`` and gated in like the others. Left None, the seam
    stays open (``seam_open=True``) and fusion runs on imaging + plasma only.

    ``n_repeats`` averages each modality's (and the calibration concat's) OOF
    P(AD) over that many split seeds (``probe.RANDOM_STATE + r``) — the small-n
    ensemble analog of :func:`probe.cross_val_oof`. The default 1 keeps the
    historical single-split number, so shipped attention-fusion metrics don't
    move unless a caller opts into the stabilized estimate (e.g.
    ``probe.N_REPEATS_ENSEMBLE``).

    Degrades without raising: returns an ``error``-stamped result when the slice
    is too thin or fewer than two modalities are usable.
    """
    mb = _build_modality_blocks(df, imaging_embedding)
    if mb is None:
        return AttentionFusionResult(
            error=("attention late fusion needs >= 2 usable modalities (imaging + "
                   "plasma); plasma/tabular block unavailable on this slice"),
            verdict=("attention fusion not run — the plasma modality is "
                     "unavailable, so there is nothing to gate against imaging"),
            seam_open=imaging_embedding is None,
        )

    names = list(mb.blocks.keys())

    # 1) Leakage-free per-modality OOF P(AD), aligned to one common row set.
    per_scores: dict = {}
    common = np.ones(mb.n, dtype=bool)
    y_codes = mb.y_codes
    for m in names:
        out = _oof_binary_score(mb.blocks[m], y_codes, mb.groups,
                                random_state=random_state, n_repeats=n_repeats)
        if out is None:
            return AttentionFusionResult(
                n=mb.n, n_ad=mb.n_ad, n_cn=mb.n_cn, n_sites=mb.n_sites,
                modality_names=names, seam_open=mb.seam_open,
                neurojepa_wired=MODALITY_NEUROJEPA in names,
                error=(f"modality {m!r} produced no out-of-fold scores (slice too "
                       f"thin for site-disjoint CV)"),
                verdict="attention fusion not run — a modality could not be scored")
        score_full, evaluated, _ = out
        per_scores[m] = score_full
        common &= evaluated

    if common.sum() < 4 or len(np.unique(y_codes[common])) < 2:
        return AttentionFusionResult(
            n=mb.n, n_ad=mb.n_ad, n_cn=mb.n_cn, n_sites=mb.n_sites,
            modality_names=names, seam_open=mb.seam_open,
            neurojepa_wired=MODALITY_NEUROJEPA in names,
            error="too few commonly-evaluated rows across modalities",
            verdict="attention fusion not run — modalities share too few OOF rows")

    yc = y_codes[common]
    grp = mb.groups[common]
    raw = {m: per_scores[m][common] for m in names}

    # standardize each modality's OOF score so the gate weights are comparable.
    std: dict = {}
    for m in names:
        s = raw[m]
        sd = s.std()
        std[m] = (s - s.mean()) / sd if sd > 1e-12 else s - s.mean()

    # 2) Per-modality standalone AUC (frozen-score CI/perm) + contributions.
    modalities: dict = {}
    contributions = []
    for m in names:
        modalities[m] = _frozen_score_ci_perm(yc, raw[m], grp, n_boot, n_perm,
                                               random_state=random_state)
        auc_m = modalities[m]["auc"] or 0.5
        contributions.append(max(float(auc_m) - 0.5, 0.0))
    contributions = np.asarray(contributions, dtype=float)

    # 3) Attention gate (numpy default; torch seam degrades to numpy honestly).
    Zmat = np.column_stack([std[m] for m in names])
    gate_vec, gate_used = learn_attention_gates(
        contributions, Zmat, yc, temperature=temperature, learner=learner)
    gate = {m: float(gate_vec[i]) for i, m in enumerate(names)}

    # 4) Fused score + its CI/perm; out-of-fold fused probability for calibration.
    fused_score = _fuse(std, gate, names)
    fused = _frozen_score_ci_perm(yc, fused_score, grp, n_boot, n_perm,
                                  random_state=random_state)

    # Calibration uses the genuinely-OOF concat-fusion P(AD) (leakage-free per
    # fold) as the fused probability — the attention score is not a probability.
    concat = np.column_stack([mb.blocks[m] for m in names])
    cal_out = _oof_binary_score(concat, y_codes, mb.groups,
                                random_state=random_state, n_repeats=n_repeats)
    if cal_out is not None:
        cs, cev, _ = cal_out
        calibration = calibration_metrics(y_codes[cev], cs[cev])
    else:
        calibration = calibration_metrics(np.asarray([]), np.asarray([]))

    # 5) Ablation + attribution: leave-one-modality-out fused AUC drop, with CIs.
    attribution = []
    fused_auc = fused.get("auc")
    for m in names:
        others = [o for o in names if o != m]
        row = {
            "modality": m,
            "gate": round(float(gate[m]), 4),
            "standalone_auc": modalities[m]["auc"],
            "standalone_ci_lo": modalities[m]["ci_lo"],
            "standalone_ci_hi": modalities[m]["ci_hi"],
        }
        if others:
            # regate over the remaining modalities and refuse the dropped one.
            sub_contrib = np.asarray(
                [max((modalities[o]["auc"] or 0.5) - 0.5, 0.0) for o in others],
                dtype=float)
            sub_gate_vec = _numpy_attention_gate(sub_contrib, temperature)
            sub_gate = {o: float(sub_gate_vec[i]) for i, o in enumerate(others)}
            loo_score = _fuse(std, sub_gate, others)
            loo = _frozen_score_ci_perm(yc, loo_score, grp, n_boot, n_perm,
                                        random_state=random_state)
            row["loo_fused_auc"] = loo["auc"]
            row["loo_fused_ci_lo"] = loo["ci_lo"]
            row["loo_fused_ci_hi"] = loo["ci_hi"]
            if fused_auc is not None and loo["auc"] is not None:
                row["attribution_delta"] = round(float(fused_auc) - float(loo["auc"]), 4)
            else:
                row["attribution_delta"] = None
        else:
            row["loo_fused_auc"] = None
            row["loo_fused_ci_lo"] = None
            row["loo_fused_ci_hi"] = None
            row["attribution_delta"] = None
        attribution.append(row)

    # top modality by attribution (drop-one impact), tie-broken by gate weight.
    def _attr_key(r):
        d = r["attribution_delta"]
        return (d if d is not None else -1.0, r["gate"])
    top = max(attribution, key=_attr_key)["modality"] if attribution else ""

    # 6) Honest verdict: only claim a fusion gain the CI supports.
    best_single = max(names, key=lambda m: (modalities[m]["auc"] or 0.0))
    verdict = _attention_verdict(fused, modalities[best_single], best_single,
                                 gate, top)

    return AttentionFusionResult(
        n=int(common.sum()), n_ad=int((yc == 1).sum()), n_cn=int((yc == 0).sum()),
        n_sites=int(len(np.unique(grp))),
        modality_names=names, gates=gate, gate_learner=gate_used,
        gate_temperature=float(temperature), modalities=modalities, fused=fused,
        attribution=attribution, calibration=calibration, top_modality=top,
        neurojepa_wired=MODALITY_NEUROJEPA in names, seam_open=mb.seam_open,
        verdict=verdict, error=mb.note,
    )


def _attention_verdict(fused: dict, best: dict, best_name: str,
                       gate: dict, top: str) -> str:
    """State only what the fused CI supports; report the driving modality."""
    if not fused.get("ci_excludes_chance"):
        return ("attention-fused AUC not distinguishable from chance (95% CI "
                "includes 0.5) — no usable fused signal")
    fa, ba = fused.get("auc"), best.get("auc")
    gate_str = ", ".join(f"{m}={gate[m]:.2f}" for m in gate)
    lead = (f"attention gate [{gate_str}]; {top!r} drives the largest "
            f"leave-one-out AUC drop")
    if fa is not None and ba is not None:
        delta = round(float(fa) - float(ba), 4)
        overlap = _cis_overlap(fused, best)
        if delta > 0 and overlap is False:
            return (f"{lead}. Fused superior to best single modality "
                    f"{best_name} (AUC {ba}): delta=+{delta:.4f}, non-overlapping "
                    f"95% CIs")
        if delta > 0 and fused.get("ci_lo") is not None and \
                fused["ci_lo"] > ba:
            return (f"{lead}. Fused likely improves on {best_name} (AUC {ba}): "
                    f"delta=+{delta:.4f}, fused CI lower bound clears the single "
                    f"point estimate but CIs overlap — suggestive")
        sign = "+" if delta >= 0 else ""
        return (f"{lead}. No CI-supported gain over best single modality "
                f"{best_name} (AUC {ba}): delta={sign}{delta:.4f}")
    return f"{lead}. Fused AUC {fa}"


# ---------------------------------------------------------------------------
# Local convenience (NOT used in CI — reads the gated ADNI export)
# ---------------------------------------------------------------------------


def fit_fusion_for_dataset(name: str = "adni", *, seed: int = 0,
                           n_boot: int = probe.N_BOOT,
                           n_perm: int = probe.N_PERM) -> FusionResult:
    """Load a registered dataset and run the fusion comparison (local use only).

    Lazily imports ``data.loaders`` so importing this module stays offline-safe.
    Reads the GATED ADNI export when ``name='adni'`` — never call this in CI.
    """
    from ..data import loaders  # lazy: keep module import network/gated-free
    df = loaders.load(name, seed=seed)
    return compare_fusion_vs_single(df, n_boot=n_boot, n_perm=n_perm)
