"""
cross_attention — a from-scratch, offline, deterministic MULTI-HEAD CROSS-ATTENTION
FEATURE fusion for AD-vs-CN.

The pipeline diagram promises "MULTIMODAL TRANSFORMER FUSION (Cross-Attention &
Graph Networks)". The sibling ``fusion`` module ships an honest numpy/sklearn
softmax-gated LATE fusion — principled, but NOT a transformer and with NO
cross-attention. This module closes that gap HONESTLY: it implements the core
transformer operation — scaled-dot-product MULTI-HEAD CROSS-ATTENTION — from
scratch in numpy and uses it as a FIXED, non-trained FEATURE TRANSFORMER over two
modality-token blocks (the frozen imaging embedding + the plasma/tabular block),
then hands the resulting cross-modal interaction features to the ONE reused,
leakage-honest classifier head (``probe.auc_ci_perm``).

What IS and ISN'T fitted (read this before quoting a number):
  * The cross-attention block is NOT trained. Its Q/K/V projections and the
    modality tokenizers are FIXED, seeded Gaussian random projections. The
    attention WEIGHTS are still genuinely data-dependent — softmax of the scaled
    dot product of each subject's own token projections — but nothing in the
    attention is learned. It is a deterministic FEATURE MAP, not a model.
  * The ONLY thing fitted is the downstream logistic head, and its AUC is
    leakage-free: StandardScaler + automatic PCA-10 + logistic fit INSIDE each
    site-disjoint CV fold (``probe.auc_ci_perm``), with a bootstrap 95% CI and a
    within-site permutation null. We never imply the attention itself was
    validated.

Leakage-free by construction: the feature map is STRICTLY per-subject. Each
subject is tokenized and per-token LayerNorm'd using ONLY that subject's own token
statistics; attention is computed per subject; tokens are mean-pooled per subject.
No cross-subject statistic ever enters a subject's feature row, so a subset's
transform equals the full-matrix transform sliced to that subset (tested).

Three aligned views (identical rows, fair like-for-like comparison):
  * ``cross_attention`` — the bidirectional cross-attention interaction features.
  * ``emb_only``        — the frozen embedding block, through the identical head.
  * ``plasma_only``     — the plasma/tabular block, through the identical head.

Honesty stamps: ``source="cross_attention_fusion"``,
``model="numpy_cross_attention_fusion"`` — deliberately DISTINCT from the fitted
late-fusion (``fitted_fusion`` / ``adni_late_fusion`` /
``adni_attention_late_fusion``) and the surrogate (``offline_surrogate`` /
``surrogate_logistic``). This is NOT the published vkola-lab/ncomms2025 ADRD
transformer (that path needs torch + GPU + gated weights) and is NOT a trained
end-to-end model. The verdict claims fusion superiority ONLY when the 95% CIs do
not overlap. ADNI-only decision support; NOT outcome-validated against known AD
drugs.

Import-time contract: numpy / pandas / sklearn only. No torch, no network, no GPU
anywhere on the default path. Degrades honestly (no raise, no fabricated numbers)
when the plasma modality is absent — cross-attention needs >= 2 modalities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .. import contract, probe  # noqa: F401  (contract used transitively / in docs)
from . import fusion

# ---------------------------------------------------------------------------
# Provenance stamps (single source of truth) — DISTINCT from every sibling.
# ---------------------------------------------------------------------------

#: Provenance for the from-scratch numpy cross-attention FEATURE fusion. Never
#: collides with fitted_fusion / offline_surrogate.
SOURCE_CROSS_ATTENTION = "cross_attention_fusion"
MODEL_CROSS_ATTENTION = "numpy_cross_attention_fusion"

#: The three aligned modality views this head compares (identical rows).
VIEW_NAMES = ("cross_attention", "emb_only", "plasma_only")

CROSS_ATTENTION_DISCLAIMER = (
    "FROM-SCRATCH numpy multi-head cross-attention used as a FIXED, non-trained "
    "FEATURE transformer: the Q/K/V projections and modality tokenizers are "
    "seeded random projections, so the cross-attention weights are data-dependent "
    "but nothing in the attention block is learned. The ONLY fitted component is "
    "the downstream logistic head, whose AUC is leakage-free (StandardScaler + "
    "PCA-10 + logistic fit inside each site-disjoint CV fold, bootstrap 95% CI, "
    "within-site permutation null via probe.auc_ci_perm). This is NOT the "
    "published vkola-lab/ncomms2025 ADRD transformer (that needs torch + GPU + "
    "gated weights) and is NOT a trained end-to-end model. ADNI-only decision "
    "support; NOT outcome-validated against known AD drugs."
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CrossAttentionConfig:
    """Fixed hyper-parameters of the cross-attention FEATURE transformer.

    None of these are learned — they only shape the deterministic feature map.
    ``seed`` threads through every random projection so the whole transform is
    byte-reproducible and byte-different across seeds.
    """
    n_heads: int = 4
    d_model: int = 32
    n_tokens_imaging: int = 4
    n_tokens_plasma: int = 3
    seed: int = 0
    residual: bool = True

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def d_attn(self) -> int:
        """Effective attention width = n_heads * d_head (== d_model when divisible)."""
        return self.n_heads * self.d_head

    def to_dict(self) -> dict:
        return {
            "n_heads": int(self.n_heads),
            "d_model": int(self.d_model),
            "d_head": int(self.d_head),
            "d_attn": int(self.d_attn),
            "n_tokens_imaging": int(self.n_tokens_imaging),
            "n_tokens_plasma": int(self.n_tokens_plasma),
            "seed": int(self.seed),
            "residual": bool(self.residual),
        }


DEFAULT_CONFIG = CrossAttentionConfig()


# ---------------------------------------------------------------------------
# Core transformer ops (from scratch, numpy, batched over leading axes)
# ---------------------------------------------------------------------------


def _softmax_last(x: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax over the LAST axis (rows sum to 1)."""
    x = np.asarray(x, dtype=float)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    s = np.sum(e, axis=-1, keepdims=True)
    return e / np.where(s <= 0, 1.0, s)


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray,
                                 V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scaled-dot-product attention, batched over all leading axes.

    ``Q`` is (..., t_q, d_k), ``K`` is (..., t_kv, d_k), ``V`` is (..., t_kv, d_v).
    Returns ``(out, weights)`` where ``weights = softmax(Q Kᵀ / sqrt(d_k))`` (each
    query row a probability distribution over the ``t_kv`` keys, rows sum to 1) and
    ``out = weights @ V`` is (..., t_q, d_v). This is the exact transformer
    operation — no approximation.
    """
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)
    d_k = Q.shape[-1]
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(max(d_k, 1))
    weights = _softmax_last(scores)
    out = np.matmul(weights, V)
    return out, weights


def _seeded_projection(rng: np.random.Generator, d_in: int,
                       d_out: int) -> np.ndarray:
    """Fixed Gaussian random projection (d_in, d_out), scaled ~1/sqrt(d_in).

    Johnson–Lindenstrauss-style scaling keeps the projected magnitudes stable
    across differing input widths (imaging 768-d vs plasma 6-d).
    """
    W = rng.standard_normal((d_in, d_out))
    return W / np.sqrt(max(d_in, 1))


def _layernorm(tokens: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Per-token LayerNorm over the LAST (feature) axis.

    Strictly per-subject, per-token: uses only that token's own mean/var, so NO
    cross-subject statistic enters — the leakage-free guarantee depends on this.
    """
    mu = np.mean(tokens, axis=-1, keepdims=True)
    var = np.var(tokens, axis=-1, keepdims=True)
    return (tokens - mu) / np.sqrt(var + eps)


def _tokenize(X: np.ndarray, n_tokens: int, d_model: int,
              seed: int) -> np.ndarray:
    """Tokenize a modality matrix (n, d_in) into (n, n_tokens, d_model) tokens.

    Each of the ``n_tokens`` tokens is a distinct FIXED seeded Gaussian random
    projection of the subject's modality vector, followed by per-token LayerNorm.
    Deterministic in ``seed``; strictly per-subject.
    """
    X = np.asarray(X, dtype=float)
    n, d_in = X.shape
    rng = np.random.default_rng(seed)
    W = _seeded_projection(rng, d_in, n_tokens * d_model)  # (d_in, n_tokens*d_model)
    tokens = X @ W                                         # (n, n_tokens*d_model)
    tokens = tokens.reshape(n, n_tokens, d_model)
    return _layernorm(tokens)


def multi_head_cross_attention(tokens_q: np.ndarray, tokens_kv: np.ndarray,
                               config: CrossAttentionConfig,
                               seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Multi-head cross-attention with FIXED seeded Q/K/V projections.

    ``tokens_q`` is (n, t_q, d_model); ``tokens_kv`` is (n, t_kv, d_model). Returns
    ``(attended, weights)`` where ``attended`` is (n, t_q, d_attn) (heads
    concatenated) and ``weights`` is (n, n_heads, t_q, t_kv) — the per-head
    attention distributions. Nothing is learned: Wq/Wk/Wv are seeded random
    projections; the softmax attention is computed per subject from that subject's
    own token projections.
    """
    tokens_q = np.asarray(tokens_q, dtype=float)
    tokens_kv = np.asarray(tokens_kv, dtype=float)
    n, t_q, d_model = tokens_q.shape
    t_kv = tokens_kv.shape[1]
    h, dh = config.n_heads, config.d_head
    d_attn = h * dh

    rng = np.random.default_rng(seed)
    Wq = _seeded_projection(rng, d_model, d_attn)
    Wk = _seeded_projection(rng, d_model, d_attn)
    Wv = _seeded_projection(rng, d_model, d_attn)

    Q = (tokens_q @ Wq).reshape(n, t_q, h, dh).transpose(0, 2, 1, 3)   # (n,h,t_q,dh)
    K = (tokens_kv @ Wk).reshape(n, t_kv, h, dh).transpose(0, 2, 1, 3)  # (n,h,t_kv,dh)
    V = (tokens_kv @ Wv).reshape(n, t_kv, h, dh).transpose(0, 2, 1, 3)  # (n,h,t_kv,dh)

    out, weights = scaled_dot_product_attention(Q, K, V)  # (n,h,t_q,dh),(n,h,t_q,t_kv)
    # concat heads: (n,h,t_q,dh) -> (n,t_q,h*dh)
    attended = out.transpose(0, 2, 1, 3).reshape(n, t_q, d_attn)
    return attended, weights


# ---------------------------------------------------------------------------
# The deterministic, leakage-free FEATURE TRANSFORMER
# ---------------------------------------------------------------------------


@dataclass
class CrossAttentionTransform:
    """Output of the cross-attention feature transformer.

    ``features`` is (n_subjects, feature_dim) — the per-subject cross-modal
    interaction vector fed to the reused head. ``attention`` holds the
    interpretability report (per-head/per-direction mean weight matrices + a
    per-modality attention-mass scalar). Both are JSON-safe via nested lists /
    floats.
    """
    features: np.ndarray
    attention: dict
    feature_dim: int
    config: CrossAttentionConfig


def _attention_report(w_i2p: np.ndarray, w_p2i: np.ndarray) -> dict:
    """Assemble the per-head / per-direction interpretability report.

    ``w_i2p`` is (n, n_heads, t_img, t_plasma) — imaging queries attending to
    plasma keys — and ``w_p2i`` is (n, n_heads, t_plasma, t_img). We average over
    subjects to get per-head weight matrices (each query row a distribution that
    sums to ~1) and a per-modality "attention mass" scalar: the mean peak
    attention weight (max over the attended modality's tokens), a concentration
    summary in [1/t_kv, 1] — how sharply each modality attends to the other.
    """
    mean_i2p = w_i2p.mean(axis=0)   # (n_heads, t_img, t_plasma)
    mean_p2i = w_p2i.mean(axis=0)   # (n_heads, t_plasma, t_img)
    # attention mass: average (over subjects, heads, query tokens) of the peak
    # weight the modality places on a single token of the other modality.
    mass_imaging = float(w_i2p.max(axis=-1).mean())
    mass_plasma = float(w_p2i.max(axis=-1).mean())
    return {
        "n_heads": int(w_i2p.shape[1]),
        "imaging_to_plasma": {
            "shape": [int(s) for s in mean_i2p.shape],
            "weights": np.round(mean_i2p, 6).tolist(),
        },
        "plasma_to_imaging": {
            "shape": [int(s) for s in mean_p2i.shape],
            "weights": np.round(mean_p2i, 6).tolist(),
        },
        "attention_mass": {
            "imaging": round(mass_imaging, 6),
            "plasma": round(mass_plasma, 6),
        },
    }


def cross_attention_features(X_imaging: np.ndarray, X_plasma: np.ndarray,
                             config: CrossAttentionConfig = DEFAULT_CONFIG
                             ) -> CrossAttentionTransform:
    """The deterministic, leakage-free cross-attention FEATURE TRANSFORMER.

    Steps (all numpy, all strictly per-subject so the feature map cannot leak):
      1. TOKENIZE each modality into fixed seeded token sets, per-token LayerNorm.
      2. bidirectional MULTI-HEAD CROSS-ATTENTION (imaging->plasma AND
         plasma->imaging) with fixed seeded Q/K/V projections; attention weights
         are the per-subject softmax of the scaled dot product.
      3. POOL attended tokens (mean over tokens) for each direction, and — when
         ``config.residual`` — also mean-pool the raw pre-attention tokens of each
         modality, so the head always still sees each modality's own signal.
      4. Concatenate into a per-subject feature vector of width
         ``(4 if residual else 2) * d_attn``.

    Returns a :class:`CrossAttentionTransform`. Nothing here is trained.
    """
    X_imaging = np.asarray(X_imaging, dtype=float)
    X_plasma = np.asarray(X_plasma, dtype=float)

    img_tokens = _tokenize(X_imaging, config.n_tokens_imaging, config.d_model,
                           seed=config.seed + 101)
    plasma_tokens = _tokenize(X_plasma, config.n_tokens_plasma, config.d_model,
                              seed=config.seed + 202)

    # bidirectional cross-attention (distinct seeds per direction).
    att_i2p, w_i2p = multi_head_cross_attention(
        img_tokens, plasma_tokens, config, seed=config.seed + 303)
    att_p2i, w_p2i = multi_head_cross_attention(
        plasma_tokens, img_tokens, config, seed=config.seed + 404)

    # pool attended tokens (mean over the token axis) per direction.
    pool_i2p = att_i2p.mean(axis=1)   # (n, d_attn)
    pool_p2i = att_p2i.mean(axis=1)   # (n, d_attn)

    parts = [pool_i2p, pool_p2i]
    if config.residual:
        # residual raw-token summaries: project the LayerNorm'd tokens into the
        # attention width so the head still sees each modality's own signal even
        # if attention discards information. Uses fixed seeded projections; still
        # strictly per-subject.
        rng = np.random.default_rng(config.seed + 505)
        Wri = _seeded_projection(rng, config.d_model, config.d_attn)
        Wrp = _seeded_projection(rng, config.d_model, config.d_attn)
        res_img = (img_tokens @ Wri).mean(axis=1)      # (n, d_attn)
        res_plasma = (plasma_tokens @ Wrp).mean(axis=1)  # (n, d_attn)
        parts += [res_img, res_plasma]

    features = np.concatenate(parts, axis=1)
    attention = _attention_report(w_i2p, w_p2i)
    return CrossAttentionTransform(
        features=features, attention=attention,
        feature_dim=int(features.shape[1]), config=config)


# ---------------------------------------------------------------------------
# Fitted-and-validated result (mirrors fusion.FusionResult's honest shape)
# ---------------------------------------------------------------------------


@dataclass
class CrossAttentionResult:
    """A cross-attention fusion comparison, stamped with honest provenance.

    ``views`` maps each of :data:`VIEW_NAMES` to a ``probe.auc_ci_perm`` dict (or
    an ``{unavailable: True}`` marker). ``delta_auc`` is fused minus the best
    single modality; ``verdict`` states only what the CIs support.
    """
    source: str = SOURCE_CROSS_ATTENTION
    model: str = MODEL_CROSS_ATTENTION
    target: str = "dx_binary"
    n: int = 0
    n_ad: int = 0
    n_cn: int = 0
    n_sites: int = 0
    views: dict = field(default_factory=dict)   # view -> auc_ci_perm dict
    best_single: str = ""
    delta_auc: Optional[float] = None
    ci_overlap: Optional[bool] = None
    attention: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    feature_dim: int = 0
    plasma_available: bool = True
    verdict: str = ""
    disclaimer: str = CROSS_ATTENTION_DISCLAIMER
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
            "attention": dict(self.attention),
            "config": dict(self.config),
            "feature_dim": int(self.feature_dim),
            "plasma_available": bool(self.plasma_available),
            "verdict": self.verdict,
            "disclaimer": self.disclaimer,
            "error": self.error,
        }


def _cross_attention_verdict(fused: dict, single: dict, single_name: str,
                             delta: Optional[float],
                             overlap: Optional[bool]) -> str:
    """State only what the confidence intervals support — never overclaim.

    Superiority is asserted ONLY when the fused 95% CI does not overlap the best
    single modality's CI. A positive delta with overlapping CIs is, at most,
    suggestive.
    """
    if not fused.get("ci_excludes_chance"):
        return ("cross-attention fused AUC not distinguishable from chance (95% "
                "CI includes 0.5) — no usable fused signal")
    d = 0.0 if delta is None else delta
    tag = f"{single_name} (AUC {single.get('auc')})"
    if d > 0 and overlap is False:
        return (f"cross-attention fusion superior to best single modality {tag}: "
                f"delta=+{d:.4f}, non-overlapping 95% CIs")
    if d > 0 and fused.get("ci_lo") is not None and \
            single.get("auc") is not None and fused["ci_lo"] > single["auc"]:
        return (f"cross-attention fusion likely improves on {tag}: delta=+{d:.4f} "
                f"and fused 95% CI lower bound clears the single-modality point "
                f"estimate, but CIs overlap — treat as suggestive")
    if overlap:
        sign = "+" if d >= 0 else ""
        return (f"no CI-supported difference between cross-attention fusion and "
                f"best single modality {tag}: delta={sign}{d:.4f} within "
                f"overlapping 95% CIs")
    if d < 0:
        return (f"cross-attention fusion does not improve on best single modality "
                f"{tag}: delta={d:.4f}")
    return (f"cross-attention fusion vs best single modality {tag}: delta=+{d:.4f} "
            f"(CI relationship indeterminate)")


def cross_attention_fusion(df: pd.DataFrame, *,
                           config: Optional[CrossAttentionConfig] = None,
                           n_boot: int = probe.N_BOOT,
                           n_perm: int = probe.N_PERM,
                           random_state: int = probe.RANDOM_STATE,
                           n_repeats: int = probe.N_REPEATS) -> CrossAttentionResult:
    """Run the cross-attention feature fusion and compare it to each single view.

    Referee-consumable entry point. Reuses :func:`fusion.build_fusion_views` to get
    the AD/CN + complete-plasma slice, builds the deterministic cross-attention
    interaction features, and validates all three views through the IDENTICAL
    leakage-honest head (``probe.auc_ci_perm``) on the IDENTICAL rows. Returns a
    :class:`CrossAttentionResult` with per-view AUC/CI/p_perm, delta-AUC,
    CI-overlap, the attention interpretability report, and a CI-honest verdict.

    Degrades honestly (no raise, no fabricated numbers) when the plasma modality
    is absent: cross-attention needs >= 2 modalities, so the cross_attention /
    plasma_only views are marked unavailable, emb_only is still fitted, and the
    verdict says fusion was not run.
    """
    cfg = config or DEFAULT_CONFIG
    v = fusion.build_fusion_views(df)

    def _fit(X: Optional[np.ndarray]) -> dict:
        if X is None:
            return {"auc": None, "ci_lo": None, "ci_hi": None, "p_perm": None,
                    "n": v.n, "n_boot_ok": 0, "ci_excludes_chance": False,
                    "unavailable": True}
        return probe.auc_ci_perm(X, v.y, v.groups, n_boot=n_boot, n_perm=n_perm,
                                 random_state=random_state, n_repeats=n_repeats)

    # --- degraded path: plasma modality absent -> cross-attention not run ----
    if not v.plasma_available:
        emb = _fit(v.emb_only)
        result = CrossAttentionResult(
            n=v.n, n_ad=v.n_ad, n_cn=v.n_cn, n_sites=v.n_sites,
            views={
                "cross_attention": {"unavailable": True},
                "emb_only": emb,
                "plasma_only": {"unavailable": True},
            },
            best_single="emb_only", delta_auc=None, ci_overlap=None,
            attention={}, config=cfg.to_dict(), feature_dim=0,
            plasma_available=False,
            error=("cross-attention needs >= 2 modalities; the plasma/tabular "
                   "block is unavailable on this slice, so only the emb_only view "
                   "was fitted"),
            verdict=("cross-attention not run — plasma modality unavailable; only "
                     f"the emb_only view was fitted (AUC {emb.get('auc')})"),
        )
        return result

    # --- full path: build cross-attention features + compare three views -----
    transform = cross_attention_features(v.emb_only, v.plasma_tabular, cfg)

    views = {
        "cross_attention": _fit(transform.features),
        "emb_only": _fit(v.emb_only),
        "plasma_only": _fit(v.plasma_tabular),
    }

    singles = ("emb_only", "plasma_only")
    best_name = max(singles, key=lambda k: (views[k].get("auc") or 0.0))
    best = views[best_name]
    fused = views["cross_attention"]

    delta = None
    if fused.get("auc") is not None and best.get("auc") is not None:
        delta = round(float(fused["auc"]) - float(best["auc"]), 4)
    overlap = fusion._cis_overlap(fused, best)

    return CrossAttentionResult(
        n=v.n, n_ad=v.n_ad, n_cn=v.n_cn, n_sites=v.n_sites,
        views=views, best_single=best_name, delta_auc=delta, ci_overlap=overlap,
        attention=transform.attention, config=cfg.to_dict(),
        feature_dim=transform.feature_dim, plasma_available=True,
        verdict=_cross_attention_verdict(fused, best, best_name, delta, overlap),
    )


# ---------------------------------------------------------------------------
# Local convenience (NOT used in CI — reads the gated ADNI export)
# ---------------------------------------------------------------------------


def fit_cross_attention_for_dataset(name: str = "adni", *, seed: int = 0,
                                    n_boot: int = probe.N_BOOT,
                                    n_perm: int = probe.N_PERM
                                    ) -> CrossAttentionResult:
    """Load a registered dataset and run the cross-attention fusion (local only).

    Lazily imports ``data.loaders`` so importing this module stays offline-safe.
    Reads the GATED ADNI export when ``name='adni'`` — never call this in CI.
    """
    from ..data import loaders  # lazy: keep module import network/gated-free
    df = loaders.load(name, seed=seed)
    return cross_attention_fusion(df, n_boot=n_boot, n_perm=n_perm)
