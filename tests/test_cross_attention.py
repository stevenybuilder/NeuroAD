"""
Offline, deterministic tests for the from-scratch cross-attention FEATURE fusion
(integrations.cross_attention).

Every test runs with NO network and NO gated file: the AD/CN + plasma fixture is
built synthetically in-process (a local copy of the fusion fixture). These assert:
the core transformer op is correct; the feature map is deterministic and strictly
leakage-free (subset transform == full transform sliced); the interpretability
report is well-formed; the three views produce valid probe.auc_ci_perm dicts; the
honesty stamps/disclaimer are correct and distinct from every sibling; the verdict
never overclaims; and the head degrades gracefully when plasma is absent.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from neuroad import contract
from neuroad.integrations import cross_attention as ca
from neuroad.integrations.cross_attention import (
    CROSS_ATTENTION_DISCLAIMER,
    DEFAULT_CONFIG,
    MODEL_CROSS_ATTENTION,
    SOURCE_CROSS_ATTENTION,
    VIEW_NAMES,
    CrossAttentionConfig,
    CrossAttentionResult,
    CrossAttentionTransform,
    cross_attention_features,
    cross_attention_fusion,
    multi_head_cross_attention,
    scaled_dot_product_attention,
)


# --- fixture (local copy; offline, no gated file) -------------------------

def _synthetic_adcn(n: int = 120, d: int = 16, n_sites: int = 6,
                    seed: int = 0, with_plasma: bool = True) -> pd.DataFrame:
    """A small contract-shaped AD/CN cohort with a learnable emb+plasma signal."""
    rng = np.random.default_rng(seed)
    y = np.array([1, 0] * (n // 2))
    dx = np.where(y == 1, "AD", "CN")

    X = rng.normal(0.0, 1.0, size=(n, d))
    X[:, :4] += (y[:, None] * 0.9)

    cols = {f"{contract.EMBED_PREFIX}{i}": X[:, i] for i in range(d)}
    df = pd.DataFrame(cols)
    df["subject_id"] = [f"S{i:04d}" for i in range(n)]
    df["dx"] = pd.Categorical(dx, categories=contract.DX_LEVELS)
    df["conversion"] = pd.array([pd.NA] * n, dtype="Int8")
    df["age"] = rng.normal(72, 6, n) + y * 3.0
    df["sex"] = pd.Categorical(rng.choice(["M", "F"], n),
                               categories=contract.SEX_LEVELS)
    df["site"] = pd.Categorical([f"site{i % n_sites}" for i in range(n)])
    df["scanner"] = pd.Categorical(rng.choice(["1.5T", "3T"], n))
    df["amyloid"] = pd.array(rng.integers(0, 2, n), dtype="Int8")
    df["apoe4"] = pd.array(rng.integers(0, 3, n), dtype="Int8")

    if with_plasma:
        df["p_tau217"] = rng.normal(1.0, 0.4, n) + y * 1.2
        df["gfap"] = rng.normal(150, 40, n) + y * 80.0
        df["nfl"] = rng.normal(30, 8, n) + y * 12.0
    else:
        for c in ("p_tau217", "gfap", "nfl"):
            df[c] = np.nan

    contract.validate_table(df)
    return df


# --- core transformer op --------------------------------------------------

def test_scaled_dot_product_attention_hand_computed():
    # Q attends to two identical-key rows -> uniform 0.5/0.5 weights.
    Q = np.array([[1.0, 0.0]])
    K = np.array([[1.0, 0.0], [1.0, 0.0]])
    V = np.array([[10.0, 0.0], [0.0, 20.0]])
    out, w = scaled_dot_product_attention(Q, K, V)
    assert w.shape == (1, 2)
    np.testing.assert_allclose(w, [[0.5, 0.5]], atol=1e-9)
    np.testing.assert_allclose(out, [[5.0, 10.0]], atol=1e-9)


def test_scaled_dot_product_weights_are_distributions():
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(5, 3, 4))   # batched leading axis
    K = rng.normal(size=(5, 6, 4))
    V = rng.normal(size=(5, 6, 7))
    out, w = scaled_dot_product_attention(Q, K, V)
    assert w.shape == (5, 3, 6)
    assert out.shape == (5, 3, 7)
    assert np.all(w >= 0.0)
    np.testing.assert_allclose(w.sum(axis=-1), np.ones((5, 3)), atol=1e-9)


def test_multi_head_shapes():
    cfg = CrossAttentionConfig()
    n, t_q, t_kv = 8, cfg.n_tokens_imaging, cfg.n_tokens_plasma
    tq = np.random.default_rng(1).normal(size=(n, t_q, cfg.d_model))
    tkv = np.random.default_rng(2).normal(size=(n, t_kv, cfg.d_model))
    attended, w = multi_head_cross_attention(tq, tkv, cfg, seed=3)
    assert attended.shape == (n, t_q, cfg.d_attn)
    assert w.shape == (n, cfg.n_heads, t_q, t_kv)
    np.testing.assert_allclose(w.sum(axis=-1), np.ones((n, cfg.n_heads, t_q)),
                               atol=1e-9)


# --- determinism ----------------------------------------------------------

def test_feature_transform_is_deterministic():
    X_img = np.random.default_rng(10).normal(size=(30, 16))
    X_pl = np.random.default_rng(11).normal(size=(30, 6))
    a = cross_attention_features(X_img, X_pl)
    b = cross_attention_features(X_img, X_pl)
    np.testing.assert_array_equal(a.features, b.features)
    assert a.attention == b.attention


def test_different_seed_changes_features():
    X_img = np.random.default_rng(10).normal(size=(30, 16))
    X_pl = np.random.default_rng(11).normal(size=(30, 6))
    a = cross_attention_features(X_img, X_pl, CrossAttentionConfig(seed=0))
    b = cross_attention_features(X_img, X_pl, CrossAttentionConfig(seed=1))
    assert not np.allclose(a.features, b.features)


# --- KEY: leakage-free feature map ---------------------------------------

def test_feature_map_is_leakage_free_per_subject():
    # transform of a row subset must equal transform of the full matrix sliced
    # to those rows -> proves NO cross-subject statistic enters a feature row.
    X_img = np.random.default_rng(20).normal(size=(50, 16))
    X_pl = np.random.default_rng(21).normal(size=(50, 6))
    full = cross_attention_features(X_img, X_pl).features
    idx = np.array([0, 5, 9, 13, 27, 41])
    sub = cross_attention_features(X_img[idx], X_pl[idx]).features
    np.testing.assert_allclose(sub, full[idx], atol=1e-10)


# --- shape / feature_dim tracks config -----------------------------------

def test_feature_dim_tracks_config():
    X_img = np.random.default_rng(30).normal(size=(20, 16))
    X_pl = np.random.default_rng(31).normal(size=(20, 6))

    t = cross_attention_features(X_img, X_pl, CrossAttentionConfig())
    assert isinstance(t, CrossAttentionTransform)
    assert t.features.shape == (20, t.feature_dim)
    assert np.all(np.isfinite(t.features))
    # residual -> 4 * d_attn; no residual -> 2 * d_attn.
    cfg = CrossAttentionConfig()
    assert t.feature_dim == 4 * cfg.d_attn

    t_nores = cross_attention_features(X_img, X_pl,
                                       CrossAttentionConfig(residual=False))
    assert t_nores.feature_dim == 2 * cfg.d_attn
    assert t_nores.feature_dim != t.feature_dim

    # larger d_model -> wider features.
    t_wide = cross_attention_features(X_img, X_pl,
                                      CrossAttentionConfig(d_model=64))
    assert t_wide.feature_dim > t.feature_dim


# --- attention interpretability structure --------------------------------

def test_attention_report_structure():
    cfg = CrossAttentionConfig()
    X_img = np.random.default_rng(40).normal(size=(25, 16))
    X_pl = np.random.default_rng(41).normal(size=(25, 6))
    att = cross_attention_features(X_img, X_pl, cfg).attention

    assert att["n_heads"] == cfg.n_heads
    i2p = np.array(att["imaging_to_plasma"]["weights"])
    p2i = np.array(att["plasma_to_imaging"]["weights"])
    assert i2p.shape == (cfg.n_heads, cfg.n_tokens_imaging, cfg.n_tokens_plasma)
    assert p2i.shape == (cfg.n_heads, cfg.n_tokens_plasma, cfg.n_tokens_imaging)
    # every query row is a valid distribution over the attended tokens.
    np.testing.assert_allclose(i2p.sum(axis=-1),
                               np.ones((cfg.n_heads, cfg.n_tokens_imaging)),
                               atol=1e-5)
    np.testing.assert_allclose(p2i.sum(axis=-1),
                               np.ones((cfg.n_heads, cfg.n_tokens_plasma)),
                               atol=1e-5)
    mass = att["attention_mass"]
    assert set(mass) == {"imaging", "plasma"}
    assert 0.0 <= mass["imaging"] <= 1.0
    assert 0.0 <= mass["plasma"] <= 1.0


# --- end-to-end fusion: valid auc_ci_perm dicts --------------------------

def test_three_views_produce_valid_auc_ci_perm_dicts():
    df = _synthetic_adcn()
    res = cross_attention_fusion(df, n_boot=200, n_perm=200)
    assert isinstance(res, CrossAttentionResult)
    for name in VIEW_NAMES:
        d = res.views[name]
        for key in ("auc", "ci_lo", "ci_hi", "p_perm", "n",
                    "ci_excludes_chance"):
            assert key in d
        assert d["auc"] is not None and 0.0 <= d["auc"] <= 1.0
        assert d["ci_lo"] is not None and d["ci_hi"] is not None
        assert d["ci_lo"] <= d["auc"] <= d["ci_hi"]


def test_all_views_learnable_above_chance():
    df = _synthetic_adcn(n=160, seed=1)
    res = cross_attention_fusion(df, n_boot=200, n_perm=200)
    for name in VIEW_NAMES:
        assert res.views[name]["auc"] > 0.5


# --- honesty stamps -------------------------------------------------------

def test_honest_source_and_model_stamps_distinct():
    df = _synthetic_adcn()
    res = cross_attention_fusion(df, n_boot=100, n_perm=100)
    assert res.source == SOURCE_CROSS_ATTENTION == "cross_attention_fusion"
    assert res.model == MODEL_CROSS_ATTENTION == "numpy_cross_attention_fusion"
    # distinct from every sibling stamp.
    for bad in ("fitted_fusion", "offline_surrogate"):
        assert res.source != bad
    for bad in ("adni_late_fusion", "adni_attention_late_fusion",
                "surrogate_logistic"):
        assert res.model != bad


def test_disclaimer_is_scrupulously_honest():
    d = CROSS_ATTENTION_DISCLAIMER.lower()
    assert "from-scratch" in d
    assert "not a trained" in d
    assert "not the published vkola" in d
    assert "cross-attention" in d
    assert "adni-only" in d
    assert "not outcome-validated" in d


# --- verdict never overclaims --------------------------------------------

def test_verdict_never_overclaims_without_ci_support():
    df = _synthetic_adcn()
    res = cross_attention_fusion(df, n_boot=200, n_perm=200)
    assert res.best_single in ("emb_only", "plasma_only")
    assert isinstance(res.verdict, str) and res.verdict
    if "superior" in res.verdict.lower():
        assert res.ci_overlap is False


def test_delta_is_fused_minus_best_single():
    df = _synthetic_adcn(seed=2)
    res = cross_attention_fusion(df, n_boot=200, n_perm=200)
    best = res.views[res.best_single]["auc"]
    fused = res.views["cross_attention"]["auc"]
    assert res.delta_auc == round(fused - best, 4)


# --- graceful degradation when plasma absent -----------------------------

def test_graceful_when_plasma_absent():
    df = _synthetic_adcn(with_plasma=False)
    res = cross_attention_fusion(df, n_boot=100, n_perm=100)
    assert isinstance(res, CrossAttentionResult)
    assert res.plasma_available is False
    assert res.error                                   # non-empty, no raise
    assert res.views["emb_only"]["auc"] is not None    # emb view still fit
    assert res.views["cross_attention"].get("unavailable") is True
    assert res.views["plasma_only"].get("unavailable") is True
    assert res.best_single == "emb_only"
    assert res.delta_auc is None
    assert "not run" in res.verdict.lower()
    # stamps stay honest even in the degraded path.
    assert res.source == "cross_attention_fusion"


# --- serialization --------------------------------------------------------

def test_to_dict_is_json_safe():
    df = _synthetic_adcn()
    res = cross_attention_fusion(df, n_boot=100, n_perm=100)
    d = res.to_dict()
    json.loads(json.dumps(d))                           # must not raise
    assert d["source"] == "cross_attention_fusion"
    assert d["model"] == "numpy_cross_attention_fusion"
    assert set(d["views"]) == set(VIEW_NAMES)
    # attention + config nested structures round-trip.
    assert "imaging_to_plasma" in d["attention"]
    assert d["config"]["n_heads"] == DEFAULT_CONFIG.n_heads
    assert d["feature_dim"] > 0


def test_config_to_dict_round_trips():
    cfg = CrossAttentionConfig(n_heads=4, d_model=32)
    d = cfg.to_dict()
    json.loads(json.dumps(d))
    assert d["d_head"] == cfg.d_head == 8
    assert d["d_attn"] == cfg.d_attn == 32


# --- offline / deterministic guard ---------------------------------------

def test_module_imports_without_torch():
    # importing the module must NOT force a torch import on the default path.
    assert "neuroad.integrations.cross_attention" in sys.modules
    # (torch may be present in some envs, but this module never imports it.)
    import inspect
    src = inspect.getsource(ca)
    assert "import torch" not in src
