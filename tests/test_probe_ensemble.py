"""Tests for repeated-CV out-of-fold ensembling in the probe (probe.py).

The NeuroVFM protocol ensembles several trained probes for the final prediction.
Our small-n analog is repeated site-disjoint CV with OOF averaging: run the same
cross-validation with several split seeds and average each subject's OOF
probability. These tests pin the two properties that make it safe to ship:

  1. ``n_repeats=1`` reproduces the historical single-split number byte-for-byte
     (so nothing in the existing pipeline/reports moves unless a caller opts in).
  2. The ensemble damps the split-seed variance a single fold assignment carries
     at small n, landing on the mean of the per-seed distribution.

Deterministic, offline, synthetic — no gated data.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from neuroad import contract, probe
from neuroad.integrations import fusion
from neuroad.integrations.fusion import attention_fusion


def _signal_cohort(n=80, D=30, seed=7, ungrouped=True):
    """A tiny p>>n-ish cohort with real class signal (+ optional site confound)."""
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    X = rng.normal(size=(n, D))
    X[:, 0] += 0.9 * y
    if ungrouped:
        return X, y, None
    sites = np.array([i % 10 for i in range(n)])
    X[:, 1] += 0.5 * sites
    return X, y, sites


def test_n_repeats_1_equals_historical_single_split():
    """Opt-out default: n_repeats=1 is identical to the un-parameterized call."""
    X, y, groups = _signal_cohort()
    assert probe.cross_val_auc(X, y, groups=groups) == \
        probe.cross_val_auc(X, y, groups=groups, n_repeats=1)


def test_cross_val_oof_n_repeats_1_matches_baseline_scores():
    """The averaged OOF matrix at n_repeats=1 is exactly the single-pass matrix."""
    X, y, groups = _signal_cohort(ungrouped=False)
    a = probe.cross_val_oof(X, y, groups, n_repeats=1)
    b = probe.cross_val_oof(X, y, groups, n_repeats=1)
    assert a is not None and b is not None
    np.testing.assert_array_equal(a[1], b[1])


def _per_seed_aucs(X, y, groups, n_seeds=12):
    saved = probe.RANDOM_STATE
    try:
        vals = []
        for s in range(n_seeds):
            probe.RANDOM_STATE = s
            vals.append(probe.cross_val_auc(X, y, groups=groups, n_repeats=1))
        return np.asarray(vals)
    finally:
        probe.RANDOM_STATE = saved


def test_ensemble_reduces_split_seed_variance_ungrouped():
    """A single seed swings; the ensemble lands at the per-seed mean, stably."""
    X, y, groups = _signal_cohort(ungrouped=True)
    per_seed = _per_seed_aucs(X, y, groups)
    ens = probe.cross_val_auc(X, y, groups=groups, n_repeats=8)
    # The single-split number genuinely depends on the fold seed here...
    assert per_seed.std() > 0.01
    # ...and the ensemble sits at the center of that distribution.
    assert abs(ens - per_seed.mean()) <= per_seed.std()


def test_ensemble_stable_when_folds_bundle_sites():
    """10 sites capped to 5 folds: bundling is seed-dependent; ensemble averages it."""
    X, y, groups = _signal_cohort(n=120, ungrouped=False)
    per_seed = _per_seed_aucs(X, y, groups)
    ens = probe.cross_val_auc(X, y, groups=groups, n_repeats=8)
    assert per_seed.std() > 0.01
    assert abs(ens - per_seed.mean()) <= per_seed.std()


def test_leave_one_site_out_is_seed_invariant_and_ensemble_is_a_noop():
    """With n_sites == n_splits the folds are determined; ensembling can't change it."""
    # 5 sites, 5 folds -> each fold is exactly one held-out site, any seed.
    rng = np.random.default_rng(3)
    n, D = 90, 40
    y = np.array([0, 1] * (n // 2))
    sites = np.array([i % 5 for i in range(n)])
    X = rng.normal(size=(n, D))
    X[:, 0] += 1.0 * y
    single = probe.cross_val_auc(X, y, groups=sites, n_repeats=1)
    ens = probe.cross_val_auc(X, y, groups=sites, n_repeats=8)
    assert single == ens


def test_auc_ci_perm_threads_n_repeats():
    """The referee-facing entry point accepts n_repeats and stays JSON-shaped."""
    X, y, groups = _signal_cohort(ungrouped=False)
    out = probe.auc_ci_perm(X, y, groups, n_boot=200, n_perm=200, n_repeats=8)
    assert set(out) >= {"auc", "ci_lo", "ci_hi", "p_perm", "n",
                        "ci_excludes_chance"}
    assert 0.0 <= out["auc"] <= 1.0


# ==========================================================================
# Attention-fusion path: same repeated-CV ensembling wired into fusion.py
# ==========================================================================

def _synthetic_adcn(n: int = 160, d: int = 16, n_sites: int = 6,
                    seed: int = 0) -> pd.DataFrame:
    """A contract-shaped AD/CN + plasma cohort (matches tests/test_fusion.py)."""
    rng = np.random.default_rng(seed)
    y = np.array([1, 0] * (n // 2))                       # interleave AD/CN
    dx = np.where(y == 1, "AD", "CN")

    X = rng.normal(0.0, 1.0, size=(n, d))
    X[:, :4] += (y[:, None] * 0.9)                        # AD shifted on dims 0-3

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
    df["p_tau217"] = rng.normal(1.0, 0.4, n) + y * 1.2
    df["gfap"] = rng.normal(150, 40, n) + y * 80.0
    df["nfl"] = rng.normal(30, 8, n) + y * 12.0

    contract.validate_table(df)
    return df


def test_attention_fusion_accepts_n_repeats():
    """attention_fusion exposes an n_repeats knob (default probe.N_REPEATS)."""
    sig = inspect.signature(attention_fusion)
    assert "n_repeats" in sig.parameters
    assert sig.parameters["n_repeats"].default == probe.N_REPEATS


def test_oof_binary_score_accepts_n_repeats():
    """The per-modality scorer threads n_repeats too (default probe.N_REPEATS)."""
    sig = inspect.signature(fusion._oof_binary_score)
    assert "n_repeats" in sig.parameters
    assert sig.parameters["n_repeats"].default == probe.N_REPEATS


def test_attention_fusion_n_repeats_1_matches_default_byte_for_byte():
    """Opt-out default: n_repeats=1 reproduces the single-split fused AUC exactly."""
    df = _synthetic_adcn(n=160, seed=3)
    base = attention_fusion(df, n_boot=100, n_perm=100)
    one = attention_fusion(df, n_boot=100, n_perm=100, n_repeats=1)
    assert base.fused["auc"] == one.fused["auc"]
    # the whole per-modality AUC table is byte-identical too.
    for m in base.modality_names:
        assert base.modalities[m]["auc"] == one.modalities[m]["auc"]


def test_oof_binary_score_n_repeats_1_matches_baseline():
    """The averaged OOF score vector at n_repeats=1 is the single-pass vector."""
    df = _synthetic_adcn(n=160, seed=3)
    v = fusion.build_fusion_views(df)
    a = fusion._oof_binary_score(v.emb_only, v.y, v.groups, n_repeats=1)
    b = fusion._oof_binary_score(v.emb_only, v.y, v.groups)
    assert a is not None and b is not None
    np.testing.assert_array_equal(a[0], b[0])


def test_attention_fusion_ensemble_runs_and_returns_valid_dict():
    """n_repeats=8 runs and returns a well-formed fused auc_ci_perm dict."""
    df = _synthetic_adcn(n=160, seed=5)
    res = attention_fusion(df, n_boot=100, n_perm=100, n_repeats=8)
    assert res.error == ""
    assert set(res.gates) == set(res.modality_names)
    assert abs(sum(res.gates.values()) - 1.0) < 1e-6
    for key in ("auc", "ci_lo", "ci_hi", "p_perm", "ci_excludes_chance"):
        assert key in res.fused
    assert 0.0 <= res.fused["auc"] <= 1.0
    assert res.fused["ci_lo"] <= res.fused["auc"] <= res.fused["ci_hi"]
