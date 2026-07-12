"""
Offline, deterministic tests for the FITTED late-fusion head (integrations.fusion).

Every test runs with NO network and NO gated file: the AD/CN + plasma fixture is
built synthetically in-process. These assert that the three views produce valid
``probe.auc_ci_perm`` dicts, that the honesty stamps are correct and distinct from
the surrogate, and that the head degrades gracefully when the plasma block is
absent. The real ADNI numbers are recorded out-of-band, never in CI.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from neuroad import contract
from neuroad.integrations import fusion
from neuroad.integrations.fusion import (
    AttentionFusionResult,
    FusionResult,
    FusionViews,
    MODALITY_IMAGING,
    MODALITY_NEUROJEPA,
    MODALITY_PLASMA,
    MODEL_ATTENTION_FUSION,
    MODEL_LATE_FUSION,
    SOURCE_FITTED,
    attention_fusion,
    build_fusion_views,
    calibration_metrics,
    compare_fusion_vs_single,
    learn_attention_gates,
)


# --- fixture --------------------------------------------------------------

def _synthetic_adcn(n: int = 120, d: int = 16, n_sites: int = 6,
                    seed: int = 0, with_plasma: bool = True) -> pd.DataFrame:
    """A small contract-shaped AD/CN cohort with a learnable emb+plasma signal.

    Half AD (dx=AD, label 1), half CN. AD rows get a mean shift in the embedding
    and elevated plasma markers so every view carries real (but not perfect)
    signal; sites are assigned round-robin so site-disjoint CV has >= 2 groups
    per class. Deterministic under ``seed``.
    """
    rng = np.random.default_rng(seed)
    y = np.array([1, 0] * (n // 2))                     # interleave AD/CN
    dx = np.where(y == 1, "AD", "CN")

    # Embedding: shared noise + class-dependent shift on a few dims.
    X = rng.normal(0.0, 1.0, size=(n, d))
    X[:, :4] += (y[:, None] * 0.9)                       # AD shifted on dims 0-3

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
        # Columns exist but are entirely missing -> plasma block uncompletable.
        for c in ("p_tau217", "gfap", "nfl"):
            df[c] = np.nan

    contract.validate_table(df)                          # must satisfy the contract
    return df


# --- view construction ----------------------------------------------------

def test_build_views_aligned_and_site_disjoint():
    df = _synthetic_adcn()
    v = build_fusion_views(df)
    assert isinstance(v, FusionViews)
    assert v.plasma_available
    # y / groups / all three X blocks share the same row count.
    assert v.emb_only.shape[0] == v.n == len(v.y) == len(v.groups)
    assert v.plasma_tabular.shape == (v.n, len(fusion.FUSION_FEATURES))
    assert v.fusion.shape == (v.n, v.emb_dim + len(fusion.FUSION_FEATURES))
    # fusion == emb concatenated with plasma (identical rows, in order).
    np.testing.assert_allclose(v.fusion[:, :v.emb_dim], v.emb_only)
    np.testing.assert_allclose(v.fusion[:, v.emb_dim:], v.plasma_tabular)
    # only AD/CN, at least two sites, both classes present.
    assert v.n_ad > 0 and v.n_cn > 0
    assert v.n_sites >= 2


def test_views_exclude_mci():
    df = _synthetic_adcn()
    df.loc[:9, "dx"] = "MCI"                              # inject some MCI rows
    v = build_fusion_views(df)
    assert v.n == (df["dx"].astype("string").isin(["AD", "CN"]).sum())


# --- fitted comparison: valid auc_ci_perm dicts ---------------------------

def test_three_views_produce_valid_auc_ci_perm_dicts():
    df = _synthetic_adcn()
    res = compare_fusion_vs_single(df, n_boot=200, n_perm=200)
    assert isinstance(res, FusionResult)
    for name in ("emb_only", "plasma_tabular", "fusion"):
        d = res.views[name]
        for key in ("auc", "ci_lo", "ci_hi", "p_perm", "n",
                    "ci_excludes_chance"):
            assert key in d
        assert d["auc"] is not None and 0.0 <= d["auc"] <= 1.0
        assert d["ci_lo"] is not None and d["ci_hi"] is not None
        assert d["ci_lo"] <= d["auc"] <= d["ci_hi"]


def test_signal_is_learnable_above_chance():
    # The fixture has real signal, so every view should clear chance here.
    df = _synthetic_adcn(n=160, seed=1)
    res = compare_fusion_vs_single(df, n_boot=200, n_perm=200)
    for name in ("emb_only", "plasma_tabular", "fusion"):
        assert res.views[name]["auc"] > 0.5


# --- honesty stamps -------------------------------------------------------

def test_honest_source_and_model_stamps():
    df = _synthetic_adcn()
    res = compare_fusion_vs_single(df, n_boot=100, n_perm=100)
    assert res.source == SOURCE_FITTED == "fitted_fusion"
    assert res.model == MODEL_LATE_FUSION == "adni_late_fusion"
    # Must be clearly DISTINCT from the non-fitted surrogate.
    assert res.source != "offline_surrogate"
    assert res.model != "surrogate_logistic"
    assert "NOT outcome-validated" in res.disclaimer
    assert "ADNI-only" in res.disclaimer


def test_verdict_never_overclaims_without_ci_support():
    df = _synthetic_adcn()
    res = compare_fusion_vs_single(df, n_boot=200, n_perm=200)
    assert res.best_single in ("emb_only", "plasma_tabular")
    assert res.delta_auc is not None
    assert isinstance(res.verdict, str) and res.verdict
    # If the word "superior" appears, the CIs must be non-overlapping.
    if "superior" in res.verdict.lower():
        assert res.ci_overlap is False


def test_delta_is_fusion_minus_best_single():
    df = _synthetic_adcn(seed=2)
    res = compare_fusion_vs_single(df, n_boot=200, n_perm=200)
    best = res.views[res.best_single]["auc"]
    fus = res.views["fusion"]["auc"]
    assert res.delta_auc == round(fus - best, 4)


# --- graceful degradation when plasma is absent ---------------------------

def test_graceful_when_plasma_columns_all_missing():
    df = _synthetic_adcn(with_plasma=False)
    res = compare_fusion_vs_single(df, n_boot=100, n_perm=100)
    assert res.plasma_available is False
    assert res.views["emb_only"]["auc"] is not None      # emb view still fit
    assert res.views["plasma_tabular"].get("unavailable") is True
    assert res.views["fusion"].get("unavailable") is True
    assert res.best_single == "emb_only"
    assert res.delta_auc is None
    assert "unavailable" in res.verdict.lower()
    # Stamps stay honest even in the degraded path.
    assert res.source == "fitted_fusion"


def test_graceful_when_plasma_columns_dropped_entirely():
    df = _synthetic_adcn()
    df = df.drop(columns=["p_tau217", "gfap", "nfl"])
    v = build_fusion_views(df)
    assert v.plasma_available is False
    assert v.plasma_tabular is None and v.fusion is None
    assert v.emb_only.shape[0] == v.n > 0


# --- serialization + package export ---------------------------------------

def test_to_dict_is_json_safe():
    df = _synthetic_adcn()
    d = compare_fusion_vs_single(df, n_boot=100, n_perm=100).to_dict()
    json.loads(json.dumps(d))                            # must not raise
    assert d["source"] == "fitted_fusion"
    assert d["model"] == "adni_late_fusion"
    assert set(d["views"]) == {"emb_only", "plasma_tabular", "fusion"}


def test_package_exports_fusion_symbols():
    from neuroad import integrations
    for sym in ("FusionResult", "FusionViews", "build_fusion_views",
                "compare_fusion_vs_single", "MODEL_LATE_FUSION",
                "SOURCE_FITTED"):
        assert hasattr(integrations, sym)


def test_build_views_carries_row_mask():
    # The row_mask seam must select exactly the kept rows so a third modality
    # can be aligned by subject to the same subjects.
    df = _synthetic_adcn()
    v = build_fusion_views(df)
    assert v.row_mask is not None
    assert v.row_mask.dtype == bool
    assert int(v.row_mask.sum()) == v.n == v.emb_only.shape[0]


# ==========================================================================
# ATTENTION-WEIGHTED LATE FUSION
# ==========================================================================

def _neurojepa_frame(df: pd.DataFrame, signal: float = 0.8, d: int = 768,
                     seed: int = 7, subject_ids=None) -> pd.DataFrame:
    """A NeuroJEPA-shaped external embedding frame (subject_id + 768 emb_* cols).

    Carries a class-dependent shift so the third modality is learnable. When
    ``subject_ids`` is given it overrides the subject alignment (used to exercise
    the non-overlapping-subjects seam path).
    """
    y = df["dx"].astype("string").map({"AD": 1, "CN": 0}).fillna(0).to_numpy()
    rng = np.random.default_rng(seed)
    Z = rng.normal(0.0, 1.0, size=(len(df), d))
    Z[:, :8] += (y[:, None] * signal)
    frame = pd.DataFrame({f"{contract.EMBED_PREFIX}{i}": Z[:, i] for i in range(d)})
    frame["subject_id"] = (list(subject_ids) if subject_ids is not None
                           else df["subject_id"].to_numpy())
    return frame


# --- structure + honesty --------------------------------------------------

def test_attention_fusion_structure_and_stamps():
    df = _synthetic_adcn(n=160, seed=3)
    res = attention_fusion(df, n_boot=200, n_perm=200)
    assert isinstance(res, AttentionFusionResult)
    # provenance is distinct from both the concat head and the surrogate.
    assert res.source == SOURCE_FITTED == "fitted_fusion"
    assert res.model == MODEL_ATTENTION_FUSION == "adni_attention_late_fusion"
    assert res.model != MODEL_LATE_FUSION
    assert "NOT a transformer" in res.disclaimer
    assert "NOT outcome-validated" in res.disclaimer
    # two modalities by default (imaging + plasma), gates sum to 1.
    assert res.modality_names == [MODALITY_IMAGING, MODALITY_PLASMA]
    assert set(res.gates) == {MODALITY_IMAGING, MODALITY_PLASMA}
    assert abs(sum(res.gates.values()) - 1.0) < 1e-6
    assert all(0.0 <= g <= 1.0 for g in res.gates.values())
    # per-modality + fused auc_ci_perm dicts are well-formed.
    for name in res.modality_names:
        d = res.modalities[name]
        for key in ("auc", "ci_lo", "ci_hi", "p_perm", "ci_excludes_chance"):
            assert key in d
        assert 0.0 <= d["auc"] <= 1.0
        assert d["ci_lo"] <= d["auc"] <= d["ci_hi"]
    assert 0.0 <= res.fused["auc"] <= 1.0
    assert res.fused["ci_lo"] <= res.fused["auc"] <= res.fused["ci_hi"]


def test_attention_gate_learner_is_numpy_when_torch_absent():
    # torch is not installed here: the learner MUST honestly report numpy.
    df = _synthetic_adcn(seed=4)
    res = attention_fusion(df, learner="auto", n_boot=50, n_perm=50)
    assert res.gate_learner == "numpy"


def test_torch_gate_degrades_to_none_without_torch():
    from neuroad.integrations import fusion as F
    pytest.importorskip  # noqa: keep import-safe
    try:
        import torch  # noqa: F401
        has_torch = True
    except Exception:
        has_torch = False
    g = F._torch_attention_gate(np.zeros((10, 2)), np.array([0, 1] * 5), 0.1)
    if not has_torch:
        assert g is None
    # dispatch falls back to a valid numpy gate regardless.
    gate, used = learn_attention_gates(np.array([0.4, 0.1]), np.zeros((10, 2)),
                                       np.array([0, 1] * 5), temperature=0.1,
                                       learner="auto")
    if not has_torch:
        assert used == "numpy"
    assert abs(gate.sum() - 1.0) < 1e-6


def test_numpy_gate_monotonic_in_contribution():
    from neuroad.integrations import fusion as F
    # A stronger above-chance modality must receive a larger gate weight.
    gate = F._numpy_attention_gate(np.array([0.30, 0.05]), temperature=0.1)
    assert gate[0] > gate[1]
    assert abs(gate.sum() - 1.0) < 1e-6
    # below-chance contributions are clipped, never up-weighted for being worse.
    gate2 = F._numpy_attention_gate(np.array([-0.2, 0.0]), temperature=0.1)
    assert abs(gate2[0] - gate2[1]) < 1e-9


# --- ablation + attribution ----------------------------------------------

def test_attribution_table_shape_and_top_modality():
    df = _synthetic_adcn(n=160, seed=5)
    res = attention_fusion(df, n_boot=200, n_perm=200)
    assert len(res.attribution) == len(res.modality_names)
    seen = set()
    for row in res.attribution:
        for key in ("modality", "gate", "standalone_auc", "standalone_ci_lo",
                    "standalone_ci_hi", "loo_fused_auc", "attribution_delta"):
            assert key in row
        seen.add(row["modality"])
    assert seen == set(res.modality_names)
    assert res.top_modality in res.modality_names


def test_attribution_delta_matches_fused_minus_loo():
    df = _synthetic_adcn(n=160, seed=6)
    res = attention_fusion(df, n_boot=200, n_perm=200)
    fused_auc = res.fused["auc"]
    for row in res.attribution:
        if row["attribution_delta"] is not None and row["loo_fused_auc"] is not None:
            assert row["attribution_delta"] == round(fused_auc - row["loo_fused_auc"], 4)


# --- calibration ----------------------------------------------------------

def test_calibration_metrics_perfect_and_miscalibrated():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    # perfectly calibrated / perfectly separated -> zero error.
    perfect = calibration_metrics(y, y.astype(float))
    assert perfect["brier"] == 0.0
    assert perfect["ece"] == 0.0
    # over-confident: predict 0.9 for everyone but only half are positive.
    over = calibration_metrics(y, np.full(len(y), 0.9))
    assert over["brier"] > 0.0
    # single occupied bin: confidence 0.9 vs accuracy 0.5 -> ece ~ 0.4.
    assert abs(over["ece"] - 0.4) < 1e-6
    assert 0.0 <= over["mce"] <= 1.0


def test_calibration_reported_in_attention_result():
    df = _synthetic_adcn(n=160, seed=7)
    res = attention_fusion(df, n_boot=100, n_perm=100)
    cal = res.calibration
    for key in ("brier", "ece", "mce", "n_bins", "reliability"):
        assert key in cal
    assert 0.0 <= cal["brier"] <= 1.0
    assert 0.0 <= cal["ece"] <= 1.0
    assert isinstance(cal["reliability"], list) and cal["reliability"]


def test_calibration_metrics_empty_is_safe():
    cal = calibration_metrics(np.asarray([]), np.asarray([]))
    assert cal["brier"] is None and cal["n"] == 0


# --- third-modality (NeuroJEPA) seam -------------------------------------

def test_seam_open_by_default_without_third_modality():
    df = _synthetic_adcn(seed=8)
    res = attention_fusion(df, n_boot=50, n_perm=50)
    assert res.seam_open is True
    assert res.neurojepa_wired is False
    assert MODALITY_NEUROJEPA not in res.modality_names


def test_third_modality_wired_when_embedding_frame_supplied():
    df = _synthetic_adcn(n=160, seed=9)
    nj = _neurojepa_frame(df, signal=0.8)
    res = attention_fusion(df, imaging_embedding=nj, n_boot=100, n_perm=100)
    assert res.neurojepa_wired is True
    assert res.seam_open is False
    assert res.modality_names == [MODALITY_IMAGING, MODALITY_PLASMA,
                                  MODALITY_NEUROJEPA]
    assert set(res.gates) == set(res.modality_names)
    assert abs(sum(res.gates.values()) - 1.0) < 1e-6
    assert len(res.attribution) == 3
    # the fused view still yields a well-formed CI.
    assert res.fused["ci_lo"] <= res.fused["auc"] <= res.fused["ci_hi"]


def test_seam_stays_open_when_subjects_do_not_overlap():
    df = _synthetic_adcn(n=120, seed=10)
    # give the NeuroJEPA frame disjoint subject_ids -> nothing to align.
    disjoint = [f"Z{i:04d}" for i in range(len(df))]
    nj = _neurojepa_frame(df, subject_ids=disjoint)
    res = attention_fusion(df, imaging_embedding=nj, n_boot=50, n_perm=50)
    assert res.neurojepa_wired is False
    assert res.seam_open is True
    assert MODALITY_NEUROJEPA not in res.modality_names
    assert "overlap" in res.error.lower() or "third modality" in res.error.lower()


# --- graceful degradation + serialization --------------------------------

def test_attention_fusion_graceful_when_plasma_absent():
    df = _synthetic_adcn(with_plasma=False)
    res = attention_fusion(df, n_boot=50, n_perm=50)
    assert isinstance(res, AttentionFusionResult)
    assert res.error                        # non-empty, no raise
    assert not res.gates                    # nothing to gate
    assert "not run" in res.verdict.lower()


def test_attention_fusion_verdict_never_overclaims():
    df = _synthetic_adcn(n=160, seed=11)
    res = attention_fusion(df, n_boot=200, n_perm=200)
    assert isinstance(res.verdict, str) and res.verdict
    if "superior" in res.verdict.lower():
        best = max(res.modality_names,
                   key=lambda m: (res.modalities[m]["auc"] or 0.0))
        assert res.fused["ci_lo"] > res.modalities[best]["ci_hi"]


def test_attention_result_to_dict_is_json_safe():
    df = _synthetic_adcn(n=160, seed=12)
    nj = _neurojepa_frame(df, signal=0.7)
    res = attention_fusion(df, imaging_embedding=nj, n_boot=100, n_perm=100)
    d = res.to_dict()
    json.loads(json.dumps(d))               # must not raise
    assert d["source"] == "fitted_fusion"
    assert d["model"] == "adni_attention_late_fusion"
    assert d["neurojepa_wired"] is True
    assert set(d["gates"]) == set(d["modality_names"])
    assert "attribution" in d and "calibration" in d
