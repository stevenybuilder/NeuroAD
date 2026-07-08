"""
Unit tests for the M2 data layer. Self-contained: uses sklearn directly (never
agent-1's probe) so it runs without any other module's new code.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neuroad import contract
from neuroad import calibration as cal
from neuroad.data import synthetic, real, loaders


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _cv_auc(X: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y)
    groups = np.arange(len(y))  # one row per subject -> grouping is a no-op
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    proba = cross_val_predict(clf, X, y, cv=GroupKFold(5), groups=groups,
                              method="predict_proba")[:, 1]
    return roc_auc_score(y, proba)


def _conversion_auc(df) -> float:
    X = contract.embedding_matrix(df)
    m = df["conversion"].notna().to_numpy()
    y = df["conversion"][m].astype(int).to_numpy()
    return _cv_auc(X[m], y)


def _site_auc(df) -> float:
    X = contract.embedding_matrix(df)
    sites = df["site"].astype(str).to_numpy()
    y = (sites == sites[0]).astype(int)  # binary one-vs-rest on the first site
    return _cv_auc(X, y)


# --------------------------------------------------------------------------- #
# synthetic: contract-validity + coverage
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("preset", ["SURVIVOR", "KILL"])
def test_synthetic_is_contract_valid(preset):
    df = synthetic.generate_cohort(preset, seed=0)
    contract.validate_table(df)  # raises on any violation
    assert len(df) == synthetic.PRESETS[preset].n_subjects
    assert len(contract.embedding_columns(df)) == synthetic.PRESETS[preset].embed_dim
    # all three dx levels present
    assert set(df["dx"].dropna().unique()) == set(contract.DX_LEVELS)


@pytest.mark.parametrize("preset", ["SURVIVOR", "KILL"])
def test_synthetic_coverage(preset):
    df = synthetic.generate_cohort(preset, seed=0)
    # conversion is defined only for MCI -> partial coverage by design
    conv_cov = df["conversion"].notna().mean()
    assert 0.20 <= conv_cov <= 0.60
    # both converters and non-converters exist
    conv = df["conversion"].dropna().astype(int)
    assert conv.sum() > 0 and (conv == 0).sum() > 0
    # realistic p-tau217 missingness (~56%) -> coverage ~44%
    ptau_cov = df["p_tau217"].notna().mean()
    assert abs((1 - ptau_cov) - cal.PTAU217_MISSINGNESS) < 0.10
    # gfap fully observed; every biomarker column exists
    for b in contract.BIOMARKER_COLUMNS:
        assert b in df.columns


@pytest.mark.parametrize("preset", ["SURVIVOR", "KILL"])
def test_synthetic_is_deterministic(preset):
    a = synthetic.generate_cohort(preset, seed=7)
    b = synthetic.generate_cohort(preset, seed=7)
    # labels are bit-identical for the same seed; embeddings match to machine
    # epsilon (Apple Accelerate BLAS reductions are not last-bit reproducible).
    np.testing.assert_allclose(
        contract.embedding_matrix(a), contract.embedding_matrix(b),
        rtol=0, atol=1e-12)
    assert a["dx"].tolist() == b["dx"].tolist()
    assert a["conversion"].tolist() == b["conversion"].tolist()
    assert a["subject_id"].tolist() == b["subject_id"].tolist()
    # a different seed changes the draw materially
    c = synthetic.generate_cohort(preset, seed=8)
    assert np.abs(contract.embedding_matrix(a)
                  - contract.embedding_matrix(c)).max() > 1e-3


def test_biomarkers_track_disease_axis():
    """p-tau217 and GFAP must rise with diagnosis severity (the anchor signal)."""
    df = synthetic.generate_cohort("SURVIVOR", seed=0)
    dx_ord = df["dx"].map({"CN": 0, "MCI": 1, "AD": 2}).astype(float)
    for marker in ("p_tau217", "gfap"):
        m = df[marker].notna()
        r = np.corrcoef(df[marker][m].to_numpy(), dx_ord[m].to_numpy())[0, 1]
        assert r > 0.10, f"{marker} should correlate positively with severity"


# --------------------------------------------------------------------------- #
# synthetic: the intended verdict mechanics (directional, loose thresholds)
# --------------------------------------------------------------------------- #
def test_survivor_outcome_beats_confound():
    df = synthetic.generate_cohort("SURVIVOR", seed=0)
    conv_auc = _conversion_auc(df)
    site_auc = _site_auc(df)
    # calibrated conversion band, and a positive leakage margin
    assert 0.66 <= conv_auc <= 0.86
    assert conv_auc > site_auc, "survivor: outcome must exceed the confound"


def test_kill_confound_dominates():
    df = synthetic.generate_cohort("KILL", seed=0)
    conv_auc = _conversion_auc(df)
    site_auc = _site_auc(df)
    # site leakage is strong and at/above the outcome (the punchline)
    assert site_auc >= 0.85
    assert site_auc >= conv_auc, "kill: confound must meet or exceed outcome"


# --------------------------------------------------------------------------- #
# real OASIS feeder
# --------------------------------------------------------------------------- #
def test_load_oasis_is_contract_valid():
    df = real.load_oasis("both")
    contract.validate_table(df)
    assert len(contract.embedding_columns(df)) >= 3
    dx = set(df["dx"].dropna().unique())
    assert "CN" in dx and "AD" in dx
    # at least one real converter (OASIS-2 Group == 'Converted')
    assert int((df["conversion"] == 1).sum()) >= 1
    # both pseudo-sites present -> the cohort/batch leakage star is runnable
    assert set(df["site"].astype(str).unique()) == {"OASIS1", "OASIS2"}
    # OASIS has no plasma markers -> those columns are all <NA>
    for marker in ("p_tau217", "gfap", "nfl"):
        assert df[marker].notna().sum() == 0


@pytest.mark.parametrize("which", ["oasis1", "oasis2"])
def test_load_oasis_single_cohort(which):
    df = real.load_oasis(which)
    contract.validate_table(df)
    assert not df["subject_id"].duplicated().any()


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["synthetic:SURVIVOR", "synthetic:KILL", "oasis"])
def test_loaders_dispatch(name):
    df = loaders.load(name)
    contract.validate_table(df)
    assert len(df) > 0


def test_loaders_unknown_raises():
    with pytest.raises(ValueError):
        loaders.load("does-not-exist")


# --------------------------------------------------------------------------- #
# gated stubs are schema-shaped and clearly-marked placeholders
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("stub", ["adni", "oasis3", "nacc", "epad"])
def test_stub_matches_contract_columns(stub):
    import pandas as pd
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "real" / "_stubs" / f"{stub}_stub.csv",
                     comment="#")
    for col in contract.METADATA_COLUMNS:
        assert col in df.columns, f"{stub} stub missing contract column {col}"
    assert any(c.startswith(contract.EMBED_PREFIX) for c in df.columns)
    assert set(df["dx"].dropna().unique()) <= set(contract.DX_LEVELS)
