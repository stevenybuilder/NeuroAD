"""
Tests for the OpenBHB real-data feeder — the STAR scanner-leakage star on
healthy-controls-only real data.
"""
from __future__ import annotations

import numpy as np
import pytest

from neuroad import contract
from neuroad.data import openbhb, loaders


def test_load_openbhb_is_contract_valid():
    df = openbhb.load_openbhb()
    contract.validate_table(df)  # raises on any violation
    assert len(df) >= 3000
    assert len(contract.embedding_columns(df)) == 4  # tiv, csfv, gmv, wmv
    assert not df["subject_id"].duplicated().any()


def test_openbhb_is_healthy_controls_only():
    df = openbhb.load_openbhb()
    # every subject is CN -> the disease probe cannot run, by design
    assert set(df["dx"].dropna().unique()) == {"CN"}


def test_openbhb_has_multiple_field_strengths():
    df = openbhb.load_openbhb()
    # the STAR test needs >= 2 scanner (field-strength) classes
    assert df["scanner"].nunique(dropna=True) >= 2
    assert df["site"].nunique(dropna=True) > 1
    # scanner labels are field-strength strings like '1.5T'/'3.0T'
    for lab in df["scanner"].dropna().unique():
        assert str(lab).endswith("T")


def test_openbhb_has_no_molecular_markers():
    df = openbhb.load_openbhb()
    for marker in ("p_tau217", "gfap", "nfl"):
        assert df[marker].notna().sum() == 0
    for col in ("conversion", "amyloid", "apoe4"):
        assert df[col].notna().sum() == 0


def test_openbhb_age_not_in_embedding():
    """Age is a covariate, not a structural feature — it must stay out of emb."""
    df = openbhb.load_openbhb()
    X = contract.embedding_matrix(df)
    age = df["age"].to_numpy(float)
    # no embedding column should be (nearly) perfectly correlated with age
    for j in range(X.shape[1]):
        r = abs(np.corrcoef(X[:, j], age)[0, 1])
        assert r < 0.98


def test_real_scanner_leakage_is_strong():
    """The real batch effect: structural embedding predicts the scanner at
    AUC clearly > 0.7 in healthy subjects with NO disease."""
    result = openbhb.real_scanner_leakage()
    assert result["healthy_controls_only"] is True
    assert result["scanner_auc"] > 0.7, (
        f"expected strong real scanner leakage, got {result['scanner_auc']}")
    # prior art must be cited (we ship the tool, we don't claim the mechanism)
    assert len(result["prior_art"]) >= 1
    assert "AUC" in result["message"]


def test_loaders_dispatch_openbhb():
    df = loaders.load("openbhb")
    contract.validate_table(df)
    assert len(df) > 0
    assert "openbhb" in loaders.AVAILABLE
