"""Tests for the attentive MLP probe (attentive_probe.py) + the probe_factory hook.

The NeuroVFM-style choice: a nonlinear MLP head on the frozen embedding run
through the SAME leakage-honest machinery as the linear probe, plus leave-one-
group-out attribution for interpretable grounding. Deterministic, offline,
synthetic — small n and low n_repeats so the MLP fits stay fast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from neuroad import contract, probe
from neuroad import attentive_probe as ap


def _signal(n=80, d=24, seed=1):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    X = rng.normal(size=(n, d))
    X[:, 0] += 1.2 * y
    return X, y


# --- MLPProbe honors the shared probe contract ---------------------------

def test_mlpprobe_fit_predict_contract():
    X, y = _signal()
    p = ap.MLPProbe().fit(X, y)
    proba = p.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert set(p.classes_) == {0, 1}
    scores = p.decision_scores(X)
    assert scores.shape == (len(y),)


def test_mlpprobe_plugs_into_cross_val_auc():
    X, y = _signal()
    auc = probe.cross_val_auc(X, y, n_repeats=1, probe_factory=ap._mlp_factory)
    assert 0.5 <= auc <= 1.0        # recovers the planted signal


# --- probe_factory hook is backward-compatible ---------------------------

def test_probe_factory_none_equals_linear_default():
    """Threading probe_factory=None must not change the default linear result."""
    X, y = _signal()
    a = probe.cross_val_auc(X, y, n_repeats=1)
    b = probe.cross_val_auc(X, y, n_repeats=1, probe_factory=None)
    assert a == b


# --- evaluate: honest MLP-vs-linear comparison ---------------------------

def _contract(n=120, d=16, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([1, 0] * (n // 2))
    X = rng.normal(size=(n, d))
    X[:, :3] += y[:, None] * 0.9
    cols = {f"{contract.EMBED_PREFIX}{i}": X[:, i] for i in range(d)}
    df = pd.DataFrame(cols)
    df["subject_id"] = [f"S{i:04d}" for i in range(n)]
    df["dx"] = pd.Categorical(np.where(y == 1, "AD", "CN"),
                              categories=contract.DX_LEVELS)
    df["conversion"] = pd.array([pd.NA] * n, dtype="Int8")
    df["age"] = rng.normal(72, 6, n) + y * 3
    df["sex"] = pd.Categorical(rng.choice(["M", "F"], n),
                               categories=contract.SEX_LEVELS)
    df["site"] = pd.Categorical([f"s{i % 5}" for i in range(n)])
    df["scanner"] = pd.Categorical(rng.choice(["1.5T", "3T"], n))
    df["amyloid"] = pd.array(rng.integers(0, 2, n), dtype="Int8")
    df["apoe4"] = pd.array(rng.integers(0, 3, n), dtype="Int8")
    df["p_tau217"] = rng.normal(1.0, 0.4, n) + y * 1.0
    df["gfap"] = rng.normal(150, 40, n) + y * 60
    df["nfl"] = rng.normal(30, 8, n) + y * 10
    contract.validate_table(df)
    return df


def test_evaluate_returns_comparison_and_verdict():
    df = _contract()
    res = ap.evaluate(df, "dx_binary", n_repeats=2, n_boot=100, n_perm=100)
    assert set(res) >= {"linear", "mlp", "delta_auc_mlp_minus_linear", "verdict"}
    assert 0.0 <= res["mlp"]["auc"] <= 1.0
    assert 0.0 <= res["linear"]["auc"] <= 1.0
    assert isinstance(res["verdict"], str) and res["verdict"]


# --- feature grounding: interpretable LOO attribution --------------------

def test_feature_grounding_attributes_named_features():
    df = _contract()
    g = ap.feature_grounding(df, "dx_binary", n_repeats=2)     # linear (fast)
    assert g["groups"][:1] == ["embedding"] or "embedding" in g["groups"]
    names = {r["group"] for r in g["attribution"]}
    assert "embedding" in names and "p_tau217" in names
    assert g["top_driver"] in names
    assert 0.0 <= g["full_auc"] <= 1.0


def test_feature_grounding_empty_without_dx():
    df = _contract()
    df["dx"] = pd.Categorical([pd.NA] * len(df), categories=contract.DX_LEVELS)
    assert ap.feature_grounding(df, "dx_binary") == {}
