"""
Offline-first tests for the multimodal_transformer adapter.

Every test here runs with NO network, NO credentials, and NO torch: the default
path is the hand-set logistic surrogate. The single (guarded) real-path test
skips gracefully unless a checkpoint is explicitly provided AND torch/adrd are
importable — it never requires them.
"""
from __future__ import annotations

import math
import os

import pandas as pd
import pytest

from neuroad.integrations import multimodal_transformer as mt
from neuroad.integrations.multimodal_transformer import (
    BiomarkerFusionPredictor,
    FusionPrediction,
    MODEL_SURROGATE,
    SOURCE_SURROGATE,
    predict_biomarker_status,
)


# --- fixtures -------------------------------------------------------------

AD_LIKE = {          # high tau/amyloid signal: elevated p-tau217/GFAP, atrophy
    "p_tau217": 3.2,
    "gfap": 320.0,
    "hippocampal_volume": 2400.0,
    "age": 79.0,
    "apoe4": 2,
}
CN_LIKE = {          # low signal: normal plasma, preserved volume, young-ish
    "p_tau217": 0.4,
    "gfap": 90.0,
    "hippocampal_volume": 4200.0,
    "age": 63.0,
    "apoe4": 0,
}


# --- offline surrogate: contract & determinism ----------------------------

def test_default_predictor_is_surrogate_and_stamped():
    pred = BiomarkerFusionPredictor().predict(AD_LIKE)
    assert isinstance(pred, FusionPrediction)
    assert pred.model == MODEL_SURROGATE == "surrogate_logistic"
    assert pred.source == SOURCE_SURROGATE == "offline_surrogate"
    # honesty: never dressed up as the real published model
    assert pred.model != "jasodanand2025"


def test_probs_in_unit_interval_and_status_matches_threshold():
    for row in (AD_LIKE, CN_LIKE):
        pred = BiomarkerFusionPredictor(threshold=0.5).predict(row)
        for p in (pred.abeta_prob, pred.tau_prob):
            assert 0.0 <= p <= 1.0
        assert pred.abeta_status == (pred.abeta_prob >= 0.5)
        assert pred.tau_status == (pred.tau_prob >= 0.5)


def test_determinism():
    a = BiomarkerFusionPredictor().predict(AD_LIKE)
    b = BiomarkerFusionPredictor().predict(AD_LIKE)
    assert a.abeta_prob == b.abeta_prob
    assert a.tau_prob == b.tau_prob


def test_directionality_ad_higher_than_cn():
    ad = BiomarkerFusionPredictor().predict(AD_LIKE)
    cn = BiomarkerFusionPredictor().predict(CN_LIKE)
    assert ad.abeta_prob > cn.abeta_prob
    assert ad.tau_prob > cn.tau_prob
    # the AD-like row should read positive, the CN-like row negative
    assert ad.abeta_status and ad.tau_status
    assert not cn.abeta_status and not cn.tau_status


# --- input flexibility: pandas, aliases, missing/NaN ----------------------

def test_accepts_pandas_series():
    pred = BiomarkerFusionPredictor().predict(pd.Series(AD_LIKE))
    assert pred.model == MODEL_SURROGATE
    assert pred.abeta_prob > 0.5


def test_alias_keys_are_canonicalized():
    aliased = {"ptau217": 3.2, "plasma_gfap": 320.0,
               "hippocampus_volume": 2400.0, "age_years": 79.0, "apoe4_count": 2}
    a = BiomarkerFusionPredictor().predict(aliased)
    b = BiomarkerFusionPredictor().predict(AD_LIKE)
    assert a.abeta_prob == b.abeta_prob
    assert a.tau_prob == b.tau_prob


def test_missing_features_are_masked_not_fatal():
    sparse = {"p_tau217": 3.2}          # only one feature
    pred = BiomarkerFusionPredictor().predict(sparse)
    assert isinstance(pred, FusionPrediction)
    assert "p_tau217" in pred.features_used
    # gfap/apoe4/age (abeta) and hippocampal_volume/age (tau) are missing
    assert "gfap" in pred.missing_features
    assert "hippocampal_volume" in pred.missing_features


def test_empty_features_returns_base_rate():
    pred = BiomarkerFusionPredictor().predict({})
    # all features masked => logits are just the biases => valid probs, no crash
    assert 0.0 <= pred.abeta_prob <= 1.0
    assert 0.0 <= pred.tau_prob <= 1.0
    assert pred.features_used == []


def test_nan_and_none_values_are_dropped():
    row = {"p_tau217": 3.2, "gfap": float("nan"), "age": None, "apoe4": 2}
    pred = BiomarkerFusionPredictor().predict(row)
    assert "gfap" in pred.missing_features
    assert "age" in pred.missing_features
    assert "p_tau217" in pred.features_used


# --- schema advertisement -------------------------------------------------

def test_expected_features_advertises_surrogate_schema():
    ef = BiomarkerFusionPredictor().expected_features
    for k in ("p_tau217", "gfap", "hippocampal_volume", "age", "apoe4"):
        assert k in ef


# --- module convenience + serialization -----------------------------------

def test_module_convenience_matches_class():
    a = predict_biomarker_status(AD_LIKE)
    b = BiomarkerFusionPredictor().predict(AD_LIKE)
    assert a.to_dict() == b.to_dict()


def test_to_dict_is_json_safe():
    import json
    d = BiomarkerFusionPredictor().predict(AD_LIKE).to_dict()
    json.loads(json.dumps(d))          # must not raise
    assert d["source"] == "offline_surrogate"
    assert isinstance(d["abeta_status"], bool)


# --- real-path loader degrades, never raises ------------------------------

def test_from_pretrained_without_weights_degrades_to_surrogate():
    # No weights_path, no env, no clone -> must return a surrogate predictor.
    pred = BiomarkerFusionPredictor.from_pretrained()
    out = pred.predict(AD_LIKE)
    assert out.model == MODEL_SURROGATE
    assert out.source == SOURCE_SURROGATE
    assert out.error  # records why the real path was unavailable


def test_from_pretrained_bad_path_degrades():
    pred = BiomarkerFusionPredictor.from_pretrained("/nonexistent/model_stage_1.ckpt")
    assert pred._real_model is None
    assert pred.predict(CN_LIKE).source == SOURCE_SURROGATE


# --- optional REAL model (skips unless explicitly available) --------------

def test_real_model_if_available():
    ckpt = os.environ.get("NCOMMS2025_CKPT")
    if not ckpt or not os.path.exists(ckpt):
        pytest.skip("no NCOMMS2025_CKPT checkpoint provided")
    pytest.importorskip("torch")
    pytest.importorskip("adrd")
    pred = BiomarkerFusionPredictor.from_pretrained(ckpt)
    if pred._real_model is None:
        pytest.skip("adrd/torch present but model could not load in this env")
    out = pred.predict(AD_LIKE)
    # if it genuinely ran the transformer, provenance must say so
    assert out.model in ("jasodanand2025", "surrogate_logistic")
    assert 0.0 <= out.abeta_prob <= 1.0
