"""
test_boltz — offline-first contract tests for the Boltz-2 targeting adapter.

Every test MUST pass with NO network, NO GPU, and NO ``boltz`` install. Boltz-2 is
a REAL open (MIT) AlphaFold3-class predictor, but it needs a GPU we don't have
here, so the shipping contract is: read a committed PRECOMPUTED-results snapshot if
present, else return an honest, non-fabricated 'deferred' object. These tests
assert the adapter (a) never invents coords/confidence/affinity, (b) provenance-
stamps every record, and (c) surfaces REAL numbers only from an injected/committed
snapshot. Any ``boltz`` invocation is mocked.
"""
from __future__ import annotations

import json

import pytest

from neuroad.integrations import boltz as bz
from neuroad.integrations.boltz import (
    AD_TARGETS,
    BOLTZ_LABEL,
    BoltzClient,
    BoltzTargeting,
    boltz_targeting,
    has_precomputed_results,
    ligand_affinity,
)


# ---------------------------------------------------------------------------
# A monkeypatch that hard-fails any network access, to prove no-network safety.
# ---------------------------------------------------------------------------

@pytest.fixture
def no_network(monkeypatch):
    import requests

    def _boom(*args, **kwargs):
        raise AssertionError("network access attempted in an offline test")

    monkeypatch.setattr(requests, "get", _boom)
    monkeypatch.setattr(requests, "post", _boom)
    return monkeypatch


# A synthetic snapshot with REAL-shaped (but test-fabricated, clearly-in-test)
# results for exercising the precomputed path. This is a TEST FIXTURE, never the
# committed data — the committed snapshot ships empty (see the honesty test below).
_FIXTURE_SNAPSHOT = {
    "_provenance": {"model": "Boltz-2", "license": "MIT", "captured": "2026-07-11T00:00:00Z"},
    "complexes": {
        "APP|MAPT": {
            "gene_a": "APP", "gene_b": "MAPT",
            "iptm": 0.82, "ptm": 0.78, "pae": 6.4, "confidence_score": 0.80,
        }
    },
    "affinities": {
        "BACE1::verubecestat": {
            "gene_a": "BACE1", "ligand_id": "verubecestat",
            "ligand_smiles": "O=C(N)c1ccccc1",
            "iptm": 0.71, "ptm": 0.69, "confidence_score": 0.70,
            "binding_affinity": 7.9, "binding_probability": 0.88,
        }
    },
}


# ---------------------------------------------------------------------------
# Import / resolution / snapshot integrity
# ---------------------------------------------------------------------------

def test_module_imports_and_client_constructs_offline():
    client = BoltzClient(prefer_offline=True)
    assert isinstance(client, BoltzClient)
    assert len(AD_TARGETS) == 12


def test_resolve_symbol_normalizes_and_rejects_garbage():
    c = BoltzClient(prefer_offline=True)
    assert c.resolve_symbol("app") == "APP"
    assert c.resolve_symbol("  MAPT ") == "MAPT"
    assert c.resolve_symbol("") is None
    assert c.resolve_symbol("   ") is None
    assert c.resolve_symbol("bad symbol!") is None


def test_committed_snapshot_has_real_complexes_and_defers_unfolded():
    # A REAL Boltz-2 GPU run (scripts/boltz_fold_colab.py on A100) has been
    # committed: the hero AD complexes carry real confidence scalars, while any
    # pair NOT folded still returns an honest deferred (never fabricated).
    assert has_precomputed_results() is True
    ev = boltz_targeting("APP", "MAPT", prefer_offline=True)   # folded pair
    assert ev["source"] == "precomputed_snapshot"
    assert ev["status"] == "predicted"
    assert 0.0 <= ev["iptm"] <= 1.0 and 0.0 <= ev["ptm"] <= 1.0
    assert ev["provenance"].get("model") == "Boltz-2"
    # A pair with no committed fold is still honestly deferred.
    unfolded = boltz_targeting("APP", "TREM2", prefer_offline=True)
    assert unfolded["source"] == "deferred"
    assert unfolded["status"] == "deferred"
    assert "GPU run required" in unfolded["note"]


# ---------------------------------------------------------------------------
# Deferred path — honest, never fabricated
# ---------------------------------------------------------------------------

def test_deferred_complex_has_all_none_scalars(no_network):
    # Empty snapshot injected -> deferred; must not touch the network.
    c = BoltzClient(prefer_offline=True, snapshot={"complexes": {}, "affinities": {}})
    t = c.predict_complex("APP", "MAPT")
    assert t.source == "deferred" and t.status == "deferred"
    assert t.note == bz.DEFERRED_NOTE
    for scalar in (t.iptm, t.ptm, t.pae, t.confidence_score,
                   t.binding_affinity, t.binding_probability):
        assert scalar is None
    assert t.provenance.get("model") == "Boltz-2"
    assert t.provenance.get("license") == "MIT"


def test_deferred_affinity_has_none_scalars():
    c = BoltzClient(prefer_offline=True, snapshot={})
    t = c.predict_affinity("BACE1", "O=C(N)c1ccccc1", ligand_id="verubecestat")
    assert t.kind == "target_ligand"
    assert t.status == "deferred"
    assert t.binding_affinity is None and t.binding_probability is None
    assert t.ligand_id == "verubecestat"


def test_self_pair_is_deferred_not_folded():
    t = BoltzClient(prefer_offline=True, snapshot={}).predict_complex("APP", "APP")
    assert t.status == "deferred"
    assert "self-pair" in t.error


def test_malformed_gene_is_deferred_stub():
    t = BoltzClient(prefer_offline=True, snapshot={}).predict_complex("APP", "")
    assert t.status == "deferred"
    assert "could not resolve" in t.error
    assert t.iptm is None


# ---------------------------------------------------------------------------
# Precomputed snapshot path — REAL numbers surfaced + provenance stamped
# ---------------------------------------------------------------------------

def test_injected_snapshot_complex_is_precomputed_and_labeled(no_network):
    c = BoltzClient(prefer_offline=True, snapshot=_FIXTURE_SNAPSHOT)
    t = c.predict_complex("APP", "MAPT")
    assert t.source == "precomputed_snapshot"
    assert t.status == "predicted"
    assert t.note == BOLTZ_LABEL
    assert t.iptm == pytest.approx(0.82)
    assert t.ptm == pytest.approx(0.78)
    assert t.pae == pytest.approx(6.4)
    assert t.confidence_score == pytest.approx(0.80)
    # Provenance from the snapshot's _provenance block is stamped through.
    assert t.provenance.get("model") == "Boltz-2"
    assert t.provenance.get("captured") == "2026-07-11T00:00:00Z"


def test_injected_snapshot_complex_is_order_independent():
    c = BoltzClient(prefer_offline=True, snapshot=_FIXTURE_SNAPSHOT)
    a = c.predict_complex("APP", "MAPT")
    b = c.predict_complex("MAPT", "APP")
    assert a.iptm == b.iptm == pytest.approx(0.82)
    assert a.source == b.source == "precomputed_snapshot"


def test_injected_snapshot_affinity_surfaces_binding_scalars():
    c = BoltzClient(prefer_offline=True, snapshot=_FIXTURE_SNAPSHOT)
    t = c.predict_affinity("BACE1", ligand_id="verubecestat")
    assert t.source == "precomputed_snapshot"
    assert t.kind == "target_ligand"
    assert t.binding_affinity == pytest.approx(7.9)
    assert t.binding_probability == pytest.approx(0.88)
    assert t.ligand_smiles == "O=C(N)c1ccccc1"


def test_has_precomputed_results_true_for_populated_snapshot(tmp_path):
    p = tmp_path / "snap.json"
    p.write_text(json.dumps(_FIXTURE_SNAPSHOT))
    assert has_precomputed_results(str(p)) is True


def test_to_dict_roundtrip_shape():
    d = BoltzClient(prefer_offline=True, snapshot=_FIXTURE_SNAPSHOT).predict_complex(
        "APP", "MAPT").to_dict()
    assert set(d) == {
        "gene_a", "gene_b", "kind", "ligand_id", "ligand_smiles", "iptm", "ptm",
        "pae", "confidence_score", "binding_affinity", "binding_probability",
        "source", "status", "note", "provenance", "error",
    }
    assert d["gene_a"] == "APP" and d["gene_b"] == "MAPT"
    assert d["source"] == "precomputed_snapshot"


# ---------------------------------------------------------------------------
# _num helper — never fabricates a 0.0 for a missing value
# ---------------------------------------------------------------------------

def test_num_coerces_and_none_is_preserved():
    assert bz._num(None) is None
    assert bz._num("not a number") is None
    assert bz._num(0.5) == pytest.approx(0.5)
    assert bz._num("0.5") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# YAML builder + output parser (pure, offline, unit-testable)
# ---------------------------------------------------------------------------

def test_build_boltz_yaml_protein_pair():
    y = bz._build_boltz_yaml([("protein", "MSEQ"), ("protein", "NSEQ")])
    assert "version: 1" in y
    assert "id: A" in y and "id: B" in y
    assert "sequence: MSEQ" in y and "sequence: NSEQ" in y
    assert "affinity" not in y


def test_build_boltz_yaml_target_ligand_with_affinity():
    y = bz._build_boltz_yaml(
        [("protein", "MSEQ"), ("ligand", "O=C(N)c1ccccc1")], affinity=True)
    assert "smiles: 'O=C(N)c1ccccc1'" in y
    assert "properties:" in y and "binder: B" in y


def test_parse_boltz_outputs_reads_confidence_and_affinity(tmp_path):
    out = tmp_path / "out" / "predictions" / "job"
    out.mkdir(parents=True)
    (out / "confidence_job_model_0.json").write_text(json.dumps(
        {"confidence_score": 0.8, "ptm": 0.78, "iptm": 0.82}))
    (out / "affinity_job.json").write_text(json.dumps(
        {"affinity_pred_value": 7.9, "affinity_probability_binary": 0.88}))
    conf, aff = bz._parse_boltz_outputs(tmp_path)
    assert conf["iptm"] == pytest.approx(0.82)
    assert aff["affinity_pred_value"] == pytest.approx(7.9)


def test_parse_boltz_outputs_missing_is_none(tmp_path):
    assert bz._parse_boltz_outputs(tmp_path) is None


# ---------------------------------------------------------------------------
# Local-run path is LAZY and degrades honestly when boltz is absent
# ---------------------------------------------------------------------------

def test_run_boltz_cli_returns_none_without_boltz_installed(monkeypatch):
    # boltz is not installed in this environment -> find_spec returns None -> None.
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert bz._run_boltz_cli([("protein", "MSEQ")]) is None


def test_local_run_requested_but_boltz_absent_degrades_to_deferred(monkeypatch):
    # Even with allow_local_run, a missing boltz install degrades to deferred,
    # never fabricating a prediction.
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    c = BoltzClient(prefer_offline=False, allow_local_run=True,
                    snapshot={"complexes": {}, "affinities": {}})
    t = c.predict_complex("APP", "MAPT")
    assert t.status == "deferred"
    assert t.iptm is None


def test_local_run_mocked_success_is_labeled_boltz_live(monkeypatch):
    # Mock the whole CLI run so no real boltz/GPU/network is touched.
    monkeypatch.setattr(bz, "_sequences_for", lambda genes: ["MSEQ", "NSEQ"])
    monkeypatch.setattr(bz, "_run_boltz_cli",
                        lambda *a, **k: ({"iptm": 0.9, "ptm": 0.85,
                                          "confidence_score": 0.88}, None))
    c = BoltzClient(prefer_offline=False, allow_local_run=True,
                    snapshot={"complexes": {}, "affinities": {}})
    t = c.predict_complex("APP", "MAPT")
    assert t.source == "boltz_live" and t.status == "predicted"
    assert t.iptm == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# No-network safety on the offline default path
# ---------------------------------------------------------------------------

def test_prefer_offline_never_touches_network(no_network):
    c = BoltzClient(prefer_offline=True, snapshot=_FIXTURE_SNAPSHOT)
    assert c.predict_complex("APP", "MAPT").source == "precomputed_snapshot"
    assert c.predict_complex("APP", "TREM2").source == "deferred"


# ---------------------------------------------------------------------------
# Module wrappers
# ---------------------------------------------------------------------------

def test_module_wrappers_return_dicts():
    assert boltz_targeting("APP", "MAPT")["kind"] == "complex"
    assert ligand_affinity("BACE1", "O=C(N)c1ccccc1",
                           ligand_id="x")["kind"] == "target_ligand"


# ---------------------------------------------------------------------------
# Additive wiring into the translation loop
# ---------------------------------------------------------------------------

def test_translation_surfaces_real_boltz_for_folded_mechanism():
    # The committed GPU run folded APP complexes -> the amyloid_cascade lead
    # (top target APP) surfaces a REAL precomputed boltz_targeting.
    from neuroad.harness import translation
    lead = translation.translate("amyloid_cascade", prefer_offline=True)
    assert lead["status"] == "translated"
    bt = lead["structure"].get("boltz_targeting")
    assert bt is not None and bt["status"] == "predicted"
    assert bt["source"] == "precomputed_snapshot"
    assert 0.0 <= bt["iptm"] <= 1.0
    assert lead["provenance"].get("boltz") == "precomputed_snapshot"


def test_translation_defers_boltz_for_unfolded_mechanism():
    # A mechanism whose top target has NO committed fold (glial -> HRAS) still
    # omits the field honestly — no fabrication, provenance notes 'deferred'.
    from neuroad.harness import translation
    lead = translation.translate("glial", prefer_offline=True)
    assert lead["status"] == "translated"
    assert "boltz_targeting" not in lead["structure"]
    assert lead["provenance"].get("boltz") == "deferred"


def test_translation_attaches_boltz_field_when_result_exists(monkeypatch):
    # When a REAL precomputed/GPU result exists, the promoted target's card carries
    # a boltz_targeting field. Mock the client so no boltz/GPU/network is touched.
    from neuroad.harness import translation

    fake = BoltzTargeting(
        gene_a="APP", gene_b="MAPT", kind="complex", iptm=0.82, ptm=0.78,
        pae=6.4, confidence_score=0.80, source="precomputed_snapshot",
        status="predicted")
    monkeypatch.setattr(BoltzClient, "predict_complex",
                        lambda self, a, b: fake)
    lead = translation.translate("amyloid_cascade", prefer_offline=True)
    bt = lead["structure"].get("boltz_targeting")
    assert bt is not None
    assert bt["source"] == "precomputed_snapshot"
    assert bt["iptm"] == pytest.approx(0.82)
    assert lead["provenance"].get("boltz") == "precomputed_snapshot"
