"""
test_opentargets — offline-first contract tests for the Open Targets adapter.

Every test here MUST pass with NO network and NO credentials. The one live test
probes reachability first and skips gracefully when the API host is unreachable.
"""
from __future__ import annotations

import socket

import pytest

from neuroad.integrations import opentargets as ot
from neuroad.integrations.opentargets import (
    AD_DISEASE_ID,
    AD_TARGET_ENSEMBL,
    OpenTargetsClient,
    TargetAssociation,
    ad_target_evidence,
)


# ---------------------------------------------------------------------------
# Reachability probe for the (single) guarded live test.
# ---------------------------------------------------------------------------

def _api_reachable() -> bool:
    try:
        socket.create_connection(
            ("api.platform.opentargets.org", 443), timeout=3).close()
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Import / constants
# ---------------------------------------------------------------------------

def test_module_imports_offline():
    # Constructing the offline client must not touch the network or raise.
    client = OpenTargetsClient(prefer_offline=True)
    assert isinstance(client, OpenTargetsClient)


def test_ad_disease_id_is_the_mondo_id():
    # EFO_0000249 is stale (returns null); MONDO_0004975 is the correct AD id.
    assert AD_DISEASE_ID == "MONDO_0004975"


def test_engine_genes_have_ensembl_ids():
    for sym in ["APP", "MAPT", "APOE", "PSEN1", "PSEN2", "BACE1",
                "TREM2", "HRAS", "MAPK1", "ESR1", "CLU", "BIN1"]:
        assert AD_TARGET_ENSEMBL[sym].startswith("ENSG")


# ---------------------------------------------------------------------------
# Offline disease -> targets ranking (provenance + integrity)
# ---------------------------------------------------------------------------

def test_disease_targets_offline_labeled_and_ranked():
    client = OpenTargetsClient(prefer_offline=True)
    targets = client.disease_targets(top_n=8)
    assert len(targets) == 8
    for t in targets:
        assert isinstance(t, TargetAssociation)
        assert t.source == "offline_snapshot"          # honestly labeled fallback
        assert 0.0 <= t.association_score <= 1.0
        assert t.ensembl_id.startswith("ENSG")
    # Ranked descending by association score.
    scores = [t.association_score for t in targets]
    assert scores == sorted(scores, reverse=True)
    # The canonical top AD targets surface at the top.
    top_genes = {t.gene for t in targets[:4]}
    assert {"APP", "PSEN1", "APOE"} <= top_genes


def test_disease_targets_datatype_breakdown_present():
    client = OpenTargetsClient(prefer_offline=True)
    app = next(t for t in client.disease_targets(top_n=10) if t.gene == "APP")
    assert app.datatype_scores  # non-empty evidence breakdown
    for v in app.datatype_scores.values():
        assert 0.0 <= v <= 1.0
    assert app.n_known_drugs > 0  # APP has clinical/known drugs in the snapshot


def test_disease_targets_top_n_zero_is_empty():
    assert OpenTargetsClient(prefer_offline=True).disease_targets(top_n=0) == []


# ---------------------------------------------------------------------------
# Offline single-target association
# ---------------------------------------------------------------------------

def test_target_association_offline_labeled():
    a = OpenTargetsClient(prefer_offline=True).target_association("BACE1")
    assert a is not None
    assert a.source == "offline_snapshot"
    assert a.gene == "BACE1"
    assert a.ensembl_id == "ENSG00000186318"
    assert 0.0 <= a.association_score <= 1.0
    assert a.datatype_scores  # has an evidence breakdown


def test_target_association_case_insensitive():
    a = OpenTargetsClient(prefer_offline=True).target_association("apoe")
    assert a is not None and a.gene == "APOE"


def test_target_association_score_always_in_unit_interval():
    client = OpenTargetsClient(prefer_offline=True)
    for sym in AD_TARGET_ENSEMBL:
        a = client.target_association(sym)
        assert a is not None, f"{sym} missing from snapshot"
        assert 0.0 <= a.association_score <= 1.0
        for v in a.datatype_scores.values():
            assert 0.0 <= v <= 1.0


def test_unknown_gene_degrades_gracefully():
    client = OpenTargetsClient(prefer_offline=True)
    assert client.target_association("NOT_A_REAL_GENE") is None
    assert client.known_drugs("NOT_A_REAL_GENE") == []
    assert client.target_association("") is None


# ---------------------------------------------------------------------------
# Offline known drugs
# ---------------------------------------------------------------------------

def test_known_drugs_offline_shape():
    drugs = OpenTargetsClient(prefer_offline=True).known_drugs("BACE1")
    assert drugs, "BACE1 should have known drugs in the snapshot"
    for d in drugs:
        assert set(d) >= {"drug", "phase", "mechanism", "drug_type"}
    names = {d["drug"].upper() for d in drugs}
    assert "VERUBECESTAT" in names  # a real BACE1 inhibitor captured live


def test_to_dict_roundtrip():
    d = OpenTargetsClient(prefer_offline=True).target_association("APP").to_dict()
    assert d["source"] == "offline_snapshot"
    assert d["gene"] == "APP"
    assert 0.0 <= d["association_score"] <= 1.0
    assert isinstance(d["datatype_scores"], dict)
    assert isinstance(d["n_known_drugs"], int)


def test_ad_target_evidence_convenience_offline():
    ev = ad_target_evidence("BACE1", prefer_offline=True)
    assert ev["gene"] == "BACE1"
    assert ev["disease_id"] == AD_DISEASE_ID
    assert ev["association"] is not None
    assert 0.0 <= ev["association_score"] <= 1.0
    assert ev["n_known_drugs"] == len(ev["known_drugs"])
    assert ev["source"] == "offline_snapshot"


def test_ad_target_evidence_unknown_gene():
    ev = ad_target_evidence("NOT_A_REAL_GENE", prefer_offline=True)
    assert ev["association"] is None
    assert ev["known_drugs"] == []


# ---------------------------------------------------------------------------
# Live path falls back to snapshot when the network is monkeypatched away.
# ---------------------------------------------------------------------------

def test_network_failure_falls_back_to_snapshot(monkeypatch):
    client = OpenTargetsClient(prefer_offline=False)

    def _boom(*args, **kwargs):
        raise OSError("network down")

    import requests
    monkeypatch.setattr(requests, "post", _boom)
    targets = client.disease_targets(top_n=5)
    assert targets and all(t.source == "offline_snapshot" for t in targets)
    a = client.target_association("APP")
    assert a is not None and a.source == "offline_snapshot"


def test_non_200_falls_back_to_snapshot(monkeypatch):
    client = OpenTargetsClient(prefer_offline=False)

    class _Resp:
        status_code = 503

        def json(self):
            return {}

    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
    targets = client.disease_targets(top_n=3)
    assert targets and targets[0].source == "offline_snapshot"


def test_graphql_error_falls_back_to_snapshot(monkeypatch):
    # A 200 response carrying GraphQL "errors" must degrade, never surface.
    client = OpenTargetsClient(prefer_offline=False)

    class _Resp:
        status_code = 200

        def json(self):
            return {"errors": [{"message": "bad query"}], "data": None}

    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
    a = client.target_association("PSEN1")
    assert a is not None and a.source == "offline_snapshot"


def test_monkeypatched_live_response_is_labeled_live(monkeypatch):
    client = OpenTargetsClient(prefer_offline=False)

    class _Resp:
        status_code = 200

        def json(self):
            return {"data": {"disease": {
                "id": AD_DISEASE_ID, "name": "Alzheimer disease",
                "associatedTargets": {"count": 1, "rows": [{
                    "target": {"id": "ENSG00000142192", "approvedSymbol": "APP"},
                    "score": 0.9,
                    "datatypeScores": [{"id": "genetic_association", "score": 0.92}],
                }]}}}}

    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
    targets = client.disease_targets(top_n=1)
    assert len(targets) == 1
    assert targets[0].source == "live"                 # honestly labeled as live
    assert targets[0].gene == "APP"
    assert targets[0].association_score == pytest.approx(0.9)
    assert targets[0].datatype_scores["genetic_association"] == pytest.approx(0.92)


def test_scores_are_clamped_to_unit_interval():
    # Defensive: out-of-range/garbage scores are coerced into [0, 1].
    assert ot._clamp01(1.7) == 1.0
    assert ot._clamp01(-0.3) == 0.0
    assert ot._clamp01("not a number") == 0.0
    assert ot._clamp01(0.42) == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Guarded LIVE test — skips when the Open Targets API is unreachable.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _api_reachable(), reason="Open Targets API unreachable")
def test_live_disease_targets_smoke():
    targets = OpenTargetsClient(prefer_offline=False).disease_targets(top_n=8)
    assert targets
    # Online we expect live labels + plausible AD targets; a transient API failure
    # still yields an honest offline fallback (never a raise, never a mislabel).
    for t in targets:
        assert t.source in ("live", "offline_snapshot")
        assert 0.0 <= t.association_score <= 1.0
        assert t.ensembl_id.startswith("ENSG")
    if all(t.source == "live" for t in targets):
        assert {"APP", "PSEN1", "APOE"} <= {t.gene for t in targets}
