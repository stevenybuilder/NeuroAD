"""
test_string_ppi — offline-first contract tests for the STRING PPI adapter.

Every test here MUST pass with NO network and NO credentials. The one live test
probes reachability first and skips gracefully when the STRING host is
unreachable. STRING gives INTERACTION EVIDENCE, not de-novo complex folding, and
these tests assert the adapter labels it that way and never fabricates a score.
"""
from __future__ import annotations

import socket

import pytest

from neuroad.integrations import string_ppi as sp
from neuroad.integrations.string_ppi import (
    AD_TARGETS,
    EVIDENCE_LABEL,
    InteractionEvidence,
    StringPPIClient,
    complex_evidence,
    interaction_evidence,
)


# ---------------------------------------------------------------------------
# Reachability probe for the (single) guarded live test.
# ---------------------------------------------------------------------------

def _string_reachable() -> bool:
    try:
        socket.create_connection(("string-db.org", 443), timeout=3).close()
        return True
    except OSError:
        return False


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


# ---------------------------------------------------------------------------
# Import / resolution / snapshot integrity
# ---------------------------------------------------------------------------

def test_module_imports_offline():
    # Constructing the client must not touch the network or raise.
    client = StringPPIClient(prefer_offline=True)
    assert isinstance(client, StringPPIClient)


def test_twelve_targets_all_have_hub_partners_in_snapshot():
    client = StringPPIClient(prefer_offline=True)
    assert len(AD_TARGETS) == 12
    for g in AD_TARGETS:
        partners = client.interaction_partners(g)
        assert partners, f"{g} has no hub partners in snapshot"
        for p in partners:
            assert p.source == "offline_snapshot"
            assert p.gene_a == g
            assert 0.0 < (p.combined_score or 0.0) <= 1.0


def test_resolve_symbol_normalizes_and_rejects_garbage():
    client = StringPPIClient(prefer_offline=True)
    assert client.resolve_symbol("app") == "APP"
    assert client.resolve_symbol("  MAPT ") == "MAPT"
    assert client.resolve_symbol("") is None
    assert client.resolve_symbol("   ") is None
    assert client.resolve_symbol("bad symbol!") is None


# ---------------------------------------------------------------------------
# Offline pair evidence — provenance, channels, honesty label
# ---------------------------------------------------------------------------

def test_offline_pair_is_labeled_and_populated():
    ev = complex_evidence("APP", "BACE1", prefer_offline=True)
    assert ev["source"] == "offline_snapshot"          # honest fallback label
    assert ev["evidence_type"] == EVIDENCE_LABEL        # NOT de-novo folding
    assert "NOT de-novo" in ev["evidence_type"]
    assert ev["combined_score"] == pytest.approx(0.999)
    assert ev["channels"]["experimental"] == pytest.approx(0.86)
    assert ev["channels"]["database"] == pytest.approx(0.75)
    assert ev["channels"]["textmining"] == pytest.approx(0.999)


def test_offline_pair_is_order_independent():
    a = StringPPIClient(prefer_offline=True).pair_evidence("APP", "BACE1")
    b = StringPPIClient(prefer_offline=True).pair_evidence("BACE1", "APP")
    assert a.combined_score == b.combined_score == pytest.approx(0.999)
    assert a.channels == b.channels


def test_offline_pair_with_no_edge_is_honest_none_not_zero():
    # HRAS <-> APP has no STRING edge above cutoff in the real capture.
    ev = StringPPIClient(prefer_offline=True).pair_evidence("HRAS", "APP")
    assert ev.source == "offline_snapshot"
    assert ev.combined_score is None                   # honest, not a fabricated 0
    assert "no STRING interaction evidence" in ev.error


def test_pair_unknown_gene_does_not_raise():
    ev = complex_evidence("APP", "ZZZ_NOT_A_GENE", prefer_offline=True)
    # ZZZ_NOT_A_GENE resolves shape-wise but has no snapshot edge -> honest none.
    assert ev["source"] == "offline_snapshot"
    assert ev["combined_score"] is None


def test_pair_malformed_gene_is_honest_stub():
    ev = StringPPIClient(prefer_offline=True).pair_evidence("APP", "")
    assert ev.source == "offline_snapshot"
    assert ev.combined_score is None
    assert "could not resolve" in ev.error


def test_self_pair_is_honest_stub():
    ev = StringPPIClient(prefer_offline=True).pair_evidence("APP", "APP")
    assert ev.combined_score is None
    assert "self-pair" in ev.error


# ---------------------------------------------------------------------------
# Offline hub partners — ranking, determinism, honesty
# ---------------------------------------------------------------------------

def test_offline_partners_ranked_desc_and_deterministic():
    p1 = interaction_evidence("MAPT", prefer_offline=True)
    p2 = interaction_evidence("MAPT", prefer_offline=True)
    assert p1 == p2                                     # deterministic
    assert p1["source"] == "offline_snapshot"
    assert p1["note"] == EVIDENCE_LABEL
    scores = [p["combined_score"] for p in p1["partners"]]
    assert scores == sorted(scores, reverse=True)       # ranked desc
    # Known top-confidence MAPT interactors in the real STRING capture.
    names = [p["gene_b"] for p in p1["partners"]]
    assert "GSK3B" in names and "CDK5" in names


def test_partners_limit_is_respected():
    partners = StringPPIClient(prefer_offline=True).interaction_partners(
        "APOE", limit=3)
    assert len(partners) == 3


def test_partners_unknown_gene_is_empty_not_raise():
    partners = StringPPIClient(prefer_offline=True).interaction_partners(
        "ZZZ_NOT_A_GENE")
    assert partners == []


def test_to_dict_roundtrip_shape():
    ev = StringPPIClient(prefer_offline=True).interaction_partners("HRAS")[0]
    d = ev.to_dict()
    assert set(d) == {"gene_a", "gene_b", "combined_score", "channels",
                      "source", "evidence_type", "error"}
    assert d["gene_a"] == "HRAS"
    assert d["source"] == "offline_snapshot"


# ---------------------------------------------------------------------------
# No-network safety: prefer_offline must not touch requests at all.
# ---------------------------------------------------------------------------

def test_prefer_offline_never_touches_network(no_network):
    client = StringPPIClient(prefer_offline=True)
    assert client.pair_evidence("APP", "MAPT").source == "offline_snapshot"
    assert client.interaction_partners("APP")[0].source == "offline_snapshot"


# ---------------------------------------------------------------------------
# Live path (mocked) — provenance labeling + fallback behavior
# ---------------------------------------------------------------------------

_NETWORK_TSV = (
    "stringId_A\tstringId_B\tpreferredName_A\tpreferredName_B\tncbiTaxonId\t"
    "score\tnscore\tfscore\tpscore\tascore\tescore\tdscore\ttscore\n"
    "9606.ENSP1\t9606.ENSP2\tAPP\tMAPT\t9606\t0.5\t0\t0\t0\t0\t0.4\t0.3\t0.2\n"
)


def test_monkeypatched_live_pair_is_labeled_live(monkeypatch):
    class _Resp:
        status_code = 200
        text = _NETWORK_TSV

    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    ev = StringPPIClient(prefer_offline=False).pair_evidence("APP", "MAPT")
    assert ev.source == "live"
    assert ev.combined_score == pytest.approx(0.5)      # from live TSV, not snapshot
    assert ev.channels["experimental"] == pytest.approx(0.4)
    assert ev.channels["database"] == pytest.approx(0.3)
    assert ev.channels["textmining"] == pytest.approx(0.2)


def test_network_failure_falls_back_to_snapshot(monkeypatch):
    def _boom(*args, **kwargs):
        raise OSError("network down")

    import requests
    monkeypatch.setattr(requests, "get", _boom)
    ev = StringPPIClient(prefer_offline=False).pair_evidence("APP", "BACE1")
    assert ev.source == "offline_snapshot"              # degraded, never raised
    assert ev.combined_score == pytest.approx(0.999)    # from bundled snapshot
    assert "unavailable" in ev.error


def test_non_200_falls_back_to_snapshot(monkeypatch):
    class _Resp:
        status_code = 503
        text = ""

    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    partners = StringPPIClient(prefer_offline=False).interaction_partners("APOE")
    assert partners and partners[0].source == "offline_snapshot"


_PARTNERS_TSV = (
    "stringId_A\tstringId_B\tpreferredName_A\tpreferredName_B\tncbiTaxonId\t"
    "score\tnscore\tfscore\tpscore\tascore\tescore\tdscore\ttscore\n"
    "9606.ENSP1\t9606.ENSPx\tAPP\tSORL1\t9606\t0.9\t0\t0\t0\t0\t0.8\t0.7\t0.6\n"
    "9606.ENSP1\t9606.ENSPy\tAPP\tAPBB1\t9606\t0.95\t0\t0\t0\t0\t0.85\t0.75\t0.65\n"
)


def test_monkeypatched_live_partners_labeled_live_and_ranked(monkeypatch):
    class _Resp:
        status_code = 200
        text = _PARTNERS_TSV

    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    partners = StringPPIClient(prefer_offline=False).interaction_partners("APP")
    assert [p.source for p in partners] == ["live", "live"]
    assert partners[0].gene_b == "APBB1"               # 0.95 ranked above 0.90
    assert partners[0].gene_a == "APP"


# ---------------------------------------------------------------------------
# TSV parser (pure, no network)
# ---------------------------------------------------------------------------

def test_parse_tsv_handles_empty_and_short_rows():
    assert sp._parse_tsv("") == []
    rows = sp._parse_tsv("a\tb\n1\t2\nBAD\n3\t4\n")
    assert rows == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]


# ---------------------------------------------------------------------------
# Guarded LIVE test — skips when the STRING host is unreachable.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _string_reachable(), reason="STRING-db unreachable")
def test_live_pair_smoke():
    ev = StringPPIClient(prefer_offline=False).pair_evidence("APP", "BACE1")
    # Online we expect a live label + plausible score; a transient failure still
    # returns an honest offline fallback.
    assert ev.source in ("live", "offline_snapshot")
    assert isinstance(ev, InteractionEvidence)
    if ev.source == "live":
        assert ev.combined_score is not None and 0.0 < ev.combined_score <= 1.0
