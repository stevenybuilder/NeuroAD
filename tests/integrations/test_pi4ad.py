"""
test_pi4ad — offline-first contract tests for the PI4AD prioritization adapter.

Every test here MUST pass with NO network and NO credentials. The one live test
probes reachability first (HTTP-only host) and skips gracefully when unreachable.
"""
from __future__ import annotations

import socket

import pytest

from neuroad.integrations import pi4ad as p
from neuroad.integrations.pi4ad import (
    GenePriority,
    PI4AD,
    gene_priority,
    rank_ad_targets,
)

# The 10 mandated AD-priority genes the snapshot must carry.
MANDATED = ["APP", "ESR1", "HRAS", "MAPK1", "APOE",
            "MAPT", "TREM2", "BIN1", "CLU", "PSEN1"]


# ---------------------------------------------------------------------------
# Reachability probe for the (single) guarded live test.
# ---------------------------------------------------------------------------

def _portal_reachable() -> bool:
    try:  # HTTP-only host — probe port 80, never 443 (HTTPS times out)
        socket.create_connection(("www.genetictargets.com", 80), timeout=3).close()
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Import / construction
# ---------------------------------------------------------------------------

def test_module_imports_offline():
    client = PI4AD(prefer_offline=True)
    assert isinstance(client, PI4AD)


def test_snapshot_loads_and_is_provenance_labeled():
    table = PI4AD(prefer_offline=True)._load()
    assert len(table) >= 60
    assert all(isinstance(g, GenePriority) for g in table)
    assert all(g.source == "offline_snapshot" for g in table)


# ---------------------------------------------------------------------------
# rank_genes — whole-disease ranking
# ---------------------------------------------------------------------------

def test_rank_genes_sorted_by_ascending_rank():
    top = PI4AD(prefer_offline=True).rank_genes(top_n=25)
    assert len(top) == 25
    ranks = [g.rank for g in top]
    assert ranks == sorted(ranks)
    assert top[0].rank == 1
    assert top[0].gene == "EGFR"          # verified portal #1, score 10.0
    assert top[0].priority_score == pytest.approx(10.0)


def test_scores_are_on_zero_to_ten_scale():
    for g in PI4AD(prefer_offline=True).rank_genes(top_n=1000):
        assert 0.0 <= g.priority_score <= 10.0


def test_rank_genes_non_positive_returns_empty():
    assert PI4AD(prefer_offline=True).rank_genes(top_n=0) == []
    assert PI4AD(prefer_offline=True).rank_genes(top_n=-5) == []


def test_module_level_rank_ad_targets():
    top = rank_ad_targets(top_n=5)
    assert [g.gene for g in top] == ["EGFR", "SRC", "GRB2", "AKT1", "CD4"]
    assert all(g.source == "offline_snapshot" for g in top)


# ---------------------------------------------------------------------------
# priority — single-gene lookup
# ---------------------------------------------------------------------------

def test_all_mandated_genes_present_with_verified_values():
    client = PI4AD(prefer_offline=True)
    # (rank, score) cross-validated vs portal + paper PMC12491700.
    expected = {
        "APP": (18, 8.597), "HRAS": (45, 8.19), "TREM2": (49, 8.135),
        "ESR1": (61, 7.992), "MAPK1": (64, 7.966), "MAPT": (151, 7.304),
        "APOE": (185, 7.151), "BIN1": (287, 6.754), "CLU": (292, 6.738),
        "PSEN1": (492, 6.211),
    }
    for gene in MANDATED:
        rec = client.priority(gene)
        assert rec is not None, f"{gene} missing from snapshot"
        assert rec.source == "offline_snapshot"
        r, s = expected[gene]
        assert rec.rank == r
        assert rec.priority_score == pytest.approx(s)


def test_priority_is_case_insensitive():
    a = gene_priority("app")
    b = gene_priority("APP")
    assert a is not None and b is not None
    assert a.gene == b.gene == "APP"
    assert a.rank == 18


def test_priority_unknown_gene_returns_none_not_raise():
    assert gene_priority("ZZZ_NOT_A_GENE") is None
    assert gene_priority("") is None


def test_evidence_note_carries_category_and_full_name():
    rec = gene_priority("APP")
    assert "Core" in rec.evidence_note
    assert "amyloid" in rec.evidence_note.lower()


def test_to_dict_roundtrip():
    d = gene_priority("TREM2").to_dict()
    assert d["gene"] == "TREM2"
    assert d["source"] == "offline_snapshot"
    assert d["rank"] == 49
    assert d["priority_score"] == pytest.approx(8.135)


# ---------------------------------------------------------------------------
# Portal HTML parsing (pure-Python, no network) + live-fallback behaviour
# ---------------------------------------------------------------------------

# A minimal transposed DataTables payload mirroring the real portal shape:
# col0=symbols (HTML-wrapped), col1=scores, col2=ranks, col3=category,
# cols 4-6 filler, col7=gene full name.
_FAKE_PAGE = (
    '<html><script>var t = {"data":['
    "[\"<a href='x'>EGFR</a>\", \"<a href='x'>APP</a>\"],"
    "[10, 8.597],"
    "[1, 18],"
    "['Core', 'Core'],"
    "[null, null],[null, null],[null, null],"
    "['epidermal growth factor receptor', 'amyloid beta precursor protein']"
    "]};</script></html>"
)


def test_parse_portal_html_extracts_live_records():
    recs = p._parse_portal_html(_FAKE_PAGE)
    assert len(recs) == 2
    assert recs[0].gene == "EGFR"
    assert recs[0].source == "live"          # parsed page => honestly "live"
    assert recs[0].priority_score == pytest.approx(10.0)
    assert recs[1].gene == "APP"
    assert recs[1].rank == 18
    assert "amyloid" in recs[1].evidence_note.lower()


def test_parse_portal_html_garbage_returns_empty():
    assert p._parse_portal_html("no data array here") == []


def test_live_fetch_success_is_labeled_live(monkeypatch):
    client = PI4AD(prefer_offline=False)

    class _Resp:
        status_code = 200
        text = _FAKE_PAGE

    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    top = client.rank_genes(top_n=2)
    assert [g.gene for g in top] == ["EGFR", "APP"]
    assert all(g.source == "live" for g in top)


def test_network_failure_falls_back_to_snapshot(monkeypatch):
    client = PI4AD(prefer_offline=False)

    def _boom(*a, **k):
        raise OSError("network down")

    import requests
    monkeypatch.setattr(requests, "get", _boom)
    # Degrades to snapshot, never raises, honestly labeled.
    top = client.rank_genes(top_n=3)
    assert len(top) == 3
    assert all(g.source == "offline_snapshot" for g in top)
    assert top[0].gene == "EGFR"


def test_non_200_falls_back_to_snapshot(monkeypatch):
    client = PI4AD(prefer_offline=False)

    class _Resp:
        status_code = 503
        text = ""

    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    rec = client.priority("APP")
    assert rec is not None
    assert rec.source == "offline_snapshot"
    assert rec.rank == 18


# ---------------------------------------------------------------------------
# Guarded LIVE test — skips when the HTTP-only portal is unreachable.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _portal_reachable(),
                    reason="PI4AD portal (HTTP-only host) unreachable")
def test_live_portal_smoke():
    top = PI4AD(prefer_offline=False).rank_genes(top_n=5)
    assert len(top) == 5
    # Online we expect live labels; a transient portal failure still yields an
    # honest offline fallback — either way the top gene and 0-10 scale hold.
    assert all(g.source in ("live", "offline_snapshot") for g in top)
    assert top[0].rank == 1
    assert all(0.0 <= g.priority_score <= 10.0 for g in top)
