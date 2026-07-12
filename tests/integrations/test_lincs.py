"""Offline / deterministic tests for the LINCS efficacy-proxy adapter.

No network, no creds: they exercise the offline-first contract (never raises,
degrades to the snapshot / empty), the parsing of a mocked live response, and the
per-gene reversal aggregation. The live path is covered by a network smoke test that
is skipped when offline.
"""
from __future__ import annotations

import pytest

from neuroad.integrations import lincs as L


def test_signature_is_nonempty_and_cited():
    assert len(L.AD_SIGNATURE_UP) >= 10 and len(L.AD_SIGNATURE_DOWN) >= 10
    # up and down sets are disjoint (a gene can't be both up and down)
    assert not (set(L.AD_SIGNATURE_UP) & set(L.AD_SIGNATURE_DOWN))
    assert "Zhang" in L._SIGNATURE_CITATION and "Mathys" in L._SIGNATURE_CITATION


def test_offline_client_never_raises_and_degrades():
    c = L.LincsClient(prefer_offline=True)
    proxy = c.ad_reversal_efficacy()            # empty snapshot -> {} (no raise)
    assert isinstance(proxy, dict)
    uni = c.reversal_universe()
    assert isinstance(uni, dict)


def test_convenience_map_offline():
    m = L.efficacy_proxy_map(prefer_offline=True)
    assert isinstance(m, dict)


def test_live_parsing_with_mocked_endpoints(monkeypatch):
    """Drive ad_reversal_efficacy with mocked REST responses (no network)."""
    c = L.LincsClient(prefer_offline=False)

    def fake_post(url, body):
        if url.endswith("/entities/find"):
            syms = body["filter"]["where"]["meta.symbol"]["inq"]
            return [{"id": f"uuid-{s}", "meta": {"symbol": s}} for s in syms]
        if url.endswith("/enrich/ranktwosided"):
            # two reversers (MTOR strong, GSK3B weak) + one mimicker (ACTB)
            return {"results": [
                {"uuid": "sig-mtor", "type": "reversers", "z-sum": -9.0},
                {"uuid": "sig-gsk3b", "type": "reversers", "z-sum": -6.0},
                {"uuid": "sig-actb", "type": "mimickers", "z-sum": 7.0},
            ]}
        if url.endswith("/signatures/find"):
            ids = body["filter"]["where"]["id"]["inq"]
            meta = {"sig-mtor": ("MTOR", "ES2", "CRISPR Knockout"),
                    "sig-gsk3b": ("GSK3B", "MCF7", "CRISPR Knockout"),
                    "sig-actb": ("ACTB", "HT29", "CRISPR Knockout")}
            return [{"id": i, "meta": {"pert_name": meta[i][0],
                                        "cell_line": meta[i][1],
                                        "pert_type": meta[i][2]}}
                    for i in ids if i in meta]
        return None

    monkeypatch.setattr(c, "_post", fake_post)

    proxy = c.ad_reversal_efficacy(limit=10)
    # reversers become efficacy hits; the strongest is MTOR (|z-sum|=9)
    assert "MTOR" in proxy and "GSK3B" in proxy
    assert proxy["MTOR"].reversal_score == pytest.approx(9.0)
    assert proxy["MTOR"].source == "live"
    # a mimicker is NOT an inhibition target -> excluded from the efficacy proxy
    assert "ACTB" not in proxy

    uni = c.reversal_universe(limit=10)
    # the signed universe keeps the mimicker as a negative-score background
    assert uni["MTOR"] > 0 and uni["ACTB"] < 0


def test_reversal_universe_empty_when_no_entities(monkeypatch):
    c = L.LincsClient(prefer_offline=False)
    c._snapshot = {"genes": []}                               # no snapshot to fall back to
    monkeypatch.setattr(c, "_post", lambda url, body: None)   # all calls fail
    assert c.reversal_universe() == {}
    assert c.ad_reversal_efficacy() == {}                     # degrades to empty, no raise


def test_ad_reversal_efficacy_degrades_to_snapshot_on_live_failure(monkeypatch):
    # Offline-first contract: a LIVE failure falls back to the committed snapshot
    # (non-empty in this repo), rather than losing all data.
    c = L.LincsClient(prefer_offline=False)
    c._snapshot = {"genes": [{"gene": "MTOR", "reversal_score": 9.0,
                              "n_signatures": 1, "best_database": "l1000_xpr",
                              "best_cell_line": "ES2"}]}
    monkeypatch.setattr(c, "_post", lambda url, body: None)   # all live calls fail
    proxy = c.ad_reversal_efficacy()
    assert "MTOR" in proxy and proxy["MTOR"].source == "offline_snapshot"
