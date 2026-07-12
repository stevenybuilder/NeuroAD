"""Smoke tests for the interactive backend (app/server.py).

Offline / deterministic: spins the stdlib server on an ephemeral port in a
background thread and drives it over real HTTP with urllib. Uses the synthetic
cohort so no gated data or network is required. The live-Claude path is never
exercised (claude_live is False without a key), matching the honesty contract.
"""
from __future__ import annotations

import json
import threading
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer

import pytest

from app import server


@pytest.fixture()
def base_url():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def _post(url: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_health_reports_offline_claude(base_url):
    d = _get(f"{base_url}/api/health")
    assert d["status"] == "ok"
    # No key in the test env -> the honest live signal must be False.
    assert d["claude_live"] is False
    assert isinstance(d["datasets"], list) and d["datasets"]


def test_investigate_synthetic_returns_refereed_card_with_translation(base_url):
    status, d = _post(f"{base_url}/api/investigate", {
        "hypothesis": "Does the embedding predict MCI to AD conversion?",
        "dataset": "synthetic:SURVIVOR",
    })
    assert status == 200
    assert "Neuro-JEPA" not in d["substrate"]        # honest substrate over HTTP
    assert d["verdict"] and d["robustness_score"] is not None
    assert d["_meta"]["claude_live"] is False          # never faked live
    if d["promoted"]:
        assert d["translation"]["status"] == "translated"
        assert d["translation"]["top_target"]


def test_investigate_returns_rich_case_for_tree(base_url):
    """The additive `case` enrichment carries the rich shape the tree/story UI
    renders, without dropping any existing top-level card key."""
    status, d = _post(f"{base_url}/api/investigate", {
        "hypothesis": "Does the embedding predict MCI to AD conversion?",
        "dataset": "synthetic:SURVIVOR",
    })
    assert status == 200
    # Existing plain-card keys are still present (purely additive).
    for k in ("substrate", "verdict", "robustness_score", "promoted",
              "novelty_class", "_meta"):
        assert k in d
    case = d["case"]
    assert isinstance(case, dict)
    # Real gauntlet: five tests, each with a result the tree colors nodes by.
    assert len(case["tests"]) == 5
    assert {t["key"] for t in case["tests"]} == {
        "age_sex", "site_scanner", "brain_age", "biomarker_anchor", "replication"}
    for t in case["tests"]:
        assert t["result"] in (
            "passed", "weakened", "mixed", "failed", "not_available")
    # Cohort summary + leakage margin + score/verdict the panels read.
    assert case["cohort"]["n_subjects"]
    assert case["cohort"]["badge"]
    assert "outcome_auc" in case["leakage_margin"]
    assert case["score"] is not None and case["verdict"]
    # Normalized decision tree (audit-complete): hypothesis -> gates -> verdict.
    tree = case["tree"]
    assert tree["root"] == "hypothesis"
    assert any(n["type"] == "verdict" for n in tree["nodes"])
    # The over-HTTP case stays honest about its substrate label.
    assert "Neuro-JEPA" not in case["claim"]["substrate"]


def test_out_of_scope_hypothesis_refused_over_http(base_url):
    status, d = _post(f"{base_url}/api/investigate", {
        "hypothesis": "tau-PET SUVR trajectory",
        "dataset": "synthetic:SURVIVOR",
    })
    assert status == 200
    assert d["promoted"] is False
    assert d["novelty_class"] == "unsupported"


def test_unknown_dataset_is_400(base_url):
    status, d = _post(f"{base_url}/api/investigate", {
        "hypothesis": "x", "dataset": "not_a_dataset",
    })
    assert status == 400
    assert "unknown dataset" in d["error"]


def test_missing_hypothesis_is_400(base_url):
    status, d = _post(f"{base_url}/api/investigate", {"dataset": "synthetic:SURVIVOR"})
    assert status == 400
