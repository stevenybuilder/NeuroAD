"""
Router eval — free-text hypothesis -> target enum.

Two layers, matching docs/RESEARCH_llm_router_plan.md:

  * ALWAYS-ON (no API key needed, deterministic):
      - route_target == keyword _infer_target offline (no regression: the router
        is never worse than the old regex when the LLM path is unavailable).
      - the keyword baseline meets documented per-class recall floors on the
        golden set (a regression guard on the deterministic backstop itself).

  * LIVE EVAL (gated on ANTHROPIC_API_KEY *and* NEUROAD_ROUTER_EVAL=1, because it
    spends real API calls): clears the routing cache, routes the golden set live
    on Sonnet-5, and asserts the LLM router is >= the keyword baseline on every
    class, materially better on the adversarial collision bucket, and makes zero
    conversion<->dx_binary confusions on the clean examples. This is the whole
    point of the router: it kills the "predicts -> conversion" misroute.
"""
from __future__ import annotations

import collections
import json
import os
from pathlib import Path

import pytest

from neuroad.claude import _client, claim_parser, router
from neuroad.contract import LABEL_TARGETS

_CLASSES = tuple(LABEL_TARGETS)  # conversion, dx_binary, site, scanner
_GOLDEN = Path(__file__).parent / "data" / "router_golden.jsonl"


def _load_golden() -> list[dict]:
    return [json.loads(l) for l in _GOLDEN.read_text().splitlines() if l.strip()]


def _keyword(text: str) -> str:
    return claim_parser._infer_target(text, None)


def _recall_by_class(rows, route_fn) -> dict:
    per = collections.defaultdict(lambda: [0, 0])  # class -> [correct, total]
    for r in rows:
        exp = r["target"]
        if exp not in _CLASSES:
            continue
        per[exp][1] += 1
        if route_fn(r["text"]) == exp:
            per[exp][0] += 1
    return {c: (cor / tot if tot else 1.0) for c, (cor, tot) in per.items()}


def _collision_accuracy(rows, route_fn) -> float:
    scored = [r for r in rows if r["target"] in _CLASSES]
    hits = sum(1 for r in scored if route_fn(r["text"]) == r["target"])
    return hits / len(scored) if scored else 1.0


# ---------------------------------------------------------------------------
# Always-on: no-regression + keyword baseline floors
# ---------------------------------------------------------------------------


def test_golden_set_present_and_stratified():
    rows = _load_golden()
    assert len(rows) >= 55, "golden set should be ~60 items"
    buckets = collections.Counter(r["bucket"] for r in rows)
    assert buckets["clean"] >= 30
    assert buckets["collision"] >= 12
    assert buckets["unsupported"] >= 6
    for r in rows:
        assert r["target"] in set(_CLASSES) | {"unsupported"}


def test_router_offline_matches_keyword(monkeypatch):
    """With no live key, route_target must exactly equal the keyword backstop
    (the router is never worse than today's regex offline)."""
    monkeypatch.setattr(_client, "USING_LIVE_API", False, raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Ensure no shipped router cache masks the offline path for this assertion.
    monkeypatch.setattr(router, "_cache", {}, raising=False)
    monkeypatch.setattr(router, "_cache_mtime", -1.0, raising=False)
    monkeypatch.setattr(router, "_load", lambda: {}, raising=False)
    for r in _load_golden():
        assert router.route_target(r["text"], None) == _keyword(r["text"]), r["text"]


def test_route_target_never_raises_and_returns_enum(monkeypatch):
    monkeypatch.setattr(_client, "USING_LIVE_API", False, raising=False)
    for r in _load_golden():
        out = router.route_target(r["text"], None)
        assert out in _CLASSES, f"{r['text']} -> {out}"


def test_keyword_baseline_floors():
    """Regression guard on the deterministic backstop. These are the MEASURED
    floors the keyword router meets today; the live LLM router must beat them."""
    rows = _load_golden()
    clean = [r for r in rows if r["bucket"] == "clean"]
    rec = _recall_by_class(clean, _keyword)
    # measured 2026-07-13: conversion .80, dx_binary .80, site .62, scanner .62
    assert rec["conversion"] >= 0.80
    assert rec["dx_binary"] >= 0.80
    assert rec["site"] >= 0.60
    assert rec["scanner"] >= 0.60
    # the known weakness the LLM fixes: dx_binary collision recall is low
    coll = [r for r in rows if r["bucket"] == "collision"]
    assert _collision_accuracy(coll, _keyword) <= 0.85  # room to improve


# ---------------------------------------------------------------------------
# Live eval (gated): the LLM router must beat the keyword baseline
# ---------------------------------------------------------------------------

_LIVE = bool(os.environ.get("ANTHROPIC_API_KEY")) and os.environ.get("NEUROAD_ROUTER_EVAL") == "1"


@pytest.mark.skipif(not _LIVE, reason="set ANTHROPIC_API_KEY and NEUROAD_ROUTER_EVAL=1 to run the live router eval")
def test_llm_router_beats_baseline(tmp_path, monkeypatch):
    rows = _load_golden()
    clean = [r for r in rows if r["bucket"] == "clean"]
    coll = [r for r in rows if r["bucket"] == "collision"]

    # Route live with a FRESH cache so we measure the model, not a warmed answer.
    monkeypatch.setattr(router, "_CACHE_FILE", tmp_path / "router_cache.json", raising=False)
    monkeypatch.setattr(router, "_cache", None, raising=False)
    monkeypatch.setattr(router, "_cache_mtime", -1.0, raising=False)

    def _llm(text):
        return router.route_target(text, None)

    kw_clean = _recall_by_class(clean, _keyword)
    llm_clean = _recall_by_class(clean, _llm)
    for c in _CLASSES:
        assert llm_clean.get(c, 1.0) >= kw_clean.get(c, 1.0) - 1e-9, (
            f"LLM regressed on clean {c}: {llm_clean.get(c)} < keyword {kw_clean.get(c)}"
        )

    # No conversion<->dx_binary confusion on clean examples (the flagship bug class).
    for r in clean:
        if r["target"] in ("conversion", "dx_binary"):
            got = _llm(r["text"])
            assert got == r["target"], f"clean confusion: {r['text']} -> {got} (want {r['target']})"

    # Materially better on the adversarial collision bucket.
    kw_coll = _collision_accuracy(coll, _keyword)
    llm_coll = _collision_accuracy(coll, _llm)
    assert llm_coll >= kw_coll + 0.15, f"collision acc: llm {llm_coll:.2f} vs keyword {kw_coll:.2f}"
    assert llm_coll >= 0.90, f"collision acc {llm_coll:.2f} below 0.90 floor"
