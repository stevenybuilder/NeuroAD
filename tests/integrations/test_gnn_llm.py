"""Offline tests for the GNN/LLM drug-repurposing adapter.

All tests pass with NO network and NO credentials: they exercise the bundled
curated snapshot path and the deterministic evidence template. The optional
live TxGNN path is verified to be OFF by default (returns None), and the live
Claude rationale path is only asserted when ANTHROPIC_API_KEY is present.
"""
from __future__ import annotations

import os

import pytest

from neuroad.integrations.gnn_llm import (
    ALZHEIMER_MONDO_NODES,
    ENABLE_TXGNN_ENV,
    RepurposingCandidate,
    RepurposingEngine,
    resolve_disease_nodes,
)


@pytest.fixture()
def engine() -> RepurposingEngine:
    return RepurposingEngine()


def test_snapshot_loads_and_ranks_disease_level(engine: RepurposingEngine) -> None:
    cands = engine.rank_compounds("Alzheimer disease", top_n=5)
    assert 1 <= len(cands) <= 5
    assert all(isinstance(c, RepurposingCandidate) for c in cands)
    # Provenance is honestly stamped as the offline snapshot, never faux-live.
    assert all(c.source == "offline_snapshot" for c in cands)
    # Ranked best-first by curated evidence strength.
    strengths = [c.evidence_strength for c in cands]
    assert strengths == sorted(strengths, reverse=True)


def test_top_n_is_respected(engine: RepurposingEngine) -> None:
    assert len(engine.rank_compounds("Alzheimer disease", top_n=3)) == 3
    assert len(engine.rank_compounds("Alzheimer disease", top_n=0)) == 0


def test_evidence_strength_in_unit_interval(engine: RepurposingEngine) -> None:
    for c in engine.rank_compounds("Alzheimer disease", top_n=50):
        assert 0.0 <= c.evidence_strength <= 1.0


def test_no_efficacy_claims_in_notes(engine: RepurposingEngine) -> None:
    banned = ("cures", "proven effective", "guarantees", "reverses alzheimer")
    for c in engine.rank_compounds("Alzheimer disease", top_n=50):
        low = c.mechanism_note.lower()
        assert not any(b in low for b in banned)


def test_gene_target_matches_related_genes(engine: RepurposingEngine) -> None:
    # MAPT (tau) is a related gene of the tau-directed candidates in the snapshot.
    cands = engine.rank_compounds("MAPT", top_n=10)
    assert cands, "expected at least one MAPT-related candidate"
    for c in cands:
        genes = {c.target_gene.lower()} | {g.lower() for g in c.related_genes}
        assert "mapt" in genes


def test_unknown_gene_falls_back_to_disease_level(engine: RepurposingEngine) -> None:
    cands = engine.rank_compounds("ZZZ_NOT_A_GENE", top_n=5)
    assert cands, "unknown gene should still yield disease-level hypotheses"
    assert all(c.source == "offline_snapshot" for c in cands)
    assert all("disease-level hypothesis" in c.mechanism_note for c in cands)


def test_resolve_disease_nodes() -> None:
    nodes = resolve_disease_nodes("Alzheimer disease")
    assert nodes == ALZHEIMER_MONDO_NODES
    assert "4975" in nodes  # base MONDO Alzheimer-disease node
    # A snapshot gene resolves to the AD group; a random string does not.
    assert resolve_disease_nodes("GLP1R") == ALZHEIMER_MONDO_NODES
    assert resolve_disease_nodes("nonsense_token") == {}


def test_candidates_carry_disease_node_ids(engine: RepurposingEngine) -> None:
    for c in engine.rank_compounds("Alzheimer disease", top_n=3):
        assert c.disease_node_ids
        assert "4975" in c.disease_node_ids


def test_synthesize_evidence_offline_template(engine: RepurposingEngine, monkeypatch) -> None:
    # Force the offline branch regardless of the ambient environment.
    import neuroad.claude._client as client

    monkeypatch.setattr(client, "USING_LIVE_API", False, raising=False)
    cand = engine.rank_compounds("Alzheimer disease", top_n=1)[0]
    text = engine.synthesize_evidence(cand)
    assert isinstance(text, str) and text
    assert cand.rationale == text
    assert cand.rationale_source == "offline_template"
    assert cand.compound.lower() in text.lower()
    assert "not evidence of clinical benefit" in text.lower()


def test_txgnn_path_off_by_default(engine: RepurposingEngine, monkeypatch) -> None:
    monkeypatch.delenv(ENABLE_TXGNN_ENV, raising=False)
    # Private helper returns None (falls back) when the live path is disabled.
    assert engine._try_txgnn("Alzheimer disease", 5) is None


def test_to_dict_roundtrip(engine: RepurposingEngine) -> None:
    c = engine.rank_compounds("Alzheimer disease", top_n=1)[0]
    d = c.to_dict()
    assert d["source"] == "offline_snapshot"
    assert d["compound"] == c.compound
    assert set(["compound", "target_gene", "mechanism_note",
                "evidence_strength", "source"]).issubset(d)


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="live Claude rationale requires ANTHROPIC_API_KEY",
)
def test_synthesize_evidence_live_when_key_present(engine: RepurposingEngine) -> None:
    cand = engine.rank_compounds("Alzheimer disease", top_n=1)[0]
    text = engine.synthesize_evidence(cand)
    assert isinstance(text, str) and text
    assert cand.rationale_source in {"live_llm", "offline_template"}
