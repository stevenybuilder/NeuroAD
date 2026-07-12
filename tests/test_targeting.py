"""Tests for L6 targeting — offline, deterministic evidence-fusion ranking."""
from __future__ import annotations

import math
import sys

import pytest

from neuroad.integrations import targeting
from neuroad.integrations.targeting import (
    DEFAULT_WEIGHTS,
    LigandDruggability,
    TARGETING_LABEL,
    TargetDruggability,
    TargetingComponent,
    TargetingEngine,
    druggability_ranking,
    ligand_druggability_summary,
    target_druggability,
)
from neuroad.integrations.boltz import AD_TARGETS


# ---------------------------------------------------------------------------
# Offline / no-network / no-torch guard
# ---------------------------------------------------------------------------


def test_no_torch_imported():
    # Importing + running the default path must not pull in torch.
    TargetingEngine().rank_targets()
    assert "torch" not in sys.modules


def test_no_network_on_default_path(monkeypatch):
    import requests

    def _boom(*a, **k):  # pragma: no cover - must never be hit
        raise AssertionError("network access attempted on the offline path")

    monkeypatch.setattr(requests, "get", _boom)
    monkeypatch.setattr(requests, "post", _boom, raising=False)
    eng = TargetingEngine()
    rows = eng.rank_targets()
    eng.ligand_druggability()
    assert len(rows) == len(AD_TARGETS)


# ---------------------------------------------------------------------------
# score_target — present + provenance
# ---------------------------------------------------------------------------


def test_score_app_complex_and_struct_present():
    eng = TargetingEngine()
    row = eng.score_target("APP")
    by = {c.name: c for c in row.components}

    cc = by["complex_confidence"]
    assert cc.present is True
    # APP participates in 3 committed complexes; max confidence_score is APP|BACE1.
    assert cc.value_raw == pytest.approx(0.587441086769104)
    assert cc.source == "precomputed_snapshot"
    assert cc.model == "Boltz-2"

    st = by["struct_plddt"]
    assert st.present is True
    assert st.value_norm == pytest.approx(st.value_raw / 100.0)
    assert st.source == "offline_snapshot"
    assert st.model == "AlphaFold-DB"

    assert row.composite_score is not None
    assert 0.0 <= row.composite_score <= 1.0
    assert row.source == "targeting_fusion"
    # every component provenance-stamped
    for c in row.components:
        assert c.source and c.model


def test_absent_complex_is_none_not_zero():
    eng = TargetingEngine()
    row = eng.score_target("TREM2")  # no committed complex
    by = {c.name: c for c in row.components}
    assert by["complex_confidence"].present is False
    assert by["complex_confidence"].value_raw is None
    assert by["complex_confidence"].value_norm is None
    # still scored from present components (pLDDT + PI4AD)
    assert row.composite_score is not None
    assert abs(sum(row.effective_weights.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Empty-snapshot injection
# ---------------------------------------------------------------------------


def test_empty_snapshot_drops_boltz_components():
    eng = TargetingEngine(boltz_snapshot={})
    for gene in AD_TARGETS:
        row = eng.score_target(gene)
        by = {c.name: c for c in row.components}
        assert by["complex_confidence"].present is False
        assert by["ligand_binding"].present is False
        # composite still derived from pLDDT (+ PI4AD)
        assert row.composite_score is not None
    assert eng.ligand_druggability() == []


# ---------------------------------------------------------------------------
# Fully-degraded target
# ---------------------------------------------------------------------------


def test_fully_degraded_target_none_composite():
    eng = TargetingEngine(boltz_snapshot={})
    row = eng.score_target("NOTAGENE")
    assert row.composite_score is None
    assert row.n_components == 0
    assert "no" in row.note.lower() and "evidence" in row.note.lower()
    for c in row.components:
        assert c.value_raw is None
        assert c.value_norm is None


def test_unknown_and_empty_gene_do_not_raise():
    eng = TargetingEngine()
    for g in ("", "NOTAGENE"):
        row = eng.score_target(g)
        assert isinstance(row, TargetDruggability)
        # empty snapshot not injected here, but neither resolves any evidence
        # except possibly nothing -> composite None
        assert row.composite_score is None


# ---------------------------------------------------------------------------
# rank_targets ordering
# ---------------------------------------------------------------------------


def test_rank_targets_sorted_and_none_sinks():
    eng = TargetingEngine()
    rows = eng.rank_targets()
    assert len(rows) == len(AD_TARGETS)
    scored = [r for r in rows if r.composite_score is not None]
    nones = [r for r in rows if r.composite_score is None]
    # scored come first, descending
    vals = [r.composite_score for r in scored]
    assert vals == sorted(vals, reverse=True)
    # any None-composite rows are at the tail
    assert rows[len(scored):] == nones
    for r in rows:
        assert r.source == "targeting_fusion"


def test_rank_targets_none_stable_gene_order():
    genes = ["APP", "ZZZFAKE", "AAAFAKE"]
    eng = TargetingEngine(boltz_snapshot={})
    rows = eng.rank_targets(genes)
    none_genes = [r.gene for r in rows if r.composite_score is None]
    assert none_genes == sorted(none_genes)


# ---------------------------------------------------------------------------
# Weight renormalization + math
# ---------------------------------------------------------------------------


def test_weight_renormalization_and_composite_math():
    eng = TargetingEngine()
    for gene in AD_TARGETS:
        row = eng.score_target(gene)
        if row.n_components == 0:
            continue
        assert abs(sum(row.effective_weights.values()) - 1.0) < 1e-9
        present = [c for c in row.components if c.present]
        recomputed = sum(row.effective_weights[c.name] * c.value_norm
                         for c in present)
        assert row.composite_score == pytest.approx(recomputed)


def test_normalization_scales():
    eng = TargetingEngine()
    row = eng.score_target("APP")
    by = {c.name: c for c in row.components}
    st = by["struct_plddt"]
    assert st.value_norm == pytest.approx(st.value_raw / 100.0)
    assert 0.0 <= st.value_norm <= 1.0
    pr = by["pi4ad_priority"]
    if pr.present:
        assert pr.value_norm == pytest.approx(pr.value_raw / 10.0)
        assert 0.0 <= pr.value_norm <= 1.0
    cc = by["complex_confidence"]
    assert 0.0 <= cc.value_norm <= 1.0  # confidence already in [0,1]


# ---------------------------------------------------------------------------
# Ligand druggability summary
# ---------------------------------------------------------------------------


def test_ligand_summary_ranked_by_probability():
    eng = TargetingEngine()
    ligs = eng.ligand_druggability()
    assert len(ligs) == 2
    # Bexarotene (0.9324) ranks above Nilotinib (0.6725) despite Nilotinib's
    # more-negative affinity.
    assert ligs[0].ligand_id == "Bexarotene"
    assert ligs[0].rank == 1
    assert ligs[0].binding_probability == pytest.approx(0.9324398636817932)
    assert ligs[1].ligand_id == "Nilotinib"
    assert ligs[1].rank == 2
    assert ligs[1].binding_affinity == pytest.approx(-1.5244178771972656)
    # affinity direction disagrees with probability -> Nilotinib more negative
    assert ligs[1].binding_affinity < ligs[0].binding_affinity
    for lg in ligs:
        assert lg.source == "precomputed_snapshot"
        assert "binding_probability" in lg.note


def test_ligand_targets_outside_ad_targets_no_fake_join():
    eng = TargetingEngine()
    lig_genes = {lg.gene.upper() for lg in eng.ligand_druggability()}
    assert lig_genes == {"ABL1", "RXRA"}
    assert lig_genes.isdisjoint({t.upper() for t in AD_TARGETS})
    # no AD target picked up a ligand_binding component from them
    for gene in AD_TARGETS:
        row = eng.score_target(gene)
        by = {c.name: c for c in row.components}
        assert by["ligand_binding"].present is False


# ---------------------------------------------------------------------------
# Serialization / frame
# ---------------------------------------------------------------------------


def test_to_dict_json_serializable_and_none_preserved():
    import json

    row = target_druggability("APP")
    s = json.dumps(row)  # must not raise
    assert '"source": "targeting_fusion"' in s
    # absent components serialize None, never NaN
    trem = target_druggability("TREM2")
    for c in trem["components"]:
        if not c["present"]:
            assert c["value_raw"] is None
            assert c["value_norm"] is None


def test_to_frame_one_row_per_target_no_nan():
    eng = TargetingEngine()
    df = eng.to_frame()
    assert len(df) == len(AD_TARGETS)
    assert "composite_score" in df.columns
    for name in DEFAULT_WEIGHTS:
        assert f"{name}_norm" in df.columns
        assert f"{name}_present" in df.columns
    # object cells: absent norms are None, never float NaN
    for col in [f"{n}_norm" for n in DEFAULT_WEIGHTS]:
        for v in df[col].tolist():
            assert v is None or not (isinstance(v, float) and math.isnan(v))


def test_module_conveniences():
    ranking = druggability_ranking()
    assert isinstance(ranking, list) and ranking
    assert all(r["source"] == "targeting_fusion" for r in ranking)
    ligs = ligand_druggability_summary()
    assert [l["ligand_id"] for l in ligs] == ["Bexarotene", "Nilotinib"]
    assert TARGETING_LABEL and "NOT new folding" in TARGETING_LABEL


def test_dataclass_to_dict_shapes():
    comp = TargetingComponent("x", 1.0, 0.5, 0.3, True, "s", "m")
    assert comp.to_dict()["name"] == "x"
    lig = LigandDruggability("G", "L", "C", 0.5, -1.0, 0.6, 0.9, 1, "src")
    assert lig.to_dict()["gene"] == "G"
