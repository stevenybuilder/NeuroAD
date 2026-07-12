"""Tests for integrations.pathway_enrichment — offline ORA over curated AD sets."""
from __future__ import annotations

import builtins
import sys

import pytest

from neuroad.integrations import pathway_enrichment as pe


# ---------------------------------------------------------------------------
# Offline / deterministic import
# ---------------------------------------------------------------------------

def test_import_is_offline_no_torch():
    # Default import + a real enrichment must not force torch into sys.modules.
    pe.enrich_genes(["APP", "PSEN1", "APOE"])
    assert "torch" not in sys.modules


# ---------------------------------------------------------------------------
# Snapshot integrity
# ---------------------------------------------------------------------------

def test_snapshot_integrity():
    sets = pe.load_pathway_gene_sets()
    assert len(sets) >= 8
    for gs in sets:
        assert gs.genes, f"{gs.id} has empty genes"
        assert gs.source in {"KEGG", "Reactome", "GO"}
        assert gs.source_id
        d = gs.to_dict()
        assert d["id"] == gs.id and d["genes"] == gs.genes


def test_snapshot_has_provenance_block():
    import json
    with open(pe._SNAPSHOT_PATH, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    prov = blob.get("provenance")
    assert isinstance(prov, dict)
    assert prov.get("snapshot_source") == "ad_pathway_snapshot_v1"
    assert "not fetched live" in prov.get("disclaimer", "").lower()
    assert "sources" in prov


def test_kegg_ad_set_membership():
    sets = {gs.source_id: gs for gs in pe.load_pathway_gene_sets()}
    ad = sets.get("hsa05010")
    assert ad is not None and ad.source == "KEGG"
    for g in ("APP", "PSEN1", "APOE", "MAPT"):
        assert g in ad.genes


# ---------------------------------------------------------------------------
# Known-positive enrichment
# ---------------------------------------------------------------------------

def test_enrich_mechanism_amyloid_top_kegg_ad():
    res = pe.enrich_mechanism("amyloid_cascade")
    assert res, "expected non-empty enrichment"
    top = res[0]
    assert top.source_id == "hsa05010"
    assert top.p_value < 0.05
    assert "APP" in top.overlap_genes and "PSEN1" in top.overlap_genes
    assert top.overlap_size <= min(top.pathway_size, top.query_size)


def test_enrich_mechanism_unknown_defaults_amyloid():
    unknown = pe.enrich_mechanism("does_not_exist")
    amyloid = pe.enrich_mechanism("amyloid_cascade")
    assert unknown and amyloid
    assert [r.source_id for r in unknown] == [r.source_id for r in amyloid]


# ---------------------------------------------------------------------------
# Determinism + order/case invariance
# ---------------------------------------------------------------------------

def test_determinism_and_input_invariance():
    a = pe.enrich_genes(["APP", "PSEN1", "APOE", "MAPT", "BACE1"])
    b = pe.enrich_genes(["APP", "PSEN1", "APOE", "MAPT", "BACE1"])
    assert [r.to_dict() for r in a] == [r.to_dict() for r in b]

    c = pe.enrich_genes(["app", "PSEN1"])
    d = pe.enrich_genes(["PSEN1", "APP"])
    assert [(r.source_id, r.p_value, r.q_value, tuple(r.overlap_genes)) for r in c] == \
           [(r.source_id, r.p_value, r.q_value, tuple(r.overlap_genes)) for r in d]


# ---------------------------------------------------------------------------
# Method agreement
# ---------------------------------------------------------------------------

def test_hypergeom_vs_fisher_agreement():
    q = ["APP", "PSEN1", "APOE", "MAPT", "BACE1", "MAPK1", "HRAS"]
    h = {r.source_id: r for r in pe.enrich_genes(q, method="hypergeom")}
    f = {r.source_id: r for r in pe.enrich_genes(q, method="fisher")}
    assert set(h) == set(f)
    for sid in h:
        assert h[sid].overlap_size == f[sid].overlap_size
        assert h[sid].overlap_genes == f[sid].overlap_genes
        assert h[sid].p_value == pytest.approx(f[sid].p_value, abs=1e-9)
    # ranking consistency at the top
    top_h = min(h.values(), key=lambda r: (r.p_value, -r.overlap_size)).source_id
    top_f = min(f.values(), key=lambda r: (r.p_value, -r.overlap_size)).source_id
    assert top_h == top_f


# ---------------------------------------------------------------------------
# Hand-computed hypergeometric sanity (off-by-one guard)
# ---------------------------------------------------------------------------

def test_hypergeom_off_by_one_hand_computed():
    from scipy.stats import hypergeom
    # Deterministic small universe via background override; recompute one row.
    q = ["APP", "PSEN1", "APOE", "MAPT", "BACE1"]
    res = {r.source_id: r for r in pe.enrich_genes(q)}
    ad = res["hsa05010"]
    expected = float(hypergeom.sf(ad.overlap_size - 1, ad.background_size,
                                  ad.pathway_size, ad.query_size))
    assert ad.p_value == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# BH-FDR correctness
# ---------------------------------------------------------------------------

def test_bh_fdr_monotone_and_ge_p():
    q = ["APP", "PSEN1", "APOE", "MAPT", "BACE1", "MAPK1", "HRAS", "TREM2"]
    res = pe.enrich_genes(q)
    assert res
    prev_q = -1.0
    for r in res:
        assert r.q_value >= r.p_value - 1e-12
        assert r.q_value >= prev_q - 1e-12  # non-decreasing in BH (p asc) order
        prev_q = r.q_value


# ---------------------------------------------------------------------------
# Empty / degenerate
# ---------------------------------------------------------------------------

def test_empty_query_returns_empty():
    assert pe.enrich_genes([]) == []
    assert pe.enrich_genes(["   ", ""]) == []


def test_no_universe_overlap_returns_empty():
    assert pe.enrich_genes(["ZZZ_FAKE1", "ZZZ_FAKE2"]) == []


def test_min_overlap_excludes_low_overlap():
    q = ["APP", "PSEN1", "APOE", "MAPT", "BACE1"]
    lo = pe.enrich_genes(q, min_overlap=1)
    hi = pe.enrich_genes(q, min_overlap=4)
    assert all(r.overlap_size >= 4 for r in hi)
    assert len(hi) <= len(lo)


# ---------------------------------------------------------------------------
# Degrade-never-raise
# ---------------------------------------------------------------------------

def test_missing_snapshot_degrades(monkeypatch, tmp_path):
    pe._GENESET_CACHE = None
    monkeypatch.setattr(pe, "_SNAPSHOT_PATH", tmp_path / "nope.json")
    try:
        assert pe.load_pathway_gene_sets() == []
        assert pe.enrich_genes(["APP", "PSEN1"]) == []
    finally:
        pe._GENESET_CACHE = None  # reset cache for other tests


def test_missing_scipy_degrades(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("simulated missing scipy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert pe.enrich_genes(["APP", "PSEN1", "APOE"]) == []


# ---------------------------------------------------------------------------
# Provenance stamps + background override
# ---------------------------------------------------------------------------

def test_provenance_stamps_and_background_override():
    res = pe.enrich_genes(["APP", "PSEN1", "APOE", "MAPT"])
    assert res
    r0 = res[0]
    d = r0.to_dict()
    for key in ("snapshot_source", "source", "source_id", "method", "model",
                "background_size", "pathway_size", "query_size"):
        assert key in d
    assert d["snapshot_source"] == "ad_pathway_snapshot_v1"
    assert d["model"] == "ORA_hypergeometric_BH"
    default_bg = r0.background_size

    # Explicit override is honored and recorded.
    over = pe.enrich_genes(["APP", "PSEN1", "APOE", "MAPT"], background_size=20000)
    assert over and over[0].background_size == 20000
    assert over[0].background_size > default_bg


# ---------------------------------------------------------------------------
# Integration with propagation
# ---------------------------------------------------------------------------

def test_enrich_propagation_glial_seeds():
    res = pe.enrich_propagation(["TREM2", "APOE", "CLU", "MAPK1", "HRAS"])
    assert res, "propagation enrichment should not be empty"
    ids = {r.source_id for r in res}
    # neuroinflammation (GO:0150076) and/or MAPK/Ras (hsa04010/hsa04014)
    assert ids & {"GO:0150076", "hsa04010", "hsa04014"}
    # overlap_genes are exactly the set intersection with the pathway
    sets = {gs.name: set(gs.genes) for gs in pe.load_pathway_gene_sets()}
    for r in res:
        assert set(r.overlap_genes) <= sets[r.pathway]
        assert r.overlap_size == len(r.overlap_genes)
