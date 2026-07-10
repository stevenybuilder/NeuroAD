"""
test_alphafold — offline-first contract tests for the AlphaFold DB adapter.

Every test here MUST pass with NO network and NO credentials. The one live test
probes reachability first and skips gracefully when the EBI host is unreachable.
"""
from __future__ import annotations

import socket

import pytest

from neuroad.integrations import alphafold as af
from neuroad.integrations.alphafold import (
    AD_PROTEIN_MAP,
    AlphaFoldClient,
    AlphaFoldStructure,
    structural_confidence,
)


# ---------------------------------------------------------------------------
# Reachability probe for the (single) guarded live test.
# ---------------------------------------------------------------------------

def _ebi_reachable() -> bool:
    try:
        socket.create_connection(("alphafold.ebi.ac.uk", 443), timeout=3).close()
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Import / resolution
# ---------------------------------------------------------------------------

def test_module_imports_offline():
    # Simply constructing the client must not touch the network or raise.
    client = AlphaFoldClient(prefer_offline=True)
    assert isinstance(client, AlphaFoldClient)


def test_all_twelve_targets_in_map_and_snapshot():
    client = AlphaFoldClient(prefer_offline=True)
    # 12 canonical AD symbols (TAU is an alias of MAPT, so it is extra, not counted).
    canonical = {k for k in AD_PROTEIN_MAP if k != "TAU"}
    assert len(canonical) == 12
    for sym in canonical:
        acc = client.resolve_uniprot(sym)
        assert acc is not None
        assert acc in client._snapshot, f"{sym} -> {acc} missing from snapshot"


def test_resolve_symbol_and_accession_and_alias():
    client = AlphaFoldClient(prefer_offline=True)
    assert client.resolve_uniprot("APP") == "P05067"
    assert client.resolve_uniprot("app") == "P05067"          # case-insensitive
    assert client.resolve_uniprot("TAU") == "P10636"          # alias
    assert client.resolve_uniprot("MAPT") == "P10636"
    assert client.resolve_uniprot("P05067") == "P05067"       # pass-through
    assert client.resolve_uniprot("NOT_A_GENE") is None


# ---------------------------------------------------------------------------
# Offline fetch — provenance + data integrity
# ---------------------------------------------------------------------------

def test_offline_fetch_is_labeled_and_populated():
    s = structural_confidence("APP", prefer_offline=True)
    assert isinstance(s, AlphaFoldStructure)
    assert s.source == "offline_snapshot"          # honestly labeled as fallback
    assert s.uniprot == "P05067"
    assert s.gene_symbol == "APP"
    assert s.mean_plddt == pytest.approx(67.38)    # from snapshot / globalMetricValue
    assert s.cif_url.endswith("AF-P05067-F1-model_v6.cif")
    assert s.model_url.endswith("AF-P05067-F1-model_v6.pdb")
    assert s.plddt_recomputed is False


def test_offline_fetch_by_accession_reverse_populates_symbol():
    s = AlphaFoldClient(prefer_offline=True).fetch_structure("P10636")
    assert s.source == "offline_snapshot"
    assert s.gene_symbol == "MAPT"                  # reverse-mapped from accession
    # tau is intrinsically disordered -> low but valid mean pLDDT (expected).
    assert 0 < s.mean_plddt < 60


def test_every_offline_target_has_confidence_and_urls():
    client = AlphaFoldClient(prefer_offline=True)
    for sym, acc in AD_PROTEIN_MAP.items():
        s = client.fetch_structure(sym)
        assert s.source == "offline_snapshot"
        assert s.uniprot == acc
        assert 0 < (s.mean_plddt or 0) <= 100
        assert s.cif_url and s.model_url


def test_unknown_query_does_not_raise():
    s = structural_confidence("ZZZ_UNKNOWN", prefer_offline=True)
    assert s.source == "offline_snapshot"
    assert s.uniprot == ""
    assert "could not resolve" in s.error


def test_to_dict_roundtrip():
    d = structural_confidence("BACE1", prefer_offline=True).to_dict()
    assert d["source"] == "offline_snapshot"
    assert d["gene_symbol"] == "BACE1"
    assert d["mean_plddt"] == pytest.approx(87.5)


# ---------------------------------------------------------------------------
# Live path falls back to snapshot when the network is monkeypatched away.
# ---------------------------------------------------------------------------

def test_network_failure_falls_back_to_snapshot(monkeypatch):
    client = AlphaFoldClient(prefer_offline=False)

    def _boom(*args, **kwargs):
        raise OSError("network down")

    # Even the default (non-offline) path degrades, never raises.
    import requests
    monkeypatch.setattr(requests, "get", _boom)
    s = client.fetch_structure("APOE")
    assert s.source == "offline_snapshot"
    assert s.uniprot == "P02649"
    assert "unavailable" in s.error


def test_non_200_falls_back_to_snapshot(monkeypatch):
    client = AlphaFoldClient(prefer_offline=False)

    class _Resp:
        status_code = 503

        def json(self):
            return []

    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    s = client.fetch_structure("PSEN1")
    assert s.source == "offline_snapshot"
    assert s.uniprot == "P49768"


def test_monkeypatched_live_response_is_labeled_live(monkeypatch):
    client = AlphaFoldClient(prefer_offline=False)

    class _Resp:
        status_code = 200

        def json(self):
            return [{
                "uniprotAccession": "P05067",
                "gene": "APP",
                "globalMetricValue": 67.38,
                "cifUrl": "https://alphafold.ebi.ac.uk/files/AF-P05067-F1-model_v6.cif",
                "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P05067-F1-model_v6.pdb",
                "bcifUrl": "x", "entryId": "AF-P05067-F1", "latestVersion": 6,
            }]

    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    s = client.fetch_structure("APP")
    assert s.source == "live"
    assert s.mean_plddt == pytest.approx(67.38)
    assert s.model_version == "v6"
    assert s.gene_symbol == "APP"


# ---------------------------------------------------------------------------
# CIF pLDDT parser (pure-Python, no network)
# ---------------------------------------------------------------------------

def test_mean_ca_plddt_from_cif_parses_bfactor_column():
    # Minimal synthetic mmCIF _atom_site loop: two residues, CA B-factors 40 & 60.
    cif = "\n".join([
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.B_iso_or_equiv",
        "ATOM N   MET 10.0",
        "ATOM CA  MET 40.0",
        "ATOM CB  MET 11.0",
        "ATOM CA  ALA 60.0",
        "#",
    ])
    mean, n = af._mean_ca_plddt_from_cif(cif)
    assert n == 2
    assert mean == pytest.approx(50.0)


def test_mean_ca_plddt_handles_garbage():
    mean, n = af._mean_ca_plddt_from_cif("no atom site loop here")
    assert mean is None and n is None


# ---------------------------------------------------------------------------
# Guarded LIVE test — skips when the EBI host is unreachable.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _ebi_reachable(), reason="EBI AlphaFold DB unreachable")
def test_live_fetch_app_smoke():
    s = AlphaFoldClient(prefer_offline=False).fetch_structure("APP")
    # Online we expect a live label + a plausible mean pLDDT; but if the API
    # transiently fails the adapter still returns an honest offline fallback.
    assert s.uniprot == "P05067"
    assert s.source in ("live", "offline_snapshot")
    if s.source == "live":
        assert s.mean_plddt is not None and 0 < s.mean_plddt <= 100
        assert s.cif_url.endswith(".cif")
