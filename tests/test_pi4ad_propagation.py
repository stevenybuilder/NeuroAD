"""Tests for the in-repo STRING v12.0 network propagation (PI4AD sibling).

All offline / deterministic (NO network, NO credentials). Covers:
  (a) determinism        — same seeds -> byte-identical scores across runs and
                           regardless of seed order;
  (b) offline path       — the bundled STRING CSV is used and the live fetch is
                           NEVER touched in the default (prefer_offline) mode;
  (c) known connectivity — the real STRING Ras edges surface MAPK1 from an HRAS
                           seed, and glial seeds surface non-seed network hubs;
  (d) RWR convergence    — the closed-form solve matches an independent power-
                           iteration of the same random walk (series converges).

Provenance note under test: this is an in-repo RWR/heat-diffusion over a bundled
real STRING subgraph — it is NOT PI4AD's proprietary R (dTarget/oSubneterGenes)
pipeline, and the tests assert only what the real STRING edges actually support.
"""
from __future__ import annotations

import numpy as np
import pytest

from neuroad.integrations import pi4ad
from neuroad.integrations.pi4ad import (
    PI4AD,
    PropagatedNode,
    fetch_string_subgraph,
    propagate_hits,
)

AMYLOID = ["APP", "MAPT", "PSEN1", "BACE1", "APOE", "ESR1"]
GLIAL = ["TREM2", "APOE", "CLU", "MAPK1", "HRAS"]


# --------------------------------------------------------------------------
# Bundled subgraph loads with real STRING content
# --------------------------------------------------------------------------

def test_bundled_subgraph_loads_and_is_string_v12():
    edges, source = fetch_string_subgraph()  # offline default
    assert source == "string_v12_snapshot"
    assert len(edges) > 1000                 # real subgraph, not a stub
    # The canonical Ras edge MAPK1-HRAS is a very high-confidence STRING edge.
    lookup = {(a, b): s for a, b, s in edges}
    lookup.update({(b, a): s for a, b, s in edges})
    assert ("MAPK1", "HRAS") in lookup
    assert lookup[("MAPK1", "HRAS")] >= 900  # 0.979 -> 979 on the 0-1000 scale


# --------------------------------------------------------------------------
# (a) Determinism
# --------------------------------------------------------------------------

def test_propagation_is_deterministic_across_runs():
    a = [n.to_dict() for n in propagate_hits(AMYLOID)]
    b = [n.to_dict() for n in propagate_hits(AMYLOID)]
    assert a == b
    assert a, "expected a non-empty ranked node list"


def test_propagation_is_invariant_to_seed_order():
    a = [n.to_dict() for n in propagate_hits(GLIAL)]
    b = [n.to_dict() for n in propagate_hits(list(reversed(GLIAL)))]
    assert a == b


def test_case_insensitive_seeds():
    a = [n.to_dict() for n in propagate_hits(["hras"])]
    b = [n.to_dict() for n in propagate_hits(["HRAS"])]
    assert a == b


# --------------------------------------------------------------------------
# (b) Offline path — bundled CSV, live fetch never touched
# --------------------------------------------------------------------------

def test_offline_never_calls_live(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("live STRING fetch must not run in offline mode")

    monkeypatch.setattr(pi4ad, "_fetch_string_live", _boom)
    nodes = propagate_hits(AMYLOID, prefer_offline=True)
    assert nodes
    assert all(n.source == "string_v12_snapshot" for n in nodes)


def test_fetch_offline_degrades_when_live_would_fail(monkeypatch):
    # Even in prefer_offline=False, a failing live fetch degrades to the snapshot.
    monkeypatch.setattr(pi4ad, "_fetch_string_live", lambda *a, **k: None)
    edges, source = fetch_string_subgraph(["APP"], prefer_offline=False)
    assert source == "string_v12_snapshot"
    assert edges


# --------------------------------------------------------------------------
# (c) Known connectivity over the real STRING edges
# --------------------------------------------------------------------------

def test_hras_seed_surfaces_mapk1_via_ras_edge():
    # HRAS's strongest AD-universe partners are the other Ras GTPases and the ERK
    # kinase MAPK1 (direct STRING edge 0.979); it must land in the top non-seed set.
    nodes = propagate_hits(["HRAS"])
    non_seed = [n.gene for n in nodes if not n.is_seed]
    assert "MAPK1" in non_seed[:6]
    # NRAS/KRAS (the Ras cluster) rank at the very top of the non-seed nodes.
    assert {"NRAS", "KRAS"} & set(non_seed[:3])


def test_glial_seeds_surface_non_seed_hubs():
    nodes = propagate_hits(GLIAL)
    hubs = [n for n in nodes if n.is_hub]
    assert hubs, "glial propagation should surface at least one network hub"
    # Hubs are non-seeds with high propagated mass AND high degree.
    seed_up = {g.upper() for g in GLIAL}
    for h in hubs:
        assert h.gene.upper() not in seed_up
        assert not h.is_seed
    # APP is the highest-degree hub the glial cluster lights up.
    assert "APP" in {h.gene for h in hubs}


def test_seeds_outrank_periphery():
    nodes = propagate_hits(["HRAS"])
    by_gene = {n.gene: n for n in nodes}
    # The seed itself holds the most restart mass -> rank 1.
    assert by_gene["HRAS"].rank == 1
    assert by_gene["HRAS"].is_seed


# --------------------------------------------------------------------------
# (d) RWR convergence / correctness vs an independent power iteration
# --------------------------------------------------------------------------

def _reference_rwr(seed: str, restart: float = 0.5, threshold: float = 0.4):
    """Independent, plain-numpy power-iteration RWR over the bundled edges."""
    edges, _ = fetch_string_subgraph()
    idx: dict[str, int] = {}
    order: list[str] = []

    def _i(sym: str) -> int:
        u = sym.upper()
        if u not in idx:
            idx[u] = len(order)
            order.append(u)
        return idx[u]

    pairs = []
    for a, b, s in edges:
        w = s / 1000.0
        if w < threshold:
            continue
        pairs.append((_i(a), _i(b), w))
    n = len(order)
    A = np.zeros((n, n))
    for ia, ib, w in pairs:
        A[ia, ib] = w
        A[ib, ia] = w
    deg = A.sum(1)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    W = A * dinv[:, None] * dinv[None, :]
    s_vec = np.zeros(n)
    s_vec[idx[seed.upper()]] = 1.0
    # Power iteration: p_{k+1} = (1-r) s + r W p_k. Converges iff spectral radius
    # of rW < 1 (true here: sym-normalized W has spectral radius <= 1, r=0.5).
    p = s_vec.copy()
    for _ in range(2000):
        p_next = (1.0 - restart) * s_vec + restart * (W @ p)
        if np.max(np.abs(p_next - p)) < 1e-12:
            p = p_next
            break
        p = p_next
    return {g: p[idx[g.upper()]] for g in order}


def test_closed_form_matches_power_iteration():
    ref = _reference_rwr("HRAS")
    nodes = {n.gene: n.propagated_score for n in propagate_hits(["HRAS"])}
    # Same genes, and scores agree to closed-form precision.
    assert set(nodes) == set(ref)
    for g, score in nodes.items():
        assert score == pytest.approx(ref[g], abs=1e-5)


def test_row_normalized_walk_is_convergent():
    # The RWR series sum_k (rW)^k converges; the reference iteration above hit its
    # fixed point, so the propagated vector is finite and positive on the seed.
    ref = _reference_rwr("HRAS")
    assert all(np.isfinite(v) for v in ref.values())
    assert ref["HRAS"] > 0


# --------------------------------------------------------------------------
# Heat-diffusion variant + robustness
# --------------------------------------------------------------------------

def test_heat_variant_is_deterministic_and_labeled():
    a = [n.to_dict() for n in propagate_hits(["HRAS"], method="heat")]
    b = [n.to_dict() for n in propagate_hits(["HRAS"], method="heat")]
    assert a == b
    assert a and all(n["method"] == "heat" for n in a)


def test_empty_and_unknown_seeds_degrade_to_empty():
    assert propagate_hits([]) == []
    assert propagate_hits(["NOT_A_REAL_GENE"]) == []
    assert propagate_hits(["APP"], restart=1.5) == []  # invalid restart


def test_pi4ad_method_delegates():
    pi = PI4AD()
    nodes = pi.propagate(["HRAS"])
    assert nodes and isinstance(nodes[0], PropagatedNode)


# --------------------------------------------------------------------------
# Translation wiring — additive network_hubs, offline, non-perturbing
# --------------------------------------------------------------------------

def test_translation_attaches_network_hubs_offline():
    from neuroad.harness import translation

    out = translation.translate("glial")
    assert "network_hubs" in out
    assert isinstance(out["network_hubs"], list)
    # Every attached hub is provenance-stamped from the STRING network.
    for h in out["network_hubs"]:
        assert h["is_hub"] is True
        assert h["source"] == "string_v12_snapshot"
        assert "propagated_score" in h


def test_translation_network_hub_failure_is_non_fatal(monkeypatch):
    from neuroad.harness import translation

    monkeypatch.setattr(translation, "_network_hubs", lambda *a, **k: [])
    out = translation.translate("glial")
    assert out["network_hubs"] == []
    # The rest of the chain is unaffected.
    assert out["mechanism"] == "glial"
