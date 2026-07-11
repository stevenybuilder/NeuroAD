"""Tests for the active-learning experiment loop (harness/discovery_loop.py).

Deterministic, offline. Covers: composite-prior seeding, the Bayesian update
direction (a hit raises the posterior, a miss lowers it), acquisition selection,
ranking, and persistence round-trip.
"""
from __future__ import annotations

import json

from neuroad.harness.discovery_loop import DiscoveryLoop, TargetBelief


def _fresh(tmp_path):
    return DiscoveryLoop(state_path=tmp_path / "loop.json")


def test_seed_produces_beta_priors_from_evidence(tmp_path):
    loop = _fresh(tmp_path)
    seeded = loop.seed_mechanism("amyloid_cascade")
    assert seeded, "expected candidate genes for amyloid_cascade"
    for tb in seeded:
        assert 0.0 < tb.prior_score < 1.0
        assert tb.alpha > 0 and tb.beta > 0
        assert 0.0 <= tb.mean <= 1.0
        # prior is a composite of >=1 evidence source (pi4ad/structure/opentargets)
        assert tb.evidence


def test_hit_raises_posterior_miss_lowers_it(tmp_path):
    loop = _fresh(tmp_path)
    loop.seed_mechanism("amyloid_cascade")
    spec = loop.propose_next_experiment()
    before = loop.beliefs[spec.gene].mean
    loop.record_result(spec.experiment_id, hit=True, effect_size=1.5)
    after_hit = loop.beliefs[spec.gene].mean
    assert after_hit > before                       # a hit raises belief

    spec2 = loop.propose_next_experiment()
    g2 = spec2.gene
    b2 = loop.beliefs[g2].mean
    loop.record_result(spec2.experiment_id, hit=False, effect_size=1.5)
    assert loop.beliefs[g2].mean < b2               # a miss lowers belief


def test_uncertainty_strategy_explores_distinct_targets(tmp_path):
    loop = _fresh(tmp_path)
    loop.seed_mechanism("amyloid_cascade")
    picks = []
    for _ in range(3):
        spec = loop.propose_next_experiment(strategy="uncertainty")
        picks.append(spec.gene)
        loop.record_result(spec.experiment_id, hit=False)  # shrink its variance
    # pure information-gain should not keep hammering the same target
    assert len(set(picks)) >= 2


def test_experiment_spec_is_falsifiable(tmp_path):
    loop = _fresh(tmp_path)
    loop.seed_mechanism("glial")
    spec = loop.propose_next_experiment()
    assert spec.gene and spec.model_system and spec.readout
    assert "kill" in spec.kill_criterion.lower() or "downweight" in spec.kill_criterion.lower()


def test_persistence_roundtrip(tmp_path):
    loop = _fresh(tmp_path)
    loop.seed_mechanism("amyloid_cascade")
    spec = loop.propose_next_experiment()
    loop.record_result(spec.experiment_id, hit=True)
    path = loop.save()
    assert path.exists()

    reloaded = DiscoveryLoop.load(state_path=path)
    assert reloaded.beliefs.keys() == loop.beliefs.keys()
    assert reloaded.beliefs[spec.gene].n_experiments == 1
    assert reloaded._counter == loop._counter


def test_agent_tools_drive_the_loop(tmp_path, monkeypatch):
    # Point the loop's default state at a temp file so tools persist there.
    import neuroad.harness.discovery_loop as dl
    monkeypatch.setattr(dl, "_STATE_PATH", tmp_path / "loop.json")
    from neuroad.harness import agent

    seeded = json.loads(agent.seed_experiment_targets("amyloid_cascade"))
    assert seeded["seeded"]
    spec = json.loads(agent.propose_experiment("amyloid_cascade"))
    assert spec["gene"] and spec["experiment_id"]
    out = json.loads(agent.record_wetlab_result(spec["experiment_id"], True, 1.2))
    assert out["updated"]["n_experiments"] == 1
    rank = json.loads(agent.target_ranking())
    assert rank["n_experiments_completed"] == 1
