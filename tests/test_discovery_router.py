"""
Tests for the Stage-2 harness lane (agent A, T3/T4):

  * discovery_router.route — novel-pattern -> unsupervised Detective;
    named-contrast -> supervised probe/gauntlet; deterministic + offline.
  * experiment_card.build_experiment_card — wraps a ClaimCard with the three
    Stage-2 annotations, always stamps a non-empty novelty_class + honesty_rung
    (the anti-overclaim contract), and merges cleanly in to_dict().
"""
from __future__ import annotations

import pytest

from neuroad import contract
from neuroad.contract import Claim, ClaimCard, Verdict
from neuroad.harness import discovery_router as router
from neuroad.harness import experiment_card as ec


# --------------------------------------------------------------------------- #
# discovery_router
# --------------------------------------------------------------------------- #
def test_novel_pattern_string_routes_to_detective():
    d = router.route("Are there hidden MCI subtypes / novel phenotypes?")
    assert d.mode == "unsupervised"
    assert not d.supervised
    assert d.engine == "neuroad.discovery.discover_and_referee"
    assert d.target is None
    # detective_cfg must be dispatchable verbatim to discover_and_referee.
    assert d.detective_cfg == {"method": "kmeans", "B": 50}
    assert d.signals  # at least one matched novel-pattern keyword


@pytest.mark.parametrize("text", [
    "discover data-driven subgroups of converters",
    "cluster the embeddings to find latent phenotypes",
    "is there hidden structure / an emergent stratification?",
])
def test_various_novel_hypotheses_route_unsupervised(text):
    assert router.route(text).mode == "unsupervised"


def test_named_contrast_string_routes_to_probe():
    d = router.route("Does the embedding predict MCI->AD conversion?")
    assert d.mode == "supervised"
    assert d.supervised
    assert d.engine == "neuroad.pipeline.run_referee"
    assert d.target == "conversion"
    assert d.detective_cfg == {}


def test_structured_named_claim_uses_its_target():
    claim = Claim(
        claim_id="c1",
        claim_text="AD vs CN diagnosis from structural embeddings",
        target="dx_binary",
        group_a="AD", group_b="CN",
    )
    d = router.route(claim)
    assert d.mode == "supervised"
    assert d.target == "dx_binary"


def test_unknown_target_falls_back_to_conversion():
    claim = Claim(claim_id="c2", claim_text="separate the two groups",
                  target="not_a_real_target", group_a="A", group_b="B")
    d = router.route(claim)
    assert d.mode == "supervised"
    assert d.target == "conversion"        # fallback to a valid LABEL_TARGET
    assert d.target in contract.LABEL_TARGETS


def test_novel_language_wins_over_default_groups():
    # A structured claim whose TEXT asks for discovery routes unsupervised even
    # though it carries default converter/non-converter groups.
    claim = Claim(
        claim_id="c3",
        claim_text="Do MCI converters fall into distinct phenotype clusters?",
        target="conversion",
        group_a="MCI converters", group_b="MCI non-converters",
    )
    assert router.route(claim).mode == "unsupervised"


def test_decision_is_deterministic_and_serializable():
    a = router.route("predict conversion").to_dict()
    b = router.route("predict conversion").to_dict()
    assert a == b
    assert set(a) == {"mode", "engine", "target", "detective_cfg",
                      "rationale", "signals"}


def test_router_engine_targets_resolve_to_real_callables():
    from neuroad import discovery, pipeline
    assert callable(discovery.discover_and_referee)
    assert callable(pipeline.run_referee)


# --------------------------------------------------------------------------- #
# experiment_card
# --------------------------------------------------------------------------- #
def _make_card(*, score: int, verdict: Verdict, promoted: bool) -> ClaimCard:
    return ClaimCard(
        claim=Claim(claim_id="k", claim_text="a hunch", target="conversion"),
        naive_effect={"metric": "AUC", "value": 0.74},
        tests=[],
        score=score,
        verdict=verdict,
        promoted=promoted,
    )


def test_build_stamps_defaults_and_mirrors_onto_claimcard():
    card = _make_card(score=90, verdict=Verdict.STRONG, promoted=True)
    xc = ec.build_experiment_card(card)
    # never blank — the anti-overclaim invariant
    assert xc.novelty_class == "unclassified"
    assert xc.honesty_rung == "replication-ready"
    # stamped back onto the frozen-contract fields
    assert card.novelty_class == "unclassified"
    assert card.honesty_rung == "replication-ready"


def test_honesty_rung_never_claims_proven_or_validated():
    for score, verdict, promoted, expected in [
        (10, Verdict.FRAGILE, False, "artifact-suspected"),
        (55, Verdict.PARTIALLY_ROBUST, False, "exploratory"),
        (78, Verdict.ROBUST_FOLLOWUP, False, "candidate-signal"),
        (78, Verdict.ROBUST_FOLLOWUP, True, "corroborated-candidate"),
        (90, Verdict.STRONG, True, "replication-ready"),
    ]:
        card = _make_card(score=score, verdict=verdict, promoted=promoted)
        rung = ec.build_experiment_card(card).honesty_rung
        assert rung == expected
        assert rung in ec.HONESTY_LADDER
        assert "proven" not in rung and "validated" not in rung


def test_explicit_annotations_and_to_dict_merge():
    card = _make_card(score=90, verdict=Verdict.STRONG, promoted=True)
    xc = ec.build_experiment_card(
        card,
        novelty_class="Novel",
        atn_profile={"A": "+", "T": "+", "N": "?"},
        honesty_rung="corroborated-candidate",
        discovery_provenance={"mode": "unsupervised", "stability": 0.71},
    )
    d = xc.to_dict()
    # ClaimCard fields still present (merge, not replace)
    assert d["claim_id"] == "k"
    assert d["robustness_score"] == 90
    # harness block
    assert d["novelty_class"] == "novel"          # normalized lower-case
    assert d["atn_profile"] == {"A": "+", "T": "+", "N": "?"}
    assert d["honesty_rung"] == "corroborated-candidate"
    assert d["discovery_provenance"] == {"mode": "unsupervised", "stability": 0.71}


def test_card_atn_field_is_read_when_arg_omitted():
    card = _make_card(score=50, verdict=Verdict.PARTIALLY_ROBUST, promoted=False)
    card.atn_profile = {"A": "-", "T": "-", "N": "+"}
    card.novelty_class = "adjacent"
    xc = ec.build_experiment_card(card)
    assert xc.novelty_class == "adjacent"
    assert xc.atn_profile == {"A": "-", "T": "-", "N": "+"}
    assert xc.honesty_rung == "exploratory"
