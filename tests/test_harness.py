"""
End-to-end tests for the L5 orchestrator (neuroad.harness.orchestrator):

  * investigate() runs fully OFFLINE (api=False, no ANTHROPIC_API_KEY) on the
    synthetic harness and returns an ExperimentCard that always carries the
    honesty fields (non-empty novelty_class + honesty_rung) drawn from the L3
    novelty_rubric ladder;
  * the biomarker-anchor HARD GATE + honesty ladder are wired to real card
    evidence (SURVIVOR climbs the ladder; KILL, which fails a STAR confound, is
    capped low);
  * the HONESTY GUARD blocks a planted overclaim.
"""
from __future__ import annotations

import pytest

from neuroad.contract import Claim, ClaimCard, TestEvidence, TestResult, Verdict
from neuroad.harness import experiment_card as ec
from neuroad.harness import orchestrator
from neuroad.harness.orchestrator import (
    HonestyViolation,
    ExperimentCard,
    honesty_guard,
    investigate,
)

_SURVIVOR_CLAIM = (
    "MCI patients who convert to AD show a distinct structural-MRI signature "
    "in their frozen embeddings versus non-converters."
)


# --------------------------------------------------------------------------- #
# end-to-end investigate() on the synthetic harness (offline)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dataset", ["synthetic:SURVIVOR", "synthetic:KILL"])
def test_investigate_end_to_end_offline(dataset, monkeypatch):
    # Guarantee the offline path even if the environment has a key.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    xcard = investigate(_SURVIVOR_CLAIM, dataset, api=False)

    # An ExperimentCard wrapping a real refereed ClaimCard.
    assert isinstance(xcard, ExperimentCard)
    assert isinstance(xcard.card, ClaimCard)

    # Honesty fields are always present and non-empty.
    assert xcard.novelty_class and xcard.novelty_class.strip()
    assert xcard.honesty_rung and xcard.honesty_rung.strip()
    # The rung comes from the novelty_rubric ladder (not the experiment_card default).
    assert xcard.honesty_rung in orchestrator._novelty_rungs()

    # Provenance records mode + the pre-registered kill criterion + anchor gate.
    prov = xcard.discovery_provenance
    assert prov["mode"] == "supervised"          # named-contrast hypothesis
    assert prov["dataset"] == dataset
    assert prov["kill_criterion"]                # a falsifier was pre-registered
    assert prov["expected_direction"]
    assert prov["anchor_gate"]["gate"] == "biomarker_anchor"

    # Serializes cleanly with the harness block merged in.
    d = xcard.to_dict()
    assert d["novelty_class"] == xcard.novelty_class
    assert d["honesty_rung"] == xcard.honesty_rung
    assert d["discovery_provenance"]["mode"] == "supervised"

    # And it passes its own guard (investigate ran it, but assert idempotency).
    assert honesty_guard(xcard) is xcard


def test_investigate_survivor_outclimbs_kill():
    """SURVIVOR survives the confounds and should sit strictly higher on the
    honesty ladder than KILL, which fails a STAR leakage test."""
    rungs = orchestrator._novelty_rungs()
    surv = investigate(_SURVIVOR_CLAIM, "synthetic:SURVIVOR", api=False)
    kill = investigate(_SURVIVOR_CLAIM, "synthetic:KILL", api=False)
    assert rungs.index(surv.honesty_rung) > rungs.index(kill.honesty_rung)
    # KILL is capped at/below the confound-survivor rung (it fails a STAR test).
    assert rungs.index(kill.honesty_rung) <= rungs.index("confound_survivor")


def test_investigate_is_deterministic_offline():
    a = investigate(_SURVIVOR_CLAIM, "synthetic:SURVIVOR", api=False).to_dict()
    b = investigate(_SURVIVOR_CLAIM, "synthetic:SURVIVOR", api=False).to_dict()
    assert a["honesty_rung"] == b["honesty_rung"]
    assert a["novelty_class"] == b["novelty_class"]
    assert a["robustness_score"] == b["robustness_score"]


# --------------------------------------------------------------------------- #
# HONESTY GUARD
# --------------------------------------------------------------------------- #
def _wrap(card: ClaimCard) -> ExperimentCard:
    return ec.build_experiment_card(
        card, novelty_class="novel", honesty_rung="confound_survivor")


def _plain_card() -> ClaimCard:
    return ClaimCard(
        claim=Claim(claim_id="k", claim_text="a hunch", target="conversion"),
        naive_effect={"metric": "AUC", "value": 0.74},
        tests=[],
        score=50,
        verdict=Verdict.PARTIALLY_ROBUST,
        promoted=False,
    )


def test_guard_passes_a_clean_card():
    assert honesty_guard(_wrap(_plain_card())) is not None


@pytest.mark.parametrize("overclaim", [
    "This is a proven biomarker for Alzheimer's.",
    "A validated biomarker of conversion.",
    "The signature is clinically validated.",
    "The probe detects preclinical disease.",
    "A structural cure for dementia.",
])
def test_guard_blocks_planted_overclaim(overclaim):
    card = _plain_card()
    card.biology_hypothesis = overclaim        # planted into a rendered field
    with pytest.raises(HonestyViolation):
        honesty_guard(_wrap(card))


def test_guard_blocks_overclaim_in_narration_attr():
    card = _plain_card()
    card.narration = "In practice this is a proven biomarker."
    with pytest.raises(HonestyViolation):
        honesty_guard(_wrap(card))


def test_guard_requires_novelty_and_rung():
    card = _plain_card()
    xc = ExperimentCard(card=card, novelty_class="", honesty_rung="stable_cluster")
    with pytest.raises(HonestyViolation):
        honesty_guard(xc)
    xc2 = ExperimentCard(card=card, novelty_class="novel", honesty_rung="")
    with pytest.raises(HonestyViolation):
        honesty_guard(xc2)


def test_guard_word_boundary_cure_does_not_false_positive():
    # "secure"/"accurate" contain the letters of "cure" but must NOT trip it.
    card = _plain_card()
    card.biology_hypothesis = ("A secure, accurate structural probe with reproducible "
                               "out-of-fold scores.")
    assert honesty_guard(_wrap(card)) is not None


# --------------------------------------------------------------------------- #
# HARD GATE + honesty ladder unit behavior
# --------------------------------------------------------------------------- #
def _card_with_anchor(anchor: TestResult, *, promoted: bool,
                      score: int = 50) -> ClaimCard:
    tests = [TestEvidence("biomarker_anchor", anchor,
                          stats={"ptau217_ci_lo": 0.20 if anchor == TestResult.PASSED
                                 else -0.05})]
    return ClaimCard(
        claim=Claim(claim_id="g", claim_text="x", target="conversion"),
        naive_effect={"metric": "AUC", "value": 0.72},
        tests=tests, score=score, verdict=Verdict.PARTIALLY_ROBUST,
        promoted=promoted,
    )


def test_hard_gate_blocks_promoted_failed_anchor():
    card = _card_with_anchor(TestResult.FAILED, promoted=True)
    gate = orchestrator.apply_biomarker_anchor_gate(card, mechanism="amyloid_cascade")
    assert gate["blocked_promotion"] is True
    assert card.promoted is False               # HARD GATE flipped it
    assert any("HARD GATE" in c for c in card.caveats)


def test_hard_gate_passes_clean_anchor():
    card = _card_with_anchor(TestResult.PASSED, promoted=True)
    gate = orchestrator.apply_biomarker_anchor_gate(card, mechanism="amyloid_cascade")
    assert gate["blocked_promotion"] is False
    assert gate["passed"] is True
    assert card.promoted is True


def test_hard_gate_na_anchor_is_not_condemned():
    card = _card_with_anchor(TestResult.NA, promoted=True)
    gate = orchestrator.apply_biomarker_anchor_gate(card)
    assert gate["blocked_promotion"] is False
    assert card.promoted is True                # NA neither credits nor condemns
