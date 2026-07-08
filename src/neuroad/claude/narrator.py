"""
narrator — plain-language verdict, naming the assumption that would break it.

narrate(card) -> str

A good referee tells a scientist not just "partially robust", but the *one
assumption* under which the finding stops being true — the thing they should go
check before spending a quarter on it. The narration cites the batch-effect
prior art rather than claiming the leakage insight.
"""
from __future__ import annotations

from typing import Optional

from ..contract import ClaimCard, TestEvidence, TestResult, GAUNTLET_BY_KEY
from ..calibration import PRIOR_ART
from . import _client

SYSTEM = (
    "Persona: NARRATOR. In one tight paragraph, tell the scientist what the "
    "verdict means in plain language and — most importantly — name the single "
    "assumption under which this finding stops being true, so they know exactly "
    "what to go check. Cite the published batch-effect prior art for the leakage "
    "mechanic; do not present it as your own discovery. Keep the verdict hedged."
)


def narrate(card: ClaimCard) -> str:
    """Return a plain-language verdict paragraph for a ClaimCard."""
    if _client.USING_LIVE_API:
        try:
            txt = _client.complete(SYSTEM, _prompt(card))
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
        except Exception:
            pass
    return _fallback(card)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _by_key(tests: list[TestEvidence], key: str) -> Optional[TestEvidence]:
    for t in tests:
        if t.key == key:
            return t
    return None


def _breaking_assumption(card: ClaimCard) -> str:
    """The single load-bearing assumption behind this verdict."""
    tests = card.tests
    site = _by_key(tests, "site_scanner")
    if site is not None:
        s = site.stats or {}
        margin = s.get("margin")
        if margin is None and "outcome_auc" in s and "scanner_auc" in s:
            margin = float(s["outcome_auc"]) - float(s["scanner_auc"])
        if margin is not None and margin <= 0.15:
            return (
                "that the scanner/site mix in any new cohort matches this one — "
                f"the leakage margin is only {float(margin):+.2f} AUC, so a "
                "different scanner distribution could erase the effect"
            )
    brain = _by_key(tests, "brain_age")
    if brain is not None and brain.result in (TestResult.WEAKENED, TestResult.MIXED, TestResult.FAILED):
        return (
            "that the signal is more than accelerated brain-age; a chunk of it "
            "co-varies with the brain-age control, so in a cohort matched on "
            "brain-age gap the separation would shrink"
        )
    anchor = _by_key(tests, "biomarker_anchor")
    if anchor is None or anchor.result == TestResult.NA:
        return (
            "that a molecular anchor exists at all — with no p-tau217 / GFAP "
            "correlation measured here, the finding rests on imaging alone and "
            "would not survive a demand for orthogonal pathology"
        )
    repl = _by_key(tests, "replication")
    if repl is not None and repl.result in (TestResult.WEAKENED, TestResult.MIXED, TestResult.FAILED):
        return (
            "that the effect reproduces off the cohort it was found in — it "
            "weakened on the held-out site, so it may be partly cohort-specific"
        )
    return (
        "that the held-out cohort resembles this one on scanner, age and sex — "
        "the effect is calibrated to the population it was measured in"
    )


def _fallback(card: ClaimCard) -> str:
    v = card.verdict.value
    score = card.score
    naive = card.naive_effect or {}
    metric = naive.get("metric", "AUC")
    value = naive.get("value")
    prior = PRIOR_ART[0]

    lead = f"The claim “{card.claim.claim_text}” lands at “{v}” ({score}/100)."
    if value is not None:
        lead += (
            f" The naive effect ({metric} {float(value):.2f} separating "
            f"{card.claim.group_a} from {card.claim.group_b}) is real on paper, "
            "but the referee's job is to ask what else could produce it."
        )

    if card.promoted:
        stance = (
            " It cleared the gauntlet's promotion floor: the effect exceeds the "
            "scanner-prediction floor on subject-disjoint splits and is worth a "
            "confirmatory follow-up — as a hypothesis, not a settled result."
        )
    else:
        stance = (
            " It sits below the promotion floor, so the honest read is that "
            "acquisition, demographics or generic aging can plausibly account "
            "for it, and it should not consume a scientist's quarter without "
            "stronger evidence."
        )

    where = (
        f" Where it stops being true: {_breaking_assumption(card)}."
    )
    cite = (
        f" That the same frozen embeddings can predict scanner/site as well as "
        f"outcome is documented prior art (“{prior[0]},” {prior[1]}); "
        "NeuroAD Discovery Engine only runs the audit and issues the verdict."
    )
    return (lead + stance + where + cite).strip()


def _prompt(card: ClaimCard) -> str:
    lines = [
        f"Claim: {card.claim.claim_text}",
        f"Populations: {card.claim.group_a} vs {card.claim.group_b}.",
        f"Naive effect: {card.naive_effect}.",
        f"Robustness score: {card.score}/100 -> verdict '{card.verdict.value}'. "
        f"Promoted: {card.promoted}.",
        "Gauntlet:",
    ]
    for t in card.tests:
        dim = GAUNTLET_BY_KEY.get(t.key)
        label = dim.label if dim else t.key
        lines.append(f"  - {label}: {t.result.value}. {t.detail} stats={t.stats}")
    lines.append(
        "Write the plain-language verdict paragraph and explicitly name the one "
        "assumption under which this finding stops being true."
    )
    return "\n".join(lines)
