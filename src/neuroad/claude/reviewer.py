"""
reviewer — a referee that referees itself.

review(card) -> {'critique': [...], 'revised_caveats': [...]}

The reviewer argues AGAINST the tool's own verdict: it flags the proxy nature of
the brain-age control, plasma-biomarker missingness, the gap between "partially
robust" and "robust", small N, and the fact that the leakage test uses the same
probe family it is auditing. It cites the batch-effect prior art rather than
claiming the insight.
"""
from __future__ import annotations

from typing import Optional

from ..contract import ClaimCard, TestEvidence, TestResult, Verdict
from ..calibration import PRIOR_ART
from . import _client

SYSTEM = (
    "Persona: ADVERSARIAL REVIEWER. Peer-review the referee's OWN verdict and "
    "argue against it. Raise the weaknesses a sceptical reviewer would: the "
    "brain-age control is a proxy (embedding-derived, not a gold standard — quote "
    "THIS card's own R2/MAE, never a generic number); plasma p-tau217/GFAP "
    "coverage is partial so the anchor may "
    "rest on a small complete-case subset; 'partially robust' is not 'robust'; "
    "small N inflates apparent effects; and the leakage test uses the same probe "
    "family it audits. Cite the batch-effect prior art rather than claiming it. "
    "Then give revised, more conservative caveats."
)

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "critique": {"type": "array", "items": {"type": "string"}},
        "revised_caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["critique", "revised_caveats"],
}


def review(card: ClaimCard) -> dict:
    """Return an adversarial critique of the card's own verdict + tighter caveats.

    DETERMINISTIC: the referee never calls Claude. This produces the critique from
    the card's own TestEvidence stats. (Claude's only role in the engine is the
    orchestrator — see harness/agent.py.)"""
    return _fallback(card)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _by_key(tests: list[TestEvidence], key: str) -> Optional[TestEvidence]:
    for t in tests:
        if t.key == key:
            return t
    return None


def _sample_n(card: ClaimCard) -> Optional[int]:
    naive = card.naive_effect or {}
    for k in ("n", "n_subjects", "N"):
        if k in naive:
            try:
                return int(naive[k])
            except (TypeError, ValueError):
                pass
    for t in card.tests:
        for k in ("n", "n_complete", "n_subjects"):
            if k in (t.stats or {}):
                try:
                    return int(t.stats[k])
                except (TypeError, ValueError):
                    pass
    return None


def _fallback(card: ClaimCard) -> dict:
    prior_batch = PRIOR_ART[0]
    prior_frozen = PRIOR_ART[1]
    critique: list[str] = []

    # 1. partially robust != robust
    if card.verdict == Verdict.PARTIALLY_ROBUST:
        critique.append(
            "‘Partially robust’ is not ‘robust’. The score renormalizes over the "
            "tests that ran, so a middling verdict can hide a dimension that "
            "quietly failed — read the per-test results, not just the headline."
        )
    elif card.verdict in (Verdict.ROBUST_FOLLOWUP, Verdict.STRONG):
        critique.append(
            f"A ‘{card.verdict.value}’ verdict is a licence to follow up, not to "
            "conclude — the referee promotes hypotheses, it does not confirm them."
        )

    # 2. brain-age control is a proxy — quote THIS card's real fit, never a fixed
    #    calibration string (which could contradict the numbers on the same card).
    brain = _by_key(card.tests, "brain_age")
    if brain is not None and brain.result != TestResult.NA:
        bs = brain.stats or {}
        r2, mae = bs.get("r2"), bs.get("mae_yr")
        fit = (f" (this cohort's fit: R2={r2:.2f}, MAE={mae:.1f}yr)"
               if isinstance(r2, (int, float)) and isinstance(mae, (int, float)) else "")
        critique.append(
            "The brain-age control is a proxy, not a gold standard: it is "
            f"embedding-derived{fit}, so residual generic-aging signal can survive "
            "the adjustment and masquerade as disease-specific."
        )

    # 3. biomarker missingness
    anchor = _by_key(card.tests, "biomarker_anchor")
    if anchor is None or anchor.result == TestResult.NA:
        critique.append(
            "There is no usable plasma p-tau217 / GFAP anchor here, so the "
            "molecular gate is unmet — the finding rests on imaging alone."
        )
    else:
        n_anchor = (anchor.stats or {}).get("n")
        extra = f" (complete-case n={n_anchor})" if n_anchor else ""
        critique.append(
            "The biomarker anchor likely rests on a partial complete-case subset"
            f"{extra}; with realistic p-tau217 missingness the correlation is "
            "estimated on far fewer subjects than the headline cohort."
        )

    # 4. small N
    n = _sample_n(card)
    if n is not None and n < 200:
        critique.append(
            f"N is small (~{n}); at this size AUCs are noisy and a single "
            "site's idiosyncrasy can dominate — treat effect sizes as wide-"
            "interval estimates."
        )
    else:
        critique.append(
            "Confirm the effective sample size per split — subject-disjoint "
            "cross-validation on a modest cohort leaves each fold thin."
        )

    # 5. own-probe leakage risk
    critique.append(
        "The leakage test audits the embeddings with the same linear-probe "
        "family it uses for the outcome, so it bounds — but cannot fully rule "
        f"out — shared confounding; this is the mechanic quantified in "
        f"‘{prior_frozen[0]}’ ({prior_frozen[1]}), which we cite rather than claim."
    )

    revised_caveats = [
        "Verdict is provisional and conditional on the scanner/site distribution "
        "of any replication cohort.",
        "Brain-age adjustment uses a proxy control; residual aging signal cannot "
        "be excluded.",
        "Any biomarker anchor is complete-case only and may not generalize.",
        (
            "Leakage is bounded, not eliminated — see the published batch-effect "
            f"audits (‘{prior_batch[0]},’ {prior_batch[1]})."
        ),
    ]
    if card.promoted:
        revised_caveats.append(
            "Promotion authorizes one confirmatory experiment, not adoption of "
            "the finding."
        )

    return {"critique": critique, "revised_caveats": revised_caveats}


def _prompt(card: ClaimCard) -> str:
    n = _sample_n(card)
    lines = [
        f"Verdict under review: '{card.verdict.value}' ({card.score}/100), "
        f"promoted={card.promoted}.",
        f"Claim: {card.claim.claim_text}.",
        f"Naive effect: {card.naive_effect}.",
        f"Approx N: {n}.",
        "Gauntlet:",
    ]
    for t in card.tests:
        lines.append(f"  - {t.key}: {t.result.value}. {t.detail} stats={t.stats}")
    lines.append(
        "Argue against this verdict and then list tightened, more conservative "
        "caveats."
    )
    return "\n".join(lines)
