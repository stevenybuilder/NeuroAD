"""End-to-end orchestration for the NeuroAD Discovery Engine.

`run_referee(df, claim)` chains the whole loop:

    probe (naive effect)
      -> gauntlet.run_gauntlet (the 5 adversarial tests)
      -> scoring.build_claim_card (weighted verdict)
      -> if promoted: courtroom.adjudicate + bridge.propose_biology
      -> reviewer.review (argue against the verdict)
      -> narrator.narrate (plain-language summary)

The core engine (`probe`, `gauntlet`, `scoring`) is required. The Claude
reasoning layer (`neuroad.claude.*`) is OPTIONAL and every call into it is
wrapped so a missing package or an offline API never crashes a run — the demo
must complete fully offline. All imports of sibling modules are lazy so this
module imports cleanly even before the rest of the engine has landed.
"""
from __future__ import annotations

from typing import Optional, Union

import pandas as pd

from neuroad import contract
from neuroad.contract import Claim, ClaimCard, TestEvidence, TestResult


# ---------------------------------------------------------------------------
# Claude-layer shims — every one degrades to a safe default, never raises.
# ---------------------------------------------------------------------------

def _parse_claim(text: str, df: Optional[pd.DataFrame]) -> Claim:
    """NL hunch -> structured Claim, with a deterministic fallback."""
    try:
        from neuroad.claude import claim_parser
        claim = claim_parser.parse_claim(text, df)
        if isinstance(claim, Claim):
            return claim
    except Exception:
        pass
    # Fallback: a sensible default claim keyed to conversion.
    return Claim(
        claim_id="claim-fallback",
        claim_text=text,
        target="conversion",
        group_a="MCI converters",
        group_b="MCI non-converters",
    )


def _adjudicate(claim: Claim, tests: list[TestEvidence]) -> Optional[dict]:
    try:
        from neuroad.claude import courtroom
        result = courtroom.adjudicate(claim, tests)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    return None


def _propose_biology(card: ClaimCard, df: pd.DataFrame) -> Optional[dict]:
    try:
        from neuroad.claude import bridge
        result = bridge.propose_biology(card, df)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    return None


def _review(card: ClaimCard) -> Optional[dict]:
    try:
        from neuroad.claude import reviewer
        result = reviewer.review(card)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    return None


def _narrate(card: ClaimCard) -> str:
    try:
        from neuroad.claude import narrator
        text = narrator.narrate(card)
        if isinstance(text, str) and text.strip():
            return text
    except Exception:
        pass
    return _fallback_narration(card)


def _fallback_narration(card: ClaimCard) -> str:
    metric = card.naive_effect.get("metric", "AUC")
    value = card.naive_effect.get("value", "?")
    return (
        f"Naive {metric} = {value}. After the adversarial gauntlet the claim "
        f"scores {card.score}/100 -> verdict: {card.verdict.value}. "
        + ("Promoted to the biology step." if card.promoted
           else "Not promoted; treat as an artifact until it survives more tests.")
    )


# ---------------------------------------------------------------------------
# Naive effect — point the reused head at the claim's target.
# ---------------------------------------------------------------------------

def _naive_effect(df: pd.DataFrame, claim: Claim) -> dict:
    """Cross-validated, subject-disjoint probe AUC for the claim's target."""
    from neuroad import probe
    target = claim.target if claim.target in contract.LABEL_TARGETS else "conversion"
    X, y, groups = probe.point_head(df, target)
    auc = probe.cross_val_auc(X, y, groups=groups)
    return {
        "metric": "AUC",
        "value": round(float(auc), 3),
        "target": target,
        "n": int(len(y)),
        "head": claim.head,
        "substrate": claim.substrate,
    }


# ---------------------------------------------------------------------------
# The referee.
# ---------------------------------------------------------------------------

def run_referee(df: pd.DataFrame, claim: Union[Claim, str]) -> ClaimCard:
    """Run the full referee loop and return the exported ClaimCard.

    `claim` may be a structured `contract.Claim` or a raw NL string (which is
    parsed via the Claude claim-parser, with a deterministic fallback).
    """
    from neuroad import gauntlet, scoring

    contract.validate_table(df)

    if isinstance(claim, str):
        claim = _parse_claim(claim, df)

    # 1. Naive effect (before any challenge).
    naive_effect = _naive_effect(df, claim)

    # 2. The adversarial gauntlet.
    tests = gauntlet.run_gauntlet(df, claim)

    # 3. First-pass card to learn the verdict / promotion decision.
    card = scoring.build_claim_card(claim, naive_effect, tests)

    # 4. Survivors only -> Claude adjudication + biology bridge.
    adjudication = None
    biology = None
    if card.promoted:
        adjudication = _adjudicate(claim, tests)
        biology = _propose_biology(card, df)

    # 5. Reviewer argues against the verdict (always runs).
    reviewer_out = _review(card)

    # 6. Rebuild the card so scoring folds in biology + reviewer caveats.
    card = scoring.build_claim_card(
        claim, naive_effect, tests, biology=biology, reviewer=reviewer_out,
    )

    # 7. Attach narration + adjudication as read-only side artifacts for the UI.
    #    (ClaimCard has no dedicated slots; set as dynamic attributes so the
    #    exporter/UI can pick them up without changing the frozen contract.)
    try:
        card.narration = _narrate(card)
    except Exception:
        card.narration = _fallback_narration(card)
    if adjudication is not None:
        card.adjudication = adjudication
    # Expose the reviewer critique + biology dicts for the UI/exporter (the
    # frozen ClaimCard has no dedicated slots; these are read-only side artifacts).
    if reviewer_out is not None:
        card.reviewer = reviewer_out
    if biology is not None:
        card.biology = biology
    # Raw test evidence, so downstream (UI/exporter) can adjudicate or re-render
    # any case — including refused ones — without re-running the gauntlet.
    card.tests_evidence = tests

    return card
