"""
courtroom — Claude as ADJUDICATOR: Prosecution, Defense, Judge.

Three personas each make a consequential call on the same gauntlet evidence:

  - Prosecution argues the signal is an ARTIFACT — cites which gauntlet tests
    failed/weakened and the subject-disjoint leakage margin.
  - Defense argues it is REAL biology — cites the survivors and the plasma
    biomarker anchor.
  - Judge renders reasoning consistent with the *computed* robustness verdict
    (it does not get to overrule the arithmetic — it explains it).

adjudicate(claim, tests) -> {'prosecution', 'defense', 'judge_reasoning'}

Offline, the three arguments are synthesised deterministically from the
TestEvidence stats so the courtroom always returns all three parts.
"""
from __future__ import annotations

from typing import Optional

from ..contract import (
    Claim,
    TestEvidence,
    TestResult,
    GAUNTLET_BY_KEY,
    robustness_score,
    verdict_for,
)
from ..calibration import PRIOR_ART
from ..scoring import apply_honesty_caps
from . import _client

SYSTEM = (
    "Persona: COURTROOM. Run three voices over one gauntlet of adversarial "
    "tests. PROSECUTION argues the imaging signal is an artifact (scanner/site "
    "leakage, demographics, or generic aging), citing the specific tests that "
    "failed or weakened and the subject-disjoint leakage margin (outcome AUC "
    "minus scanner AUC). DEFENSE argues it is real disease biology, citing the "
    "tests it survived and any plasma-biomarker anchor. JUDGE renders reasoning "
    "that is consistent with the already-computed robustness verdict — the "
    "score is fixed arithmetic; the judge explains, it does not overrule."
)

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "prosecution": {"type": "string"},
        "defense": {"type": "string"},
        "judge_reasoning": {"type": "string"},
    },
    "required": ["prosecution", "defense", "judge_reasoning"],
}


def adjudicate(claim: Claim, tests: list[TestEvidence]) -> dict:
    """Return prosecution / defense / judge arguments for a claim + gauntlet."""
    if _client.USING_LIVE_API:
        try:
            data = _client.complete(SYSTEM, _prompt(claim, tests), schema=_SCHEMA)
            if all(data.get(k) for k in ("prosecution", "defense", "judge_reasoning")):
                return {
                    "prosecution": data["prosecution"],
                    "defense": data["defense"],
                    "judge_reasoning": data["judge_reasoning"],
                }
        except Exception:
            pass
    return _fallback(claim, tests)


# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------


def _by_key(tests: list[TestEvidence], key: str) -> Optional[TestEvidence]:
    for t in tests:
        if t.key == key:
            return t
    return None


def _label(key: str) -> str:
    dim = GAUNTLET_BY_KEY.get(key)
    return dim.label if dim else key


def _leakage_margin(tests: list[TestEvidence]) -> Optional[float]:
    te = _by_key(tests, "site_scanner")
    if te is None:
        return None
    s = te.stats or {}
    if "margin" in s:
        return float(s["margin"])
    if "outcome_auc" in s and "scanner_auc" in s:
        return float(s["outcome_auc"]) - float(s["scanner_auc"])
    return None


def _weak(tests: list[TestEvidence]) -> list[TestEvidence]:
    bad = {TestResult.FAILED, TestResult.WEAKENED}
    return [t for t in tests if t.result in bad]


def _strong(tests: list[TestEvidence]) -> list[TestEvidence]:
    return [t for t in tests if t.result == TestResult.PASSED]


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------


def _fallback(claim: Claim, tests: list[TestEvidence]) -> dict:
    results = {t.key: t.result for t in tests}
    score = robustness_score(results)
    score, verdict = apply_honesty_caps(results, score, verdict_for(score))
    margin = _leakage_margin(tests)
    weak = _weak(tests)
    strong = _strong(tests)
    anchor = _by_key(tests, "biomarker_anchor")
    prior = PRIOR_ART[0]

    # --- Prosecution ------------------------------------------------------
    p: list[str] = [
        f"The claim that {claim.substrate} separate {claim.group_a} from "
        f"{claim.group_b} does not survive scrutiny."
    ]
    if margin is not None:
        if margin <= 0.10:
            p.append(
                f"The subject-disjoint leakage margin is only {margin:+.2f} AUC — "
                "the same head predicts the scanner nearly as well as the "
                "outcome, exactly the batch-effect mechanic reported in "
                f"'{prior[0]}' ({prior[1]})."
            )
        else:
            p.append(
                f"Even granting a {margin:+.2f} AUC leakage margin, that headroom "
                "is thin enough that acquisition confounds could account for much "
                "of the apparent effect."
            )
    if weak:
        names = ", ".join(f"{_label(t.key)} ({t.result.value})" for t in weak)
        p.append(f"It failed or weakened under: {names}.")
    else:
        p.append(
            "No single dimension cleanly collapses, but absence of a knock-out "
            "is not the same as demonstrated biology."
        )
    p.append(
        "The parsimonious reading is an acquisition/aging artifact until an "
        "orthogonal molecular anchor says otherwise."
    )

    # --- Defense ----------------------------------------------------------
    d: list[str] = [
        f"The signal separating {claim.group_a} from {claim.group_b} is "
        "consistent with real disease biology."
    ]
    if strong:
        names = ", ".join(_label(t.key) for t in strong)
        d.append(f"It withstood: {names}.")
    if margin is not None and margin > 0.10:
        d.append(
            f"The outcome exceeds the scanner-prediction floor by {margin:+.2f} "
            "AUC on subject-disjoint splits — the effect is not merely which "
            "machine acquired the scan."
        )
    if anchor is not None and anchor.result in (TestResult.PASSED, TestResult.WEAKENED):
        r = (anchor.stats or {}).get("ptau217_r") or (anchor.stats or {}).get("r")
        anchor_txt = "a plasma p-tau217 / GFAP correlation on the complete subset"
        if r is not None:
            anchor_txt += f" (r={float(r):.2f})"
        d.append(f"Critically, it is anchored to {anchor_txt} — molecular pathology, not pixels.")
    else:
        d.append(
            "The molecular anchor is incomplete, so the defense rests on the "
            "imaging survivors alone and asks only for follow-up, not acceptance."
        )

    # --- Judge ------------------------------------------------------------
    j = _judge(claim, tests, score, verdict, margin, anchor)

    return {
        "prosecution": " ".join(p),
        "defense": " ".join(d),
        "judge_reasoning": j,
    }


def _judge(claim, tests, score, verdict, margin, anchor) -> str:
    parts = [
        f"Robustness score {score}/100 places this finding at "
        f"'{verdict.value}'.",
    ]
    na = [t for t in tests if t.result == TestResult.NA]
    if na:
        parts.append(
            f"{len(na)} dimension(s) could not be run and were excluded from the "
            "denominator — the verdict reflects only the evidence actually gathered."
        )
    if margin is not None:
        if margin <= 0.10:
            parts.append(
                f"The {margin:+.2f} AUC leakage margin is the deciding fact: the "
                "prosecution's artifact reading is the more defensible one."
            )
        else:
            parts.append(
                f"The {margin:+.2f} AUC leakage margin clears the acquisition "
                "confound, which is why the defense retains standing."
            )
    if verdict.value in ("robust enough for follow-up", "strong candidate"):
        gate = "meets" if anchor and anchor.result != TestResult.NA else "provisionally meets"
        parts.append(
            f"The finding {gate} the promotion floor and may proceed to the "
            "biomarker-gated biology step — as a hypothesis to test, not a "
            "conclusion to bank."
        )
    else:
        parts.append(
            "The finding sits below the promotion floor; it is not carried "
            "forward to a mechanism, and would need a stronger leakage margin or "
            "molecular anchor to be worth a scientist's quarter."
        )
    return " ".join(parts)


def _prompt(claim: Claim, tests: list[TestEvidence]) -> str:
    results = {t.key: t.result for t in tests}
    score = robustness_score(results)
    score, verdict = apply_honesty_caps(results, score, verdict_for(score))
    margin = _leakage_margin(tests)
    lines = [
        f"Claim: {claim.claim_text}",
        f"Populations: {claim.group_a} vs {claim.group_b}; substrate: {claim.substrate}.",
        f"Computed robustness score: {score}/100 -> verdict '{verdict.value}'.",
    ]
    if margin is not None:
        lines.append(f"Subject-disjoint leakage margin (outcome AUC - scanner AUC): {margin:+.2f}.")
    lines.append("Gauntlet results:")
    for t in tests:
        lines.append(f"  - {_label(t.key)}: {t.result.value}. {t.detail} stats={t.stats}")
    lines.append(
        "Write the prosecution, the defense, and a judge reasoning consistent "
        "with the computed verdict above."
    )
    return "\n".join(lines)
