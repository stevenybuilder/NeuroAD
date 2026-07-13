"""
Scoring — assemble the exported `ClaimCard` from the gauntlet evidence.

The score, verdict and promotion decision come straight from the contract
(`robustness_score`, `verdict_for`, `is_promoted`). On top of that we enforce
the product's HARD GATE: a claim cannot be *promoted* to the biology step
without a passing (or at least weakened) plasma-biomarker anchor, no matter how
high the weighted score climbs. A referee that lets an unanchored signal through
is not a referee.
"""
from __future__ import annotations

from typing import Optional

from . import contract
from .contract import (Claim, ClaimCard, TestEvidence, TestResult, Verdict,
                       is_promoted, robustness_score, verdict_for)
from .calibration import PTAU217_MISSINGNESS

# stat key -> (metric label) used to surface one headline number per test.
_LEDGER_METRIC = {
    "age_sex": ("retained", "effect retained (age/sex)"),
    "site_scanner": ("margin", "leakage margin (outcome-scanner AUC)"),
    "brain_age": ("retained", "effect retained (brain-age)"),
    "biomarker_anchor": ("ptau217_r", "p-tau217 correlation r"),
    "replication": ("test_auc", "held-out cohort AUC"),
}


def _ledger_row(t: TestEvidence) -> dict:
    key, (stat_key, label) = t.key, _LEDGER_METRIC.get(t.key, ("value", t.key))
    value = t.stats.get(stat_key)
    if value is None and t.key == "biomarker_anchor":
        value = t.stats.get("gfap_r")
    n_used = t.stats.get("n") or t.stats.get("n_test") or t.stats.get("ptau217_n") \
        or t.stats.get("n_healthy")
    n_missing = None
    row_extra: dict = {}
    if t.key == "biomarker_anchor":
        used = t.stats.get("ptau217_n") or 0
        # completeness gap is informational; total isn't carried here
        n_used = used
        # Report r with its Fisher-z CI lower bound + provenance, so a downloaded
        # ledger never presents a calibrated correlation as a measurement.
        row_extra = {
            "ci_lo": t.stats.get("ptau217_ci_lo") if t.stats.get("ptau217_r") is not None
            else t.stats.get("gfap_ci_lo"),
            "synthetic": bool(t.stats.get("synthetic")),
            "source": ("synthetic (calibration target)" if t.stats.get("synthetic")
                       else ("measured" if t.stats.get("ptau217_r") is not None
                             or t.stats.get("gfap_r") is not None else "unavailable")),
        }
    return {
        "test": key,
        "metric": label,
        "value": value,
        "result": t.result.value,
        "n_used": n_used,
        "n_missing": n_missing,
        "detail": t.detail,
        **row_extra,
    }


def apply_honesty_caps(results: dict, score: int, verdict: Verdict) -> tuple[int, Verdict]:
    """Two honesty caps shared by the card scorer and the courtroom judge so a
    single finding never gets two different scores/verdicts (the judge_reasoning
    string must agree with the meter the viewer sees).

    CAP 1 (generalized) — the top "strong candidate" band requires a COMPLETE
    gauntlet. ANY NA test (the molecular anchor when a cohort ships no plasma, an
    uninformative brain-age control, an unrun scanner star) is dropped from the
    denominator, so an otherwise clean card can renormalize up to 100 on the tests
    that DID run and read "strong candidate". A signal with an unrun test is not a
    strong candidate: if any test is NA, cap just below the STRONG band (85) so the
    verdict lands at "robust enough for follow-up". A genuine 5/5 (no NA) is
    untouched and earns up to 100. Only bites when the score would reach the top band.

    CAP 2 — a FAILED scanner/site star test is a likely-artifact signal.
    Replication cannot rescue it (a batch artifact "replicates" in every cohort
    that shares the confound), so cap the verdict language to "fragile".
    """
    # CAP 1 (generalized) - the top "strong candidate" band requires a COMPLETE
    # gauntlet. Every NA test is dropped from the denominator, so a card missing
    # an artifact control (brain-age / scanner star) OR the molecular anchor can
    # renormalize up to 100 on the tests that DID run. If ANY test is NA, cap just
    # below STRONG (85 -> 84) so no unrun test can launder a signal into the
    # promote band. A genuine 5/5 (no NA) is untouched and earns up to 100.
    if any(r == TestResult.NA for r in results.values()) and score >= 85:
        score = 84
        verdict = verdict_for(score)
    if results.get("site_scanner") == TestResult.FAILED and verdict != Verdict.FRAGILE:
        score = min(score, 39)
        verdict = verdict_for(score)
    return score, verdict


def build_claim_card(claim: Claim, naive_effect: dict, tests: list[TestEvidence],
                     biology: Optional[dict] = None,
                     reviewer: Optional[dict] = None) -> ClaimCard:
    """Assemble a `ClaimCard` from claim + naive effect + gauntlet evidence.

    * score/verdict via the contract's weighted, NA-renormalized scheme.
    * promotion via `is_promoted`, additionally gated on the biomarker anchor.
    * biology (dict) and reviewer (dict) attached when supplied.
    * evidence_ledger: one row per test (test, metric, value, n_used, n_missing).
    """
    results = {t.key: t.result for t in tests}
    score = robustness_score(results)
    verdict = verdict_for(score)
    promoted = is_promoted(verdict)

    # HARD GATE: a promoted finding must be CORROBORATED by an independent line
    # of evidence, not just score high. Two accepted corroboration paths:
    #   (1) MOLECULAR (strongest): a passing plasma p-tau217/GFAP anchor. Requires
    #       gated cohorts (no open dataset pairs MRI with plasma markers).
    #   (2) REPLICATION (open-data): when no molecular marker exists (anchor NA),
    #       a PASSED held-out cross-cohort replication corroborates the finding on
    #       real, open data. This is weaker than a molecular anchor but is genuine
    #       independent evidence — not synthetic, not gated. CRUCIAL GUARD: it only
    #       counts if the finding also PASSED the scanner/site leakage test — a
    #       scanner artifact "replicates" too (the confound is in both cohorts), so
    #       replication corroborates only a signal that is not a batch artifact.
    # A FAILED molecular anchor is a real refutation and always blocks promotion.
    anchor = results.get("biomarker_anchor", TestResult.NA)
    replication = results.get("replication", TestResult.NA)
    leakage_ok = results.get("site_scanner") == TestResult.PASSED

    # Honesty caps (shared with the courtroom judge so the reasoning agrees with
    # the meter): an unrun molecular anchor can't renormalize to the top band, and
    # a scanner-failed star test caps the verdict to "fragile". See apply_honesty_caps.
    score, verdict = apply_honesty_caps(results, score, verdict)
    promoted = is_promoted(verdict)
    # Molecular corroboration = a usable plasma anchor (PASSED or WEAKENED, i.e.
    # anything that isn't a hard FAILED or an unavailable NA) — matches the
    # original hard gate.
    molecular_ok = anchor not in (TestResult.FAILED, TestResult.NA)
    replication_ok = (anchor == TestResult.NA
                      and replication == TestResult.PASSED
                      and leakage_ok)
    corroboration = ("molecular" if molecular_ok
                     else "replication" if replication_ok else None)
    if not (molecular_ok or replication_ok):
        promoted = False

    caveats: list[str] = []
    if anchor == TestResult.FAILED:
        caveats.append(
            "Biomarker anchor failed: no molecular correlate found on the "
            "complete subset — treat as unanchored (blocks promotion).")
    elif corroboration == "replication":
        caveats.append(
            "Corroborated by REAL cross-cohort replication (held-out cohort), not "
            "a molecular marker: no open dataset pairs MRI with plasma p-tau217/"
            "GFAP. Replication is genuine independent evidence but weaker than a "
            "molecular anchor — a plasma-anchored confirmation needs gated data.")
    elif anchor == TestResult.NA and not replication_ok:
        caveats.append(
            "Not corroborated: no molecular anchor (no open plasma data) and no "
            "passing cross-cohort replication — cannot be promoted.")
    n_na = sum(1 for r in results.values() if r == TestResult.NA)
    if n_na:
        caveats.append(
            f"Completeness: {n_na}/5 gauntlet tests could not run; score is "
            "renormalized over the tests that did.")
    anchor_stats = next((t.stats for t in tests if t.key == "biomarker_anchor"), {})
    if anchor_stats.get("synthetic"):
        caveats.append(
            "SYNTHETIC HARNESS: the p-tau217 / GFAP anchor is a CALIBRATION TARGET "
            "(drawn to sit inside a literature range in calibration.CAL), not a "
            "measured plasma value — no open cohort pairs MRI with plasma markers. "
            "The molecular anchor demonstrates the gate MECHANIC, not a real result.")
    if anchor_stats.get("ptau217_r") is not None:
        caveats.append(
            f"p-tau217 measured on a partial subset (~{PTAU217_MISSINGNESS:.0%} "
            "missing in a realistic cohort) — anchor is suggestive, not confirmatory.")
    if verdict == Verdict.PARTIALLY_ROBUST:
        caveats.append("'Partially robust' is not 'robust' — survives some but "
                       "not all challenges.")

    card = ClaimCard(
        claim=claim,
        naive_effect=naive_effect,
        tests=tests,
        score=score,
        verdict=verdict,
        promoted=promoted,
        evidence_ledger=[_ledger_row(t) for t in tests],
        caveats=caveats,
    )

    if biology:
        card.biology_hypothesis = biology.get("hypothesis", "")
        card.next_experiment = list(biology.get("next_experiment", []))
        card.falsification = list(biology.get("falsification", []))
    if reviewer:
        card.caveats.extend(reviewer.get("revised_caveats", []))
        card.caveats.extend(reviewer.get("critique", []))

    return card
