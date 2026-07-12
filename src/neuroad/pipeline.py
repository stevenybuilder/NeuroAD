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

import logging
from typing import Optional, Union

import pandas as pd

from neuroad import contract
from neuroad.contract import Claim, ClaimCard, TestEvidence, TestResult

#: Debug-level logger. The Claude shims below degrade gracefully to keep the
#: offline demo alive, but we no longer swallow the reason silently — a
#: misconfigured live key is diagnosable with ``logging.getLogger('neuroad')``
#: set to DEBUG, without ever breaking the guaranteed-offline path.
_log = logging.getLogger("neuroad.pipeline")


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
    except Exception as exc:
        _log.debug("claim_parser unavailable/failed, using fallback claim: %r", exc)
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
    except Exception as exc:
        _log.debug("courtroom.adjudicate failed, skipping adjudication: %r", exc)
    return None


def _propose_biology(card: ClaimCard, df: pd.DataFrame) -> Optional[dict]:
    try:
        from neuroad.claude import bridge
        result = bridge.propose_biology(card, df)
        if isinstance(result, dict):
            return result
    except Exception as exc:
        _log.debug("bridge.propose_biology failed, skipping biology: %r", exc)
    return None


def _translate(card: ClaimCard, df: pd.DataFrame) -> Optional[dict]:
    """Promoted survivors ONLY -> chain PI4AD/AlphaFold/repurposing off the
    Bridge's mechanism routing. Offline-first and exception-safe: a failure here
    NEVER affects the score/verdict — translation is a read-only side artifact."""
    try:
        from neuroad.claude import bridge
        from neuroad.harness import translation
        mechanism = bridge._route(df)
        result = translation.translate(mechanism, df)
        if isinstance(result, dict):
            return result
    except Exception as exc:
        _log.debug("translation.translate failed, skipping translation: %r", exc)
    return None


def _review(card: ClaimCard) -> Optional[dict]:
    try:
        from neuroad.claude import reviewer
        result = reviewer.review(card)
        if isinstance(result, dict):
            return result
    except Exception as exc:
        _log.debug("reviewer.review failed, skipping reviewer critique: %r", exc)
    return None


def _narrate(card: ClaimCard) -> str:
    try:
        from neuroad.claude import narrator
        text = narrator.narrate(card)
        if isinstance(text, str) and text.strip():
            return text
    except Exception as exc:
        _log.debug("narrator.narrate failed, using fallback narration: %r", exc)
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
    """Cross-validated, subject-disjoint probe AUC for the claim's target.

    The headline card AUC uses the REPEATED-CV ensemble (``N_REPEATS_ENSEMBLE``
    split seeds, out-of-fold scores averaged) so the number the researcher reads
    is the de-noised, reproducible estimate rather than one that rides on the
    luck of a single fold shuffle at small n. The gauntlet's internal
    before/after residualization comparisons deliberately stay single-split:
    they are paired ratios where the split-seed cancels, and they anchor the
    calibrated retained-fraction thresholds.
    """
    import numpy as np
    from neuroad import probe
    target = claim.target if claim.target in contract.LABEL_TARGETS else "conversion"
    X, y, groups = probe.point_head(df, target)
    auc = probe.cross_val_auc(X, y, groups=groups,
                              n_repeats=probe.N_REPEATS_ENSEMBLE)

    # Age/sex-residualized primary AUC (fold-honest: the nuisance regression is
    # fit inside each fold on train rows only). Reported ALONGSIDE the naive AUC
    # so a headline driven by demographic confounding is visible at a glance;
    # None when there is no age/sex variation to adjust for.
    auc_adjusted = None
    if target in ("conversion", "dx_binary"):
        keep = (pd.to_numeric(df["conversion"], errors="coerce").notna().to_numpy()
                if target == "conversion"
                else df["dx"].astype("string").map({"AD": 1, "CN": 0}).notna().to_numpy())
        sub = df.loc[keep]
        cov_cols = []
        age = sub["age"].to_numpy(float)
        if np.isfinite(age).sum() >= 3 and np.nanstd(age) > 0:
            cov_cols.append(np.nan_to_num(age, nan=np.nanmean(age)))
        if sub["sex"].nunique(dropna=True) > 1:
            cov_cols.append((sub["sex"].astype("string") == "F").to_numpy(float))
        if cov_cols:
            C = np.column_stack(cov_cols)
            auc_adjusted = round(float(probe.residualized_cross_val_auc(
                X, C, y, groups, kind="covariate")), 3)

    return {
        "metric": "AUC",
        "value": round(float(auc), 3),
        "value_adjusted": auc_adjusted,
        "adjusted_note": "age/sex-residualized (fold-honest)",
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
    translation = None
    if card.promoted:
        adjudication = _adjudicate(claim, tests)
        biology = _propose_biology(card, df)
        translation = _translate(card, df)

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
    # Translation lead (molecule/wet-lab follow-up) — promoted survivors only,
    # read-only side artifact; never affects score/verdict.
    if translation is not None:
        card.translation = translation
    # Raw test evidence, so downstream (UI/exporter) can adjudicate or re-render
    # any case — including refused ones — without re-running the gauntlet.
    card.tests_evidence = tests

    # Live-vs-offline Claude badge: a truthful descriptor of whether the reasoning
    # text came from the live API or the deterministic offline template. Printed by
    # the CLI and written into every report so the "Claude as adjudicator"
    # differentiator is verifiable rather than assumed.
    try:
        from neuroad.claude import _client
        card.claude = _client.model_badge()
    except Exception as exc:  # noqa: BLE001
        _log.debug("claude model_badge unavailable: %r", exc)
        card.claude = {"live": False, "model": "offline-template",
                       "path": "deterministic offline template"}

    # 8. STAR trust features — double dissociation + confound leaderboard.
    #    Pure pandas/numpy (leakage.py), no API; guarded so the deterministic
    #    offline referee path never breaks. These populate card.to_dict() and the
    #    written reports, so all four headline trust features ship (not demo-only).
    _attach_leakage_features(card, df, claim)

    return card


def seed_sweep(preset: str = "KILL", n_seeds: int = 20,
               n_boot: int = 200, n_perm: int = 200) -> dict:
    """Verdict-stability sweep across ``n_seeds`` seeds of a synthetic cohort.

    Re-runs the gauntlet + scoring on freshly generated synthetic cohorts (one per
    seed) and reports the score distribution and the verdict-flip rate — turning
    the single-seed determinism from a hidden risk into a demonstrated robustness
    claim ('verdict stable across N seeds'). Uses reduced bootstrap/permutation
    counts for speed (the star decision is stable well below the headline 1000).

    Returns a JSON-safe dict:
        {preset, n_seeds, scores, verdicts, promoted, mean, std, min, max,
         modal_verdict, flip_rate, promotion_rate, promotion_line}
    """
    import numpy as np

    from neuroad import gauntlet, scoring
    from neuroad.contract import Claim
    from neuroad.data import synthetic

    target = "conversion"
    scores: list[int] = []
    verdicts: list[str] = []
    promoted: list[bool] = []
    claim = Claim(claim_id=f"sweep-{preset}", claim_text="seed-stability sweep",
                  target=target, group_a="MCI converters", group_b="MCI non-converters")
    for s in range(n_seeds):
        df = synthetic.generate_cohort(preset, seed=s)
        tests = gauntlet.run_gauntlet(df, claim, n_boot=n_boot, n_perm=n_perm)
        card = scoring.build_claim_card(claim, {"metric": "AUC"}, tests)
        scores.append(int(card.score))
        verdicts.append(card.verdict.value)
        promoted.append(bool(card.promoted))

    arr = np.asarray(scores, dtype=float)
    modal = max(set(verdicts), key=verdicts.count) if verdicts else None
    flip_rate = round(1.0 - (verdicts.count(modal) / len(verdicts)), 3) if verdicts else 0.0
    promo_rate = round(sum(promoted) / len(promoted), 3) if promoted else 0.0
    stable = flip_rate == 0.0
    promotion_line = (
        f"Verdict stable across {n_seeds} seeds — every seed lands on "
        f"'{modal}' (score {int(arr.min())}-{int(arr.max())})."
        if stable else
        f"Verdict flips on {int(round(flip_rate * n_seeds))}/{n_seeds} seeds "
        f"(modal '{modal}', {int((1 - flip_rate) * 100)}% agreement).")
    return {
        "preset": preset,
        "n_seeds": int(n_seeds),
        "scores": scores,
        "verdicts": verdicts,
        "promoted": promoted,
        "mean": round(float(arr.mean()), 2) if arr.size else None,
        "std": round(float(arr.std()), 2) if arr.size else None,
        "min": int(arr.min()) if arr.size else None,
        "max": int(arr.max()) if arr.size else None,
        "modal_verdict": modal,
        "flip_rate": flip_rate,
        "promotion_rate": promo_rate,
        "promotion_line": promotion_line,
    }


def _attach_leakage_features(card: ClaimCard, df: pd.DataFrame,
                             claim: Claim) -> None:
    """Compute double_dissociation + confound_leaderboard for the claim's target
    and attach them to the card. Failure degrades to the empty defaults."""
    from neuroad import leakage
    target = claim.target if claim.target in contract.LABEL_TARGETS else "conversion"
    try:
        dd = leakage.double_dissociation(df, target)
        if isinstance(dd, dict):
            card.double_dissociation = dd
    except Exception as exc:
        _log.debug("leakage.double_dissociation failed: %r", exc)
    try:
        board = leakage.confound_leaderboard(df, target)
        if isinstance(board, list):
            card.confound_leaderboard = board
    except Exception as exc:
        _log.debug("leakage.confound_leaderboard failed: %r", exc)
