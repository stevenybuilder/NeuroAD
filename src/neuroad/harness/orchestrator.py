"""
orchestrator — the L5 entry point (Stage-2 blueprint §4).

``investigate("<hypothesis>", "<dataset>")`` is the one call a researcher drives
the whole instrument with. It turns a plain-language hunch into a defensible,
honesty-stamped ``ExperimentCard`` by chaining the harness pieces:

    1. PARSE   the free-text hypothesis into a structured ``contract.Claim``
               (claude.claim_parser, with its deterministic offline fallback),
               ENRICHED from the L3 policy layer with a novelty_class, a
               pre-registered expected_direction and a pre-registered
               kill_criterion (hypothesis_schema + novelty_rubric + biomarker_routing).
    2. ROUTE   the claim to a discovery MODE (harness.discovery_router):
               novel-pattern -> the unsupervised Detective; named-contrast ->
               the supervised probe/gauntlet.
    3. REFEREE run the engine for that mode (discovery.discover_and_referee for
               novel-pattern, pipeline.run_referee for named-contrast).
    4. GATE    apply the biomarker-anchor HARD GATE (policy atn_framework +
               biomarker_routing): a promoted finding whose molecular anchor is
               present-but-FAILED is blocked; the mechanism routing is recorded.
    5. CARD    wrap the refereed ClaimCard into an ``ExperimentCard``
               (harness.experiment_card), stamping the honesty_rung derived from
               the novelty_rubric ladder.

Finally the HONESTY GUARD runs before ANYTHING is returned: it asserts every
card carries a non-empty novelty_class and honesty_rung and refuses to emit a
card whose rendered text contains a forbidden overclaim.

OFFLINE / DETERMINISTIC: with ``api=False`` (the default) and no
ANTHROPIC_API_KEY the entire path uses deterministic fallbacks — no network. Every
policy read falls back to today's hardcoded constants if ``policy/`` is missing or
malformed, so this module never depends on the L3 docs being present.
"""
from __future__ import annotations

import collections
import copy
import json
import logging
import os
import re
import threading
from typing import Optional, Union

import pandas as pd

from ..contract import (
    Claim,
    ClaimCard,
    TestEvidence,
    TestResult,
    Verdict,
    is_promoted,
    verdict_for,
)
from . import discovery_router, experiment_card, policy
from .experiment_card import ExperimentCard, HONESTY_LADDER

_log = logging.getLogger("neuroad.harness.orchestrator")


# ===========================================================================
# Stage-level base-card memo — the live-compute fallback for genuine cache
# MISSES. The referee's expensive work (naive_effect ~11-17s + gauntlet
# ~15-19s + leakage) is ANCHOR-INVARIANT: the chosen fluid biomarker only
# routes the read-only translation/atn side artifacts, never a probe input,
# a gauntlet test, the score, the verdict, or the promotion decision. So one
# base card computed with anchor=None is the genuine engine result for EVERY
# anchor at the same coordinate (dataset, target, region, seed, api); a
# different-anchor request deep-copies it and re-applies ONLY the ~0.5s
# translation. This is exact reuse, not an approximation — the frozen-seam
# honesty pattern, one rung below the disk grid: a disk MISS that shares its
# (dataset,target,region) with any already-computed anchor now costs ~0.5s
# instead of ~30s. N_BOOT is NOT the lever here (gauntlet time is flat in
# N_BOOT); reusing the whole anchor-invariant stage is.
# ===========================================================================

#: LRU cap on distinct (dataset,target,region,seed,api) base cards held in RAM.
_BASE_MEMO_CAP = int(os.environ.get("NEUROAD_BASE_MEMO_CAP", "256"))
_base_memo: "collections.OrderedDict[tuple, ClaimCard]" = collections.OrderedDict()
_base_memo_lock = threading.Lock()


def _base_memo_key(dataset: str, target: str, region: str, seed: int,
                   api: bool) -> tuple:
    return (dataset, target or "", region or "", int(seed), bool(api))


def _base_memo_get(key: tuple) -> Optional[ClaimCard]:
    with _base_memo_lock:
        card = _base_memo.get(key)
        if card is not None:
            _base_memo.move_to_end(key)          # LRU touch
            return copy.deepcopy(card)           # caller mutates (gate); keep ours pristine
    return None


def _base_memo_put(key: tuple, card: ClaimCard) -> None:
    with _base_memo_lock:
        _base_memo[key] = copy.deepcopy(card)    # store a pristine snapshot
        _base_memo.move_to_end(key)
        while len(_base_memo) > _BASE_MEMO_CAP:
            _base_memo.popitem(last=False)       # evict least-recently-used


def _region_slug_for_key(claim: Claim, df: pd.DataFrame) -> str:
    """The region slug the referee would resolve (deterministic, ~0ms; a "" for
    non-ROI cohorts). Used only as a memo-key axis, so a mismatch can only ever
    cause a recompute, never a wrong number."""
    if getattr(claim, "region", None):
        return claim.region
    try:
        from .region import extract_region
        slug, _cols = extract_region(claim.claim_text or "", df)
        return slug or ""
    except Exception:  # noqa: BLE001
        return ""


# ===========================================================================
# HONESTY GUARD — the anti-overclaim contract every card must satisfy.
# ===========================================================================

#: Multi-word overclaim PHRASES the engine must never emit (case-insensitive
#: substring). The pitch is calibrated honesty: the card says how far a finding
#: has been DEFENDED, never that it is true, clinically actionable, or superior
#: to an established assay.
FORBIDDEN_OVERCLAIMS = (
    "proven biomarker",
    "validated biomarker",
    "confirmed biomarker",
    "established biomarker",
    "clinically validated",
    "clinically proven",
    "clinically actionable",
    "ready for clinical use",
    "detects preclinical",
    "better than plasma",
    "replaces pet",
    "replace pet",
)

#: Single overclaim WORDS matched on a word boundary so legitimate substrings
#: never trip. IMPORTANT: bare "discovery"/"discovered" are deliberately NOT here
#: — this is the "Discovery Engine" and the Detective legitimately discovers
#: clusters; only unambiguous claim-of-truth words belong in this list.
FORBIDDEN_WORDS = (
    "cure",
    "breakthrough",
    "definitive",
    "definitively",
)


class HonestyViolation(AssertionError):
    """Raised when an ExperimentCard breaks the anti-overclaim contract."""


def _rendered_text(xcard: ExperimentCard) -> str:
    """Concatenate everything a reader could see off this card into one string.

    Covers the serialized card (claim, caveats, biology, next experiment,
    falsification, provenance) plus the read-only Claude side-artifacts the
    referee attaches as dynamic attributes (narration / biology / reviewer /
    adjudication)."""
    parts = [json.dumps(xcard.to_dict(), default=str)]
    card = xcard.card
    for attr in ("narration", "biology", "reviewer", "adjudication"):
        v = getattr(card, attr, None)
        if not v:
            continue
        parts.append(v if isinstance(v, str) else json.dumps(v, default=str))
    return "\n".join(parts)


def honesty_guard(xcard: ExperimentCard) -> ExperimentCard:
    """Assert the anti-overclaim contract and return the card, or raise.

    Invariants:
      * ``novelty_class`` is present and non-empty,
      * ``honesty_rung`` is present and non-empty,
      * the rendered card contains NONE of ``FORBIDDEN_OVERCLAIMS`` (phrases)
        or ``FORBIDDEN_WORDS`` (word-boundary matched).

    ``investigate`` always runs this before returning; call it directly to vet a
    card built by hand."""
    if not (xcard.novelty_class and str(xcard.novelty_class).strip()):
        raise HonestyViolation("ExperimentCard is missing a novelty_class")
    if not (xcard.honesty_rung and str(xcard.honesty_rung).strip()):
        raise HonestyViolation("ExperimentCard is missing an honesty_rung")

    text = _rendered_text(xcard).lower()
    hits: list[str] = []
    for phrase in FORBIDDEN_OVERCLAIMS:
        if phrase in text:
            hits.append(phrase)
    for word in FORBIDDEN_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", text):
            hits.append(word)
    if hits:
        raise HonestyViolation(
            "forbidden overclaim(s) in rendered ExperimentCard: "
            + ", ".join(sorted(set(hits))))
    return xcard


# ===========================================================================
# Step 1 — parse + enrich the hypothesis.
# ===========================================================================

#: Keyword hints for the novelty class (fallback for policy/novelty_rubric.md's
#: closed vocabulary). "known" = published prior art we re-measure (e.g. that
#: frozen embeddings leak scanner/site); "novel" = asks the data for structure
#: with no clean precedent; otherwise "adjacent" (a known mechanism extended).
_KNOWN_HINTS = ("scanner", "site", "leak", "batch", "field strength",
                "acquisition", "prior art")
_NOVEL_HINTS = ("novel", "hidden", "unknown", "undiscovered", "latent",
                "emergent", "phenotype", "subtype", "sub-type", "subgroup",
                "cluster", "stratif")


def _novelty_values() -> tuple:
    """Closed novelty vocabulary, from policy with a hardcoded fallback."""
    try:
        vals = policy.table("novelty_rubric").get("novelty_class", {}).get("values")
        if isinstance(vals, list) and vals:
            return tuple(str(v).strip().lower() for v in vals)
    except Exception as exc:  # pragma: no cover - fallback path
        _log.debug("novelty vocabulary policy read failed: %r", exc)
    return ("known", "adjacent", "novel")


def _classify_novelty(text: str) -> str:
    """Deterministic novelty_class for a hunch, constrained to the policy vocab."""
    allowed = _novelty_values()
    low = f" {(text or '').lower()} "
    guess = "adjacent"
    if any(h in low for h in _KNOWN_HINTS):
        guess = "known"
    elif any(h in low for h in _NOVEL_HINTS):
        guess = "novel"
    return guess if guess in allowed else (allowed[0] if allowed else "adjacent")


def _mechanism_enrichment(df: Optional[pd.DataFrame],
                          anchor: Optional[str] = None) -> tuple:
    """Pre-register (mechanism, expected_direction, kill_criterion).

    Routes to a mechanism with the same biomarker-dominance rule the Bridge uses,
    then reads its expected_direction + kill_criterion from policy/biomarker_routing
    (fallback: bridge._MECHANISMS). These are FIXED BEFORE the experiment runs —
    the pre-registered falsifier the honesty ladder rests on."""
    mech = "amyloid_cascade"
    try:
        from ..claude.bridge import _route
        mech = _route(df, anchor)
    except Exception as exc:  # pragma: no cover - fallback path
        _log.debug("bridge routing unavailable, defaulting mechanism: %r", exc)

    direction = kill = ""
    try:
        row = policy.table("biomarker_routing").get("mechanisms", {}).get(mech, {})
        direction = (row.get("expected_direction") or row.get("direction") or "")
        kill = (row.get("kill_criterion") or row.get("kill") or "")
    except Exception as exc:  # pragma: no cover - fallback path
        _log.debug("biomarker_routing policy read failed: %r", exc)

    if not (direction and kill):
        try:
            from ..claude.bridge import _MECHANISMS
            m = _MECHANISMS.get(mech, {})
            direction = direction or m.get("direction", "")
            kill = kill or m.get("kill", "")
        except Exception as exc:  # pragma: no cover - fallback path
            _log.debug("bridge._MECHANISMS unavailable: %r", exc)

    return mech, str(direction).strip(), str(kill).strip()


def _parse_claim(text: str, df: Optional[pd.DataFrame], *, api: bool) -> Claim:
    """NL hunch -> structured ``Claim`` with a deterministic offline fallback.

    ``api=True`` asks Claude for a strict structured parse (falling back on any
    failure); ``api=False`` (the default) goes straight to the deterministic
    keyword router so the path is guaranteed offline."""
    try:
        from ..claude import claim_parser
    except Exception as exc:
        _log.debug("claim_parser import failed, using default claim: %r", exc)
        return _default_claim(text)

    if api:
        try:
            claim = claim_parser.parse_claim(text, df)
            if isinstance(claim, Claim):
                return claim
        except Exception as exc:
            _log.debug("live claim parse failed, using fallback: %r", exc)
    try:
        claim = claim_parser._fallback(text, df)
        if isinstance(claim, Claim):
            return claim
    except Exception as exc:
        _log.debug("deterministic claim fallback failed: %r", exc)
    return _default_claim(text)


def _default_claim(text: str) -> Claim:
    return Claim(
        claim_id="claim-fallback",
        claim_text=text or "Structural embeddings predict conversion",
        target="conversion",
        group_a="MCI converters",
        group_b="MCI non-converters",
    )


# ===========================================================================
# Step 2b — refuse out-of-scope targets instead of silently coercing them.
# ===========================================================================

#: Vocabulary that names a target the head CANNOT point at (no such label column
#: and no regression head). The engine's supported targets are exactly
#: {conversion, dx_binary, site, scanner}. Anything below was, historically,
#: silently re-interpreted as `conversion` — a real honesty hazard.
_OUT_OF_SCOPE_TARGETS = (
    "tau-pet", "tau pet", "amyloid-pet", "amyloid pet", "suvr", "centiloid",
    "csf ", "cerebrospinal", "amyloid-positive", "amyloid positive",
    "amyloid-negative", "amyloid status", "amyloid positivity",
    "apoe", "genotype", "genome-wide", "gwas", "allele", "polygenic",
    "regression", "predict age", "brain-age prediction", "cognitive score",
    "trajectory", "rate of decline", "slope of", "continuous outcome",
)

#: If any of these in-scope anchors appear, the hypothesis is plausibly a
#: supported contrast (or a novel-pattern discovery) — do NOT refuse.
_IN_SCOPE_ANCHORS = (
    "conversion", "convert", "mci", "ad vs", "ad-vs", "cn", "diagnos", " dx",
    "progress to", "site", "scanner", "leakage", "subtype", "phenotype",
    "cluster", "stratif", "discover",
)


def _out_of_scope_reason(text: str) -> str:
    """Return a reason string if ``text`` names an unsupported target, else ""."""
    low = f" {(text or '').lower()} "
    if any(a in low for a in _IN_SCOPE_ANCHORS):
        return ""
    hit = next((t for t in _OUT_OF_SCOPE_TARGETS if t in low), "")
    if not hit:
        return ""
    return (
        f"names an unsupported target ('{hit.strip()}'): this engine's reused "
        "head can only be pointed at {conversion, dx_binary, site, scanner} and "
        "has no regression / molecular / genetic head. Not analyzed (rather than "
        "silently answering a different question)."
    )


def _unsupported_card(hypothesis: str, dataset: str, reason: str) -> ClaimCard:
    """A refused card: honest, non-promoted, no fabricated effect or tests."""
    claim = Claim(
        claim_id="claim-unsupported",
        claim_text=hypothesis or "",
        target="unsupported",
        substrate="(not analyzed — unsupported target)",
    )
    return ClaimCard(
        claim=claim,
        naive_effect={"metric": "n/a", "value": None,
                      "note": "not analyzed — unsupported target"},
        tests=[],
        score=0,
        verdict=Verdict.FRAGILE,
        promoted=False,
        caveats=[f"Unsupported hypothesis: {reason}"],
    )


def _run_supervised(df: pd.DataFrame, claim: Claim, use_claude: bool = True,
                    anchor: Optional[str] = None, *, dataset: str = "",
                    seed: int = 0) -> ClaimCard:
    """Named-contrast: point the reused head at the target via the full referee.

    Fast path: a base card for this coordinate (dataset, target, region, seed,
    api) — computed once with anchor=None — is memoized. Because the anchor only
    routes the read-only translation/atn artifacts (never a number, score, or
    verdict), a second anchor at the same coordinate deep-copies the base and
    re-applies ONLY the ~0.5s translation instead of re-running the ~30s referee.
    The key is region-aware, so a per-region hypothesis reuses its own base, not
    another region's. ``dataset=""`` (e.g. a direct unit-test call) disables the
    memo and always computes live — identical numbers, just no reuse."""
    from .. import pipeline

    if not dataset:
        return pipeline.run_referee(df, claim, use_claude=use_claude, anchor=anchor)

    region = _region_slug_for_key(claim, df)
    key = _base_memo_key(dataset, claim.target, region, seed, use_claude)
    base = _base_memo_get(key)
    if base is None:
        # Cold coordinate: compute the anchor-invariant base ONCE (anchor=None),
        # memoize a pristine snapshot, then work on the freshly-computed card
        # directly. (Don't re-get from the memo — a pathological cap<1 would evict
        # the just-put entry and hand back None, crashing the downstream gate.)
        base = pipeline.run_referee(df, claim, use_claude=use_claude, anchor=None)
        _base_memo_put(key, base)                # stores its own deepcopy; `base` stays ours to mutate

    # Re-apply the anchor-specific translation onto the reused base. A promoted
    # base already carries the anchor=None (cohort-dominance) translation; only
    # overwrite it when the researcher chose a specific anchor. Never touches the
    # score/verdict/tests — those are the base's genuine anchor-invariant result.
    if anchor and getattr(base, "translation", None) is not None:
        try:
            tx = pipeline._translate(base, df, anchor)
            if isinstance(tx, dict):
                base.translation = tx
        except Exception as exc:  # noqa: BLE001
            _log.debug("anchor translation re-apply failed, keeping base: %r", exc)

    # Display-only personalisation: the base card carries the FIRST requester's
    # phrasing at this coordinate. Carry the current hypothesis text onto it so
    # the rendered claim matches what the user typed. The computed target/region/
    # region_columns are the base's genuine values and are left untouched.
    try:
        if getattr(base, "claim", None) is not None and claim.claim_text:
            base.claim.claim_text = claim.claim_text
    except Exception:  # noqa: BLE001
        pass
    return base


def _card_from_cluster(claim: Claim, cluster: dict) -> ClaimCard:
    """Reconstruct a ClaimCard from a discover_and_referee cluster summary.

    The Detective already refereed each cluster with the SAME five-test gauntlet;
    we rebuild the ClaimCard (with its TestEvidence list intact) from that summary
    so the biomarker-anchor gate + honesty ladder read the same evidence a
    supervised card carries — no re-running the gauntlet on a tiny sub-cohort."""
    g = cluster.get("gauntlet", {}) or {}
    tests: list[TestEvidence] = []
    for key, val in (g.get("tests") or {}).items():
        try:
            tests.append(TestEvidence(key, TestResult(val)))
        except Exception:
            continue
    score = int(g.get("score", 0))
    verdict = verdict_for(score)
    promoted = bool(g.get("promoted", is_promoted(verdict)))
    return ClaimCard(
        claim=claim,
        naive_effect=dict(cluster.get("naive_effect") or {}),
        tests=tests,
        score=score,
        verdict=verdict,
        promoted=promoted,
    )


def _run_unsupervised(df: pd.DataFrame, claim: Claim,
                      decision) -> tuple[ClaimCard, dict]:
    """Novel-pattern: run the Detective, pick its most-defended cluster, and
    represent it as a ClaimCard. Provenance records the discovery context."""
    from .. import discovery
    disc = discovery.discover_and_referee(df, **(decision.detective_cfg or {}))
    clusters = disc.get("clusters", []) or []

    if clusters:
        best = max(clusters, key=lambda c: (
            bool(c.get("gauntlet", {}).get("promoted")),
            int(c.get("gauntlet", {}).get("score", 0)),
            int(c.get("n", 0)),
        ))
        card = _card_from_cluster(claim, best)
        provenance = {
            "mode": "unsupervised",
            "engine": decision.engine,
            "rationale": decision.rationale,
            "cluster": best.get("cluster"),
            "n": best.get("n"),
            "stability": best.get("stability"),
            "status": best.get("status"),
            "n_clusters": len(clusters),
            "ari": disc.get("ari"),
            "ami": disc.get("ami"),
            "note": disc.get("note"),
        }
    else:  # pragma: no cover - Detective always returns >=1 cluster today
        card = _default_claim_card(claim)
        provenance = {"mode": "unsupervised", "engine": decision.engine,
                      "n_clusters": 0, "note": disc.get("note")}
    return card, provenance


def _default_claim_card(claim: Claim) -> ClaimCard:
    return ClaimCard(claim=claim, naive_effect={"metric": "AUC", "value": 0.5},
                     tests=[], score=0, verdict=verdict_for(0), promoted=False)


# ===========================================================================
# Step 4 — the biomarker-anchor HARD GATE (policy atn_framework + biomarker_routing).
# ===========================================================================

def _find_test(card: ClaimCard, key: str) -> Optional[TestEvidence]:
    return next((t for t in card.tests if t.key == key), None)


def apply_biomarker_anchor_gate(card: ClaimCard, *, mechanism: str = "",
                                expected_direction: str = "",
                                kill_criterion: str = "") -> dict:
    """The molecular HARD GATE, re-derived at the harness layer from policy.

    "Imaging finds it, proteins confirm it." A promoted finding whose plasma
    p-tau217/GFAP anchor is present-but-FAILED (CI lower bound at/below the
    policy floor) is molecular *data present, unanchored* — the gate blocks
    promotion. An NA anchor (no coverage) is neither credited nor condemned here
    (the referee's replication path may still corroborate it). PASSED/WEAKENED
    anchors clear the gate. Thresholds come from policy/atn_framework (with the
    hardcoded fallback); the routed mechanism + pre-registered kill criterion
    (policy/biomarker_routing) ride along on the returned decision.

    Mutates ``card.promoted``/``card.caveats`` when it blocks, and returns a
    serializable decision dict recorded in the card's discovery_provenance."""
    thr = policy.thresholds("anchor")          # ci_pass, ci_weak, min_n
    anchor = _find_test(card, "biomarker_anchor")
    status = anchor.result if anchor is not None else TestResult.NA
    stats = dict(anchor.stats) if (anchor is not None and anchor.stats) else {}

    blocked = False
    if card.promoted and status == TestResult.FAILED:
        card.promoted = False
        blocked = True
        msg = ("Biomarker-anchor HARD GATE: plasma marker present but its "
               f"correlation FAILED (95% CI lower bound at/below {thr['ci_weak']:.2f} "
               "— no molecular support) — promotion blocked, treat as unanchored.")
        if msg not in card.caveats:
            card.caveats.append(msg)

    return {
        "gate": "biomarker_anchor",
        "status": status.value,
        "passed": status == TestResult.PASSED,
        "blocked_promotion": blocked,
        "ci_pass": thr.get("ci_pass"),
        "ci_weak": thr.get("ci_weak"),
        "min_n": thr.get("min_n"),
        "ptau217_ci_lo": stats.get("ptau217_ci_lo"),
        "gfap_ci_lo": stats.get("gfap_ci_lo"),
        "routed_mechanism": mechanism,
        "expected_direction": expected_direction,
        "kill_criterion": kill_criterion,
    }


# ===========================================================================
# Step 5 — the honesty rung, walked up the novelty_rubric ladder.
# ===========================================================================

def _novelty_rungs() -> list:
    """Ordered rung keys of the novelty_rubric ladder (policy, with fallback)."""
    try:
        rr = policy.table("novelty_rubric").get("honesty_rung", {}).get("rungs", [])
        keys = [str(r.get("key")).strip() for r in rr if r.get("key")]
        if len(keys) == 5:
            return keys
    except Exception as exc:  # pragma: no cover - fallback path
        _log.debug("novelty_rubric ladder policy read failed: %r", exc)
    return list(HONESTY_LADDER)   # canonical fallback — single source of truth


def _stability_floor() -> float:
    try:
        from .. import detective
        return float(detective.STABILITY_FLOOR)
    except Exception:  # pragma: no cover - fallback path
        return 0.60


def _confound_survivor(tests: dict) -> bool:
    """The confound tests (age/sex + the two STAR tests) survive — none FAILED,
    and at least one was actually testable and survived (rung-3 evidence).

    NA is NEUTRAL, not disqualifying: an untestable confound (e.g. a brain-age
    control whose model is non-predictive -> NA) is missing evidence, not evidence
    against — exactly as contract.robustness_score drops NA rather than penalizing
    it. Treating NA like FAILED would let one uninformative control silently sink
    a card that cleanly survives the confounds it COULD test."""
    keys = ("age_sex", "site_scanner", "brain_age")
    results = [tests.get(k) for k in keys]
    if any(r == TestResult.FAILED for r in results):
        return False
    survived = {TestResult.PASSED, TestResult.WEAKENED, TestResult.MIXED}
    return any(r in survived for r in results)


def compute_honesty_rung(card: ClaimCard, *, mode: str,
                         provenance: Optional[dict] = None) -> str:
    """Walk the cumulative novelty_rubric ladder and return the highest rung KEY
    whose evidence the card actually carries.

    Cumulative: rung N holds only if every lower rung holds, so a card that fails
    a STAR confound never rises above rung 2, and a card lacking a passing anchor
    is capped at rung 3 — exactly the honesty coupling the rubric specifies. All
    thresholds are read from policy (anchor CI floor, replication AUC floor) with
    hardcoded fallbacks."""
    rungs = _novelty_rungs()
    provenance = provenance or {}
    thr_anchor = policy.thresholds("anchor")
    thr_rep = policy.thresholds("replication")
    tests = {t.key: t.result for t in card.tests}
    try:
        naive = float(card.naive_effect.get("value") or 0.0)
    except Exception:
        naive = 0.0

    # rung 1 — a naive effect exists (above chance).
    r1 = naive > 0.5
    # rung 2 — the pattern reproduces (unsupervised: a stable cluster; supervised:
    # a materially above-chance effect, not a coin-flip fluke).
    if mode == "unsupervised":
        stab = provenance.get("stability")
        r2 = r1 and (stab is not None and float(stab) >= _stability_floor())
    else:
        r2 = r1 and naive >= 0.55
    # rung 3 — survives the confound gauntlet (STAR tests + age/sex).
    r3 = r2 and _confound_survivor(tests)
    # rung 4 — clears the biomarker-anchor HARD GATE and is promoted.
    anchor_lo = _find_test(card, "biomarker_anchor")
    anchor_ci_lo = None
    if anchor_lo is not None and anchor_lo.stats:
        anchor_ci_lo = (anchor_lo.stats.get("ptau217_ci_lo")
                        or anchor_lo.stats.get("gfap_ci_lo"))
    anchor_pass = tests.get("biomarker_anchor") == TestResult.PASSED
    # Prefer the explicit CI-floor test when the number is present; PASSED already
    # encodes it, this just makes the policy threshold load-bearing.
    if anchor_ci_lo is not None:
        anchor_pass = anchor_pass and float(anchor_ci_lo) >= float(thr_anchor["ci_pass"])
    r4 = r3 and anchor_pass and bool(card.promoted)
    # rung 5 — EXTERNAL replication on a genuinely INDEPENDENT cohort/site.
    # The gauntlet's "replication" test is an internal held-out SPLIT of the SAME
    # cohort; passing it earns rung 4 ("ready for replication"), NOT rung 5.
    # Claiming "externally_replicated" therefore requires an explicit external-
    # cohort signal in provenance, set only by a genuine cross-cohort run — a
    # single-dataset investigate() can never honestly reach rung 5. This is the
    # honesty coupling: internal robustness ≠ external replication.
    rep = _find_test(card, "replication")
    rep_auc = rep.stats.get("test_auc") if (rep is not None and rep.stats) else None
    rep_pass = tests.get("replication") == TestResult.PASSED
    if rep_auc is not None:
        rep_pass = rep_pass and float(rep_auc) >= float(thr_rep["pass"])
    external = bool(provenance.get("external_replication")
                    or provenance.get("external_cohort"))
    r5 = r4 and rep_pass and external

    highest = 0
    for i, ok in enumerate((r1, r2, r3, r4, r5), start=1):
        if ok:
            highest = i
        else:
            break
    idx = max(highest, 1) - 1                    # floor at the lowest rung (non-empty)
    return rungs[idx] if 0 <= idx < len(rungs) else rungs[0]


def _atn_profile(card: ClaimCard, mechanism: str) -> dict:
    """A compact A/T/(N)/(+I) staging summary from the anchor stats + routing."""
    anchor = _find_test(card, "biomarker_anchor")
    stats = dict(anchor.stats) if (anchor is not None and anchor.stats) else {}
    return {
        "T_ptau217_r": stats.get("ptau217_r"),
        "T_ptau217_ci_lo": stats.get("ptau217_ci_lo"),
        "I_gfap_r": stats.get("gfap_r"),
        "I_gfap_ci_lo": stats.get("gfap_ci_lo"),
        "anchor_status": (anchor.result.value if anchor is not None
                          else TestResult.NA.value),
        "routed_mechanism": mechanism,
        "synthetic": bool(stats.get("synthetic")),
    }


# ===========================================================================
# L5 entry point.
# ===========================================================================

def investigate(hypothesis: str, dataset: str, *, api: bool = False,
                seed: int = 0, anchor: Optional[str] = None) -> ExperimentCard:
    """Turn a plain-language hypothesis into a refereed, honesty-stamped card.

    Parameters
    ----------
    hypothesis : str
        The researcher's free-text hunch.
    dataset : str
        A registered dataset name (e.g. ``"synthetic:SURVIVOR"``,
        ``"synthetic:KILL"``, ``"oasis"``) — dispatched via ``data.loaders.load``.
    api : bool, default False
        When False (and with no ANTHROPIC_API_KEY) the whole path is offline and
        deterministic. When True the claim parse may call Claude, still falling
        back on any failure.
    seed : int, default 0
        Seed forwarded to the (synthetic) feeder.
    anchor : str, optional
        The fluid biomarker the researcher CHOSE to anchor on (amyloid /
        p_tau217 / gfap / nfl). When given it routes the molecular follow-up
        mechanism (overriding cohort-dominance) and selects the anchor-congruent
        lead + organoid readout — a read-only side artifact that never touches
        the score/verdict.

    Returns
    -------
    ExperimentCard
        Always carries a non-empty ``novelty_class`` and ``honesty_rung`` and has
        passed the HONESTY GUARD (raised as ``HonestyViolation`` otherwise).
    """
    from ..data import loaders

    df = loaders.load(dataset, seed=seed)

    # 0. Refuse out-of-scope targets up front rather than silently coercing them
    #    to `conversion` (the historical honesty hazard).
    reason = _out_of_scope_reason(hypothesis)
    if reason:
        card = _unsupported_card(hypothesis, dataset, reason)
        xcard = experiment_card.build_experiment_card(
            card,
            novelty_class="unsupported",
            honesty_rung=HONESTY_LADDER[0] if HONESTY_LADDER else "unsupported",
            discovery_provenance={"dataset": dataset, "status": "unsupported_target",
                                  "reason": reason},
        )
        return honesty_guard(xcard)

    # 1. Parse + enrich the hypothesis.
    claim = _parse_claim(hypothesis, df, api=api)
    claim.substrate = loaders.honest_substrate(dataset)   # truthful per-feeder label
    novelty = _classify_novelty(claim.claim_text or hypothesis)
    mechanism, expected_direction, kill_criterion = _mechanism_enrichment(df, anchor)

    # 2. Route to a discovery mode.
    decision = discovery_router.route(claim, df)

    # 3. Referee for that mode.
    if decision.supervised:
        card = _run_supervised(df, claim, use_claude=api, anchor=anchor,
                               dataset=dataset, seed=seed)
        provenance = {
            "mode": "supervised",
            "engine": decision.engine,
            "target": decision.target,
            "rationale": decision.rationale,
            "signals": list(decision.signals),
        }
    else:
        card, provenance = _run_unsupervised(df, claim, decision)

    # 4. Biomarker-anchor HARD GATE (may block promotion; records the routing +
    #    the pre-registered kill criterion).
    gate = apply_biomarker_anchor_gate(
        card, mechanism=mechanism, expected_direction=expected_direction,
        kill_criterion=kill_criterion)

    # 5. Honesty rung (walked up the novelty_rubric ladder) + provenance.
    rung = compute_honesty_rung(card, mode=provenance["mode"], provenance=provenance)
    provenance["dataset"] = dataset
    provenance["novelty_class"] = novelty
    provenance["expected_direction"] = expected_direction
    provenance["kill_criterion"] = kill_criterion
    provenance["anchor_gate"] = gate

    xcard = experiment_card.build_experiment_card(
        card,
        novelty_class=novelty,
        atn_profile=_atn_profile(card, mechanism),
        honesty_rung=rung,
        discovery_provenance=provenance,
    )

    # ALWAYS vet the card before it leaves the harness.
    return honesty_guard(xcard)
