"""
experiment_card — the final researcher-facing artifact.

An ``ExperimentCard`` is a THIN wrapper over the frozen ``contract.ClaimCard``.
It adds the three Stage-2 annotations the harness stamps on top of a refereed
claim — ``novelty_class``, ``atn_profile``, ``honesty_rung`` — plus a
``discovery_provenance`` block (which discovery mode ran, cluster stability, ARI
vs planted truth when available). The frozen contract stays almost untouched:
those three fields already exist on ``ClaimCard`` with empty defaults, so the
builder simply fills them and mirrors them in ``to_dict()``.

Anti-overclaim contract (the pitch rests on this): every card carries a
``novelty_class`` and an ``honesty_rung``, and NO rung ever asserts a
"proven"/"validated biomarker". The rungs describe how much INDEPENDENT
corroboration a finding has, never that it is true.

Offline / deterministic: the honesty rung is derived from the card's own
verdict + promotion decision using the hardcoded ladder below. No network, no
Claude call, no policy read is required — if the L3 ``policy/`` docs are absent
this module still produces a complete, defensible card.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..contract import ClaimCard, Verdict

# ---------------------------------------------------------------------------
# Novelty taxonomy + the 5-rung calibrated-honesty ladder.
# These are the HARDCODED FALLBACK for policy/novelty_rubric.md — if that doc is
# missing or malformed the harness still stamps a defensible rung.
# ---------------------------------------------------------------------------

#: Allowed novelty classes (from contract.ClaimCard's field comment). "unclassified"
#: is the safe default so the honesty guard's "every card carries novelty_class"
#: invariant holds even before the claim_parser assigns one.
NOVELTY_CLASSES = ("known", "adjacent", "novel", "unclassified")

#: 5 rungs, lowest -> highest. Higher = more independent corroboration, NEVER a
#: claim of truth. None of these strings says "proven" or "validated biomarker".
HONESTY_LADDER = [
    "artifact-suspected",       # collapsed under the gauntlet — treat as an artifact
    "exploratory",              # survives some challenges; hypothesis-generating only
    "candidate-signal",         # robust score but not independently corroborated
    "corroborated-candidate",   # promoted: biomarker anchor OR leakage-clean replication
    "replication-ready",        # strong + corroborated: ready for a pre-stated killer test
]


def default_honesty_rung(card: ClaimCard) -> str:
    """Derive an honesty rung from the card's own verdict + promotion decision.

    Deterministic, offline, and deliberately conservative — a promoted STRONG
    finding is only ever "replication-ready", never "validated". The promotion
    gate (biomarker anchor or leakage-clean replication) already encodes the
    independent-corroboration requirement, so we key off it directly.
    """
    if card.promoted:
        return "replication-ready" if card.verdict == Verdict.STRONG \
            else "corroborated-candidate"
    if card.verdict == Verdict.FRAGILE:
        return "artifact-suspected"
    if card.verdict == Verdict.PARTIALLY_ROBUST:
        return "exploratory"
    # Robust score that nonetheless failed the corroboration gate (anchor NA /
    # no leakage-clean replication) — a genuine candidate, not yet corroborated.
    return "candidate-signal"


def _norm_novelty(value: Optional[str]) -> str:
    """Normalize a novelty class; empty/unknown -> 'unclassified' (never blank)."""
    if not value:
        return "unclassified"
    v = str(value).strip().lower()
    return v if v else "unclassified"


# ---------------------------------------------------------------------------
# The wrapper + builder.
# ---------------------------------------------------------------------------


@dataclass
class ExperimentCard:
    """Researcher-facing artifact: a refereed ClaimCard + harness annotations."""
    card: ClaimCard
    novelty_class: str = "unclassified"
    atn_profile: dict = field(default_factory=dict)
    honesty_rung: str = ""
    discovery_provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Merge the frozen ClaimCard export with the harness annotations.

        The three contract fields are re-stated here from the wrapper (they are
        the source of truth for the card view), and the discovery provenance is
        appended. Order: ClaimCard's dict first, then the harness block.
        """
        d = dict(self.card.to_dict())
        d["novelty_class"] = self.novelty_class
        d["atn_profile"] = dict(self.atn_profile)
        d["honesty_rung"] = self.honesty_rung
        d["discovery_provenance"] = dict(self.discovery_provenance)
        return d


def build_experiment_card(
    card: ClaimCard,
    *,
    novelty_class: Optional[str] = None,
    atn_profile: Optional[dict] = None,
    honesty_rung: Optional[str] = None,
    discovery_provenance: Optional[dict] = None,
) -> ExperimentCard:
    """Wrap a refereed ``ClaimCard`` into the final ``ExperimentCard``.

    Each annotation falls back, in order, to (1) the explicit argument, (2) the
    value already on the ClaimCard (e.g. set upstream by the claim_parser /
    anchor stamp), (3) a deterministic default. The chosen values are ALSO
    stamped back onto the ClaimCard's own optional fields so that a bare
    ``card.to_dict()`` (the frozen demo/report path) carries them too.

    Guarantees (the anti-overclaim contract): the returned card always has a
    non-empty ``novelty_class`` and a non-empty ``honesty_rung``.
    """
    novelty = _norm_novelty(
        novelty_class if novelty_class is not None else card.novelty_class)
    atn = dict(atn_profile) if atn_profile is not None else dict(card.atn_profile)
    rung = honesty_rung or card.honesty_rung or default_honesty_rung(card)
    provenance = dict(discovery_provenance or {})

    # Stamp back onto the frozen-contract fields so ClaimCard.to_dict() agrees.
    card.novelty_class = novelty
    card.atn_profile = atn
    card.honesty_rung = rung

    return ExperimentCard(
        card=card,
        novelty_class=novelty,
        atn_profile=atn,
        honesty_rung=rung,
        discovery_provenance=provenance,
    )
