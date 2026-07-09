"""
discovery_router — aim the instrument before the referee runs.

Given a (structured or raw) hypothesis, decide the discovery MODE:

  * a NOVEL-PATTERN hypothesis ("are there hidden MCI subtypes / phenotypes?")
    routes to the UNSUPERVISED Detective — ``discovery.discover_and_referee`` —
    which clusters the embeddings and referees each recovered phenotype;

  * a NAMED-CONTRAST hypothesis ("does the embedding predict MCI->AD
    conversion?", "AD vs CN") routes to the SUPERVISED probe / gauntlet — the
    ``pipeline.run_referee`` path — which points the one reused head at a target.

The decision is a small, serializable ``RouteDecision``. This module reads only
the keyword tables below (its hardcoded fallback for policy/hypothesis_schema.yaml)
and ``contract.LABEL_TARGETS`` — no network, no policy file required. If the L3
``policy/`` docs are absent the router still routes deterministically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from .. import contract
from ..contract import Claim

# ---------------------------------------------------------------------------
# Routing vocabulary — HARDCODED FALLBACK for policy/hypothesis_schema.yaml's
# discovery-mode selector. Novel-pattern language wins: asking the data to
# reveal structure is the Detective's job; naming a contrast is the probe's.
# ---------------------------------------------------------------------------

NOVEL_PATTERN_KEYWORDS = (
    "phenotype", "subtype", "sub-type", "subgroup", "sub-group", "cluster",
    "clustering", "novel pattern", "unsupervised", "discover", "stratif",
    "latent", "hidden", "unknown group", "distinct group", "unlabeled",
    "unlabelled", "data-driven", "data driven", "emergent",
)

NAMED_CONTRAST_KEYWORDS = (
    "predict", "prognos", " vs ", "versus", "converter", "conversion",
    "diagnos", "classif", "separate", "distinguish", "amyloid-positive",
    "site leakage", "scanner",
)

#: Default Detective configuration — mirrors discovery.discover_and_referee's
#: own defaults so the decision can be dispatched verbatim.
_DEFAULT_DETECTIVE_CFG = {"method": "kmeans", "B": 50}

_SUPERVISED_ENGINE = "neuroad.pipeline.run_referee"
_UNSUPERVISED_ENGINE = "neuroad.discovery.discover_and_referee"


@dataclass
class RouteDecision:
    """The small structured decision the router returns.

    * ``mode``          -- "supervised" | "unsupervised".
    * ``engine``        -- dotted path of the entry point that will run.
    * ``target``        -- supervised only: the LABEL_TARGET the head points at.
    * ``detective_cfg`` -- unsupervised only: kwargs for discover_and_referee.
    * ``rationale``     -- one-line, human-readable justification.
    * ``signals``       -- the matched keywords that drove the decision.
    """
    mode: str
    engine: str
    target: Optional[str] = None
    detective_cfg: dict = field(default_factory=dict)
    rationale: str = ""
    signals: list = field(default_factory=list)

    @property
    def supervised(self) -> bool:
        return self.mode == "supervised"

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "engine": self.engine,
            "target": self.target,
            "detective_cfg": dict(self.detective_cfg),
            "rationale": self.rationale,
            "signals": list(self.signals),
        }


def _unpack(claim: Union[Claim, str]) -> tuple[str, str, tuple[str, str]]:
    """Return (text, target, (group_a, group_b)) from a Claim or raw string."""
    if isinstance(claim, Claim):
        return (claim.claim_text or "", claim.target or "",
                (claim.group_a or "", claim.group_b or ""))
    return (str(claim or ""), "", ("", ""))


def route(claim: Union[Claim, str],
          df=None) -> RouteDecision:
    """Decide the discovery mode for ``claim``.

    Novel-pattern language ("phenotype / subtype / cluster / discover / hidden")
    routes to the unsupervised Detective; anything else is treated as a named
    contrast and routes to the supervised probe/gauntlet (the engine's default).
    ``df`` is accepted for signature parity with the orchestrator but the routing
    decision is text-driven and deterministic; it never touches the network.
    """
    text, target, (group_a, group_b) = _unpack(claim)
    low = f" {text.lower()} "

    novel = [k for k in NOVEL_PATTERN_KEYWORDS if k in low]
    named = [k for k in NAMED_CONTRAST_KEYWORDS if k in low]
    explicit_groups = bool(group_a and group_b and group_a != group_b)

    if novel:
        rationale = (
            "Novel-pattern hypothesis (matched: "
            f"{', '.join(sorted(set(s.strip() for s in novel)))}) -> unsupervised "
            "Detective clusters the embeddings and referees each phenotype.")
        return RouteDecision(
            mode="unsupervised",
            engine=_UNSUPERVISED_ENGINE,
            target=None,
            detective_cfg=dict(_DEFAULT_DETECTIVE_CFG),
            rationale=rationale,
            signals=[s.strip() for s in novel],
        )

    # Named contrast (default): point the one reused head at a target.
    resolved_target = target if target in contract.LABEL_TARGETS else "conversion"
    if named:
        why = f"matched: {', '.join(sorted(set(s.strip() for s in named)))}"
    elif explicit_groups:
        why = f"named groups: {group_a} vs {group_b}"
    else:
        why = "no novel-pattern language -> default named contrast"
    rationale = (
        f"Named-contrast hypothesis ({why}) -> supervised probe/gauntlet points "
        f"the head at '{resolved_target}'.")
    return RouteDecision(
        mode="supervised",
        engine=_SUPERVISED_ENGINE,
        target=resolved_target,
        detective_cfg={},
        rationale=rationale,
        signals=[s.strip() for s in named] or (["named-groups"] if explicit_groups
                                               else ["default"]),
    )
