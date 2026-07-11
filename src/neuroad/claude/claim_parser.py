"""
claim_parser — natural-language hunch -> structured, testable Claim.

This is the first *consequential* Claude decision in the loop: it turns a
scientist's free-text hunch ("I think converters look different in the
hippocampus") into a `contract.Claim` with a concrete label target, the two
populations being contrasted, and the covariates the gauntlet must adjust for.
Everything downstream (probe target, group split, age/sex adjustment) is driven
by what this step decides.

Live path asks Claude for a strict structured parse; the offline path uses a
deterministic keyword router so the demo runs without an API key.
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from ..contract import Claim, LABEL_TARGETS
from . import _client

SYSTEM = (
    "Persona: CLAIM PARSER. Convert a researcher's informal hunch about an "
    "Alzheimer's structural-MRI finding into a single structured, falsifiable "
    "claim. Choose exactly one target label column from the allowed set, name "
    "the two populations being contrasted, and list the demographic covariates "
    "the referee must adjust for (default age and sex). Do not invent numbers."
)

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "claim_text": {"type": "string"},
        "target": {"type": "string", "enum": list(LABEL_TARGETS)},
        "group_a": {"type": "string"},
        "group_b": {"type": "string"},
        "covariates": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["claim_text", "target", "group_a", "group_b", "covariates"],
}

# Default population labels per target.
_GROUPS = {
    "conversion": ("MCI converters", "MCI non-converters"),
    "dx_binary": ("AD", "CN"),
    "site": ("site A", "site B"),
    "scanner": ("scanner A", "scanner B"),
}


def parse_claim(text: str, df: Optional[pd.DataFrame] = None) -> Claim:
    """Parse a free-text hunch into a `contract.Claim`, enriched with the L3
    policy layer's pre-registered annotations.

    The deterministic template parse (live structured -> `_from_dict`, offline ->
    `_fallback`) is unchanged and remains the fallback; `_enrich` then attaches a
    novelty_class, an expected_direction and a pre-registered kill_criterion drawn
    from `policy/` (with hardcoded fallbacks), so the whole step stays offline and
    deterministic."""
    # DETERMINISTIC parse — the referee never calls Claude (Claude is
    # orchestrator-only, see harness/agent.py). Keyword router + L3 policy enrich.
    text = (text or "").strip()
    claim = _fallback(text, df)
    _enrich(claim, text, df)
    return claim


def _prompt(text: str, df: Optional[pd.DataFrame]) -> str:
    cols = ""
    if df is not None:
        present = [c for c in ("conversion", "dx", "site", "scanner") if c in df.columns]
        cols = f"\nTable has columns: {', '.join(present)}."
    allowed = "; ".join(f"{k}: {v}" for k, v in LABEL_TARGETS.items())
    return (
        f"Hunch: {text!r}{cols}\n"
        f"Allowed target columns — {allowed}.\n"
        "Return the structured claim."
    )


def _from_dict(text: str, data: dict) -> Claim:
    target = data.get("target", "conversion")
    if target not in LABEL_TARGETS:
        target = "conversion"
    ga, gb = _GROUPS.get(target, ("group A", "group B"))
    covs = data.get("covariates") or ["age", "sex"]
    return Claim(
        claim_id=_claim_id(text),
        claim_text=data.get("claim_text") or text or "Structural signal claim",
        target=target,
        group_a=data.get("group_a") or ga,
        group_b=data.get("group_b") or gb,
        covariates=[c for c in covs if isinstance(c, str)] or ["age", "sex"],
    )


# ---------------------------------------------------------------------------
# L3 policy enrichment — pre-registered annotations on the parsed Claim.
# ---------------------------------------------------------------------------
# Keyword hints for the deterministic novelty_class — the hardcoded FALLBACK for
# policy/novelty_rubric.md's closed vocabulary (mirrors harness.orchestrator so
# the parser and the orchestrator classify a hunch identically). "known" =
# published prior art we re-measure (e.g. embeddings leaking scanner/site);
# "novel" = asks the data for structure with no clean precedent; else "adjacent".
_KNOWN_HINTS = ("scanner", "site", "leak", "batch", "field strength",
                "acquisition", "prior art")
_NOVEL_HINTS = ("novel", "hidden", "unknown", "undiscovered", "latent",
                "emergent", "phenotype", "subtype", "sub-type", "subgroup",
                "cluster", "stratif")


def _enrich(claim: Claim, text: str, df: Optional[pd.DataFrame]) -> None:
    """Attach L3 policy annotations to a parsed Claim (in place).

    Sets three pre-registered attributes DRAWN FROM the policy layer, each with a
    hardcoded fallback to today's live constants so the path stays deterministic
    and never touches the network or raises:

      * ``novelty_class``       — from policy/novelty_rubric.md's closed vocabulary
      * ``expected_direction``  — the routed mechanism's direction (biomarker_routing)
      * ``kill_criterion``      — the routed mechanism's kill, phrased per the
                                  hypothesis_schema kill_criterion_format contract

    The base Claim (target, groups, covariates) is never modified; enrichment is
    purely additive, so ClaimCard serialization and the demo path are unchanged.
    """
    claim.novelty_class = _novelty_class(text)
    direction, kill = _mechanism_enrichment(df)
    claim.expected_direction = direction
    claim.kill_criterion = kill


def _novelty_values() -> tuple:
    """Closed novelty vocabulary, from policy with a hardcoded fallback."""
    try:
        from ..harness import policy
        vals = policy.table("novelty_rubric").get("novelty_class", {}).get("values")
        if isinstance(vals, list) and vals:
            return tuple(str(v).strip().lower() for v in vals)
    except Exception:
        pass
    return ("known", "adjacent", "novel")


def _novelty_class(text: str) -> str:
    """Deterministic novelty_class for a hunch, constrained to the policy vocab."""
    allowed = _novelty_values()
    low = f" {(text or '').lower()} "
    guess = "adjacent"
    if any(h in low for h in _KNOWN_HINTS):
        guess = "known"
    elif any(h in low for h in _NOVEL_HINTS):
        guess = "novel"
    return guess if guess in allowed else (allowed[0] if allowed else "adjacent")


def _mechanism_enrichment(df: Optional[pd.DataFrame]) -> tuple[str, str]:
    """Pre-register (expected_direction, kill_criterion) for the routed mechanism.

    Routes with the same biomarker-dominance rule the Bridge uses, reads the
    mechanism's expected_direction + kill_criterion from policy/biomarker_routing
    (fallback: bridge._MECHANISMS), and phrases the kill via the hypothesis_schema
    kill_criterion_format when the mechanism supplies none — all with fallbacks."""
    mech = "amyloid_cascade"
    try:
        from .bridge import _route
        mech = _route(df)
    except Exception:
        pass

    direction = kill = ""
    try:
        from ..harness import policy
        row = policy.table("biomarker_routing").get("mechanisms", {}).get(mech, {})
        direction = str(row.get("expected_direction") or row.get("direction") or "").strip()
        kill = str(row.get("kill_criterion") or row.get("kill") or "").strip()
    except Exception:
        pass

    if not (direction and kill):
        try:
            from .bridge import _MECHANISMS
            m = _MECHANISMS.get(mech, {})
            direction = direction or str(m.get("direction", "")).strip()
            kill = kill or str(m.get("kill", "")).strip()
        except Exception:
            pass

    if not kill:
        kill = _kill_criterion_format()
    return direction, kill


def _kill_criterion_format() -> str:
    """Pre-registered kill-criterion phrasing from policy/hypothesis_schema.yaml
    (kill_criterion_format.example/template), with a hardcoded fallback."""
    try:
        from ..harness import policy
        fmt = policy.table("hypothesis_schema").get("kill_criterion_format", {})
        for key in ("example", "template"):
            v = fmt.get(key)
            if isinstance(v, str) and v.strip():
                return " ".join(v.split())
    except Exception:
        pass
    return ("register one falsifiable kill criterion (metric, threshold, cohort, "
            "N, direction) before the confirmatory experiment is run")


# ---------------------------------------------------------------------------
# Deterministic offline router
# ---------------------------------------------------------------------------


def _fallback(text: str, df: Optional[pd.DataFrame]) -> Claim:
    target = _infer_target(text, df)
    ga, gb = _GROUPS.get(target, ("group A", "group B"))
    return Claim(
        claim_id=_claim_id(text),
        claim_text=text or f"Structural embeddings predict {target}",
        target=target,
        group_a=ga,
        group_b=gb,
        covariates=["age", "sex"],
    )


def _infer_target(text: str, df: Optional[pd.DataFrame]) -> str:
    low = text.lower()
    if re.search(r"conver|progress|mci.?to.?ad|declin", low):
        return "conversion"
    if re.search(r"scanner|field strength", low):
        return "scanner"
    if re.search(r"\bsite\b|acquisit", low):
        return "site"
    if re.search(r"diagnos|ad.?vs.?cn|dementia|alzheimer|patients?\b", low):
        return "dx_binary"
    # Fall back to what the table can actually support.
    if df is not None:
        if "conversion" in df.columns and df["conversion"].notna().any():
            return "conversion"
        if "dx" in df.columns:
            return "dx_binary"
    return "conversion"


def _claim_id(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:32]
    return f"claim-{slug}" if slug else "claim-unnamed"
