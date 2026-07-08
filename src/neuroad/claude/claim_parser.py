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
    """Parse a free-text hunch into a `contract.Claim`."""
    text = (text or "").strip()
    if _client.USING_LIVE_API:
        try:
            data = _client.complete(
                SYSTEM, _prompt(text, df), schema=_SCHEMA
            )
            return _from_dict(text, data)
        except Exception:
            pass
    return _fallback(text, df)


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
