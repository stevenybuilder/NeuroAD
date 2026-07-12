"""
bridge — promoted survivors ONLY -> one biomarker-routed mechanism + experiment.

The bridge is deliberately NARROW: one survivor, one likely mechanism, one
falsifiable next experiment. It never fires for a finding that did not clear the
promotion floor.

Biomarker routing (the consequential decision):
  - p-tau217 / amyloid dominant -> amyloid-cascade (tau-driven)
  - GFAP dominant, amyloid weak  -> neuroinflammatory / glial (astrogliosis)
  - NfL dominant                 -> vascular / axonal

propose_biology(card, df) -> {'hypothesis', 'next_experiment': [...], 'falsification': [...]}
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..contract import ClaimCard, BIOMARKER_COLUMNS
from ..calibration import FACTS, target
from ..harness import policy
from . import _client

SYSTEM = (
    "Persona: BRIDGE. Only for a survivor that cleared the promotion floor, "
    "name the single most likely mechanism, ROUTED by which plasma biomarker "
    "dominates the separation: p-tau217/amyloid -> amyloid-cascade; GFAP with "
    "weak amyloid -> neuroinflammatory/glial (astrogliosis); NfL -> vascular/"
    "axonal. Then propose exactly ONE falsifiable next experiment (named cohort, "
    "target N, expected direction, and an explicit kill criterion) plus the "
    "criteria that would falsify the mechanism. One survivor, one mechanism, one "
    "experiment — keep the bridge honest and narrow."
)

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "hypothesis": {"type": "string"},
        "next_experiment": {"type": "array", "items": {"type": "string"}},
        "falsification": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["hypothesis", "next_experiment", "falsification"],
}

# mechanism-key -> (label, routing biomarkers, cohort, N, direction, kill)
_MECHANISMS = {
    "amyloid_cascade": {
        "label": "amyloid-cascade (tau-driven)",
        "markers": "p-tau217 and amyloid positivity",
        "cohort": "ADNI-3 / EPAD",
        "n": 120,
        "direction": (
            "the structural probe score should track plasma p-tau217 "
            "(expected r~0.3-0.55, modest not redundant) and be enriched in "
            "amyloid-positive subjects"
        ),
        "kill": (
            "if the probe score shows no p-tau217 correlation (r<0.2) on the "
            "complete-case subset, the amyloid-cascade routing is wrong"
        ),
        "fact": FACTS["ptau217"],
    },
    "glial": {
        "label": "neuroinflammatory / glial (reactive astrogliosis)",
        "markers": "plasma GFAP with only weak amyloid coupling",
        "cohort": "ADNI-3 (GFAP subset) / a memory-clinic cohort with plasma GFAP",
        "n": 100,
        "direction": (
            "the probe score should track plasma GFAP more strongly than "
            "p-tau217, implicating early astrocyte reactivity rather than a "
            "tau-first cascade"
        ),
        "kill": (
            "if GFAP association is no stronger than p-tau217, or vanishes after "
            "adjusting for amyloid, the glial routing is wrong"
        ),
        "fact": FACTS["gfap"],
    },
    "vascular": {
        "label": "vascular / axonal",
        "markers": "plasma NfL (axonal injury), ideally alongside WMH burden",
        "cohort": "a cohort with plasma NfL and FLAIR-derived WMH (e.g. NACC/EPAD)",
        "n": 100,
        "direction": (
            "the probe score should track plasma NfL and white-matter "
            "hyperintensity burden, pointing to axonal/vascular rather than "
            "amyloid-first pathology"
        ),
        "kill": (
            "if NfL and WMH show no association with the probe score, the "
            "vascular/axonal routing is wrong"
        ),
        "fact": "Plasma NfL indexes axonal injury; WMH indexes small-vessel disease.",
    },
}


def _mechanisms() -> dict:
    """Biomarker->mechanism table, routed from the L3 policy layer
    (policy.table("biomarker_routing")) with _MECHANISMS as the exact offline
    fallback. The policy loader already guarantees that fallback when policy/ is
    missing or malformed; we mirror the doc's ("expected_direction",
    "kill_criterion", "fact_key") shape back to this module's field names and
    fall back locally if any routing target is missing, so behavior is identical
    to _MECHANISMS whenever the doc matches it."""
    try:
        raw = policy.table("biomarker_routing").get("mechanisms") or {}
        out = {}
        for key, m in raw.items():
            fact = m.get("fact")
            if fact is None and m.get("fact_key"):
                fact = FACTS.get(m["fact_key"])
            out[key] = {
                "label": m.get("label"),
                "markers": m.get("markers"),
                "cohort": m.get("cohort"),
                "n": m.get("n"),
                "direction": m.get("expected_direction"),
                "kill": m.get("kill_criterion"),
                "fact": fact,
            }
        if all(k in out for k in _MECHANISMS):       # only trust a full table
            return out
    except Exception:
        pass
    return _MECHANISMS


def propose_biology(card: ClaimCard, df: Optional[pd.DataFrame] = None) -> dict:
    """Survivors only: route to a mechanism and propose one killer experiment."""
    if not card.promoted:
        return {
            "hypothesis": (
                "Not promoted: this finding did not clear the promotion floor, so "
                "the referee does not advance it to a mechanism. Biology speaks "
                "only for survivors."
            ),
            "next_experiment": [],
            "falsification": [],
        }

    # DETERMINISTIC: the referee never calls Claude. Mechanism routing +
    # experiment come from the biomarker-dominance rule below. (Claude's only
    # role in the engine is the orchestrator — see harness/agent.py.)
    mech_key = _route(df)
    return _fallback(card, mech_key)


# ---------------------------------------------------------------------------
# Biomarker routing
# ---------------------------------------------------------------------------


def _disease_mask(df: pd.DataFrame) -> Optional[pd.Series]:
    """Boolean mask for the 'disease' pole of the contrast."""
    if "conversion" in df.columns and df["conversion"].notna().any():
        conv = pd.to_numeric(df["conversion"], errors="coerce")
        if (conv == 1).any() and (conv == 0).any():
            return conv == 1
    if "dx" in df.columns:
        dx = df["dx"].astype("string")
        if (dx == "AD").any() and (dx != "AD").any():
            return dx == "AD"
    return None


def _effect_size(values: pd.Series, mask: pd.Series) -> float:
    """Absolute standardized mean difference (Cohen's-d-like) across the mask."""
    v = pd.to_numeric(values, errors="coerce")
    a = v[mask].dropna()
    b = v[~mask].dropna()
    if len(a) < 2 or len(b) < 2:
        return 0.0
    sd = v.dropna().std()
    if not np.isfinite(sd) or sd == 0:
        return 0.0
    return float(abs(a.mean() - b.mean()) / sd)


def _route(df: Optional[pd.DataFrame]) -> str:
    """Pick a mechanism key from which biomarker dominates the separation."""
    if df is None or _disease_mask(df) is None:
        return "amyloid_cascade"
    mask = _disease_mask(df)
    scores = {
        "p_tau217": _effect_size(df["p_tau217"], mask) if "p_tau217" in df else 0.0,
        "gfap": _effect_size(df["gfap"], mask) if "gfap" in df else 0.0,
        "nfl": _effect_size(df["nfl"], mask) if "nfl" in df else 0.0,
    }
    # Amyloid: prevalence difference contributes to the tau/amyloid pole.
    if "amyloid" in df.columns:
        amy = pd.to_numeric(df["amyloid"], errors="coerce")
        a = amy[mask].dropna()
        b = amy[~mask].dropna()
        if len(a) >= 2 and len(b) >= 2:
            scores["p_tau217"] += abs(a.mean() - b.mean())
    # Triangulated plasma ensemble (contract.EXTENDED_BIOMARKER_COLUMNS): plasma
    # Aβ42/40 (amyloid "A" axis) and C2N %p-tau217 (tau axis) both reinforce the
    # amyloid-cascade / tau pole, so their separation adds to the p_tau217 score.
    for extra in ("ab42_40", "pct_ptau217"):
        if extra in df.columns:
            scores["p_tau217"] += _effect_size(df[extra], mask)

    if max(scores.values()) <= 0:
        return "amyloid_cascade"
    dominant = max(scores, key=scores.get)
    return {
        "p_tau217": "amyloid_cascade",
        "gfap": "glial",
        "nfl": "vascular",
    }[dominant]


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------


def _fallback(card: ClaimCard, mech_key: str) -> dict:
    m = _mechanisms()[mech_key]
    hypothesis = (
        f"Survivor “{card.claim.claim_text}” routes to a {m['label']} mechanism, "
        f"keyed on {m['markers']}. {m['fact']} The structural probe is read as a "
        "downstream imaging correlate of that molecular process, not an "
        "independent discovery."
    )
    next_experiment = [
        f"Cohort: {m['cohort']}.",
        f"Target N: ~{m['n']} complete-case subjects with the routing biomarker.",
        f"Direction: {m['direction']}.",
        f"Kill criterion: {m['kill']}.",
    ]
    falsification = [
        m["kill"] + ".",
        (
            "If the probe score's biomarker correlation disappears after "
            "residualizing against a scanner-predicting direction, the survivor "
            "was leakage after all."
        ),
        (
            f"If a held-out cohort fails to reproduce the effect (drop below the "
            f"promotion floor), the mechanism claim is withdrawn."
        ),
    ]
    return {
        "hypothesis": hypothesis,
        "next_experiment": next_experiment,
        "falsification": falsification,
    }


def _prompt(card: ClaimCard, df: Optional[pd.DataFrame], mech_key: str) -> str:
    cov = {}
    if df is not None:
        for b in BIOMARKER_COLUMNS:
            if b in df.columns:
                cov[b] = round(float(df[b].notna().mean()), 2)
    return (
        f"Promoted survivor: {card.claim.claim_text} ({card.verdict.value}, "
        f"{card.score}/100).\n"
        f"Populations: {card.claim.group_a} vs {card.claim.group_b}.\n"
        f"Biomarker coverage in the table: {cov}.\n"
        f"Suggested routing from marker dominance: {_mechanisms()[mech_key]['label']}.\n"
        f"Calibration anchors: conversion AUC target ~{target('conversion_auc')}, "
        f"p-tau217 correlation target ~{target('ptau217_r')}.\n"
        "Give the mechanism hypothesis, one falsifiable next experiment, and the "
        "falsification criteria."
    )
