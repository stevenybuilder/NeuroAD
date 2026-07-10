"""
translation — turn a promoted imaging survivor into a molecule / wet-lab lead.

DOWNSTREAM of the referee. This module NEVER touches embeddings, the gauntlet,
or a score/verdict. It fires ONLY for a promoted survivor (the same gate the
Bridge uses) and produces a READ-ONLY side artifact that chains the four
integration adapters into the plan's translation engine:

    mechanism (amyloid/glial/vascular — from the Bridge's biomarker-dominance
               routing, NOT from the imaging vector)
      -> PI4AD       rank the mechanism's candidate genes by priority (0-10)
      -> AlphaFold   fetch the top target's predicted structure + mean pLDDT
      -> repurposing GNN/LLM candidate compounds for the top target
      -> experiment  a falsifiable organoid readout for the lead

Every adapter is offline-first and provenance-labeled, so ``translate`` degrades
to bundled snapshots with NO network / credentials and NEVER raises — a failure
in any leg yields an empty/annotated leg, never an exception into the referee.

The mechanism -> candidate-gene map is a DELIBERATELY NARROW, literature-grounded
prior: PI4AD is what RANKS within the set. It names no gene absent from the plan
(Section 3: PI4AD recovers APP/ESR1, Ras nodes HRAS/MAPK1) or from
``integrations.alphafold.AD_PROTEIN_MAP``. Honesty contract: the returned artifact
never asserts the imaging survivor *is* a gene — only that, GIVEN a promoted
survivor routed to a mechanism, these are the prioritized molecular follow-ups.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from ..contract import ClaimCard

_log = logging.getLogger("neuroad.harness.translation")

#: Mechanism class (bridge._route keys) -> candidate AD genes for PI4AD to rank.
#: Grounded in the plan + integrations.alphafold.AD_PROTEIN_MAP; PI4AD orders
#: WITHIN each set by its 0-10 priority. Kept narrow on purpose.
MECHANISM_GENES: dict[str, list[str]] = {
    "amyloid_cascade": ["APP", "MAPT", "PSEN1", "BACE1", "APOE", "ESR1"],
    "glial": ["TREM2", "APOE", "CLU", "MAPK1", "HRAS"],
    "vascular": ["APOE", "CLU", "BIN1", "APP"],
}

#: Mechanism -> a concrete, falsifiable organoid readout for the proposed lead.
_ORGANOID_READOUT = {
    "amyloid_cascade": (
        "In Tanzi-style 3D human neural organoids, knock down the lead target "
        "(CRISPRi) and read out Aβ42/40 ratio + p-tau217 by MSD at 6 weeks; "
        "kill if neither moves beyond vehicle ±2SD."
    ),
    "glial": (
        "In iPSC-microglia + neuron co-culture organoids, perturb the lead "
        "target and read out secreted GFAP + IL-6/TNF-α and synaptic density "
        "(synaptophysin puncta) at 6 weeks; kill if glial markers are unchanged."
    ),
    "vascular": (
        "In a BBB-on-chip / organoid model, perturb the lead target and read out "
        "trans-endothelial resistance + NfL leakage at 6 weeks; kill if barrier "
        "integrity and NfL are unchanged."
    ),
}


@dataclass
class TranslationLead:
    """The molecule/wet-lab follow-up chained off one promoted survivor.

    Every field is provenance-labeled downstream (each adapter stamps its own
    ``source``); ``status`` is "translated" when a target was resolved, else a
    short reason. This is decision-support, not a claim of a validated target.
    """
    mechanism: str
    dominant_biomarker: str = ""
    status: str = ""                       # "translated" | reason it was skipped
    ranked_targets: list[dict] = field(default_factory=list)   # PI4AD priorities
    top_target: str = ""
    structure: dict = field(default_factory=dict)              # AlphaFold summary
    repurposing: list[dict] = field(default_factory=list)      # GNN/LLM compounds
    wet_lab_experiment: str = ""
    provenance: dict = field(default_factory=dict)             # per-leg source tags
    caveat: str = (
        "Decision-support only: prioritized molecular follow-ups for a promoted "
        "imaging survivor, routed via the dominant fluid biomarker — NOT a "
        "validated target, and never derived from the imaging embedding itself."
    )

    def to_dict(self) -> dict:
        return {
            "mechanism": self.mechanism,
            "dominant_biomarker": self.dominant_biomarker,
            "status": self.status,
            "ranked_targets": list(self.ranked_targets),
            "top_target": self.top_target,
            "structure": dict(self.structure),
            "repurposing": list(self.repurposing),
            "wet_lab_experiment": self.wet_lab_experiment,
            "provenance": dict(self.provenance),
            "caveat": self.caveat,
        }


def _dominant_biomarker(df: Optional[pd.DataFrame]) -> str:
    """Name the plasma biomarker that dominates the separation, for the record."""
    if df is None:
        return ""
    try:
        from ..claude import bridge
        mask = bridge._disease_mask(df)
        if mask is None:
            return ""
        scores = {}
        for col in ("p_tau217", "gfap", "nfl"):
            if col in df.columns:
                scores[col] = bridge._effect_size(df[col], mask)
        if not scores or max(scores.values()) <= 0:
            return ""
        return max(scores, key=scores.get)
    except Exception as exc:  # noqa: BLE001
        _log.debug("dominant biomarker read failed: %r", exc)
        return ""


def _rank_targets(mechanism: str, *, prefer_offline: bool = True) -> list[dict]:
    """PI4AD-rank the mechanism's candidate genes (offline snapshot by default)."""
    genes = MECHANISM_GENES.get(mechanism) or MECHANISM_GENES["amyloid_cascade"]
    ranked: list[dict] = []
    try:
        from ..integrations.pi4ad import PI4AD
        pi = PI4AD(prefer_offline=prefer_offline)
        for g in genes:
            gp = pi.priority(g)
            if gp is not None:
                ranked.append(gp.to_dict())
            else:
                # Unknown to PI4AD's table — keep it, flagged, at the bottom.
                ranked.append({
                    "gene": g, "priority_score": None, "rank": None,
                    "evidence_note": "not in PI4AD prioritisation table",
                    "source": "candidate_only",
                })
    except Exception as exc:  # noqa: BLE001
        _log.debug("PI4AD ranking failed, using unranked candidates: %r", exc)
        ranked = [{"gene": g, "priority_score": None, "rank": None,
                   "evidence_note": "PI4AD unavailable", "source": "candidate_only"}
                  for g in genes]
    # Sort by PI4AD priority desc; None priorities sink to the bottom.
    ranked.sort(key=lambda r: (r.get("priority_score") is not None,
                               r.get("priority_score") or 0.0), reverse=True)
    return ranked


def translate(
    mechanism: str,
    df: Optional[pd.DataFrame] = None,
    *,
    prefer_offline: bool = True,
    top_n_compounds: int = 5,
) -> dict:
    """Chain PI4AD -> AlphaFold -> repurposing off a promoted survivor's mechanism.

    Offline-first and exception-safe: any adapter failure degrades that leg to an
    empty/annotated result rather than raising. Returns a serializable dict
    (``TranslationLead.to_dict``). ``prefer_offline=True`` keeps the whole chain
    on bundled, provenance-labeled snapshots (no network) — the referee's default.
    """
    mech = mechanism if mechanism in MECHANISM_GENES else "amyloid_cascade"
    lead = TranslationLead(mechanism=mech, dominant_biomarker=_dominant_biomarker(df))

    ranked = _rank_targets(mech, prefer_offline=prefer_offline)
    lead.ranked_targets = ranked
    scored = [r for r in ranked if r.get("priority_score") is not None]
    if not scored:
        lead.status = "no PI4AD-ranked target for this mechanism"
        lead.wet_lab_experiment = _ORGANOID_READOUT.get(mech, "")
        lead.provenance = {"pi4ad": ranked[0]["source"] if ranked else "none"}
        return lead.to_dict()

    top = scored[0]["gene"]
    lead.top_target = top
    prov = {"pi4ad": scored[0].get("source", "offline_snapshot")}

    # AlphaFold structure for the top target (offline snapshot by default).
    try:
        from ..integrations.alphafold import AlphaFoldClient
        struct = AlphaFoldClient(prefer_offline=prefer_offline).fetch_structure(top)
        lead.structure = struct.to_dict()
        prov["alphafold"] = struct.source
    except Exception as exc:  # noqa: BLE001
        _log.debug("AlphaFold fetch failed for %s: %r", top, exc)
        lead.structure = {"gene_symbol": top, "error": "AlphaFold unavailable"}
        prov["alphafold"] = "unavailable"

    # Repurposing candidates for the top target (curated snapshot by default).
    try:
        from ..integrations.gnn_llm import RepurposingEngine
        cands = RepurposingEngine().rank_compounds(top, top_n=top_n_compounds)
        lead.repurposing = [c.to_dict() for c in cands]
        prov["repurposing"] = cands[0].source if cands else "offline_snapshot"
    except Exception as exc:  # noqa: BLE001
        _log.debug("repurposing failed for %s: %r", top, exc)
        prov["repurposing"] = "unavailable"

    lead.wet_lab_experiment = _ORGANOID_READOUT.get(mech, "")
    lead.status = "translated"
    lead.provenance = prov
    return lead.to_dict()
