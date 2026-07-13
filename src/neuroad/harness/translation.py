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
      -> STRING      the top target's ranked PPI hub partners (interaction
                     evidence, the honest stand-in for AF3 complex folding)
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
import os
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
        "(CRISPRi) and read out Aβ42/40 ratio by MSD at 6 weeks; "
        "kill if it does not move beyond vehicle ±2SD."
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

#: Fluid-biomarker anchor the researcher CHOSE -> mechanism key. The chosen
#: anchor ROUTES the molecular follow-up, overriding cohort-dominance routing.
#: amyloid and p-tau217 are the A and T poles of the SAME amyloid-cascade axis
#: (shared candidate-gene set on purpose); GFAP is glial, NfL is vascular.
#: (Mirrors the 4-entry map in claude.bridge._route — kept inline both places to
#: avoid a bridge<->translation import cycle.)
_ANCHOR_MECHANISM = {
    "amyloid": "amyloid_cascade",
    "p_tau217": "amyloid_cascade",
    "gfap": "glial",
    "nfl": "vascular",
}

#: Anchor -> the congruent LEAD gene to emphasise WITHIN the mechanism's gene set,
#: WITHOUT reordering the PI4AD-ranked panel (which stays fixed by each gene's
#: PI4AD priority). Only meaningful where two anchors share a mechanism: amyloid
#: leads on APP (the A pole), p-tau217 on MAPT (the tau/T pole). Both genes
#: already live in MECHANISM_GENES["amyloid_cascade"]. gfap/nfl carry no override
#: and fall through to their mechanism's PI4AD-top gene.
_ANCHOR_LEAD = {
    "amyloid": "APP",
    "p_tau217": "MAPT",
    "gfap": "TREM2",
}

#: Anchor -> a concrete, falsifiable organoid readout keyed to the ANCHORED
#: biomarker. Falls back to the mechanism readout when no anchor is supplied.
_ANCHOR_READOUT = {
    "amyloid": _ORGANOID_READOUT["amyloid_cascade"],
    "p_tau217": (
        "In Tanzi-style 3D human neural organoids, knock down the lead target "
        "(CRISPRi) and read out p-tau217 + total-tau by MSD with AT8 neuritic-tau "
        "burden at 6 weeks; kill if tau species do not move beyond vehicle ±2SD."
    ),
    "gfap": _ORGANOID_READOUT["glial"],
    "nfl": _ORGANOID_READOUT["vascular"],
}


def _anchor_readout(anchor: Optional[str], mechanism: str, lead_gene: str = "") -> str:
    """The organoid readout for the chosen anchor, else the mechanism default.

    When ``lead_gene`` is the real routed lead (``top_target`` / ``_ANCHOR_LEAD``
    / ``ranked_targets[0].gene``), name it in place of the generic "the lead
    target" wording. Falls back to the generic wording when no lead is known yet
    (e.g. no PI4AD-ranked target) — it never invents a gene.
    """
    text = _ANCHOR_READOUT.get(anchor, "") if anchor else ""
    text = text or _ORGANOID_READOUT.get(mechanism, "")
    if lead_gene:
        text = text.replace("the lead target", lead_gene)
    return text


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
    network_hubs: list[dict] = field(default_factory=list)     # STRING-RWR hubs
    top_target: str = ""
    structure: dict = field(default_factory=dict)              # AlphaFold summary
    repurposing: list[dict] = field(default_factory=list)      # GNN/LLM compounds
    biomarker_fusion: dict = field(default_factory=dict)       # L3 multimodal fusion
    signal_grounding: dict = field(default_factory=dict)       # attentive-probe LOO attribution
    cross_attention_fusion: dict = field(default_factory=dict)  # L3 real multi-head cross-attention fusion
    target_druggability: list[dict] = field(default_factory=list)  # L6 fused druggability ranking
    pathway_enrichment: list[dict] = field(default_factory=list)   # L5 ORA pathway over-representation
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
            "network_hubs": list(self.network_hubs),
            "top_target": self.top_target,
            "structure": dict(self.structure),
            "repurposing": list(self.repurposing),
            "biomarker_fusion": dict(self.biomarker_fusion),
            "signal_grounding": dict(self.signal_grounding),
            "cross_attention_fusion": dict(self.cross_attention_fusion),
            "target_druggability": list(self.target_druggability),
            "pathway_enrichment": list(self.pathway_enrichment),
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


def _rank_targets(mechanism: str, *, prefer_offline: bool = True,
                  method: str = "pi4ad") -> list[dict]:
    """Rank the mechanism's candidate genes (offline snapshot by default).

    ``method`` selects the signal set (opt-in, DEFAULT is unchanged for back-compat):

      * ``"pi4ad"`` (default) — single-signal PI4AD priority; each row keeps its
        ``priority_score``/``rank``/``source`` provenance. Unchanged behavior.
      * ``"composite"`` — the transparent 4-signal COMPOSITE (PI4AD priority +
        Open Targets non-genetic held-out + STRING-RWR centrality + AlphaFold
        pLDDT), shared with ``scripts/rank_candidates.py`` via ``harness.ranking``.
        Each row keeps ``composite_score`` + every raw per-signal value + a
        ``source`` stamp. Offline-first; if the composite yields nothing it falls
        back to the single-signal PI4AD path so the caller always gets a ranking.
    """
    if method == "composite":
        try:
            from . import ranking
            rows = ranking.composite_targets(mechanism, prefer_offline=prefer_offline)
            if rows:
                return rows
            _log.debug("composite ranking empty for %s; falling back to PI4AD",
                       mechanism)
        except Exception as exc:  # noqa: BLE001
            _log.debug("composite ranking failed for %s, falling back to PI4AD: %r",
                       mechanism, exc)
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


def _network_hubs(mechanism: str, *, prefer_offline: bool = True) -> list[dict]:
    """STRING-RWR hubs surfaced by propagating the mechanism's seed genes.

    Additive decision-support: after PI4AD's static ranking, seed a random-walk-
    with-restart over the bundled STRING v12.0 subgraph with this mechanism's
    ``MECHANISM_GENES`` (e.g. glial -> the TREM2/APOE/CLU/MAPK1/HRAS Ras cluster)
    and return the NON-seed network hubs the propagation lights up, each stamped
    with its ``source`` and ``propagated_score``. Offline-first and fully wrapped:
    any failure degrades to ``[]`` so it can NEVER change existing behavior."""
    genes = MECHANISM_GENES.get(mechanism) or MECHANISM_GENES["amyloid_cascade"]
    try:
        from ..integrations.pi4ad import propagate_hits
        nodes = propagate_hits(genes, prefer_offline=prefer_offline)
        return [n.to_dict() for n in nodes if n.is_hub]
    except Exception as exc:  # noqa: BLE001
        _log.debug("network propagation failed for %s: %r", mechanism, exc)
        return []


def _biomarker_fusion(df: Optional[pd.DataFrame], *,
                      prefer_offline: bool = True) -> dict:
    """L3 multimodal fusion: per-subject Abeta/tau PET status over the cohort.

    Runs the Jasodanand2025 multimodal fusion transformer (vkola-lab/ncomms2025)
    over the contract rows and summarizes the predicted amyloid/tau positivity —
    molecular-pathology corroboration for the promoted survivor's mechanism. The
    REAL model is used when its checkpoint is resolvable (``NCOMMS2025_CKPT`` /
    ``NCOMMS2025_REPO`` env, or a clone when ``NEUROAD_REAL_FUSION=1``); otherwise
    it degrades HONESTLY to the transparent hand-set surrogate, stamping which
    ran. Fully wrapped and additive: any failure returns ``{}`` and never
    perturbs the rest of the chain. ``prefer_offline`` keeps it network-free.

    Distinct from the deterministic gauntlet's plasma anchor: this is a fused
    multi-feature (plasma + volumetric + demographic) per-subject predictor, the
    diagram's L3 box, surfaced as evidence — never a referee kill/promote gate.
    """
    if df is None or len(df) == 0:
        return {}
    try:
        from ..integrations.multimodal_transformer import BiomarkerFusionPredictor
        import numpy as _np
        allow_clone = (not prefer_offline
                       and os.environ.get("NEUROAD_REAL_FUSION") == "1")
        predictor = BiomarkerFusionPredictor.from_pretrained(allow_clone=allow_clone)
        feature_cols = [c for c in
                        ("p_tau217", "gfap", "nfl", "hippocampal_volume",
                         "age", "apoe4", "sex")
                        if c in df.columns]
        if not feature_cols:
            return {}
        a_probs: list[float] = []
        t_probs: list[float] = []
        model = source = ""
        used: set[str] = set()
        for _, row in df[feature_cols].iterrows():
            pred = predictor.predict(row)
            a_probs.append(float(pred.abeta_prob))
            t_probs.append(float(pred.tau_prob))
            model, source = pred.model, pred.source
            used.update(pred.features_used)
        n = len(a_probs)
        if n == 0:
            return {}
        a_arr = _np.asarray(a_probs)
        t_arr = _np.asarray(t_probs)
        return {
            "n_subjects": int(n),
            "mean_abeta_prob": round(float(a_arr.mean()), 4),
            "mean_tau_prob": round(float(t_arr.mean()), 4),
            "abeta_positive_rate": round(float((a_arr >= 0.5).mean()), 4),
            "tau_positive_rate": round(float((t_arr >= 0.5).mean()), 4),
            "model": model,
            "source": source,
            "features_used": sorted(used),
            "note": (
                ("real vkola-lab/ncomms2025 multimodal fusion transformer predicted "
                 "amyloid/tau PET positivity over the cohort — the L3 fusion box; "
                 "surfaced as pathology corroboration, NOT a referee gate")
                if source == "live" else
                ("offline surrogate logistic (hand-set coefficients, NOT fitted) "
                 "estimated amyloid/tau PET positivity over the cohort — the L3 fusion "
                 "box; the real vkola-lab/ncomms2025 multimodal transformer is a "
                 "wired-ready seam NOT run by default (needs torch + GPU + gated weights "
                 "via NEUROAD_REAL_FUSION=1); surfaced as pathology corroboration, NOT a "
                 "referee gate")
            ),
        }
    except Exception as exc:  # noqa: BLE001
        _log.debug("biomarker fusion failed: %r", exc)
        return {}


def _cross_attention_fusion(df: Optional[pd.DataFrame]) -> dict:
    """L3 real multi-head cross-attention feature fusion over the cohort.

    Runs :func:`integrations.cross_attention.cross_attention_fusion` — genuine
    bidirectional imaging<->plasma scaled-dot-product multi-head cross-attention,
    classified by the IDENTICAL leakage-honest ``probe.auc_ci_perm`` head used by
    the referee (site-disjoint CV, PCA-in-fold, bootstrap CI, permutation null).
    Surfaced as decision-support: per-view AUC/CI/delta + a CI-honest verdict that
    never overclaims (it says so plainly when fusion does not beat the best single
    modality). Fully offline (numpy only) and additive: any failure returns ``{}``
    and never perturbs the rest of the chain. Opt-in because it runs a full CV
    sweep (~seconds), so it never slows the default/test path."""
    if df is None or len(df) == 0:
        return {}
    try:
        from ..integrations.cross_attention import cross_attention_fusion
        return cross_attention_fusion(df).to_dict()
    except Exception as exc:  # noqa: BLE001
        _log.debug("cross-attention fusion failed: %r", exc)
        return {}


def _target_druggability(mechanism: str, *, prefer_offline: bool = True) -> list[dict]:
    """L6 fused, explainable druggability ranking over the mechanism's candidates.

    Ranks ``MECHANISM_GENES[mechanism]`` by :func:`integrations.targeting.
    druggability_ranking` — a transparent weighted-mean over ONLY the present
    components (AlphaFold pLDDT + committed Boltz complex/ligand evidence + PI4AD
    priority), weights renormalized over the present subset, absent components kept
    honest (value None, never a fabricated 0.0). Offline-first and additive: any
    failure degrades to ``[]`` and never perturbs the chain. NOT a trained
    druggability model and NOT new folding — decision-support only."""
    genes = MECHANISM_GENES.get(mechanism) or MECHANISM_GENES["amyloid_cascade"]
    try:
        from ..integrations.targeting import druggability_ranking
        return druggability_ranking(list(genes), prefer_offline=prefer_offline)
    except Exception as exc:  # noqa: BLE001
        _log.debug("target druggability failed for %s: %r", mechanism, exc)
        return []


def _pathway_enrichment(mechanism: str, *, prefer_offline: bool = True) -> list[dict]:
    """L5 offline ORA pathway over-representation for the mechanism's candidates.

    Runs :func:`integrations.pathway_enrichment.enrich_mechanism` — a genuine
    one-sided hypergeometric over-representation test of ``MECHANISM_GENES`` against
    the hand-curated AD pathway snapshot, with BH-FDR q-values and full provenance
    (``snapshot_source='ad_pathway_snapshot_v1'``, background/pathway/query sizes so
    the snapshot-relative p-value is interpretable). Offline (scipy lazy-imported)
    and additive: degrades to ``[]`` on any failure and never perturbs the chain."""
    try:
        from ..integrations.pathway_enrichment import enrich_mechanism
        rows = enrich_mechanism(mechanism, prefer_offline=prefer_offline)
        return [r.to_dict() for r in rows]
    except Exception as exc:  # noqa: BLE001
        _log.debug("pathway enrichment failed for %s: %r", mechanism, exc)
        return []


def translate(
    mechanism: str,
    df: Optional[pd.DataFrame] = None,
    *,
    anchor: Optional[str] = None,
    prefer_offline: bool = True,
    top_n_compounds: int = 5,
    include_grounding: bool = False,
    include_cross_attention: bool = False,
    include_targeting: bool = False,
    include_pathways: bool = False,
) -> dict:
    """Chain PI4AD -> AlphaFold -> repurposing off a promoted survivor's mechanism.

    Offline-first and exception-safe: any adapter failure degrades that leg to an
    empty/annotated result rather than raising. Returns a serializable dict
    (``TranslationLead.to_dict``). ``prefer_offline=True`` keeps the whole chain
    on bundled, provenance-labeled snapshots (no network) — the referee's default.

    ``anchor`` is the fluid biomarker the researcher CHOSE (amyloid / p_tau217 /
    gfap / nfl). When given it (1) ROUTES the mechanism (overriding the passed
    ``mechanism``), (2) selects the anchor-congruent LEAD gene to emphasise —
    WITHOUT reordering the PI4AD-ranked panel — and (3) picks an organoid readout
    keyed to that biomarker. It never manufactures a ranking signal or reorders
    ``ranked_targets``; the anchor only influences routing + display emphasis.
    """
    anc = (str(anchor).strip().lower() or None) if anchor else None
    # The chosen anchor routes the mechanism (overrides cohort-dominance routing).
    if anc and anc in _ANCHOR_MECHANISM:
        mechanism = _ANCHOR_MECHANISM[anc]
    mech = mechanism if mechanism in MECHANISM_GENES else "amyloid_cascade"
    lead = TranslationLead(mechanism=mech, dominant_biomarker=_dominant_biomarker(df))

    ranked = _rank_targets(mech, prefer_offline=prefer_offline)
    lead.ranked_targets = ranked
    # Additive network prior: STRING-RWR hubs around the mechanism's seed genes.
    # Wrapped inside the helper so a propagation failure degrades to [] and never
    # perturbs the existing PI4AD/AlphaFold/repurposing chain below.
    lead.network_hubs = _network_hubs(mech, prefer_offline=prefer_offline)
    scored = [r for r in ranked if r.get("priority_score") is not None]
    if not scored:
        lead.status = "no PI4AD-ranked target for this mechanism"
        lead.wet_lab_experiment = _anchor_readout(anc, mech)
        lead.provenance = {"pi4ad": ranked[0]["source"] if ranked else "none",
                           "anchor": anc or ""}
        return lead.to_dict()

    # Anchor-congruent LEAD: emphasise the anchor's gene (amyloid->APP,
    # p_tau217->MAPT) when it is in the scored set, else the mechanism's PI4AD-top
    # gene. This shifts the wet-lab lead + structure + repurposing legs to the
    # tau/amyloid pole the researcher anchored on; it does NOT reorder the
    # PI4AD-ranked panel (ranked_targets stays priority-sorted above).
    top = scored[0]["gene"]
    if anc and _ANCHOR_LEAD.get(anc):
        cand = _ANCHOR_LEAD[anc]
        if any(r.get("gene") == cand for r in scored):
            top = cand
    lead.top_target = top
    prov = {"pi4ad": scored[0].get("source", "offline_snapshot"), "anchor": anc or ""}

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

    # STRING protein-protein interaction evidence for the top target's hub
    # partners — the honest stand-in for AlphaFold3 de-novo complex folding
    # (which stays intentionally unwired). Interaction evidence, NOT folding.
    try:
        from ..integrations.string_ppi import StringPPIClient, EVIDENCE_LABEL
        sc = StringPPIClient(prefer_offline=prefer_offline)
        partners = sc.interaction_partners(top)
        src = partners[0].source if partners else "offline_snapshot"
        lead.structure["interaction_evidence"] = {
            "note": EVIDENCE_LABEL,
            "top_target": top,
            "partners": [p.to_dict() for p in partners],
            "source": src,
        }
        prov["string"] = src
    except Exception as exc:  # noqa: BLE001
        _log.debug("STRING interaction evidence failed for %s: %r", top, exc)
        prov["string"] = "unavailable"

    # Boltz-2 molecular targeting — a REAL, open (MIT), AlphaFold3-class complex +
    # binding-affinity predictor (unlike STRING interaction evidence above, and
    # unlike gated AF3 which stays unwired). Boltz needs a GPU we don't have here,
    # so the client degrades HONESTLY: it attaches a `boltz_targeting` field ONLY
    # when a REAL precomputed result exists in the committed snapshot (produced by
    # scripts/boltz_fold_colab.py). When it would be 'deferred' (no GPU run yet),
    # the field is OMITTED entirely — this leg never fabricates numbers and never
    # perturbs existing behavior. Fully wrapped: any failure degrades to omission.
    _boltz_client = None
    try:
        from ..integrations.boltz import BoltzClient
        _boltz_client = BoltzClient(prefer_offline=prefer_offline)
        _pe = lead.structure.get("interaction_evidence", {}).get("partners", [])
        # Query partners in STRING-rank order and surface the FIRST one that has a
        # REAL (precomputed/GPU) Boltz complex result. This stays honest — it never
        # fabricates — while not being brittle to whether the single highest-ranked
        # partner happens to have been folded yet (some, e.g. very large receptors,
        # may be deferred). A deferred-for-all case omits the field entirely.
        bt = None
        for _p in _pe:
            gene_b = _p.get("gene_b", "")
            if not gene_b:
                continue
            cand = _boltz_client.predict_complex(top, gene_b)
            if cand.status == "predicted":  # a REAL precomputed/GPU result exists
                bt = cand
                break
        if bt is not None:
            lead.structure["boltz_targeting"] = bt.to_dict()
            prov["boltz"] = bt.source
        else:
            prov["boltz"] = "deferred"  # honest: no folded partner yet, field omitted
    except Exception as exc:  # noqa: BLE001
        _log.debug("Boltz targeting failed for %s: %r", top, exc)
        prov["boltz"] = "unavailable"

    # Repurposing candidates for the top target (curated snapshot by default).
    try:
        from ..integrations.gnn_llm import RepurposingEngine
        cands = RepurposingEngine().rank_compounds(top, top_n=top_n_compounds)
        cand_dicts = [c.to_dict() for c in cands]
        # Boltz-2 target+ligand binding affinity for each repurposing compound
        # against its canonical target (the "optimize compound / repurposing fits"
        # step). Attach the REAL predicted affinity when a GPU run exists in the
        # snapshot; otherwise leave the candidate untouched — never fabricate a fit.
        if _boltz_client is not None:
            for cd in cand_dicts:
                tgene = cd.get("target_gene", "")
                comp = cd.get("compound", "")
                if tgene and comp:
                    aff = _boltz_client.predict_affinity(tgene, ligand_id=comp)
                    if aff.status == "predicted":
                        cd["boltz_affinity"] = aff.to_dict()
                        prov["boltz_affinity"] = aff.source
        lead.repurposing = cand_dicts
        prov["repurposing"] = cands[0].source if cands else "offline_snapshot"
    except Exception as exc:  # noqa: BLE001
        _log.debug("repurposing failed for %s: %r", top, exc)
        prov["repurposing"] = "unavailable"

    # L3 multimodal fusion transformer — per-subject Abeta/tau over the cohort,
    # surfaced as molecular-pathology corroboration for the mechanism (additive).
    lead.biomarker_fusion = _biomarker_fusion(df, prefer_offline=prefer_offline)
    if lead.biomarker_fusion:
        prov["biomarker_fusion"] = lead.biomarker_fusion.get("source", "unavailable")

    # Attentive-probe interpretable grounding — which named feature drives the
    # AD-vs-CN signal (embedding vs plasma). Opt-in (a full CV sweep, ~seconds) so
    # it never slows the default/test path. Additive and fully wrapped.
    if include_grounding and df is not None:
        try:
            from ..attentive_probe import feature_grounding
            lead.signal_grounding = feature_grounding(df, "dx_binary")
            if lead.signal_grounding:
                prov["signal_grounding"] = "attentive_probe_loo"
        except Exception as exc:  # noqa: BLE001
            _log.debug("signal grounding failed: %r", exc)

    # L3 real multi-head cross-attention feature fusion — molecule/mechanism lead.
    # Opt-in (a full leakage-honest CV sweep, ~seconds); additive and fully wrapped.
    if include_cross_attention and df is not None:
        lead.cross_attention_fusion = _cross_attention_fusion(df)
        if lead.cross_attention_fusion:
            prov["cross_attention_fusion"] = lead.cross_attention_fusion.get(
                "source", "cross_attention_fusion")

    # L6 fused druggability ranking over the mechanism's candidate genes — molecule
    # lead. Opt-in (reads AlphaFold/Boltz/PI4AD snapshots); additive, degrades to [].
    if include_targeting:
        lead.target_druggability = _target_druggability(
            mech, prefer_offline=prefer_offline)
        if lead.target_druggability:
            prov["target_druggability"] = lead.target_druggability[0].get(
                "source", "targeting_fusion")

    # L5 ORA pathway over-representation for the mechanism's candidates — mechanism
    # corroboration. Opt-in (hypergeometric over the curated snapshot); degrades to [].
    if include_pathways:
        lead.pathway_enrichment = _pathway_enrichment(
            mech, prefer_offline=prefer_offline)
        if lead.pathway_enrichment:
            prov["pathway_enrichment"] = lead.pathway_enrichment[0].get(
                "snapshot_source", "ad_pathway_snapshot_v1")

    lead.wet_lab_experiment = _anchor_readout(anc, mech, lead.top_target)
    lead.status = "translated"
    lead.provenance = prov
    return lead.to_dict()
