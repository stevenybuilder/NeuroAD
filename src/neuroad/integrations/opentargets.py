"""
opentargets — quantitative AD target-disease evidence (Open Targets Platform).

Replaces hand-set molecule-side priors with REAL, sourced numbers from the open
Open Targets Platform GraphQL API (no login): overall target-disease association
scores (0-1) with a datatype breakdown (genetic_association, known_drug/clinical,
literature, animal_model, ...), plus the approved/clinical drugs hitting a target.

OFFLINE / DETERMINISTIC CONTRACT: imports and runs with NO network and NO
credentials. Any live GraphQL call is wrapped with a short (<=15s) timeout and,
on ANY failure (no network, non-200, GraphQL error, malformed JSON), degrades to
a bundled snapshot (``data/opentargets_snapshot.json``, captured live 2026-07-10)
instead of raising. Every returned record is provenance-stamped: ``source`` is
"live" (a real Open Targets fetch) or "offline_snapshot" (bundled fallback) — a
fallback is NEVER dressed up as live data.

Disease id: Alzheimer's disease is ``MONDO_0004975``. The older ``EFO_0000249``
is stale and returns null from the current API (verified), so it is not used.

Directional note (both scores are real Open Targets association scores):
  * ``disease_targets`` uses the disease->target ranking
    (``disease.associatedTargets``), the canonical "top AD targets" list.
  * ``target_association`` reports the target->disease direct association
    (``target.associatedDiseases`` filtered to AD) for one gene, which also
    carries the target's known-drug detail. The two directions can differ
    slightly (indirect evidence propagation on the disease side); each method is
    self-consistent across the live and offline paths.

Only stdlib + requests are used. ``OPENTARGETS_API_URL`` (optional env var)
overrides the GraphQL endpoint for mirrors/testing.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants + bundled snapshot
# ---------------------------------------------------------------------------
#: Correct Alzheimer's-disease MONDO id (EFO_0000249 is stale -> null).
AD_DISEASE_ID = "MONDO_0004975"

_DEFAULT_API_URL = "https://api.platform.opentargets.org/api/v4/graphql"
_SNAPSHOT_PATH = Path(__file__).with_name("data") / "opentargets_snapshot.json"
_HTTP_TIMEOUT = 15  # seconds — short, so a live call never hangs the engine

#: gene symbol -> Ensembl gene id for the engine's AD targets (captured live from
#: the Open Targets search endpoint; target ids on the platform are ENSG...).
AD_TARGET_ENSEMBL: dict[str, str] = {
    "APP": "ENSG00000142192",
    "MAPT": "ENSG00000186868",
    "APOE": "ENSG00000130203",
    "PSEN1": "ENSG00000080815",
    "PSEN2": "ENSG00000143801",
    "BACE1": "ENSG00000186318",
    "TREM2": "ENSG00000095970",
    "HRAS": "ENSG00000174775",
    "MAPK1": "ENSG00000100030",
    "ESR1": "ENSG00000091831",
    "CLU": "ENSG00000120885",
    "BIN1": "ENSG00000136717",
}


def _api_url() -> str:
    """GraphQL endpoint; overridable via OPENTARGETS_API_URL for mirrors/testing."""
    return os.environ.get("OPENTARGETS_API_URL", _DEFAULT_API_URL).rstrip("/")


def _load_snapshot() -> dict:
    """Load the bundled deterministic snapshot (real Open Targets capture)."""
    try:
        with open(_SNAPSHOT_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Structured return
# ---------------------------------------------------------------------------


@dataclass
class TargetAssociation:
    """One target's quantitative Open Targets association with Alzheimer's disease.

    ``association_score`` is Open Targets' overall target-disease association on a
    real 0-1 scale (higher = stronger aggregate evidence). ``datatype_scores`` is
    the per-evidence-type breakdown (e.g. ``genetic_association``, ``clinical``,
    ``literature``, ``animal_model``, ``affected_pathway``, ``rna_expression``),
    each also 0-1. ``n_known_drugs`` is the count of approved/clinical drugs (and
    clinical candidates) hitting the target. ``source`` is the provenance stamp:
    "live" (real fetch) or "offline_snapshot" (bundled fallback).
    """
    gene: str
    ensembl_id: str
    association_score: float
    datatype_scores: dict[str, float]
    n_known_drugs: int
    source: str
    error: str = ""  # non-fatal note (e.g. why it fell back / no AD evidence)

    def to_dict(self) -> dict:
        return {
            "gene": self.gene,
            "ensembl_id": self.ensembl_id,
            "association_score": self.association_score,
            "datatype_scores": dict(self.datatype_scores),
            "n_known_drugs": self.n_known_drugs,
            "source": self.source,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------


class OpenTargetsClient:
    """Adapter over the Open Targets Platform GraphQL API, offline-first.

    Default (``prefer_offline=False``, keyless): each method best-effort hits the
    live GraphQL API and, on ANY failure, degrades to the bundled snapshot — never
    raising, always provenance-stamped. Set ``prefer_offline=True`` to skip the
    network entirely (deterministic, e.g. in tests).
    """

    def __init__(self, prefer_offline: bool = False, *,
                 timeout: int = _HTTP_TIMEOUT) -> None:
        self.prefer_offline = prefer_offline
        self.timeout = timeout
        self._snapshot = _load_snapshot()

    # -- low-level GraphQL -------------------------------------------------

    def _post(self, query: str) -> Optional[dict]:
        """POST a GraphQL query; return ``data`` dict or None on ANY failure.

        ``requests`` is a guaranteed dep but the call is fully wrapped so a
        missing network / non-200 / GraphQL error / bad JSON degrades to the
        snapshot rather than raising."""
        try:
            import requests  # imported here to keep module import-time clean
            resp = requests.post(_api_url(), json={"query": query},
                                 timeout=self.timeout)
            if resp.status_code != 200:
                return None
            payload = resp.json()
            if not isinstance(payload, dict) or payload.get("errors"):
                return None
            data = payload.get("data")
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    # -- resolution --------------------------------------------------------

    def resolve_ensembl(self, gene_symbol: str) -> Optional[str]:
        """Map a gene symbol to an Ensembl gene id (offline-first).

        Uses the bundled AD-target map + snapshot first; falls back to the live
        Open Targets search endpoint for arbitrary symbols when online. Returns
        None if it cannot be resolved."""
        if not gene_symbol:
            return None
        q = gene_symbol.strip()
        upper = q.upper()
        if upper in AD_TARGET_ENSEMBL:
            return AD_TARGET_ENSEMBL[upper]
        # Snapshot symbol -> ensembl (top associated targets + engine genes).
        for rec in self._snapshot.get("associated_targets", []):
            if str(rec.get("gene", "")).upper() == upper:
                return rec.get("ensembl_id")
        eng = self._snapshot.get("engine_genes", {})
        if upper in {k.upper() for k in eng}:
            for k, v in eng.items():
                if k.upper() == upper:
                    return v.get("ensembl_id")
        # Already an Ensembl-shaped id?
        if upper.startswith("ENSG"):
            return upper
        # Live search resolution (best-effort; None on any failure/offline).
        if not self.prefer_offline:
            return self._resolve_ensembl_live(q)
        return None

    def _resolve_ensembl_live(self, gene_symbol: str) -> Optional[str]:
        """Resolve a symbol to an Ensembl id via the live search endpoint."""
        safe = gene_symbol.replace('"', "")
        data = self._post(
            f'{{ search(queryString:"{safe}", entityNames:["target"]){{ '
            f'hits{{ id name entity }} }} }}')
        if not data:
            return None
        hits = (data.get("search") or {}).get("hits") or []
        want = gene_symbol.strip().upper()
        for h in hits:
            if str(h.get("name", "")).upper() == want and str(h.get("id", "")).startswith("ENSG"):
                return h["id"]
        for h in hits:  # fall back to the top target hit
            if str(h.get("id", "")).startswith("ENSG"):
                return h["id"]
        return None

    # -- public API: disease -> targets ------------------------------------

    def disease_targets(self, top_n: int = 50) -> list[TargetAssociation]:
        """Top AD-associated targets, ranked by overall association score (desc).

        Live path queries ``disease.associatedTargets`` for AD (MONDO_0004975)
        and best-effort enriches each with its known-drug count; on any failure it
        degrades to the bundled snapshot. Every record is provenance-stamped."""
        if top_n <= 0:
            return []
        if not self.prefer_offline:
            live = self._disease_targets_live(top_n)
            if live:
                return live
        return self._disease_targets_offline(top_n)

    def _disease_targets_live(self, top_n: int) -> list[TargetAssociation]:
        size = max(1, min(int(top_n), 200))
        data = self._post(
            f'{{ disease(efoId:"{AD_DISEASE_ID}"){{ id name '
            f'associatedTargets(page:{{index:0,size:{size}}}){{ count rows{{ '
            f'target{{ id approvedSymbol }} score datatypeScores{{ id score }} }} }} }} }}')
        dis = (data or {}).get("disease")
        if not dis:
            return []
        rows = (dis.get("associatedTargets") or {}).get("rows") or []
        out: list[TargetAssociation] = []
        for r in rows:
            tgt = r.get("target") or {}
            out.append(TargetAssociation(
                gene=tgt.get("approvedSymbol", "") or "",
                ensembl_id=tgt.get("id", "") or "",
                association_score=_clamp01(r.get("score")),
                datatype_scores={ds["id"]: _clamp01(ds.get("score"))
                                 for ds in (r.get("datatypeScores") or [])
                                 if ds.get("id")},
                n_known_drugs=0,
                source="live",
            ))
        self._enrich_drug_counts_live(out)
        out.sort(key=lambda t: t.association_score, reverse=True)
        return out[:top_n]

    def _enrich_drug_counts_live(self, targets: list[TargetAssociation]) -> None:
        """Best-effort: fill n_known_drugs via one aliased query. Never fatal."""
        ids = [t.ensembl_id for t in targets if t.ensembl_id]
        if not ids:
            return
        aliases = " ".join(
            f'a{i}: target(ensemblId:"{e}"){{ drugAndClinicalCandidates{{ count }} }}'
            for i, e in enumerate(ids))
        data = self._post("{ " + aliases + " }")
        if not data:
            return  # counts stay 0; ranking is unaffected
        by_ens = {e: i for i, e in enumerate(ids)}
        for t in targets:
            node = data.get(f"a{by_ens.get(t.ensembl_id, -1)}")
            if isinstance(node, dict):
                cnt = (node.get("drugAndClinicalCandidates") or {}).get("count")
                if isinstance(cnt, int):
                    t.n_known_drugs = cnt

    def _disease_targets_offline(self, top_n: int) -> list[TargetAssociation]:
        rows = self._snapshot.get("associated_targets", [])
        out = [TargetAssociation(
            gene=str(r.get("gene", "")),
            ensembl_id=str(r.get("ensembl_id", "")),
            association_score=_clamp01(r.get("association_score")),
            datatype_scores={k: _clamp01(v)
                             for k, v in (r.get("datatype_scores") or {}).items()},
            n_known_drugs=int(r.get("n_known_drugs") or 0),
            source="offline_snapshot",
        ) for r in rows]
        out.sort(key=lambda t: t.association_score, reverse=True)
        return out[:top_n]

    # -- public API: one target's AD association ---------------------------

    def target_association(self, gene_symbol: str) -> Optional[TargetAssociation]:
        """One gene's AD association score + evidence breakdown + drug count.

        Returns None only when the symbol cannot be resolved to a target at all
        (an honest "unknown", never a fabricated score). A resolvable target with
        no AD evidence returns a score of 0.0 with an explanatory ``error``."""
        if not gene_symbol:
            return None
        if not self.prefer_offline:
            live = self._target_association_live(gene_symbol)
            if live is not None:
                return live
        return self._target_association_offline(gene_symbol)

    def _target_association_live(self, gene_symbol: str) -> Optional[TargetAssociation]:
        ensembl = self.resolve_ensembl(gene_symbol)
        if not ensembl:
            return None
        data = self._post(
            f'{{ target(ensemblId:"{ensembl}"){{ approvedSymbol '
            f'associatedDiseases(Bs:["{AD_DISEASE_ID}"]){{ rows{{ score '
            f'datatypeScores{{ id score }} }} }} '
            f'drugAndClinicalCandidates{{ count }} }} }}')
        tgt = (data or {}).get("target")
        if not tgt:
            return None
        rows = (tgt.get("associatedDiseases") or {}).get("rows") or []
        n_drugs = int((tgt.get("drugAndClinicalCandidates") or {}).get("count") or 0)
        gene = tgt.get("approvedSymbol") or gene_symbol.strip().upper()
        if not rows:
            return TargetAssociation(
                gene=gene, ensembl_id=ensembl, association_score=0.0,
                datatype_scores={}, n_known_drugs=n_drugs, source="live",
                error="no Open Targets AD evidence for this target")
        row = rows[0]
        return TargetAssociation(
            gene=gene, ensembl_id=ensembl,
            association_score=_clamp01(row.get("score")),
            datatype_scores={ds["id"]: _clamp01(ds.get("score"))
                             for ds in (row.get("datatypeScores") or [])
                             if ds.get("id")},
            n_known_drugs=n_drugs, source="live")

    def _target_association_offline(self, gene_symbol: str) -> Optional[TargetAssociation]:
        upper = gene_symbol.strip().upper()
        # Prefer the engine-gene record (target-side score + drug detail).
        for k, v in self._snapshot.get("engine_genes", {}).items():
            if k.upper() == upper:
                return TargetAssociation(
                    gene=v.get("gene", k),
                    ensembl_id=str(v.get("ensembl_id", "")),
                    association_score=_clamp01(v.get("association_score")),
                    datatype_scores={dk: _clamp01(dv)
                                     for dk, dv in (v.get("datatype_scores") or {}).items()},
                    n_known_drugs=int(v.get("n_known_drugs") or 0),
                    source="offline_snapshot")
        # Otherwise fall back to the disease-side top-targets record if present.
        for r in self._snapshot.get("associated_targets", []):
            if str(r.get("gene", "")).upper() == upper:
                return TargetAssociation(
                    gene=str(r.get("gene", "")),
                    ensembl_id=str(r.get("ensembl_id", "")),
                    association_score=_clamp01(r.get("association_score")),
                    datatype_scores={dk: _clamp01(dv)
                                     for dk, dv in (r.get("datatype_scores") or {}).items()},
                    n_known_drugs=int(r.get("n_known_drugs") or 0),
                    source="offline_snapshot")
        return None

    # -- public API: known drugs -------------------------------------------

    def known_drugs(self, gene_symbol: str) -> list[dict]:
        """Approved/clinical drugs (and clinical candidates) hitting the target.

        Each dict has ``drug``, ``phase`` (max clinical stage), ``mechanism``,
        and ``drug_type``. Returns [] for an unknown/unresolvable gene."""
        if not gene_symbol:
            return []
        if not self.prefer_offline:
            live = self._known_drugs_live(gene_symbol)
            if live is not None:
                return live
        return self._known_drugs_offline(gene_symbol)

    def _known_drugs_live(self, gene_symbol: str) -> Optional[list[dict]]:
        ensembl = self.resolve_ensembl(gene_symbol)
        if not ensembl:
            return None
        data = self._post(
            f'{{ target(ensemblId:"{ensembl}"){{ drugAndClinicalCandidates{{ rows{{ '
            f'maxClinicalStage drug{{ name drugType '
            f'mechanismsOfAction{{ rows{{ mechanismOfAction }} }} }} }} }} }} }}')
        tgt = (data or {}).get("target")
        if not tgt:
            return None
        rows = (tgt.get("drugAndClinicalCandidates") or {}).get("rows") or []
        out: list[dict] = []
        seen: set[str] = set()
        for r in rows:
            drug = r.get("drug") or {}
            name = drug.get("name") or ""
            if not name or name in seen:
                continue
            seen.add(name)
            moa = (drug.get("mechanismsOfAction") or {}).get("rows") or []
            out.append({
                "drug": name,
                "phase": r.get("maxClinicalStage") or "",
                "mechanism": (moa[0].get("mechanismOfAction") if moa else "") or "",
                "drug_type": drug.get("drugType") or "",
            })
        return out

    def _known_drugs_offline(self, gene_symbol: str) -> list[dict]:
        upper = gene_symbol.strip().upper()
        for k, v in self._snapshot.get("engine_genes", {}).items():
            if k.upper() == upper:
                return [dict(d) for d in (v.get("known_drugs") or [])]
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp01(value: object) -> float:
    """Coerce a score to a float clamped to [0, 1]; 0.0 if not numeric."""
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


# ---------------------------------------------------------------------------
# Module-level convenience (thin wrapper for harness callers)
# ---------------------------------------------------------------------------


def ad_target_evidence(gene_symbol: str, *,
                       prefer_offline: bool = False) -> dict:
    """One gene's evidence-based AD profile: association + known drugs.

    Returns a dict with the target's AD ``association`` (TargetAssociation.to_dict,
    or None if unresolved) and its ``known_drugs`` list — the quantitative,
    sourced replacement for a hand-set molecule-side prior. Offline-safe."""
    client = OpenTargetsClient(prefer_offline=prefer_offline)
    assoc = client.target_association(gene_symbol)
    drugs = client.known_drugs(gene_symbol)
    return {
        "gene": gene_symbol.strip().upper() if gene_symbol else "",
        "disease": "Alzheimer disease",
        "disease_id": AD_DISEASE_ID,
        "association": assoc.to_dict() if assoc else None,
        "association_score": assoc.association_score if assoc else None,
        "known_drugs": drugs,
        "n_known_drugs": len(drugs),
        "source": assoc.source if assoc else "offline_snapshot",
    }


__all__ = [
    "AD_DISEASE_ID",
    "AD_TARGET_ENSEMBL",
    "OpenTargetsClient",
    "TargetAssociation",
    "ad_target_evidence",
]
