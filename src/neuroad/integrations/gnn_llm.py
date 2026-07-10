"""
GNN/LLM hybrid drug-repurposing adapter (TxGNN + PrimeKG + Claude evidence).

Given a prioritized target gene (or the disease string "Alzheimer disease"),
return ranked repurposing-candidate compounds each with a mechanistic evidence
note. Offline-first and deterministic by contract:

  * DEFAULT (ships, no network / no creds): rank a hand-curated snapshot of
    well-documented AD-repurposing hypotheses bundled at
    ``data/repurposing_snapshot.csv``. Every returned record is stamped
    ``source="offline_snapshot"``. The notes describe the HYPOTHESIS BASIS
    (mechanism + that a trial exists), never an efficacy claim.
  * OPTIONAL live path (off by default, gated on ``NEUROAD_ENABLE_TXGNN=1``):
    lazily import torch + TxGNN and score zero-shot indications over the TxGNN
    KG. Heavy (DGL 0.5.2 + a Google-Drive checkpoint), so any missing dep /
    checkpoint / error silently degrades to the snapshot — never raises.
  * OPTIONAL evidence synthesis: ``synthesize_evidence`` reuses the existing
    Claude bridge (``neuroad.claude._client.complete``) for a one-line
    rationale when ``ANTHROPIC_API_KEY`` is set; otherwise it returns a
    deterministic template built from the mechanism note. The bridge itself
    already degrades offline — this module never reimplements Claude calling.

Disease vocabulary (the MONDO "Alzheimer disease" group node ids) is copied
from the tiny in-repo TxGNN table
``TxGNN/txgnn/data_splits/kg_grouped_diseases.csv`` (MIT), not fetched at
runtime. TxGNN ships no precomputed repurposing predictions, so the snapshot is
curated from published AD-repurposing literature / registered trials, not
captured from a model.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

# Canonical MONDO node ids for the "Alzheimer disease" group, copied verbatim
# from TxGNN/txgnn/data_splits/kg_grouped_diseases.csv (group_name == "Alzheimer
# disease"). 4975 is the base MONDO Alzheimer-disease node; the rest are typed
# subtypes (AD 1/3/4/9). Bundled inline so we never hit the live Dataverse KG.
ALZHEIMER_MONDO_NODES: dict[str, str] = {
    "4975": "Alzheimer disease",
    "11913": "Alzheimer disease 3",
    "7088": "Alzheimer disease type 1",
    "11743": "Alzheimer disease 4",
    "12153": "Alzheimer disease 9",
}

#: Env var that opts into the heavy, optional live TxGNN inference path. Absent
#: or not "1" => the default offline snapshot path (the shipping behaviour).
ENABLE_TXGNN_ENV = "NEUROAD_ENABLE_TXGNN"

_DATA_DIR = Path(__file__).resolve().parent / "data"
_SNAPSHOT_CSV = _DATA_DIR / "repurposing_snapshot.csv"


@dataclass
class RepurposingCandidate:
    """A ranked drug-repurposing hypothesis for an AD target/disease node.

    ``evidence_strength`` is a curated 0-1 prior confidence in the *hypothesis*
    (mechanistic plausibility + that a trial exists), NOT a claim of clinical
    benefit. ``source`` is the provenance stamp: "offline_snapshot" (bundled
    curated table, the default) or "live" (real TxGNN inference).
    """
    compound: str
    target_gene: str
    mechanism_note: str
    evidence_strength: float
    source: str
    disease: str = "Alzheimer disease"
    related_genes: list[str] = field(default_factory=list)
    disease_node_ids: list[str] = field(default_factory=list)
    trial_ref: str = ""
    #: One-line rationale filled in by ``synthesize_evidence`` (empty until then).
    rationale: str = ""
    #: How ``rationale`` was produced: "" | "live_llm" | "offline_template".
    rationale_source: str = ""

    def to_dict(self) -> dict:
        return {
            "compound": self.compound,
            "target_gene": self.target_gene,
            "related_genes": list(self.related_genes),
            "mechanism_note": self.mechanism_note,
            "evidence_strength": self.evidence_strength,
            "disease": self.disease,
            "disease_node_ids": list(self.disease_node_ids),
            "trial_ref": self.trial_ref,
            "rationale": self.rationale,
            "rationale_source": self.rationale_source,
            "source": self.source,
        }


def resolve_disease_nodes(target: str) -> dict[str, str]:
    """Map a target/disease string to canonical MONDO Alzheimer node ids.

    Returns the full Alzheimer group when the target names the disease (or is a
    known AD-associated gene present in the snapshot); an empty dict otherwise.
    Offline and deterministic — reads only the bundled ``ALZHEIMER_MONDO_NODES``.
    """
    t = (target or "").strip().lower()
    if not t:
        return {}
    if "alzheimer" in t or t in {"ad", "mondo:0004975", "4975"}:
        return dict(ALZHEIMER_MONDO_NODES)
    # Gene targets that appear in the AD snapshot resolve to the AD group too.
    genes = _snapshot_gene_index()
    if t in genes:
        return dict(ALZHEIMER_MONDO_NODES)
    return {}


@lru_cache(maxsize=1)
def _load_snapshot() -> pd.DataFrame:
    """Load and normalise the bundled curated snapshot (cached)."""
    df = pd.read_csv(_SNAPSHOT_CSV, dtype=str).fillna("")
    df["evidence_strength"] = pd.to_numeric(
        df["evidence_strength"], errors="coerce"
    ).fillna(0.0)
    return df


@lru_cache(maxsize=1)
def _snapshot_gene_index() -> frozenset[str]:
    """Lower-cased set of every gene named in the snapshot (target + related)."""
    df = _load_snapshot()
    genes: set[str] = set()
    for _, row in df.iterrows():
        genes.add(str(row["target_gene"]).strip().lower())
        for g in _split_genes(row.get("related_genes", "")):
            genes.add(g.lower())
    genes.discard("")
    return frozenset(genes)


def _split_genes(raw: str) -> list[str]:
    return [g.strip() for g in str(raw).split(";") if g.strip()]


class RepurposingEngine:
    """Rank AD drug-repurposing candidates for a target gene or disease.

    Default path is fully offline (curated snapshot). The optional live TxGNN
    path is gated on ``NEUROAD_ENABLE_TXGNN=1`` and degrades to the snapshot on
    any missing dependency, missing checkpoint, or error.
    """

    def __init__(self, snapshot_path: Optional[str | Path] = None) -> None:
        self._snapshot_path = Path(snapshot_path) if snapshot_path else _SNAPSHOT_CSV

    # -- snapshot access ---------------------------------------------------
    def _frame(self) -> pd.DataFrame:
        if self._snapshot_path == _SNAPSHOT_CSV:
            return _load_snapshot()
        df = pd.read_csv(self._snapshot_path, dtype=str).fillna("")
        df["evidence_strength"] = pd.to_numeric(
            df["evidence_strength"], errors="coerce"
        ).fillna(0.0)
        return df

    def _row_to_candidate(self, row: pd.Series, source: str) -> RepurposingCandidate:
        node_ids = [n.strip() for n in str(row.get("disease_node_ids", "")).split(";")
                    if n.strip()]
        return RepurposingCandidate(
            compound=str(row["compound"]).strip(),
            target_gene=str(row["target_gene"]).strip(),
            mechanism_note=str(row["mechanism_note"]).strip(),
            evidence_strength=float(row["evidence_strength"]),
            source=source,
            disease=str(row.get("mondo_group", "Alzheimer disease")).strip()
                    or "Alzheimer disease",
            related_genes=_split_genes(row.get("related_genes", "")),
            disease_node_ids=node_ids,
            trial_ref=str(row.get("trial_ref", "")).strip(),
        )

    # -- public API --------------------------------------------------------
    def rank_compounds(self, target: str, top_n: int = 10) -> list[RepurposingCandidate]:
        """Return up to ``top_n`` candidates for ``target``, best-first.

        ``target`` may be a disease string ("Alzheimer disease") or a gene
        symbol (e.g. "GFAP", "MAPT", "APOE"). Gene targets match on the
        candidate's primary or related genes; if no gene matches, the full
        AD disease-level list is returned (so the caller always gets a ranked
        hypothesis set) with the mechanism note flagged as disease-level.
        Ranked by curated ``evidence_strength`` descending.
        """
        # Optional live TxGNN path — off unless explicitly enabled, and never fatal.
        live = self._try_txgnn(target, top_n)
        if live is not None:
            return live[:max(0, top_n)]

        df = self._frame()
        t = (target or "").strip().lower()
        disease_level = (not t) or ("alzheimer" in t) or t in {"ad", "4975"}

        rows: list[pd.Series] = []
        matched_gene = False
        if not disease_level:
            for _, row in df.iterrows():
                genes = {str(row["target_gene"]).strip().lower()}
                genes.update(g.lower() for g in _split_genes(row.get("related_genes", "")))
                if t in genes:
                    rows.append(row)
            matched_gene = bool(rows)

        if disease_level or not matched_gene:
            rows = [row for _, row in df.iterrows()]

        candidates = [self._row_to_candidate(r, source="offline_snapshot") for r in rows]

        # If a gene was asked for but nothing matched, be honest: these are
        # disease-level hypotheses, not target-specific ones.
        if not disease_level and not matched_gene and target:
            for c in candidates:
                c.mechanism_note = (
                    f"[disease-level hypothesis; no direct {target} link in snapshot] "
                    + c.mechanism_note
                )

        candidates.sort(key=lambda c: c.evidence_strength, reverse=True)
        return candidates[:max(0, top_n)]

    def synthesize_evidence(self, candidate: RepurposingCandidate) -> str:
        """Return a one-line rationale for ``candidate`` (also stored on it).

        Uses the existing Claude bridge when ``ANTHROPIC_API_KEY`` is set;
        otherwise returns a deterministic template built from the mechanism
        note. Never raises: any bridge failure falls back to the template.
        ``candidate.rationale`` / ``candidate.rationale_source`` are updated.
        """
        text = self._template_rationale(candidate)
        source = "offline_template"

        try:
            from ..claude import _client  # lazy: offline path stays dependency-free
            if getattr(_client, "USING_LIVE_API", False):
                system = (
                    "You synthesize one-line drug-repurposing rationales for "
                    "Alzheimer disease. State the mechanistic hypothesis basis "
                    "only; never assert clinical efficacy. One sentence, hedged."
                )
                prompt = (
                    f"Compound: {candidate.compound}\n"
                    f"Primary target: {candidate.target_gene}\n"
                    f"Related genes: {', '.join(candidate.related_genes) or 'n/a'}\n"
                    f"Disease: {candidate.disease}\n"
                    f"Mechanism note: {candidate.mechanism_note}\n"
                    f"Trial reference: {candidate.trial_ref or 'n/a'}\n"
                    "Write ONE hedged sentence describing the repurposing "
                    "hypothesis basis (no efficacy claim)."
                )
                out = _client.complete(system, prompt)
                if getattr(_client, "LAST_CALL_LIVE", False) and isinstance(out, str):
                    stripped = out.strip()
                    if stripped:
                        text = stripped
                        source = "live_llm"
        except Exception:
            # Any import/transport error keeps the deterministic template.
            pass

        candidate.rationale = text
        candidate.rationale_source = source
        return text

    @staticmethod
    def _template_rationale(candidate: RepurposingCandidate) -> str:
        trial = f" A registered trial exists ({candidate.trial_ref})." if candidate.trial_ref else ""
        return (
            f"Repurposing hypothesis: {candidate.compound} is proposed for "
            f"{candidate.disease} via {candidate.target_gene} "
            f"(prior {candidate.evidence_strength:.2f}). {candidate.mechanism_note}"
            f"{trial} This is a mechanistic hypothesis basis, not evidence of "
            "clinical benefit."
        )

    # -- optional live path (documented, off by default) -------------------
    def _try_txgnn(self, target: str, top_n: int) -> Optional[list[RepurposingCandidate]]:
        """Attempt live TxGNN zero-shot indication scoring; None on any miss.

        Gated on ``NEUROAD_ENABLE_TXGNN=1``. Requires torch + the ``txgnn``
        package + a locally downloaded pretrained checkpoint (Google Drive,
        manual). All imports are lazy and every failure returns None so the
        caller degrades to the offline snapshot. Records would be stamped
        ``source="live"``. Kept intentionally conservative: if the heavy stack
        or checkpoint is absent (the normal case), we do nothing.
        """
        if os.environ.get(ENABLE_TXGNN_ENV) != "1":
            return None
        try:  # pragma: no cover - heavy optional stack, never runs in CI
            import importlib

            importlib.import_module("torch")
            importlib.import_module("txgnn")
            ckpt = os.environ.get("TXGNN_CKPT_DIR")
            if not ckpt or not Path(ckpt).exists():
                return None
            # A real integration would load the checkpoint + KG and score
            # indications here, mapping results into RepurposingCandidate with
            # source="live". We do not fabricate scores when we cannot run it,
            # so we fall back rather than invent a "live" result.
            return None
        except Exception:
            return None


__all__ = [
    "RepurposingCandidate",
    "RepurposingEngine",
    "resolve_disease_nodes",
    "ALZHEIMER_MONDO_NODES",
    "ENABLE_TXGNN_ENV",
]
