"""
string_ppi — protein-protein INTERACTION EVIDENCE for prioritized AD targets (STRING).

This is the honest stand-in for the L6 "molecular targeting" complex step. It does
NOT fold a novel complex: it fetches the STRING database's confidence that two
proteins interact (and each target's top hub partners), so the engine can pair a
promoted target with its ranked interactors as *decision-support*. AlphaFold 3
de-novo complex folding stays intentionally unwired (gated non-commercial weights,
see ``alphafold`` module) — STRING interaction evidence is what ships.

Given an AD gene symbol (or a pair of symbols), return STRING's ``combined_score``
(0-1) plus the per-channel breakdown that ships keyless from the public REST API:
``experimental`` (escore), ``database`` (dscore), ``textmining`` (tscore). Species
is fixed to Homo sapiens (NCBI taxon 9606).

OFFLINE / DETERMINISTIC CONTRACT: imports and runs with NO network and NO
credentials. Any function that would hit the network degrades to a bundled JSON
snapshot (the 12 AD targets: pairwise edges + top hub partners, captured live
2026-07-11) instead of raising, and stamps provenance: every returned record's
``source`` is "live" (a real STRING fetch) or "offline_snapshot" (bundled
fallback) — a fallback is NEVER dressed up as live data.

Everything here is labeled "interaction evidence, NOT de-novo folding". A high
STRING score means the interaction is well supported in the literature/databases,
NOT that this engine predicted a novel binding pose.

Only stdlib + requests are used. Endpoints (keyless):
  * network:              GET {base}/api/tsv/network?identifiers=A%0dB&species=9606
  * interaction_partners: GET {base}/api/tsv/interaction_partners?identifiers=G&...
``STRING_API_BASE_URL`` (optional env var) overrides the host for mirrors/testing.
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
#: The AD targets the engine ranks (same 12 as alphafold.AD_PROTEIN_MAP canon).
AD_TARGETS: tuple[str, ...] = (
    "APP", "MAPT", "APOE", "PSEN1", "PSEN2", "BACE1",
    "TREM2", "HRAS", "MAPK1", "ESR1", "CLU", "BIN1",
)

_DEFAULT_BASE_URL = "https://string-db.org"
_SNAPSHOT_PATH = Path(__file__).with_name("data") / "string_snapshot.json"
_HTTP_TIMEOUT = 15  # seconds — short, so a live call never hangs the engine
_SPECIES = 9606     # Homo sapiens (NCBI taxon)
_DEFAULT_PARTNER_LIMIT = 8
_DEFAULT_REQUIRED_SCORE = 400  # STRING's medium-confidence cutoff (0-1000 scale)

#: STRING interaction-channel columns we surface, keyed by their TSV field name.
_CHANNEL_COLS = {"experimental": "escore", "database": "dscore",
                 "textmining": "tscore"}

#: One-line honesty banner attached to every returned record.
EVIDENCE_LABEL = ("STRING protein-protein interaction evidence, "
                  "NOT de-novo AlphaFold3 complex folding")


def _base_url() -> str:
    """STRING host; overridable via STRING_API_BASE_URL for mirrors/testing."""
    return os.environ.get("STRING_API_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")


def _load_snapshot() -> dict:
    """Load the bundled deterministic snapshot (real STRING capture)."""
    try:
        with open(_SNAPSHOT_PATH, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return {k: v for k, v in raw.items() if not k.startswith("_")}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Structured returns
# ---------------------------------------------------------------------------


@dataclass
class InteractionEvidence:
    """STRING interaction confidence for one protein pair (or target->partner edge).

    ``combined_score`` is STRING's overall confidence on a real 0-1 scale (higher =
    better-supported interaction), or None when no edge exists above STRING's
    cutoff. ``channels`` breaks that down into the keyless public channels
    (``experimental``/``database``/``textmining``, each 0-1). ``source`` is the
    provenance stamp: "live" (real STRING fetch) or "offline_snapshot" (bundled
    fallback). ``evidence_type`` is fixed to the honesty banner — this is
    interaction evidence, not a folded complex.
    """
    gene_a: str
    gene_b: str
    combined_score: Optional[float]
    channels: dict[str, float]
    source: str
    evidence_type: str = EVIDENCE_LABEL
    error: str = ""  # non-fatal note (e.g. why it fell back / no edge)

    def to_dict(self) -> dict:
        return {
            "gene_a": self.gene_a,
            "gene_b": self.gene_b,
            "combined_score": self.combined_score,
            "channels": dict(self.channels),
            "source": self.source,
            "evidence_type": self.evidence_type,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------


class StringPPIClient:
    """Adapter over the STRING-db REST API, offline-first.

    Default (``prefer_offline=False``, keyless): each method best-effort hits the
    live STRING API and, on ANY failure (no network, non-200, malformed TSV,
    unknown gene), degrades to the bundled snapshot — never raising, always
    provenance-stamped. Set ``prefer_offline=True`` to skip the network entirely
    (deterministic, e.g. in tests).
    """

    def __init__(self, prefer_offline: bool = False, *,
                 timeout: int = _HTTP_TIMEOUT) -> None:
        self.prefer_offline = prefer_offline
        self.timeout = timeout
        self._snapshot = _load_snapshot()
        self._known = {g.upper() for g in AD_TARGETS} | {
            k.upper() for k in (self._snapshot.get("partners") or {})
        }

    # -- resolution --------------------------------------------------------

    def resolve_symbol(self, gene: str) -> Optional[str]:
        """Normalize a gene symbol to STRING form (upper-cased, alnum-ish).

        Returns None for empty/malformed input. It does not require the symbol to
        be one of the 12 AD targets — arbitrary symbols resolve for the live path;
        the offline snapshot simply has no record for unknown ones (honest stub)."""
        if not gene:
            return None
        g = gene.strip().upper()
        if not g:
            return None
        # STRING symbols are alphanumeric (plus a few punctuation chars); keep it
        # loose but reject obviously non-symbol tokens (whitespace already gone).
        if not all(c.isalnum() or c in "-_." for c in g):
            return None
        return g

    def _is_known(self, gene: str) -> bool:
        return gene.upper() in self._known

    # -- low-level TSV -----------------------------------------------------

    def _get_tsv(self, path: str, params: dict) -> Optional[list[dict]]:
        """GET a STRING TSV endpoint; return parsed rows or None on ANY failure.

        ``requests`` is a guaranteed dep but the call is fully wrapped so a missing
        network / non-200 / malformed TSV degrades to the snapshot, never raising."""
        try:
            import requests  # imported here to keep module import-time clean
            resp = requests.get(f"{_base_url()}{path}", params=params,
                                timeout=self.timeout)
            if resp.status_code != 200:
                return None
            return _parse_tsv(resp.text)
        except Exception:
            return None

    # -- public API: one pair ----------------------------------------------

    def pair_evidence(self, gene_a: str, gene_b: str) -> InteractionEvidence:
        """STRING interaction confidence + channel breakdown for a protein pair.

        Live path queries the ``network`` endpoint for the two symbols; on any
        failure (or offline) degrades to the bundled pairwise snapshot. A pair with
        no STRING edge above cutoff returns ``combined_score=None`` with an
        explanatory ``error`` (an honest "no supported interaction", never a
        fabricated score). Every record is provenance-stamped."""
        a = self.resolve_symbol(gene_a)
        b = self.resolve_symbol(gene_b)
        if a is None or b is None:
            return InteractionEvidence(
                gene_a=gene_a or "", gene_b=gene_b or "",
                combined_score=None, channels={}, source="offline_snapshot",
                error=f"could not resolve pair ('{gene_a}', '{gene_b}')")
        if a == b:
            return InteractionEvidence(
                gene_a=a, gene_b=b, combined_score=None, channels={},
                source="offline_snapshot", error="self-pair has no STRING edge")

        if not self.prefer_offline:
            live = self._pair_live(a, b)
            if live is not None:
                return live
        note = "" if self.prefer_offline else "live STRING fetch unavailable"
        return self._pair_offline(a, b, note=note)

    def _pair_live(self, a: str, b: str) -> Optional[InteractionEvidence]:
        rows = self._get_tsv(
            "/api/tsv/network",
            {"identifiers": f"{a}\r{b}", "species": _SPECIES})
        if not rows:
            return None
        for r in rows:
            names = {str(r.get("preferredName_A", "")).upper(),
                     str(r.get("preferredName_B", "")).upper()}
            if names == {a, b}:
                return _edge_from_row(a, b, r, source="live")
        return None

    def _pair_offline(self, a: str, b: str, *, note: str = "") -> InteractionEvidence:
        key = "|".join(sorted((a, b)))
        rec = (self._snapshot.get("pairwise") or {}).get(key)
        if rec is None:
            err = (note + "; " if note else "")
            err += "no STRING interaction evidence above cutoff in bundled snapshot"
            return InteractionEvidence(
                gene_a=a, gene_b=b, combined_score=None, channels={},
                source="offline_snapshot", error=err)
        return InteractionEvidence(
            gene_a=a, gene_b=b,
            combined_score=_clamp01(rec.get("combined_score")),
            channels={ch: _clamp01(rec.get(ch)) for ch in _CHANNEL_COLS},
            source="offline_snapshot", error=note)

    # -- public API: hub partners ------------------------------------------

    def interaction_partners(self, gene: str, *,
                             limit: int = _DEFAULT_PARTNER_LIMIT
                             ) -> list[InteractionEvidence]:
        """Top STRING hub partners for a target, ranked by combined_score (desc).

        Live path queries the ``interaction_partners`` endpoint; on any failure (or
        offline) degrades to the bundled partner snapshot. Returns [] for an
        unknown/unresolvable gene (honest empty), never raising. Each edge is a
        provenance-stamped ``InteractionEvidence`` (gene -> partner)."""
        g = self.resolve_symbol(gene)
        if g is None or limit <= 0:
            return []
        if not self.prefer_offline:
            live = self._partners_live(g, limit)
            if live is not None:
                return live[:limit]
        return self._partners_offline(g)[:limit]

    def _partners_live(self, gene: str,
                       limit: int) -> Optional[list[InteractionEvidence]]:
        rows = self._get_tsv(
            "/api/tsv/interaction_partners",
            {"identifiers": gene, "species": _SPECIES,
             "limit": max(1, int(limit)),
             "required_score": _DEFAULT_REQUIRED_SCORE})
        if not rows:
            return None
        out: list[InteractionEvidence] = []
        for r in rows:
            na = str(r.get("preferredName_A", "")).upper()
            nb = str(r.get("preferredName_B", "")).upper()
            partner = nb if na == gene else na
            if not partner:
                continue
            out.append(_edge_from_row(gene, partner, r, source="live"))
        out.sort(key=lambda e: e.combined_score or 0.0, reverse=True)
        return out

    def _partners_offline(self, gene: str) -> list[InteractionEvidence]:
        recs = (self._snapshot.get("partners") or {}).get(gene.upper())
        if not recs:
            return []
        out = [InteractionEvidence(
            gene_a=gene.upper(), gene_b=str(r.get("partner", "")).upper(),
            combined_score=_clamp01(r.get("combined_score")),
            channels={ch: _clamp01(r.get(ch)) for ch in _CHANNEL_COLS},
            source="offline_snapshot") for r in recs]
        out.sort(key=lambda e: e.combined_score or 0.0, reverse=True)
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_tsv(text: str) -> list[dict]:
    """Parse a STRING TSV response into a list of header-keyed row dicts."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    header = lines[0].split("\t")
    rows: list[dict] = []
    for ln in lines[1:]:
        parts = ln.split("\t")
        if len(parts) < len(header):
            continue
        rows.append(dict(zip(header, parts)))
    return rows


def _edge_from_row(a: str, b: str, row: dict, *, source: str) -> InteractionEvidence:
    """Build an InteractionEvidence from a parsed STRING TSV row."""
    return InteractionEvidence(
        gene_a=a, gene_b=b,
        combined_score=_clamp01(row.get("score")),
        channels={ch: _clamp01(row.get(col))
                  for ch, col in _CHANNEL_COLS.items()},
        source=source)


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
# Module-level convenience (thin wrappers for harness callers)
# ---------------------------------------------------------------------------


def complex_evidence(gene_a: str, gene_b: str, *,
                     prefer_offline: bool = False) -> dict:
    """One protein pair's STRING interaction evidence (the AF3-complex stand-in).

    Returns ``InteractionEvidence.to_dict`` — combined_score + channel breakdown,
    provenance-stamped, honesty-labeled. Offline-safe. This is interaction
    evidence, NOT a de-novo folded complex."""
    return StringPPIClient(prefer_offline=prefer_offline).pair_evidence(
        gene_a, gene_b).to_dict()


def interaction_evidence(gene: str, *,
                         prefer_offline: bool = False,
                         limit: int = _DEFAULT_PARTNER_LIMIT) -> dict:
    """One target's ranked STRING hub partners (interaction-evidence follow-ups).

    Returns a dict with the ``gene``, its ranked ``partners`` (each an
    ``InteractionEvidence.to_dict``), the ``source`` provenance, and the honesty
    ``note``. Offline-safe. Interaction evidence, NOT de-novo complex folding."""
    client = StringPPIClient(prefer_offline=prefer_offline)
    partners = client.interaction_partners(gene, limit=limit)
    src = partners[0].source if partners else "offline_snapshot"
    return {
        "gene": (gene or "").strip().upper(),
        "species": _SPECIES,
        "note": EVIDENCE_LABEL,
        "partners": [p.to_dict() for p in partners],
        "n_partners": len(partners),
        "source": src,
    }


__all__ = [
    "AD_TARGETS",
    "EVIDENCE_LABEL",
    "StringPPIClient",
    "InteractionEvidence",
    "complex_evidence",
    "interaction_evidence",
]
