"""
alphafold — structural layer for prioritized AD proteins (EBI AlphaFold DB).

Given an AD gene symbol or UniProt accession, return the predicted structure
(model URLs) + per-residue-confidence summary (mean pLDDT). The shipping default
is the PUBLIC, keyless EBI AlphaFold DB REST API
(GET https://alphafold.ebi.ac.uk/api/prediction/{acc}); its ``globalMetricValue``
IS the representative mean pLDDT, so the confidence signal needs zero file
download. An optional best-effort CIF download recomputes mean pLDDT by averaging
the CA-atom B-factor column.

OFFLINE / DETERMINISTIC CONTRACT: imports and runs with NO network and NO
credentials. Any function that would hit the network degrades to a bundled JSON
snapshot (12 AD targets, captured live 2026-07-10) instead of raising, and stamps
provenance: every AlphaFoldStructure.source is "live" (real fetch) or
"offline_snapshot" (fallback) — a fallback is NEVER dressed up as live data.

AlphaFold 3 (folding NOVEL PI4AD complexes) is a documented NON-DEFAULT path:
its weights are gated behind a request form + non-commercial license
(github.com/google-deepmind/alphafold3) and are intentionally NOT wired here —
the keyless DB API is the shipping path. No credentials are required for the
default path; ``AF_DB_BASE_URL`` (optional env var) may override the API/file host
for testing/mirrors.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Bundled AD protein map + offline snapshot
# ---------------------------------------------------------------------------
# symbol -> UniProt accession. These are the AD targets PI4AD ranks and the
# bridge surfaces. Source: canonical UniProt accessions (verified HTTP 200 from
# the AlphaFold DB prediction API for all 12 on 2026-07-10).
AD_PROTEIN_MAP: dict[str, str] = {
    "APP": "P05067",
    "MAPT": "P10636",
    "TAU": "P10636",       # common alias for MAPT
    "APOE": "P02649",
    "PSEN1": "P49768",
    "PSEN2": "P49810",
    "BACE1": "P56817",
    "TREM2": "Q9NZC2",
    "HRAS": "P01112",
    "MAPK1": "P28482",
    "ESR1": "P03372",
    "CLU": "P10909",
    "BIN1": "O00499",
}

_DEFAULT_BASE_URL = "https://alphafold.ebi.ac.uk"
_SNAPSHOT_PATH = Path(__file__).with_name("data") / "alphafold_snapshot.json"
_HTTP_TIMEOUT = 15  # seconds — short, so a call never hangs the engine


def _base_url() -> str:
    """API/file host; overridable via AF_DB_BASE_URL for mirrors/testing."""
    return os.environ.get("AF_DB_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")


def _load_snapshot() -> dict:
    """Load the bundled deterministic snapshot (keyed by UniProt accession)."""
    try:
        with open(_SNAPSHOT_PATH, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return {k: v for k, v in raw.items() if not k.startswith("_")}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Structured return
# ---------------------------------------------------------------------------


@dataclass
class AlphaFoldStructure:
    """Predicted structure + confidence summary for one AD protein.

    ``source`` is the provenance stamp: "live" (real AlphaFold DB fetch) or
    "offline_snapshot" (bundled fallback). ``mean_plddt`` is the representative
    per-residue confidence (0-100); it is populated on both paths (API
    globalMetricValue live, snapshot value offline). ``plddt_recomputed`` is True
    only when it was averaged from a downloaded CIF's CA B-factor column.
    """
    uniprot: str
    gene_symbol: str
    model_url: str                       # pdbUrl (or "" if unknown)
    cif_url: str
    source: str                          # "live" | "offline_snapshot"
    mean_plddt: Optional[float] = None
    model_version: Optional[str] = None  # e.g. "v6" (live) / snapshot pin
    plddt_recomputed: bool = False       # True if averaged from a downloaded CIF
    n_residues_scored: Optional[int] = None
    error: str = ""                      # non-fatal note (e.g. why fell back)
    extra: dict = field(default_factory=dict)  # spare API metadata (pae, entryId)

    def to_dict(self) -> dict:
        return {
            "uniprot": self.uniprot,
            "gene_symbol": self.gene_symbol,
            "model_url": self.model_url,
            "cif_url": self.cif_url,
            "mean_plddt": self.mean_plddt,
            "model_version": self.model_version,
            "plddt_recomputed": self.plddt_recomputed,
            "n_residues_scored": self.n_residues_scored,
            "source": self.source,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------


class AlphaFoldClient:
    """Adapter over the EBI AlphaFold DB REST API, offline-first.

    Default (keyless, no network guaranteed): ``fetch_structure`` tries the live
    prediction API and, on ANY failure (no network, non-200, malformed JSON,
    unknown accession), degrades to the bundled snapshot — never raising, always
    provenance-stamped. Set ``prefer_offline=True`` to skip the network entirely
    (deterministic, e.g. in tests).
    """

    def __init__(self, *, prefer_offline: bool = False,
                 cache_dir: Optional[str] = None,
                 timeout: int = _HTTP_TIMEOUT) -> None:
        self.prefer_offline = prefer_offline
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._snapshot = _load_snapshot()
        # reverse map accession -> a representative gene symbol (canonical name,
        # skipping the TAU alias so MAPT is reported for P10636).
        self._acc_to_symbol = {
            v["uniprot"]: v["gene_symbol"] for v in self._snapshot.values()
        }

    # -- resolution --------------------------------------------------------

    def resolve_uniprot(self, query: str) -> Optional[str]:
        """Map a gene symbol OR a UniProt accession to a UniProt accession.

        Case-insensitive on the symbol; passes an already-accession-shaped query
        through. Returns None if it cannot be resolved from the bundled map."""
        if not query:
            return None
        q = query.strip()
        upper = q.upper()
        if upper in AD_PROTEIN_MAP:
            return AD_PROTEIN_MAP[upper]
        # Already an accession we know about, or a plausibly-shaped accession.
        if upper in self._acc_to_symbol:
            return upper
        if _looks_like_accession(upper):
            return upper
        return None

    def _symbol_for(self, accession: str) -> str:
        """Best-effort gene symbol for an accession (from the bundled map)."""
        return self._acc_to_symbol.get(accession, "")

    # -- main entry --------------------------------------------------------

    def fetch_structure(self, query: str,
                        recompute_plddt: bool = False) -> AlphaFoldStructure:
        """Resolve ``query`` (gene symbol or accession) -> AlphaFoldStructure.

        Tries the live AlphaFold DB API first (unless ``prefer_offline``); on any
        failure falls back to the bundled snapshot with source="offline_snapshot".
        When ``recompute_plddt`` and online, best-effort downloads the CIF and
        averages CA B-factors for an exact mean pLDDT; the download is optional and
        never fatal (metadata mean pLDDT already covers the signal)."""
        acc = self.resolve_uniprot(query)
        if acc is None:
            # Unknown target: return an honest, non-raising stub.
            return AlphaFoldStructure(
                uniprot="", gene_symbol="", model_url="", cif_url="",
                source="offline_snapshot",
                error=f"could not resolve '{query}' to a known AD UniProt accession",
            )

        if not self.prefer_offline:
            live = self._fetch_live(acc)
            if live is not None:
                if recompute_plddt:
                    self._augment_with_cif(live)
                return live

        return self._offline(acc, note="" if self.prefer_offline
                             else "live AlphaFold DB fetch unavailable")

    # -- live path ---------------------------------------------------------

    def _fetch_live(self, accession: str) -> Optional[AlphaFoldStructure]:
        """Hit the prediction API; return a "live" structure or None on failure.

        ``requests`` is a guaranteed dependency but the call is still wrapped so
        that a missing network / non-200 / bad JSON degrades to the snapshot."""
        url = f"{_base_url()}/api/prediction/{accession}"
        try:
            import requests  # guaranteed dep; imported here to keep import-time clean
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                return None
            payload = resp.json()
            record = payload[0] if isinstance(payload, list) and payload else payload
            if not isinstance(record, dict):
                return None
        except Exception:
            return None

        cif_url = record.get("cifUrl", "") or ""
        version = None
        # Derive a version label from the model URL (…-model_v6.cif) if present.
        for tok in (cif_url.rsplit("model_", 1)[-1:] or [""]):
            v = tok.replace(".cif", "").strip()
            if v.startswith("v"):
                version = v
        gene = record.get("gene") or self._symbol_for(accession)
        mean = record.get("globalMetricValue")
        return AlphaFoldStructure(
            uniprot=record.get("uniprotAccession", accession),
            gene_symbol=gene or "",
            model_url=record.get("pdbUrl", "") or "",
            cif_url=cif_url,
            source="live",
            mean_plddt=float(mean) if isinstance(mean, (int, float)) else None,
            model_version=version,
            n_residues_scored=None,
            extra={
                "entryId": record.get("entryId"),
                "bcifUrl": record.get("bcifUrl"),
                "paeImageUrl": record.get("paeImageUrl"),
                "organism": record.get("organismScientificName"),
                "latestVersion": record.get("latestVersion"),
            },
        )

    def _augment_with_cif(self, struct: AlphaFoldStructure) -> None:
        """Best-effort: download the CIF and recompute mean pLDDT from CA B-factors.

        Purely additive — on any failure the metadata mean pLDDT stands and the
        struct is returned unchanged (plddt_recomputed stays False)."""
        if not struct.cif_url:
            return
        try:
            import requests
            resp = requests.get(struct.cif_url, timeout=self.timeout)
            if resp.status_code != 200:
                return
            text = resp.text
            if self.cache_dir is not None:
                try:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    fname = struct.cif_url.rsplit("/", 1)[-1] or f"{struct.uniprot}.cif"
                    (self.cache_dir / fname).write_text(text, encoding="utf-8")
                except Exception:
                    pass
            mean, n = _mean_ca_plddt_from_cif(text)
            if mean is not None:
                struct.mean_plddt = mean
                struct.n_residues_scored = n
                struct.plddt_recomputed = True
        except Exception:
            return

    # -- offline path ------------------------------------------------------

    def _offline(self, accession: str, note: str = "") -> AlphaFoldStructure:
        """Build an offline_snapshot structure for a known accession."""
        rec = self._snapshot.get(accession)
        if rec is None:
            return AlphaFoldStructure(
                uniprot=accession,
                gene_symbol=self._symbol_for(accession),
                model_url="", cif_url="",
                source="offline_snapshot",
                error=(note + "; " if note else "")
                + f"'{accession}' not in the bundled AD snapshot",
            )
        return AlphaFoldStructure(
            uniprot=rec["uniprot"],
            gene_symbol=rec["gene_symbol"],
            model_url=rec.get("pdb_url", ""),
            cif_url=rec.get("cif_url", ""),
            source="offline_snapshot",
            mean_plddt=(float(rec["mean_plddt"])
                        if rec.get("mean_plddt") is not None else None),
            model_version="v6",   # snapshot is pinned to AlphaFold DB model v6
            error=note,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _looks_like_accession(token: str) -> bool:
    """Loose UniProt-accession shape check (6 or 10 alnum, starts with a letter)."""
    if len(token) not in (6, 10):
        return False
    if not token[0].isalpha():
        return False
    return token.isalnum()


def _mean_ca_plddt_from_cif(cif_text: str) -> tuple[Optional[float], Optional[int]]:
    """Average the B-factor column over CA atoms of an AlphaFold mmCIF.

    Parses the _atom_site loop generically (column order is read from the header,
    not assumed). AlphaFold stores per-residue pLDDT in the B-factor column, so the
    CA-atom mean is the mean pLDDT. Returns (mean, n_ca) or (None, None) if the
    loop cannot be parsed. Pure-Python, stdlib-only — no external parser."""
    lines = cif_text.splitlines()
    headers: list[str] = []
    in_loop = False
    col_idx: dict[str, int] = {}
    total = 0.0
    n = 0
    for line in lines:
        s = line.strip()
        if s.startswith("_atom_site."):
            if not in_loop:
                in_loop = True
                headers = []
            headers.append(s)
            col_idx = {h: i for i, h in enumerate(headers)}
            continue
        if in_loop and headers:
            if s.startswith("_") or s.startswith("loop_"):
                break  # left the atom_site loop's data block
            if not s or s.startswith("#"):
                if n:
                    break
                continue
            parts = s.split()
            try:
                ci = col_idx.get("_atom_site.label_atom_id")
                bi = col_idx.get("_atom_site.B_iso_or_equiv")
                if ci is None or bi is None or bi >= len(parts):
                    continue
                if parts[ci].strip('"') != "CA":
                    continue
                total += float(parts[bi])
                n += 1
            except (ValueError, IndexError):
                continue
    if n == 0:
        return None, None
    return round(total / n, 2), n


def structural_confidence(query: str, *,
                          prefer_offline: bool = False) -> AlphaFoldStructure:
    """Module-level convenience: fetch one AD protein's structure + mean pLDDT.

    Thin wrapper over ``AlphaFoldClient().fetch_structure`` for harness callers
    that just want the structural-confidence signal for a promoted target."""
    return AlphaFoldClient(prefer_offline=prefer_offline).fetch_structure(query)
