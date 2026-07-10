"""
pi4ad — target/gene prioritization for Alzheimer's disease (PI4AD).

Given nothing (whole-disease ranking) or a gene symbol, return PI4AD priority
scores on the real 0-10 scale plus integer rank. PI4AD (Priority Index for AD,
github.com/hfang-bristol/PI4AD, paper PMC12491700) ships its ranking only as
R-binary RDS + an interactive portal; R IS NOT AND CANNOT BE USED HERE, so this
is a PURE-PYTHON adapter over a bundled priority snapshot.

OFFLINE / DETERMINISTIC CONTRACT: imports and runs with NO network and NO
credentials. The shipping default reads a bundled CSV snapshot
(``data/pi4ad_priority_snapshot.csv``, 74 top AD-priority genes scraped from the
public portal and cross-validated vs the paper). Every returned GenePriority is
provenance-stamped: ``source`` is "offline_snapshot" (bundled fallback, the
default) or "live" (real portal fetch) — a fallback is NEVER dressed up as live.

OPTIONAL live path (off by default, ``prefer_offline=False``): the portal is
HTTP-ONLY — ``http://www.genetictargets.com/PI4AD/ad`` embeds the full ranking as
a transposed DataTables JSON array in one ~3.4MB page; HTTPS to that host times
out, so the fetch MUST stay on plain http and must not upgrade. On ANY failure
(no network, non-200, unparseable page) it degrades to the snapshot, never raises.
No credentials are required; ``PI4AD_PORTAL_URL`` (optional env var) may override
the portal URL for mirrors/testing.
"""
from __future__ import annotations

import ast
import csv
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Bundled offline snapshot
# ---------------------------------------------------------------------------
# Derived from the public PI4AD portal http://www.genetictargets.com/PI4AD/ad
# (R pkg github.com/hfang-bristol/PI4AD; paper PMC12491700). 0-10 priority scale;
# ranks cross-validated against the paper (APP 18th, ESR1 61st, KIT 95th,
# PDGFRB 100th all match). The full portal table is 14,676 genes; this snapshot
# bundles the top-60 by rank plus the deeper canonical AD-genetics targets
# (ESR1, MAPK1, SORL1, MAPT, APOE, BIN1, CLU, PICALM, CD33, PSEN1, ABCA7, PSEN2).
_DEFAULT_PORTAL_URL = "http://www.genetictargets.com/PI4AD/ad"
_SNAPSHOT_PATH = Path(__file__).with_name("data") / "pi4ad_priority_snapshot.csv"
_HTTP_TIMEOUT = 15  # seconds — short, so a live call never hangs the engine


def _portal_url() -> str:
    """Portal URL; overridable via PI4AD_PORTAL_URL for mirrors/testing.

    The host is HTTP-ONLY (HTTPS times out); callers must not upgrade the scheme.
    """
    return os.environ.get("PI4AD_PORTAL_URL", _DEFAULT_PORTAL_URL)


# ---------------------------------------------------------------------------
# Structured return
# ---------------------------------------------------------------------------


@dataclass
class GenePriority:
    """One gene's PI4AD priority.

    ``priority_score`` is on the real PI4AD 0-10 scale (higher = higher priority);
    ``rank`` is the integer rank across the full ~14.7k-gene prioritisation (1 =
    top). ``source`` is the provenance stamp: "offline_snapshot" (bundled table,
    the default) or "live" (real portal fetch) — a fallback is always labeled.
    """
    gene: str
    priority_score: float          # 0-10 PI4AD priority (higher = stronger)
    rank: int                      # 1-based rank in the full prioritisation
    evidence_note: str             # Core/Peripheral category + gene full name
    source: str                    # "offline_snapshot" | "live"

    def to_dict(self) -> dict:
        return {
            "gene": self.gene,
            "priority_score": self.priority_score,
            "rank": self.rank,
            "evidence_note": self.evidence_note,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------


class PI4AD:
    """Adapter over the PI4AD Alzheimer's target prioritisation, offline-first.

    Default (``prefer_offline=True``, no network / no creds): serve the bundled
    snapshot, every record stamped ``source="offline_snapshot"``. With
    ``prefer_offline=False`` it best-effort fetches the live portal table over
    plain HTTP and, on ANY failure, degrades to the snapshot — never raising,
    always provenance-stamped.
    """

    def __init__(self, *, prefer_offline: bool = True,
                 timeout: int = _HTTP_TIMEOUT) -> None:
        self.prefer_offline = prefer_offline
        self.timeout = timeout
        self._table: Optional[list[GenePriority]] = None

    # -- table loading -----------------------------------------------------

    def _load(self) -> list[GenePriority]:
        """Load (and cache) the priority table, ascending by rank.

        Tries the live portal first unless ``prefer_offline``; on any failure
        (or offline mode) falls back to the bundled snapshot. Never raises."""
        if self._table is not None:
            return self._table
        table: Optional[list[GenePriority]] = None
        if not self.prefer_offline:
            table = self._fetch_live()
        if not table:
            table = _load_snapshot()
        table.sort(key=lambda g: g.rank)
        self._table = table
        return table

    def _fetch_live(self) -> Optional[list[GenePriority]]:
        """Fetch + parse the live portal table (source="live") or None on failure.

        ``requests`` is a guaranteed dep but the call is still fully wrapped so a
        missing network / non-200 / unparseable page degrades to the snapshot.
        Stays on the given (HTTP) URL — the host does not serve HTTPS."""
        url = _portal_url()
        try:
            import requests  # imported here to keep module import-time clean
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                return None
            parsed = _parse_portal_html(resp.text)
            return parsed or None
        except Exception:
            return None

    # -- public API --------------------------------------------------------

    def rank_genes(self, top_n: int = 25) -> list[GenePriority]:
        """Whole-disease ranking: the top ``top_n`` PI4AD priority genes.

        Sliced by ascending rank (rank 1 first). ``top_n <= 0`` returns []."""
        if top_n <= 0:
            return []
        return list(self._load()[:top_n])

    def priority(self, gene_symbol: str) -> Optional[GenePriority]:
        """Look up one gene's PI4AD priority (case-insensitive).

        Returns None if the symbol is absent from the bundled/loaded table —
        an honest "not prioritised in this snapshot", never a fabricated score."""
        if not gene_symbol:
            return None
        q = gene_symbol.strip().upper()
        for rec in self._load():
            if rec.gene.upper() == q:
                return rec
        return None


# ---------------------------------------------------------------------------
# Snapshot + portal parsing helpers
# ---------------------------------------------------------------------------


def _load_snapshot() -> list[GenePriority]:
    """Load the bundled deterministic snapshot (source="offline_snapshot")."""
    out: list[GenePriority] = []
    try:
        with open(_SNAPSHOT_PATH, "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    out.append(GenePriority(
                        gene=row["gene"].strip(),
                        priority_score=float(row["priority_score"]),
                        rank=int(row["rank"]),
                        evidence_note=row.get("evidence_note", "").strip(),
                        source="offline_snapshot",
                    ))
                except (KeyError, ValueError):
                    continue
    except Exception:
        return []
    return out


def _strip_html(s: object) -> str:
    """Drop HTML tags (the portal wraps symbols/names in <a>/<span>)."""
    return re.sub(r"<[^>]+>", "", str(s)).strip()


def _extract_data_array(html: str) -> Optional[list]:
    """Bracket-match and parse the embedded DataTables ``"data":[ ... ]`` array.

    The array is a TRANSPOSED list of columns (col0=symbols, col1=scores,
    col2=ranks, col3=Core/Peripheral, col7=gene name). Returns the parsed nested
    list or None. Uses ``ast.literal_eval`` (safe — no code execution) after
    normalising bare ``null`` tokens to ``None``."""
    key = html.find('"data":')
    if key == -1:
        return None
    start = html.find("[", key)
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    quote = ""
    end = None
    for j in range(start, len(html)):
        c = html[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == quote:
                in_str = False
            continue
        if c in ("'", '"'):
            in_str = True
            quote = c
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                end = j + 1
                break
    if end is None:
        return None
    txt = re.sub(r"(?<=[\[,])\s*null\s*(?=[,\]])", "None", html[start:end])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # invalid-escape (\/) SyntaxWarnings
            data = ast.literal_eval(txt)
    except (ValueError, SyntaxError, MemoryError):
        return None
    if not isinstance(data, list) or len(data) < 8:
        return None
    return data


def _parse_portal_html(html: str) -> list[GenePriority]:
    """Parse a live portal page into GenePriority records (source="live").

    Returns [] if the embedded table cannot be located/parsed so the caller can
    degrade to the snapshot."""
    data = _extract_data_array(html)
    if data is None:
        return []
    symbols, scores, ranks, cats, names = (
        data[0], data[1], data[2], data[3], data[7])
    n = min(len(symbols), len(scores), len(ranks), len(cats), len(names))
    out: list[GenePriority] = []
    for k in range(n):
        try:
            gene = _strip_html(symbols[k])
            score = float(scores[k])
            rank = int(ranks[k])
        except (TypeError, ValueError):
            continue
        if not gene:
            continue
        cat = _strip_html(cats[k])
        name = _strip_html(names[k])
        note = (f"{cat} PI4AD target ({name})".strip()
                if (cat or name) else "PI4AD target")
        out.append(GenePriority(gene=gene, priority_score=score, rank=rank,
                                evidence_note=note, source="live"))
    return out


# ---------------------------------------------------------------------------
# Module-level conveniences (thin wrappers for harness callers)
# ---------------------------------------------------------------------------


def rank_ad_targets(top_n: int = 25, *,
                    prefer_offline: bool = True) -> list[GenePriority]:
    """Whole-disease PI4AD ranking: top ``top_n`` prioritized AD genes."""
    return PI4AD(prefer_offline=prefer_offline).rank_genes(top_n)


def gene_priority(gene_symbol: str, *,
                  prefer_offline: bool = True) -> Optional[GenePriority]:
    """One gene's PI4AD priority (case-insensitive), or None if not prioritized."""
    return PI4AD(prefer_offline=prefer_offline).priority(gene_symbol)
