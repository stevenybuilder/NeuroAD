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

# ---------------------------------------------------------------------------
# Bundled STRING v12.0 PPI subgraph (for in-repo network propagation)
# ---------------------------------------------------------------------------
# A once-fetched STRING v12.0 subgraph over the NeuroAD gene universe (74 PI4AD
# snapshot genes UNION 50 Open Targets AD targets UNION MECHANISM_GENES = 115
# symbols). Persisted at data/string_ppi_subgraph.csv with a provenance header.
# STRING is CC BY 4.0, so bundling a subset is license-clean. This is the
# offline-first, deterministic PPI network used by ``propagate`` below — it is an
# in-repo random-walk-with-restart / heat-diffusion over real STRING edges, and
# is EXPLICITLY NOT a reproduction of PI4AD's proprietary R (dTarget /
# oSubneterGenes) subnetwork/propagation pipeline.
_STRING_SNAPSHOT_PATH = Path(__file__).with_name("data") / "string_ppi_subgraph.csv"
_STRING_API_URL = "https://string-db.org/api/tsv/network"
_STRING_EDGE_THRESHOLD = 0.4     # STRING combined-confidence cutoff (0-1 scale)
_STRING_RESTART = 0.5            # RWR restart probability (r)


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


@dataclass
class PropagatedNode:
    """One gene's score after network propagation over the STRING subgraph.

    ``propagated_score`` is the stationary mass a random-walk-with-restart (or
    heat-diffusion) seeded on the hit genes deposits on this node (higher =
    closer/more central to the seeds); ``rank`` is 1-based within the subgraph
    (rank 1 = highest propagated mass). ``degree`` is the node's STRING degree in
    the thresholded subgraph. ``is_seed`` marks a restart/seed gene; ``is_hub``
    marks a NON-seed node with both high propagated mass and high degree — a
    network hub the propagation surfaces around the seeds. ``source`` is
    "string_v12_snapshot" (bundled edges, the default) or "string_live".
    """
    gene: str
    propagated_score: float
    rank: int
    degree: int
    is_seed: bool
    is_hub: bool
    method: str                    # "rwr" | "heat"
    source: str                    # "string_v12_snapshot" | "string_live"

    def to_dict(self) -> dict:
        return {
            "gene": self.gene,
            "propagated_score": self.propagated_score,
            "rank": self.rank,
            "degree": self.degree,
            "is_seed": self.is_seed,
            "is_hub": self.is_hub,
            "method": self.method,
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

    # -- network propagation ----------------------------------------------

    def propagate(
        self,
        seed_genes: list[str],
        *,
        restart: float = _STRING_RESTART,
        method: str = "rwr",
        threshold: float = _STRING_EDGE_THRESHOLD,
        hub_top_k: int = 25,
    ) -> list[PropagatedNode]:
        """Propagate the seed hit genes over the bundled STRING v12.0 subgraph.

        Deterministic, closed-form random-walk-with-restart (``method="rwr"``) or
        heat-diffusion (``method="heat"``) over the real STRING edges — an in-repo
        network prior, NOT a reproduction of PI4AD's proprietary R subnetwork
        pipeline. Uses the same offline-first/degrade-never-raise contract as the
        rest of this module: any failure (no subgraph, empty seeds, missing
        scipy) yields ``[]`` rather than raising. Returns every subgraph node as a
        :class:`PropagatedNode`, ranked by propagated mass (rank 1 = highest)."""
        return propagate_hits(
            seed_genes,
            restart=restart,
            method=method,
            threshold=threshold,
            hub_top_k=hub_top_k,
            prefer_offline=self.prefer_offline,
            timeout=self.timeout,
        )


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


# ---------------------------------------------------------------------------
# STRING v12.0 network propagation (in-repo RWR / heat diffusion)
# ---------------------------------------------------------------------------
# Provenance-honest note: this is an in-repo random-walk-with-restart / heat-
# diffusion over a bundled real STRING v12.0 subgraph. It is NOT PI4AD's
# proprietary R (dTarget / oSubneterGenes) subnetwork/propagation pipeline, and
# does not reproduce its scores — it is a transparent, deterministic network
# prior that surfaces STRING-connected hubs around a set of seed hit genes.


def _load_string_snapshot() -> list[tuple[str, str, int]]:
    """Load the bundled STRING subgraph edges (source="string_v12_snapshot").

    Returns ``[(gene_a, gene_b, combined_score_0_1000), ...]``; skips the
    ``#``-prefixed provenance header. Never raises — returns ``[]`` on failure."""
    out: list[tuple[str, str, int]] = []
    try:
        with open(_STRING_SNAPSHOT_PATH, "r", encoding="utf-8", newline="") as fh:
            body = [ln for ln in fh if not ln.lstrip().startswith("#")]
        for row in csv.DictReader(body):
            try:
                a = row["gene_a"].strip()
                b = row["gene_b"].strip()
                score = int(float(row["combined_score"]))
            except (KeyError, ValueError, AttributeError):
                continue
            if a and b and a != b:
                out.append((a, b, score))
    except Exception:
        return []
    return out


def _fetch_string_live(genes: list[str],
                       timeout: int = _HTTP_TIMEOUT,
                       add_nodes: int = 0
                       ) -> Optional[list[tuple[str, str, int]]]:
    """Best-effort live STRING v12.0 network fetch, or None on any failure.

    Hits the free, no-credential STRING API (``/api/tsv/network``, taxon 9606,
    CC BY 4.0). Fully wrapped: a missing network / non-200 / unparseable TSV
    degrades to the bundled snapshot upstream, never raising.

    ``add_nodes`` (STRING's ``add_nodes`` param) expands the returned network with
    up to that many highest-confidence interaction partners beyond the queried
    identifiers — needed so that seeding propagation with a handful of genes yields
    a real neighborhood (not just the degenerate seed-only subgraph) to surface
    hubs in. ``0`` (default) preserves the original seed-only behavior."""
    if not genes:
        return None
    try:
        import requests  # imported here to keep module import-time clean
        data = {
            "identifiers": "\r".join(sorted({g for g in genes if g})),
            "species": 9606,
            "caller_identity": "neuroad_discovery_engine",
        }
        if add_nodes and add_nodes > 0:
            data["add_nodes"] = int(add_nodes)
        resp = requests.post(
            _STRING_API_URL,
            data=data,
            timeout=timeout,
        )
        if resp.status_code != 200:
            return None
        lines = resp.text.splitlines()
        if len(lines) < 2:
            return None
        reader = csv.DictReader(lines, delimiter="\t")
        out: list[tuple[str, str, int]] = []
        for row in reader:
            try:
                a = row["preferredName_A"].strip()
                b = row["preferredName_B"].strip()
                score = int(round(float(row["score"]) * 1000))
            except (KeyError, ValueError, TypeError, AttributeError):
                continue
            if a and b and a != b:
                out.append((a, b, score))
        return out or None
    except Exception:
        return None


def fetch_string_subgraph(
    genes: Optional[list[str]] = None,
    *,
    prefer_offline: bool = True,
    timeout: int = _HTTP_TIMEOUT,
    add_nodes: int = 0,
) -> tuple[list[tuple[str, str, int]], str]:
    """Load the STRING PPI subgraph, offline-first, and stamp its provenance.

    Returns ``(edges, source)`` where ``edges`` is a de-duplicated list of
    ``(gene_a, gene_b, combined_score_0_1000)`` and ``source`` is
    ``"string_v12_snapshot"`` (bundled, the default) or ``"string_live"``. With
    ``prefer_offline=False`` a live STRING fetch is attempted over ``genes`` (or
    the bundled universe if ``genes`` is None) and, on ANY failure, degrades to
    the bundled snapshot — never raising, always provenance-stamped."""
    edges: Optional[list[tuple[str, str, int]]] = None
    source = "string_v12_snapshot"
    if not prefer_offline:
        seeds = genes if genes else [a for a, _, _ in _load_string_snapshot()]
        live = _fetch_string_live(seeds, timeout=timeout, add_nodes=add_nodes)
        if live:
            edges, source = live, "string_live"
    if edges is None:
        edges = _load_string_snapshot()
        source = "string_v12_snapshot"
    # De-duplicate undirected edges, keeping the strongest score.
    dedup: dict[tuple[str, str], int] = {}
    for a, b, s in edges:
        key = (a, b) if a <= b else (b, a)
        if key not in dedup or s > dedup[key]:
            dedup[key] = s
    return [(a, b, s) for (a, b), s in sorted(dedup.items())], source


def propagate_hits(
    seed_genes: list[str],
    *,
    restart: float = _STRING_RESTART,
    method: str = "rwr",
    threshold: float = _STRING_EDGE_THRESHOLD,
    hub_top_k: int = 25,
    prefer_offline: bool = True,
    timeout: int = _HTTP_TIMEOUT,
    add_nodes: int = 0,
) -> list[PropagatedNode]:
    """Propagate ``seed_genes`` over the STRING subgraph; return ranked nodes.

    Builds a symmetric weighted adjacency (edge weight = combined_score/1000,
    kept iff ``>= threshold``), degree-normalizes it as ``W = D^-1/2 A D^-1/2``,
    then solves either:

      * ``method="rwr"``  — random-walk-with-restart, closed form
        ``p = (1-r)(I - r W)^-1 s`` (r = ``restart``), or
      * ``method="heat"`` — heat diffusion ``p = expm(-t (I - W)) s`` (t = ``restart``),

    with restart/heat-source ``s`` a uniform distribution over the seed genes
    present in the subgraph. Both are deterministic and closed-form. A node is
    flagged ``is_hub`` iff it is NOT a seed, its degree is at/above the subgraph's
    75th-percentile degree, and its propagated rank is within ``hub_top_k`` — a
    STRING hub the seeds' propagation lights up. Uses scipy.sparse + numpy only.

    Offline-first and degrade-never-raise: empty seeds, an empty subgraph, no
    seed present in the graph, or a missing scipy all yield ``[]``."""
    if not seed_genes or restart < 0 or restart > 1:
        return []
    seeds_up = {g.strip().upper() for g in seed_genes if g and g.strip()}
    if not seeds_up:
        return []
    try:
        import numpy as np
        import scipy.sparse as sp
        from scipy.sparse.linalg import spsolve, expm

        edges, source = fetch_string_subgraph(
            list(seeds_up), prefer_offline=prefer_offline, timeout=timeout,
            add_nodes=add_nodes)
        if not edges:
            return []

        # Build node index over the thresholded subgraph.
        kept: list[tuple[int, int, float]] = []
        index: dict[str, int] = {}
        order: list[str] = []

        def _idx(sym: str) -> int:
            if sym not in index:
                index[sym] = len(order)
                order.append(sym)
            return index[sym]

        for a, b, s in edges:
            w = s / 1000.0
            if w < threshold:
                continue
            ia, ib = _idx(a.upper()), _idx(b.upper())
            kept.append((ia, ib, w))
        n = len(order)
        if n == 0 or not kept:
            return []

        rows = np.array([e[0] for e in kept] + [e[1] for e in kept])
        cols = np.array([e[1] for e in kept] + [e[0] for e in kept])
        vals = np.array([e[2] for e in kept] + [e[2] for e in kept], dtype=float)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        degree_w = np.asarray(A.sum(axis=1)).ravel()          # weighted degree
        degree_count = np.asarray((A > 0).sum(axis=1)).ravel()  # unweighted degree
        with np.errstate(divide="ignore"):
            dinv = np.where(degree_w > 0, 1.0 / np.sqrt(degree_w), 0.0)
        D = sp.diags(dinv)
        W = D @ A @ D                                          # sym-normalized

        # Seed / restart distribution: uniform over present seeds.
        present = [index[s] for s in seeds_up if s in index]
        if not present:
            return []
        s_vec = np.zeros(n)
        s_vec[present] = 1.0 / len(present)

        if method == "heat":
            L = sp.identity(n, format="csc") - W.tocsc()
            p = np.asarray(expm(-restart * L).dot(s_vec)).ravel()
        else:  # "rwr" (default)
            M = (sp.identity(n, format="csc") - restart * W.tocsc()).tocsc()
            p = (1.0 - restart) * spsolve(M, s_vec)
            p = np.asarray(p).ravel()

        seed_idx = set(present)
        rank_order = list(np.argsort(-p))  # descending, stable
        rank_of = {int(i): r + 1 for r, i in enumerate(rank_order)}
        deg75 = float(np.percentile(degree_count, 75)) if n else 0.0

        nodes: list[PropagatedNode] = []
        for i in range(n):
            is_seed = i in seed_idx
            rk = rank_of[i]
            is_hub = (
                (not is_seed)
                and degree_count[i] >= deg75
                and rk <= hub_top_k
            )
            nodes.append(PropagatedNode(
                gene=order[i],
                propagated_score=round(float(p[i]), 6),
                rank=rk,
                degree=int(degree_count[i]),
                is_seed=is_seed,
                is_hub=bool(is_hub),
                method="heat" if method == "heat" else "rwr",
                source=source,
            ))
        nodes.sort(key=lambda x: x.rank)
        return nodes
    except Exception:
        # Degrade-never-raise: a missing scipy / numerical failure yields [].
        return []
