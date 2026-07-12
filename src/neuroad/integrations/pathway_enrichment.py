"""
pathway_enrichment — offline over-representation analysis (ORA) for AD pathways.

This closes the "Pathway Analysis" gap in the PI4AD box. ``integrations/pi4ad.py``
already does real network propagation (STRING RWR/heat) + hub detection, but it
has ZERO literal pathway-set enrichment. This module adds honest, deterministic
ORA — a one-sided hypergeometric / Fisher's-exact over-representation test of a
query gene set against a BUNDLED, provenance-stamped snapshot of curated
AD-relevant pathway gene sets (KEGG "Alzheimer disease", Reactome amyloid/APP,
tau, MAPK/Ras signaling, TLR/NF-kB neuroinflammation, autophagy, PI3K-Akt, NMDA
excitotoxicity, oxidative phosphorylation).

HONESTY CONTRACT (do not overclaim):
  * This is ORA (over-representation), NOT GSEA — there is no ranked running-sum
    statistic and no phenotype-label permutation null.
  * The bundled memberships (``data/ad_pathway_genesets.json``) are a hand-curated,
    publicly-documented SUBSET of each real pathway, NOT the complete KEGG/Reactome
    set and NOT a live Enrichr/g:Profiler query. Every result therefore carries
    ``pathway_size`` + ``background_size`` + ``snapshot_source`` + ``model`` so the
    number is interpretable rather than presented as a genome-wide Enrichr p-value.
  * Default background/universe = the deterministic UNION of all snapshot pathway
    genes UNION the query genes (self-contained, reproducible). Passing an explicit
    ``background_size`` (e.g. ~20000 protein-coding) is allowed and is RECORDED on
    every row so the assumption is never hidden.
  * DEGRADE-NEVER-RAISE: missing/corrupt snapshot -> []; missing scipy -> [];
    empty/whitespace query -> []; a query with no gene in the universe -> [];
    a pathway with overlap < ``min_overlap`` is EXCLUDED (never returned with an
    invented score). Symbols are upper-cased for matching but NO alias/synonym
    expansion is fabricated (MAPT is not silently mapped to "tau").
  * BH-FDR is computed only over the pathways actually tested (overlap>=min_overlap).

OFFLINE / DETERMINISTIC: imports and runs with numpy/pandas/sklearn only. scipy is
lazy-imported INSIDE ``enrich`` (missing scipy -> []). No torch, no network, no GPU
at import or on the default path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Bundled offline snapshot of curated AD-relevant pathway gene sets
# ---------------------------------------------------------------------------
_SNAPSHOT_PATH = Path(__file__).with_name("data") / "ad_pathway_genesets.json"
_SNAPSHOT_SOURCE = "ad_pathway_snapshot_v1"
_MODEL = "ORA_hypergeometric_BH"
_VALID_SOURCES = frozenset({"KEGG", "Reactome", "GO"})


# ---------------------------------------------------------------------------
# Structured returns (dataclass + to_dict + provenance-stamp, sibling style)
# ---------------------------------------------------------------------------


@dataclass
class PathwayGeneSet:
    """One curated pathway gene set from the bundled snapshot.

    ``genes`` is a hand-curated, publicly-documented SUBSET of the real pathway
    membership (see the JSON provenance block) — never presented as the complete
    KEGG/Reactome/GO set. ``source`` is one of KEGG|Reactome|GO and ``source_id``
    is the public identifier (e.g. hsa05010, R-HSA-977225, GO:1902949)."""
    id: str
    name: str
    source: str            # "KEGG" | "Reactome" | "GO"
    source_id: str         # e.g. "hsa05010"
    genes: list[str]
    description: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "source": self.source,
            "source_id": self.source_id,
            "genes": list(self.genes),
            "description": self.description,
        }


@dataclass
class PathwayEnrichment:
    """One pathway's over-representation result for a query gene set.

    ``p_value`` is the one-sided hypergeometric/Fisher tail P(overlap >= observed);
    ``q_value`` is the Benjamini-Hochberg FDR over the pathways actually tested
    (overlap >= min_overlap). ``odds_ratio`` is the 2x2 sample odds ratio.
    Provenance is fully stamped: ``snapshot_source`` = "ad_pathway_snapshot_v1",
    ``source``/``source_id`` = the curated set's origin, ``method`` = the test used,
    ``model`` = "ORA_hypergeometric_BH", and ``background_size``/``pathway_size``/
    ``query_size`` make the (snapshot-relative) p-value interpretable rather than a
    genome-wide Enrichr p-value."""
    pathway: str
    source: str
    source_id: str
    p_value: float
    q_value: float                 # BH-FDR
    odds_ratio: float
    overlap_genes: list[str]
    overlap_size: int
    pathway_size: int
    query_size: int
    background_size: int
    method: str                    # "ORA_hypergeometric" | "ORA_fisher"
    model: str = _MODEL            # "ORA_hypergeometric_BH"
    snapshot_source: str = _SNAPSHOT_SOURCE

    def to_dict(self) -> dict:
        return {
            "pathway": self.pathway,
            "source": self.source,
            "source_id": self.source_id,
            "p_value": self.p_value,
            "q_value": self.q_value,
            "odds_ratio": self.odds_ratio,
            "overlap_genes": list(self.overlap_genes),
            "overlap_size": self.overlap_size,
            "pathway_size": self.pathway_size,
            "query_size": self.query_size,
            "background_size": self.background_size,
            "method": self.method,
            "model": self.model,
            "snapshot_source": self.snapshot_source,
        }


# ---------------------------------------------------------------------------
# Bundled-snapshot loader (matches _load_snapshot() in pi4ad/opentargets)
# ---------------------------------------------------------------------------

_GENESET_CACHE: Optional[list[PathwayGeneSet]] = None


def load_pathway_gene_sets() -> list[PathwayGeneSet]:
    """Load (and cache) the bundled curated pathway gene sets.

    Never raises: a missing/corrupt snapshot JSON, or any set with an invalid
    source / empty genes, yields an honest [] (or is skipped) rather than a
    fabricated set. Matches the offline-first pattern in ``pi4ad._load_snapshot``.
    """
    global _GENESET_CACHE
    if _GENESET_CACHE is not None:
        return list(_GENESET_CACHE)
    out: list[PathwayGeneSet] = []
    try:
        with open(_SNAPSHOT_PATH, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        for row in blob.get("gene_sets", []):
            try:
                source = str(row["source"]).strip()
                genes = [str(g).strip().upper() for g in row["genes"]
                         if str(g).strip()]
                if source not in _VALID_SOURCES or not genes:
                    continue
                # de-duplicate genes deterministically, preserving first order
                seen: set[str] = set()
                uniq = [g for g in genes if not (g in seen or seen.add(g))]
                out.append(PathwayGeneSet(
                    id=str(row["id"]).strip(),
                    name=str(row["name"]).strip(),
                    source=source,
                    source_id=str(row["source_id"]).strip(),
                    genes=uniq,
                    description=str(row.get("description", "")).strip(),
                ))
            except (KeyError, TypeError, ValueError):
                continue
    except Exception:
        return []
    _GENESET_CACHE = out
    return list(out)


def _clean_query(query_genes) -> list[str]:
    """Upper-case + de-duplicate a query gene list (NO alias expansion).

    Deterministic: preserves first-seen order, drops blanks. Order-invariance of
    the final ranking is guaranteed downstream (results are sorted by p-value)."""
    if not query_genes:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for g in query_genes:
        if g is None:
            continue
        s = str(g).strip().upper()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _bh_fdr(p_sorted: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR q-values for p-values sorted ASCENDING.

    q_i = min over j>=i of (p_j * m / (j+1)), clamped to [p_i, 1.0]. Monotonic
    non-decreasing in BH order and always >= the raw p-value."""
    m = len(p_sorted)
    q = [0.0] * m
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = p_sorted[i] * m / rank
        prev = min(prev, val)
        q[i] = min(1.0, max(prev, p_sorted[i]))
    return q


# ---------------------------------------------------------------------------
# The analyzer
# ---------------------------------------------------------------------------


class PathwayEnrichmentAnalyzer:
    """Offline over-representation analyzer over the bundled AD pathway snapshot.

    ``prefer_offline`` is accepted for signature parity with the sibling adapters
    (pi4ad.PI4AD, opentargets); this analyzer is ALWAYS offline (there is no live
    Enrichr/g:Profiler path here — doing so would be a different, unbundled claim),
    so the flag currently only documents intent and never triggers a network call.
    """

    def __init__(self, *, prefer_offline: bool = True) -> None:
        self.prefer_offline = prefer_offline

    def load_gene_sets(self) -> list[PathwayGeneSet]:
        """Return the bundled curated pathway gene sets (offline, cached)."""
        return load_pathway_gene_sets()

    def enrich(
        self,
        query_genes: list[str],
        *,
        method: str = "hypergeom",
        background_size: Optional[int] = None,
        min_overlap: int = 1,
        fdr_alpha: float = 0.05,
    ) -> list[PathwayEnrichment]:
        """Run ORA of ``query_genes`` against every curated pathway set.

        ``method`` is "hypergeom" (default, hypergeometric tail) or "fisher"
        (Fisher's exact, one-sided greater); both give the same overlap counts and
        near-identical one-sided p-values. ``background_size`` overrides the
        default universe size (union of all snapshot genes UNION the query); it is
        stamped on every row. Pathways with overlap < ``min_overlap`` are EXCLUDED,
        and BH-FDR is computed only over the pathways actually tested.

        DEGRADE-NEVER-RAISE: empty/whitespace query -> []; query with no universe
        overlap -> []; missing scipy -> []; no snapshot -> []. ``fdr_alpha`` is
        accepted for API parity (BH q-values are returned for all rows; the caller
        decides the cutoff) and does not gate the returned rows."""
        query = _clean_query(query_genes)
        if not query:
            return []
        gene_sets = self.load_gene_sets()
        if not gene_sets:
            return []
        try:
            from scipy.stats import hypergeom, fisher_exact
        except Exception:
            # Missing scipy -> honest [] rather than a fabricated result.
            return []

        use_fisher = str(method).lower() in ("fisher", "fisher_exact", "ora_fisher")
        method_tag = "ORA_fisher" if use_fisher else "ORA_hypergeometric"

        # Universe = union of all snapshot pathway genes UNION query genes.
        universe: set[str] = set(query)
        for gs in gene_sets:
            universe.update(gs.genes)
        default_bg = len(universe)
        # Documented default; explicit override honored and stamped on every row.
        N = int(background_size) if background_size and background_size > 0 else default_bg
        N = max(N, default_bg)  # never let an override shrink below the real union

        query_set = set(query)
        n = len(query_set)      # query size (all are valid symbols)

        rows: list[PathwayEnrichment] = []
        min_ov = max(1, int(min_overlap))
        for gs in gene_sets:
            pw_genes = set(gs.genes)
            K = len(pw_genes)
            overlap = sorted(query_set & pw_genes)
            k = len(overlap)
            if k < min_ov:
                # Excluded from output AND from the BH correction — never scored.
                continue
            # Clamp to keep hypergeometric arguments valid under an override N.
            Kc = min(K, N)
            nc = min(n, N)
            kc = min(k, Kc, nc)
            if use_fisher:
                # 2x2: [[in_pw & in_q, in_pw & not_q],[not_pw & in_q, not_pw & not_q]]
                a = kc
                b = max(Kc - kc, 0)
                c = max(nc - kc, 0)
                d = max(N - Kc - nc + kc, 0)
                try:
                    odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
                    p = float(p)
                    odds_ratio = float(odds)
                except Exception:
                    continue
            else:
                # P(X >= k) = sf(k-1, N, K, n) — the sf(k-1,...) off-by-one is load-
                # bearing: sf(k,...) would silently understate significance.
                p = float(hypergeom.sf(kc - 1, N, Kc, nc))
                odds_ratio = _sample_odds_ratio(kc, Kc, nc, N)
            rows.append(PathwayEnrichment(
                pathway=gs.name,
                source=gs.source,
                source_id=gs.source_id,
                p_value=round(max(0.0, min(1.0, p)), 12),
                q_value=1.0,  # filled after BH below
                odds_ratio=round(odds_ratio, 6) if odds_ratio == odds_ratio else float("inf"),
                overlap_genes=overlap,
                overlap_size=k,
                pathway_size=K,
                query_size=n,
                background_size=N,
                method=method_tag,
            ))

        if not rows:
            return []

        # Deterministic order: p-value asc, then larger overlap, then id/name.
        rows.sort(key=lambda r: (r.p_value, -r.overlap_size, r.source_id, r.pathway))
        q_vals = _bh_fdr([r.p_value for r in rows])
        for r, q in zip(rows, q_vals):
            r.q_value = round(q, 12)
        return rows


def _sample_odds_ratio(k: int, K: int, n: int, N: int) -> float:
    """2x2 sample (cross-product) odds ratio for the ORA contingency table.

    a=k, b=n-k (query not in pathway), c=K-k (pathway not in query),
    d=N-K-n+k (neither). Returns +inf for a zero denominator (perfect
    over-representation) rather than a fabricated finite value."""
    a = k
    b = max(n - k, 0)
    c = max(K - k, 0)
    d = max(N - K - n + k, 0)
    denom = b * c
    if denom == 0:
        return float("inf") if a > 0 else 0.0
    return (a * d) / denom


# ---------------------------------------------------------------------------
# Module-level conveniences (thin wrappers for harness callers)
# ---------------------------------------------------------------------------


def enrich_genes(
    query_genes: list[str],
    *,
    method: str = "hypergeom",
    background_size: Optional[int] = None,
    min_overlap: int = 1,
    fdr_alpha: float = 0.05,
    prefer_offline: bool = True,
) -> list[PathwayEnrichment]:
    """ORA of a raw gene list against the curated AD pathway snapshot."""
    return PathwayEnrichmentAnalyzer(prefer_offline=prefer_offline).enrich(
        query_genes,
        method=method,
        background_size=background_size,
        min_overlap=min_overlap,
        fdr_alpha=fdr_alpha,
    )


def enrich_mechanism(
    mechanism: str,
    *,
    method: str = "hypergeom",
    background_size: Optional[int] = None,
    prefer_offline: bool = True,
) -> list[PathwayEnrichment]:
    """ORA of ``translation.MECHANISM_GENES[mechanism]`` (read-only).

    Reads MECHANISM_GENES read-only and defaults to ``amyloid_cascade`` on an
    unknown key (mirroring translation.py's own fallback). Any failure to import
    the mechanism map yields [] and never perturbs the translation chain."""
    try:
        from ..harness.translation import MECHANISM_GENES
    except Exception:
        return []
    genes = MECHANISM_GENES.get(mechanism) or MECHANISM_GENES.get("amyloid_cascade")
    if not genes:
        return []
    return enrich_genes(
        list(genes),
        method=method,
        background_size=background_size,
        prefer_offline=prefer_offline,
    )


def enrich_propagation(
    seed_genes: list[str],
    *,
    include_hubs: bool = True,
    method: str = "hypergeom",
    background_size: Optional[int] = None,
    prefer_offline: bool = True,
) -> list[PathwayEnrichment]:
    """ORA of the seeds (+ non-seed STRING hubs) surfaced by pi4ad.propagate_hits.

    Runs ``pi4ad.propagate_hits(seed_genes)`` and enriches the seed genes plus, when
    ``include_hubs``, the NON-seed hub genes the propagation lights up. Fully
    wrapped: if propagation fails or scipy is missing it falls back to enriching the
    raw seeds, and any failure yields [] — it never perturbs the translation chain.
    ``overlap_genes`` on each row are exactly the set intersection with the pathway."""
    query = _clean_query(seed_genes)
    if not query:
        return []
    try:
        from . import pi4ad
        nodes = pi4ad.propagate_hits(list(query), prefer_offline=prefer_offline)
        combined = list(query)
        if include_hubs:
            for nd in nodes:
                if getattr(nd, "is_hub", False):
                    combined.append(nd.gene)
        return enrich_genes(
            combined,
            method=method,
            background_size=background_size,
            prefer_offline=prefer_offline,
        )
    except Exception:
        # Degrade: still try the raw seeds; if that too fails, honest [].
        try:
            return enrich_genes(
                list(query),
                method=method,
                background_size=background_size,
                prefer_offline=prefer_offline,
            )
        except Exception:
            return []
