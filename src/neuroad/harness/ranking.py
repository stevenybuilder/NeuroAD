"""
ranking — the shared, importable multi-signal COMPOSITE target-ranking helper.

The referee's default target ranking (``harness.translation._rank_targets``) sorts a
mechanism's candidate genes by ONE signal: PI4AD priority. That is coarse. This
module factors out the transparent COMPOSITE used by ``scripts/rank_candidates.py``
so BOTH the CLI script and the referee can share ONE implementation instead of
duplicating it. It fuses four independent signals the pipeline already fetches, each
min-max-normalized to [0,1] and weighted by how much the live validation trusts it:

  signal            source                            weight   why
  ----------------  --------------------------------  ------   -----------------------
  pi4ad_priority    PI4AD portal 0-10 (14,676 genes)   0.30    prioritisation prior
  ot_assoc_heldout  Open Targets non-genetic assoc     0.35    clean non-circular signal
  net_centrality    STRING-RWR propagated score        0.20    network support from AD seeds
  struct_confidence AlphaFold mean pLDDT / 100         0.15    druggability (foldedness) proxy

Composite = Σ wᵢ·normalized_signalᵢ over the signals PRESENT (weights renormalized
so a missing signal never silently zeros a gene). Every row keeps its raw per-signal
values + how many signals were present + a ``source`` stamp, so the score is fully
auditable — decision-support, NOT an efficacy claim.

Offline-first and degrade-never-raise, matching the rest of the pipeline: any adapter
failure yields ``None`` for that signal (a missing signal, honestly stamped) rather
than raising. ``prefer_offline=True`` keeps the whole chain on bundled snapshots.
"""
from __future__ import annotations

import logging
from typing import Optional

_log = logging.getLogger("neuroad.harness.ranking")

#: Weight each normalized signal contributes to the composite. Must be the single
#: source of truth shared by the CLI script and the referee (do not duplicate).
WEIGHTS: dict[str, float] = {
    "pi4ad_priority": 0.30,
    "ot_assoc_heldout": 0.35,
    "net_centrality": 0.20,
    "struct_confidence": 0.15,
}


def _minmax(vals: dict) -> dict:
    """Min-max-normalize present values to [0,1]; ``None`` stays ``None``."""
    xs = [v for v in vals.values() if v is not None]
    if not xs:
        return {k: None for k in vals}
    lo, hi = min(xs), max(xs)
    if hi <= lo:
        return {k: (1.0 if v is not None else None) for k, v in vals.items()}
    return {k: ((v - lo) / (hi - lo) if v is not None else None)
            for k, v in vals.items()}


def _pooled_genes() -> list[str]:
    """The full candidate universe (all mechanisms pooled), sorted & de-duped."""
    from .translation import MECHANISM_GENES  # lazy: avoids import cycle
    return sorted({g for gs in MECHANISM_GENES.values() for g in gs})


def gather_signals(prefer_offline: bool, genes: Optional[list[str]] = None):
    """Fetch the four live signals for ``genes`` (default: the pooled universe).

    Returns ``(genes, sig)`` where ``sig`` maps each signal name to a
    ``{gene: raw_value_or_None}`` dict. Normalization spans whatever ``genes`` is
    passed, so pass the FULL pooled universe when you want cross-mechanism-
    comparable composites (then subset with :func:`composite`). Offline-first;
    every leg degrades to ``None`` rather than raising."""
    from .. import harness  # noqa: F401  (package init side-effects, if any)
    from . import validation as V
    from ..integrations import pi4ad as P
    from ..integrations.alphafold import AlphaFoldClient, AD_PROTEIN_MAP

    if genes is None:
        genes = _pooled_genes()

    # 1. PI4AD priority (0-10) over the table.
    pi4ad_raw: dict = {}
    pi4ad_rank: dict = {}
    try:
        pit = {g: P.gene_priority(g, prefer_offline=prefer_offline) for g in genes}
        pi4ad_raw = {g: (r.priority_score if r else None) for g, r in pit.items()}
        pi4ad_rank = {g: (r.rank if r else None) for g, r in pit.items()}
    except Exception as exc:  # noqa: BLE001
        _log.debug("PI4AD signal gather failed: %r", exc)
        pi4ad_raw = {g: None for g in genes}
        pi4ad_rank = {g: None for g in genes}

    # 2. Open Targets non-genetic (held-out) association — the clean signal.
    ot_raw: dict = {g: None for g in genes}
    try:
        ot_uni = V.opentargets_universe(prefer_offline=prefer_offline,
                                        evidence="non_genetic")
        ot_map = {g.upper(): s for g, s in zip(ot_uni.genes, ot_uni.scores)}
        ot_raw = {g: ot_map.get(g.upper()) for g in genes}
    except Exception as exc:  # noqa: BLE001
        _log.debug("Open Targets signal gather failed: %r", exc)

    # 3. STRING-RWR network centrality: propagate from ALL candidate genes, read
    #    each gene's propagated mass.
    net_raw: dict = {g: None for g in genes}
    deg_map: dict = {}
    try:
        nodes = P.propagate_hits(genes, prefer_offline=prefer_offline,
                                 add_nodes=0 if prefer_offline else 60)
        net_map = {n.gene.upper(): n.propagated_score for n in nodes}
        deg_map = {n.gene.upper(): n.degree for n in nodes}
        net_raw = {g: net_map.get(g.upper()) for g in genes}
    except Exception as exc:  # noqa: BLE001
        _log.debug("STRING-RWR signal gather failed: %r", exc)

    # 4. AlphaFold structural confidence (mean pLDDT / 100).
    struct_raw: dict = {g: None for g in genes}
    try:
        afc = AlphaFoldClient(prefer_offline=prefer_offline)
        for g in genes:
            acc = AD_PROTEIN_MAP.get(g.upper())
            s = afc.fetch_structure(acc or g, recompute_plddt=False)
            struct_raw[g] = (s.mean_plddt / 100.0
                             if s.mean_plddt is not None else None)
    except Exception as exc:  # noqa: BLE001
        _log.debug("AlphaFold signal gather failed: %r", exc)

    return genes, {
        "pi4ad_priority": pi4ad_raw, "pi4ad_rank": pi4ad_rank,
        "ot_assoc_heldout": ot_raw, "net_centrality": net_raw,
        "net_degree": deg_map, "struct_confidence": struct_raw,
    }


def composite(genes: list[str], sig: dict) -> list[dict]:
    """Fuse ``sig`` into a ranked list of composite rows for ``genes``.

    Weights are renormalized over the signals PRESENT for each gene, so a missing
    signal never silently zeros it. Every row keeps its raw per-signal values, the
    signal count, and a ``source`` stamp for full auditability."""
    norm = {k: _minmax(sig[k]) for k in WEIGHTS}
    rows: list[dict] = []
    for g in genes:
        present = {k: norm[k][g] for k in WEIGHTS if norm[k].get(g) is not None}
        wsum = sum(WEIGHTS[k] for k in present)
        score = (sum(WEIGHTS[k] * present[k] for k in present) / wsum
                 if wsum > 0 else None)
        rows.append({
            "gene": g,
            "composite_score": round(score, 4) if score is not None else None,
            "n_signals": len(present),
            "signals_present": sorted(present),
            "pi4ad_priority": sig["pi4ad_priority"].get(g),
            "pi4ad_rank": sig["pi4ad_rank"].get(g),
            "ot_assoc_heldout": (round(sig["ot_assoc_heldout"][g], 4)
                                 if sig["ot_assoc_heldout"].get(g) is not None
                                 else None),
            "net_centrality": (round(sig["net_centrality"][g], 6)
                               if sig["net_centrality"].get(g) is not None
                               else None),
            "net_degree": sig["net_degree"].get(g.upper()),
            "struct_plddt": (round(sig["struct_confidence"][g] * 100, 1)
                             if sig["struct_confidence"].get(g) is not None
                             else None),
            # per-row provenance so a composite row is self-describing like the
            # single-signal rows the referee already emits.
            "source": "composite_multi_signal",
        })
    rows.sort(key=lambda r: (r["composite_score"] is not None,
                             r["composite_score"] or 0), reverse=True)
    return rows


def composite_targets(mechanism: str, *, prefer_offline: bool = True) -> list[dict]:
    """Composite-rank ONE mechanism's candidate genes (referee entry point).

    Gathers signals over the FULL pooled universe (so scores are cross-mechanism
    comparable via a shared normalization), then returns only ``mechanism``'s
    genes, ranked. Offline-first; returns ``[]`` on total failure rather than
    raising, so callers can fall back to the single-signal path."""
    from .translation import MECHANISM_GENES  # lazy: avoids import cycle
    mech_genes = MECHANISM_GENES.get(mechanism) or MECHANISM_GENES["amyloid_cascade"]
    try:
        _all, sig = gather_signals(prefer_offline=prefer_offline)
        return composite(mech_genes, sig)
    except Exception as exc:  # noqa: BLE001
        _log.debug("composite_targets failed for %s: %r", mechanism, exc)
        return []
