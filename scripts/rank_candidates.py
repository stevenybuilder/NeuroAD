#!/usr/bin/env python3
"""
rank_candidates.py — the NARROWED, multi-signal candidate-protein ranking.

Today the referee's target ranking (harness/translation._rank_targets) sorts a
mechanism's candidate genes by ONE signal: PI4AD priority. That is coarse. This
script fleshes it out into a transparent COMPOSITE that fuses up to five independent,
live signals the pipeline already fetches, each normalized to [0,1] and weighted by
how much the live validation trusts it:

  signal            source (LIVE)                     weight   why
  ----------------  --------------------------------  ------   -----------------------
  pi4ad_priority    PI4AD portal 0-10 (14,676 genes)   0.30    prioritisation prior (residually
                                                                circular vs GWAS — capped weight)
  ot_assoc_heldout  Open Targets non-genetic assoc     0.35    the ONE clean non-circular signal
                                                                (validated AUC 0.728, p=0.003)
  net_centrality    STRING-RWR propagated score        0.20    network support from known-AD seeds
  struct_confidence AlphaFold mean pLDDT / 100         0.15    druggability proxy (foldedness)
  boltz_confidence  Boltz-2 best complex confidence    0.15    OPTIONAL structural-targeting signal:
                                                                gene's best committed Boltz-2 complex
                                                                confidence (iptm/ptm/confidence_score).
                                                                PRESENT only when a REAL GPU snapshot is
                                                                committed (boltz.has_precomputed_results);
                                                                otherwise ABSENT (None) — never fabricated.

Composite = Σ wᵢ·normalized_signalᵢ over the signals PRESENT (weights renormalized
so a missing signal never silently zeros a gene). Because renormalization is over
PRESENT signals only, the Boltz-2 5th signal is purely additive: with no committed
Boltz snapshot it is absent for every gene and the composite is IDENTICAL to the
prior 4-signal fusion (weights renormalize to the same 0.30/0.35/0.20/0.15 split).
Every row keeps its raw signals + which were present, so the score is fully
auditable — this is decision-support, NOT an efficacy claim (see
reports/target_prioritization_validation.md).

Writes reports/candidate_ranking.json (+ .md). Live by default; --offline for the
deterministic bundled snapshots.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neuroad.harness import validation as V            # noqa: E402
from neuroad.harness.translation import MECHANISM_GENES  # noqa: E402
from neuroad.integrations import pi4ad as P            # noqa: E402
from neuroad.integrations.alphafold import AlphaFoldClient, AD_PROTEIN_MAP  # noqa: E402
from neuroad.integrations import boltz as BZ            # noqa: E402

# Five-weight scheme. The first four are the original, validated live signals; the
# fifth (boltz_confidence) is an OPTIONAL structural-targeting signal that is only
# ever PRESENT when a REAL committed Boltz-2 GPU snapshot exists. Because the
# composite renormalizes weights over the signals PRESENT per gene, adding this
# entry is purely additive: when the Boltz signal is absent for every gene the
# remaining four renormalize to exactly their prior 0.30/0.35/0.20/0.15 split, so
# the 4-signal composite is unchanged.
WEIGHTS = {
    "pi4ad_priority": 0.30,
    "ot_assoc_heldout": 0.35,
    "net_centrality": 0.20,
    "struct_confidence": 0.15,
    "boltz_confidence": 0.15,
}


def _minmax(vals: dict) -> dict:
    xs = [v for v in vals.values() if v is not None]
    if not xs:
        return {k: None for k in vals}
    lo, hi = min(xs), max(xs)
    if hi <= lo:
        return {k: (1.0 if v is not None else None) for k, v in vals.items()}
    return {k: ((v - lo) / (hi - lo) if v is not None else None)
            for k, v in vals.items()}


def boltz_signal(genes, prefer_offline: bool) -> dict:
    """Per-gene Boltz-2 structural-targeting score (best complex confidence).

    HONESTY: PRESENT only when a REAL committed Boltz-2 snapshot exists
    (``boltz.has_precomputed_results()``). When no snapshot is committed the whole
    signal is ABSENT — every gene maps to None, the min-max normalizer drops it, and
    the composite renormalizes over the remaining signals (4-signal fusion unchanged).

    For a present snapshot, each gene's score is its BEST available complex confidence
    across every committed complex that gene participates in, taken via the public
    ``BoltzClient.predict_complex`` API (confidence_score, else iptm, else ptm). A
    deferred/absent pair contributes nothing — no value is ever fabricated."""
    absent = {g: None for g in genes}
    try:
        if not BZ.has_precomputed_results():
            return absent
    except Exception:
        return absent
    try:
        client = BZ.BoltzClient(prefer_offline=prefer_offline)
    except Exception:
        return absent
    # Candidate genes may form complexes with any committed AD target, so score each
    # candidate against the full target set (plus the candidate pool) and keep the best.
    partners = sorted({p.upper() for p in BZ.AD_TARGETS} | {g.upper() for g in genes})
    best = {g: None for g in genes}
    for g in genes:
        gu = g.upper()
        for p in partners:
            if p == gu:
                continue
            try:
                t = client.predict_complex(g, p)
            except Exception:
                continue
            if getattr(t, "status", "deferred") != "predicted":
                continue
            conf = t.confidence_score
            if conf is None:
                conf = t.iptm
            if conf is None:
                conf = t.ptm
            if conf is None:
                continue
            if best[g] is None or conf > best[g]:
                best[g] = conf
    return best


def gather_signals(prefer_offline: bool):
    genes = sorted({g for gs in MECHANISM_GENES.values() for g in gs})

    # 1. PI4AD priority (0-10) over the live table.
    pit = {g: (P.gene_priority(g, prefer_offline=prefer_offline)) for g in genes}
    pi4ad_raw = {g: (r.priority_score if r else None) for g, r in pit.items()}
    pi4ad_rank = {g: (r.rank if r else None) for g, r in pit.items()}

    # 2. Open Targets non-genetic (held-out) association — the clean signal.
    ot_uni = V.opentargets_universe(prefer_offline=prefer_offline,
                                    evidence="non_genetic")
    ot_map = {g.upper(): s for g, s in zip(ot_uni.genes, ot_uni.scores)}
    ot_raw = {g: ot_map.get(g.upper()) for g in genes}

    # 3. STRING-RWR network centrality: propagate from ALL candidate genes, read
    #    each gene's propagated mass (live neighborhood via add_nodes).
    nodes = P.propagate_hits(genes, prefer_offline=prefer_offline,
                             add_nodes=0 if prefer_offline else 60)
    net_map = {n.gene.upper(): n.propagated_score for n in nodes}
    deg_map = {n.gene.upper(): n.degree for n in nodes}
    net_raw = {g: net_map.get(g.upper()) for g in genes}

    # 4. AlphaFold structural confidence (mean pLDDT / 100), LIVE keyless.
    afc = AlphaFoldClient(prefer_offline=prefer_offline)
    struct_raw = {}
    for g in genes:
        acc = AD_PROTEIN_MAP.get(g.upper())
        s = afc.fetch_structure(acc or g, recompute_plddt=False)
        struct_raw[g] = (s.mean_plddt / 100.0
                         if s.mean_plddt is not None else None)

    # 5. Boltz-2 structural-targeting (OPTIONAL): best committed complex confidence
    #    per gene. Absent (all None) unless a REAL GPU snapshot is committed.
    boltz_raw = boltz_signal(genes, prefer_offline)

    return genes, {
        "pi4ad_priority": pi4ad_raw, "pi4ad_rank": pi4ad_rank,
        "ot_assoc_heldout": ot_raw, "net_centrality": net_raw,
        "net_degree": deg_map, "struct_confidence": struct_raw,
        "boltz_confidence": boltz_raw,
    }


def composite(genes, sig):
    norm = {k: _minmax(sig[k]) for k in WEIGHTS}
    rows = []
    for g in genes:
        present = {k: norm[k][g] for k in WEIGHTS if norm[k][g] is not None}
        wsum = sum(WEIGHTS[k] for k in present)
        score = (sum(WEIGHTS[k] * present[k] for k in present) / wsum
                 if wsum > 0 else None)
        rows.append({
            "gene": g,
            "composite_score": round(score, 4) if score is not None else None,
            "n_signals": len(present),
            "pi4ad_priority": sig["pi4ad_priority"][g],
            "pi4ad_rank": sig["pi4ad_rank"][g],
            "ot_assoc_heldout": (round(sig["ot_assoc_heldout"][g], 4)
                                 if sig["ot_assoc_heldout"][g] is not None else None),
            "net_centrality": (round(sig["net_centrality"][g], 6)
                               if sig["net_centrality"][g] is not None else None),
            "net_degree": sig["net_degree"].get(g.upper()),
            "struct_plddt": (round(sig["struct_confidence"][g] * 100, 1)
                             if sig["struct_confidence"][g] is not None else None),
            "boltz_confidence": (round(sig["boltz_confidence"][g], 4)
                                 if sig["boltz_confidence"][g] is not None else None),
        })
    rows.sort(key=lambda r: (r["composite_score"] is not None,
                             r["composite_score"] or 0), reverse=True)
    return rows


def to_md(payload):
    L = ["# Narrowed candidate-protein ranking (composite, multi-signal)", "",
         f"_Generated {payload['generated_utc']}; prefer_offline="
         f"{payload['prefer_offline']}._", "",
         "Composite = weighted, min-max-normalized fusion of up to five LIVE signals "
         "(weights: PI4AD 0.30, OpenTargets-heldout 0.35, STRING-RWR 0.20, "
         "AlphaFold-pLDDT 0.15, Boltz-2 complex 0.15). The Boltz-2 signal is OPTIONAL "
         "and PRESENT only when a real committed GPU snapshot exists; otherwise it is "
         "absent and the composite is the prior 4-signal fusion. Decision-support only "
         "— see `target_prioritization_validation.md` for the honesty caveats.", "",
         "## Overall shortlist (all mechanisms pooled)", "",
         "| Rank | Gene | Composite | PI4AD (rank) | OT-heldout | STRING deg | pLDDT | Boltz |",
         "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(payload["overall"], 1):
        L.append(f"| {i} | {r['gene']} | {r['composite_score']} "
                 f"| {r['pi4ad_priority']} (r{r['pi4ad_rank']}) "
                 f"| {r['ot_assoc_heldout']} | {r['net_degree']} | {r['struct_plddt']} "
                 f"| {r['boltz_confidence']} |")
    L.append("")
    for mech, rows in payload["by_mechanism"].items():
        L.append(f"## {mech} — top candidates")
        L.append("")
        top = rows[0]
        L.append(f"**Lead: {top['gene']}** (composite {top['composite_score']}, "
                 f"{top['n_signals']}/5 signals). Runners-up: "
                 + ", ".join(f"{r['gene']}({r['composite_score']})"
                             for r in rows[1:4]) + ".")
        L.append("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "..", "reports", "candidate_ranking.json"))
    args = ap.parse_args()
    prefer_offline = bool(args.offline)

    print(f"[rank] gathering live signals (prefer_offline={prefer_offline}) ...",
          flush=True)
    genes, sig = gather_signals(prefer_offline)
    overall = composite(genes, sig)
    by_mech = {m: composite(gs, sig) for m, gs in MECHANISM_GENES.items()}

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "prefer_offline": prefer_offline,
        "weights": WEIGHTS,
        "overall": overall,
        "by_mechanism": by_mech,
    }
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(payload, open(out, "w"), indent=2)
    open(out.rsplit(".", 1)[0] + ".md", "w").write(to_md(payload))
    print("\n=== OVERALL TOP 8 ===")
    for i, r in enumerate(overall[:8], 1):
        print(f"{i:2}. {r['gene']:6} comp={r['composite_score']} "
              f"pi4ad={r['pi4ad_priority']} ot_ho={r['ot_assoc_heldout']} "
              f"deg={r['net_degree']} pLDDT={r['struct_plddt']} "
              f"boltz={r['boltz_confidence']}")
    print(f"\n[wrote] {out}\n[wrote] {out.rsplit('.',1)[0]+'.md'}")


if __name__ == "__main__":
    main()
