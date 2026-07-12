#!/usr/bin/env python3
"""
run_target_prioritization_validation.py — LIVE, full-universe validation of the
discovery half of the pipeline (L5 PI4AD ranking + the Output target-prioritization
layer), plus a live PI4AD full-table / STRING-RWR network-propagation flesh-out.

The offline validation harness (harness/validation.py) is explicit that its bundled
snapshots are curation-biased (canonical AD genes deliberately included, tiny
background) and that a RIGOROUS verdict requires the LIVE run. This script IS that
run: prefer_offline=False fetches

  * the full ~14,676-gene PI4AD portal table (http://www.genetictargets.com/PI4AD/ad), and
  * the live Open Targets GraphQL AD target-association universe,

then asks the one honest question — when we rank AD proteins by priority /
association, do the genes we INDEPENDENTLY know matter (Bellenguez 2022 GWAS risk
genes; FDA-approved-drug targets) float to the top? — with precision@k, ROC-AUC,
and a 1000-shuffle label-permutation p-value. Both honesty guards from the harness
are exercised:

  Guard 1 (circularity): Open Targets' overall association_score is BUILT FROM the
  genetic/clinical datatypes that DEFINE the gold sets, so we report BOTH the naive
  (circular, optimistic) AUC and the held-out AUC (predict the GWAS gold from
  non-genetic evidence; the drug gold from non-clinical evidence). The held-out
  number is the honest one.

  Guard 2 (curation bias): eliminated here by construction — this is the full live
  universe, not the seeded offline snapshot. Every report is provenance-stamped with
  source=live and the actual background_size.

Writes reports/target_prioritization_validation.json and a human-readable .md.
Deterministic given --seed. Degrades honestly: if a live endpoint is unreachable the
underlying adapters fall back to the (clearly-labeled) offline snapshot rather than
raising, and the report's `source` field will say so.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

# repo/src on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neuroad.harness import validation as V  # noqa: E402
from neuroad.integrations import pi4ad as P  # noqa: E402
from neuroad.integrations.alphafold import AlphaFoldClient, AD_PROTEIN_MAP  # noqa: E402


def _fmt(x, nd=4):
    return None if x is None else round(float(x), nd)


def resolve_uniprot_live(symbol: str, *, timeout: int = 15) -> str:
    """Live gene-symbol -> reviewed human UniProt accession (keyless REST API).

    Swaps the hardcoded 13-gene AD_PROTEIN_MAP for real resolution so AlphaFold can
    fetch a structure for ANY ranked target. Queries rest.uniprot.org for the
    reviewed (Swiss-Prot) human entry; returns "" on any failure (caller skips it —
    never fabricates an accession)."""
    try:
        import requests
        q = (f'gene_exact:{symbol} AND organism_id:9606 AND reviewed:true')
        r = requests.get(
            "https://rest.uniprot.org/uniprotkb/search",
            params={"query": q, "fields": "accession", "format": "tsv", "size": 1},
            timeout=timeout)
        if r.status_code != 200:
            return ""
        rows = [ln for ln in r.text.splitlines() if ln.strip()]
        if len(rows) < 2:
            return ""
        return rows[1].split("\t")[0].strip()
    except Exception:
        return ""


def run_validation(prefer_offline: bool, n_perm: int, seed: int) -> dict:
    """Run the three honest tests + the two circular/optimistic comparators."""
    # --- honest (held-out / non-circular) suite -------------------------------
    pi4ad_gwas = V.validate_pi4ad_gwas(
        prefer_offline=prefer_offline, n_perm=n_perm, seed=seed)
    ot_gwas_held = V.validate_opentargets_gwas(
        prefer_offline=prefer_offline, held_out=True, n_perm=n_perm, seed=seed)
    ot_drugs_held = V.validate_opentargets_drugs(
        prefer_offline=prefer_offline, held_out=True, n_perm=n_perm, seed=seed)
    # --- circular/optimistic comparators (Guard 1 illustration) ---------------
    ot_gwas_naive = V.validate_opentargets_gwas(
        prefer_offline=prefer_offline, held_out=False, n_perm=n_perm, seed=seed)
    ot_drugs_naive = V.validate_opentargets_drugs(
        prefer_offline=prefer_offline, held_out=False, n_perm=n_perm, seed=seed)
    return {
        "honest_tests": {
            "pi4ad_vs_gwas": pi4ad_gwas.to_dict(),
            "opentargets_vs_gwas_heldout_nongenetic": ot_gwas_held.to_dict(),
            "opentargets_vs_drugs_heldout_nonclinical": ot_drugs_held.to_dict(),
        },
        "circular_comparators": {
            "opentargets_vs_gwas_overall_CIRCULAR": ot_gwas_naive.to_dict(),
            "opentargets_vs_drugs_overall_CIRCULAR": ot_drugs_naive.to_dict(),
        },
    }


def run_pi4ad_fleshout(prefer_offline: bool) -> dict:
    """Live PI4AD full-table ranking + STRING-RWR propagation from the GWAS seeds."""
    client = P.PI4AD(prefer_offline=prefer_offline)
    top = client.rank_genes(top_n=25)
    full = client.rank_genes(top_n=1_000_000)  # whole loaded universe
    table_source = full[0].source if full else "offline_snapshot"

    # Network propagation: seed the RWR with the independently-known GWAS risk
    # genes and report the NON-seed STRING hubs their propagation lights up. This
    # is the honest "what does the network surface around known genetics" view.
    # add_nodes expands the live STRING network with interaction partners beyond
    # the seeds, so there is a real neighborhood to surface hubs in (a seed-only
    # fetch yields a degenerate graph with no non-seed nodes).
    seeds = sorted(V.GWAS_GOLD.symbols)
    nodes = P.propagate_hits(seeds, prefer_offline=prefer_offline, method="rwr",
                             add_nodes=0 if prefer_offline else 60)
    hubs = [n.to_dict() for n in nodes if n.is_hub]
    prop_source = nodes[0].source if nodes else "string_v12_snapshot"

    return {
        "pi4ad_table": {
            "source": table_source,
            "n_genes_loaded": len(full),
            "top25": [g.to_dict() for g in top],
        },
        "string_rwr_propagation": {
            "source": prop_source,
            "seed_genes": seeds,
            "n_seeds_in_graph": sum(1 for n in nodes if n.is_seed),
            "n_subgraph_nodes": len(nodes),
            "n_hubs": len(hubs),
            "hubs": hubs,
        },
    }


def run_alphafold_structures(prefer_offline: bool, genes: list[str]) -> dict:
    """LIVE AlphaFold DB structural confidence for the given AD targets.

    Proves AlphaFold is ACTUALLY used (not hardcoded): the keyless EBI AlphaFold
    DB REST API (alphafold.ebi.ac.uk/api/prediction/{acc}) returns real
    AlphaFold2 monomer structures; recompute_plddt=True downloads each CIF and
    averages the CA-atom B-factor column for an exact mean pLDDT. Every record is
    provenance-stamped source=live|offline_snapshot — a fallback is never dressed
    up as live. No account/token required (that is the AF3 *Server*, a different
    product; see the target_prioritization report notes)."""
    client = AlphaFoldClient(prefer_offline=prefer_offline)
    out = []
    for g in genes:
        # Resolve to a UniProt accession: hardcoded AD map first (fast, offline),
        # else LIVE UniProt REST resolution so ANY ranked target is coverable.
        acc = AD_PROTEIN_MAP.get(g.upper())
        if not acc and not prefer_offline:
            acc = resolve_uniprot_live(g)
        query = acc or g
        s = client.fetch_structure(query, recompute_plddt=not prefer_offline)
        rec = s.to_dict()
        rec["gene_symbol"] = rec.get("gene_symbol") or g
        rec["uniprot_resolved_via"] = (
            "ad_map" if AD_PROTEIN_MAP.get(g.upper()) else
            ("uniprot_live" if acc else "unresolved"))
        out.append(rec)
    n_live = sum(1 for r in out if r.get("source") == "live")
    return {
        "api": "https://alphafold.ebi.ac.uk/api/prediction/{accession} (keyless)",
        "note": ("AlphaFold DB = free/keyless precomputed monomer structures (used "
                 "here, LIVE). AlphaFold3 de-novo COMPLEX folding is the account/"
                 "weight-gated product; the open MIT Boltz-2 GPU job is its "
                 "license-clean substitute for the L6 complex step."),
        "n_requested": len(genes),
        "n_live": n_live,
        "structures": out,
    }


#: Tests whose ranking evidence still overlaps the gold-set definition, so a high
#: AUC is partly circular even in "held-out" framings. PI4AD's Priority Index
#: integrates genetic (incl. GWAS) evidence as an INPUT layer, so PI4AD-vs-GWAS is
#: NOT a clean out-of-evidence test — flag it rather than let it read as pristine.
_RESIDUAL_CIRCULARITY = {
    "pi4ad_vs_gwas": ("PI4AD's Priority Index integrates genetic/GWAS evidence as "
                      "an input, so ranking GWAS genes high is partly circular"),
}


def _verdict(honest: dict) -> str:
    """Honest multi-part verdict distinguishing clean vs residually-circular signal."""
    clean, caveated, null = [], [], []
    for name, rep in honest.items():
        auc, p = rep.get("roc_auc"), rep.get("permutation_p")
        sig = (auc is not None and p is not None and p < 0.05 and auc > 0.5)
        tag = f"{name} (AUC={auc:.3f}, p={p:.3f})" if auc is not None else name
        if not sig:
            null.append(tag)
        elif name in _RESIDUAL_CIRCULARITY:
            caveated.append(f"{tag} — CAVEAT: {_RESIDUAL_CIRCULARITY[name]}")
        else:
            clean.append(tag)
    parts = []
    if clean:
        parts.append("CLEAN non-circular signal at the full live universe: "
                     + "; ".join(clean) + ". This is genuine, honest evidence the "
                     "ranking surfaces independently-known AD-risk genes from "
                     "out-of-evidence signal.")
    if caveated:
        parts.append("Residually-circular (strong but NOT clean): "
                     + "; ".join(caveated) + ".")
    if null:
        parts.append("At/below chance (honest negative): " + "; ".join(null) + ".")
    parts.append("Overall: a rigorously-filtered, wet-lab-testable HYPOTHESIS "
                 "ENGINE (organoid/iPSC) — not a validated efficacy predictor. The "
                 "clean held-out signal is real but is prognostic-of-relevance, not "
                 "proof a target is druggable.")
    if not clean and not caveated:
        parts.insert(0, "No test clears chance without circularity.")
    return " ".join(parts)


def to_markdown(payload: dict) -> str:
    lines = ["# Target Prioritization — Validation (LIVE full universe)", ""]
    lines.append(f"_Generated {payload['generated_utc']}; "
                 f"prefer_offline={payload['prefer_offline']}, "
                 f"n_perm={payload['n_perm']}, seed={payload['seed']}._")
    lines.append("")
    lines.append("## Honest verdict")
    lines.append("")
    lines.append(payload["verdict"])
    lines.append("")
    lines.append("## Honest tests (non-circular)")
    lines.append("")
    lines.append("| Test | Source | N univ | N gold | AUC | perm p | Circularity |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, r in payload["validation"]["honest_tests"].items():
        circ = _RESIDUAL_CIRCULARITY.get(name, "clean (out-of-evidence)")
        lines.append(
            f"| {name} | {r['source']} | {r['background_size']} | {r['n_gold']} "
            f"| {_fmt(r['roc_auc'],3)} | {_fmt(r['permutation_p'],3)} | {circ} |")
    lines.append("")
    lines.append("## Circular comparators (Guard 1 — optimistic, NOT honest)")
    lines.append("")
    lines.append("| Test | Source | AUC (circular) | perm p |")
    lines.append("|---|---|---|---|")
    for name, r in payload["validation"]["circular_comparators"].items():
        lines.append(
            f"| {name} | {r['source']} | {_fmt(r['roc_auc'],3)} "
            f"| {_fmt(r['permutation_p'],3)} |")
    lines.append("")
    fo = payload["pi4ad_fleshout"]
    lines.append("## L5 PI4AD flesh-out")
    lines.append("")
    lines.append(f"- PI4AD table: **{fo['pi4ad_table']['n_genes_loaded']} genes** "
                 f"(source={fo['pi4ad_table']['source']}).")
    prop = fo["string_rwr_propagation"]
    lines.append(f"- STRING-RWR from {len(prop['seed_genes'])} GWAS seeds "
                 f"({prop['n_seeds_in_graph']} in graph, {prop['n_subgraph_nodes']} "
                 f"subgraph nodes, source={prop['source']}): "
                 f"**{prop['n_hubs']} non-seed hubs** surfaced.")
    if prop["hubs"]:
        lines.append("")
        lines.append("  Hubs: " + ", ".join(
            f"{h['gene']}(r{h['rank']},deg{h['degree']})" for h in prop["hubs"]))
    lines.append("")
    af = payload.get("alphafold_structures", {})
    if af:
        lines.append("## L6 AlphaFold structural confidence (LIVE, keyless AF DB)")
        lines.append("")
        lines.append(f"- {af['n_live']}/{af['n_requested']} targets fetched LIVE "
                     f"from `{af['api']}`.")
        lines.append("")
        lines.append("| Target | UniProt | resolved via | mean pLDDT | residues | source |")
        lines.append("|---|---|---|---|---|---|")
        for s in af["structures"]:
            lines.append(
                f"| {s['gene_symbol'] or '?'} | {s['uniprot'] or '—'} "
                f"| {s.get('uniprot_resolved_via','?')} | {_fmt(s['mean_plddt'],2)} "
                f"| {s['n_residues_scored']} | {s['source']} |")
        lines.append("")
        lines.append(f"> {af['note']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="use bundled snapshots (curation-biased); default is LIVE")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "..", "reports",
        "target_prioritization_validation.json"))
    args = ap.parse_args()
    prefer_offline = bool(args.offline)

    print(f"[validate] prefer_offline={prefer_offline} n_perm={args.n_perm} "
          f"seed={args.seed}", flush=True)
    validation = run_validation(prefer_offline, args.n_perm, args.seed)
    print("[validate] fleshing out PI4AD live table + STRING-RWR ...", flush=True)
    fleshout = run_pi4ad_fleshout(prefer_offline)

    # AlphaFold structural confidence, LIVE, for the top PI4AD targets (UniProt
    # accessions resolved live — NOT limited to the hardcoded AD map), capped to
    # keep the CIF-download count modest.
    top_genes = [g["gene"] for g in fleshout["pi4ad_table"]["top25"]][:12]
    af_genes = list(dict.fromkeys(top_genes + sorted(V.DRUG_GOLD.symbols)))
    print(f"[alphafold] LIVE structural confidence for {len(af_genes)} targets "
          f"(live UniProt resolution) ...", flush=True)
    alphafold = run_alphafold_structures(prefer_offline, af_genes)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "prefer_offline": prefer_offline,
        "n_perm": args.n_perm,
        "seed": args.seed,
        "validation": validation,
        "pi4ad_fleshout": fleshout,
        "alphafold_structures": alphafold,
        "verdict": _verdict(validation["honest_tests"]),
    }

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    md = out.rsplit(".", 1)[0] + ".md"
    with open(md, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(payload))

    print("\n=== HONEST VERDICT ===")
    print(payload["verdict"])
    print(f"\n[wrote] {out}\n[wrote] {md}")


if __name__ == "__main__":
    main()
