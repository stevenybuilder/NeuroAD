#!/usr/bin/env python3
"""
rank_candidates_learned.py — LEARNED, calibrated candidate ranking.

Replaces the hand-set composite weights (PI4AD 0.30 / OT-heldout 0.35 / STRING 0.20 /
pLDDT 0.15) with weights LEARNED from the data: rank-normalized features + an L2 logistic
model fit to predict clean gold-gene membership, leave-one-out cross-validated, with a
bootstrap CI, permutation p, and Brier calibration — plus the new LINCS efficacy feature.

Training universe: the Open Targets non-genetic top-N (the clean held-out background),
guaranteed to include the GWAS gold set (positives). Features are gathered live (PI4AD
table, OT non-genetic, STRING-RWR centrality, AlphaFold pLDDT for mapped AD proteins,
LINCS efficacy from the committed snapshot).

HONEST FRAMING: the gold label is tiny (~15 genes) → a SENSITIVITY ANALYSIS of what the
data implies, not a definitive re-weighting. Read-only; does not touch the demo path.

Outputs: reports/candidate_ranking_learned.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neuroad.harness import validation as V  # noqa: E402
from neuroad.harness import ranking_model as M  # noqa: E402
from neuroad.harness import ranking as R  # noqa: E402
from neuroad.integrations import pi4ad as P  # noqa: E402
from neuroad.integrations import lincs as L  # noqa: E402

#: The candidate genes the pipeline actually recommends (for spotlighting in the report).
SHORTLIST = ["APP", "TREM2", "BIN1", "BACE1", "MAPT", "APOE", "PSEN1", "CLU",
             "MAPK1", "HRAS", "ESR1"]


def build_universe(prefer_offline: bool, top_n: int):
    """OT non-genetic top-N, unioned with the GWAS gold set (positives guaranteed)."""
    ot = V.opentargets_universe(prefer_offline=prefer_offline,
                                evidence="non_genetic", top_n=top_n)
    ot_map = {g.upper(): s for g, s in zip(ot.genes, ot.scores)}
    genes = list(dict.fromkeys([g.upper() for g in ot.genes]
                               + sorted(V.GWAS_GOLD.symbols)))
    return genes, ot_map, ot.source


def gather_features(genes, ot_map, prefer_offline: bool):
    """Assemble the {feature: {gene: raw_value}} matrix. Each leg is best-effort."""
    feats = {f: {} for f in M.FEATURES}
    feats["ot_assoc_heldout"] = {g: ot_map.get(g) for g in genes}

    # PI4AD priority over the full table.
    try:
        pit = V.pi4ad_universe(prefer_offline=prefer_offline)
        pmap = {g.upper(): s for g, s in zip(pit.genes, pit.scores)}
        feats["pi4ad_priority"] = {g: pmap.get(g) for g in genes}
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] PI4AD gather failed: {exc}")

    # STRING-RWR centrality over the universe (best-effort; missing -> imputed).
    try:
        nodes = P.propagate_hits(genes, prefer_offline=prefer_offline, method="rwr",
                                 add_nodes=0)
        nmap = {n.gene.upper(): n.propagated_score for n in nodes}
        feats["net_centrality"] = {g: nmap.get(g) for g in genes}
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] STRING-RWR gather failed: {exc}")

    # AlphaFold pLDDT — only for mapped AD proteins (avoids hundreds of calls).
    try:
        from neuroad.integrations.alphafold import AlphaFoldClient, AD_PROTEIN_MAP
        afc = AlphaFoldClient(prefer_offline=prefer_offline)
        pl = {}
        for g in genes:
            acc = AD_PROTEIN_MAP.get(g.upper())
            if acc is None:
                continue
            s = afc.fetch_structure(acc, recompute_plddt=False)
            if s and s.mean_plddt is not None:
                pl[g] = s.mean_plddt
        feats["struct_plddt"] = {g: pl.get(g) for g in genes}
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] AlphaFold gather failed: {exc}")

    # LINCS efficacy proxy from the committed snapshot (deterministic).
    try:
        emap = L.efficacy_proxy_map(prefer_offline=True)  # snapshot: fast + stable
        feats["lincs_efficacy"] = {g: emap.get(g.upper()) for g in genes}
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] LINCS gather failed: {exc}")
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--top-n", type=int, default=600,
                    help="OT non-genetic universe size (training background)")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    prefer_offline = bool(args.offline)
    print(f"[learned] prefer_offline={prefer_offline} top_n={args.top_n}", flush=True)

    genes, ot_map, ot_src = build_universe(prefer_offline, args.top_n)
    print(f"  universe: {len(genes)} genes (OT source={ot_src})", flush=True)
    feats = gather_features(genes, ot_map, prefer_offline)

    fit = M.fit_learned_ranker(genes, feats, V.GWAS_GOLD.symbols,
                               n_boot=args.n_boot, n_perm=args.n_perm, seed=args.seed)
    if fit is None:
        print("  [!] model fit failed (degenerate universe/labels).")
        sys.exit(1)

    ci = fit["oof_auc_ci"]
    print(f"  OOF AUC={fit['oof_auc']:.3f} "
          f"CI={('[%.3f, %.3f]' % (ci[0], ci[1])) if ci else '—'} "
          f"p={fit['permutation_p']:.4f} Brier={fit['brier']:.3f}", flush=True)
    print("  learned weights:", {k: round(v, 3) for k, v in
                                  fit["learned_weights"].items()}, flush=True)

    # Spotlight the pipeline's shortlist genes in the learned ranking.
    rank_of = {r["gene"].upper(): i + 1 for i, r in enumerate(fit["gene_scores"])}
    score_of = {r["gene"].upper(): r["learned_score"] for r in fit["gene_scores"]}
    shortlist_rows = [{"gene": g, "learned_score": score_of.get(g),
                       "learned_rank": rank_of.get(g)}
                      for g in SHORTLIST if g in rank_of]

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "prefer_offline": prefer_offline,
        "params": {"top_n": args.top_n, "n_perm": args.n_perm,
                   "n_boot": args.n_boot, "seed": args.seed},
        "hand_set_weights": dict(R.WEIGHTS),
        "fit": fit,
        "shortlist_in_learned_ranking": shortlist_rows,
        "top_learned": fit["gene_scores"][:20],
    }
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports",
                                       "candidate_ranking_learned.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(payload, open(out, "w"), indent=2)
    _write_md(out.replace(".json", ".md"), payload)
    print(f"[wrote] {out}\n[wrote] {out.replace('.json', '.md')}")


def _write_md(path, payload):
    fit = payload["fit"]
    ci = fit["oof_auc_ci"]
    hand = payload["hand_set_weights"]
    p = payload["params"]
    lines = ["# Learned, calibrated candidate ranking", "",
             f"_Generated {payload['generated_utc']}; prefer_offline="
             f"{payload['prefer_offline']}; universe={fit['n_genes']} genes "
             f"({fit['n_gold_in_universe']} gold); top_n={p['top_n']}, "
             f"n_boot={p['n_boot']}, n_perm={p['n_perm']}._", "",
             "Replaces the hand-set composite weights with weights **learned** from the "
             "data (L2 logistic on rank-normalized features, predicting clean GWAS-gold "
             "membership), leave-one-out cross-validated, plus the new LINCS efficacy "
             "feature. Read-only; does not touch the referee/demo path.", "",
             "> **Low-n caveat:** the gold label is ~15 genes, so this is a SENSITIVITY "
             "ANALYSIS of what the data implies — not a definitive re-weighting. Numbers "
             "ride with their uncertainty.", "",
             "## Honest performance (leave-one-out cross-validated)", "",
             f"- **OOF AUC: {fit['oof_auc']:.3f}** "
             f"(95% CI {('[%.3f, %.3f]' % (ci[0], ci[1])) if ci else '—'})",
             f"- Permutation p: **{fit['permutation_p']:.4f}**",
             f"- Brier calibration: **{fit['brier']:.3f}** (lower is better)", "",
             "## Learned weights vs hand-set weights", "",
             "| Feature | learned coef | hand-set weight | coverage |",
             "|---|---|---|---|"]
    hand_alias = {"pi4ad_priority": "pi4ad_priority",
                  "ot_assoc_heldout": "ot_assoc_heldout",
                  "net_centrality": "net_centrality",
                  "struct_plddt": "struct_confidence",
                  "lincs_efficacy": None}
    for f in fit["features"]:
        lw = fit["learned_weights"].get(f)
        hw = hand.get(hand_alias.get(f)) if hand_alias.get(f) else None
        cov = fit["coverage"].get(f, 0.0)
        lines.append(f"| {f} | {lw:+.3f} | "
                     f"{('%.2f' % hw) if hw is not None else '— (new)'} "
                     f"| {cov*100:.0f}% |")
    lines += ["", "A near-zero or negative learned coefficient means the data does not "
              "support that signal's positive contribution once the others are present. "
              "Coverage < 100% marks a feature that was imputed (rank 0.5) for the genes "
              "lacking it — read its coefficient with that in mind.", "",
              "## Pipeline shortlist in the learned ranking", "",
              "| Gene | learned score | learned rank |", "|---|---|---|"]
    for r in payload["shortlist_in_learned_ranking"]:
        lines.append(f"| {r['gene']} | {r['learned_score']} | {r['learned_rank']} |")
    lines += ["", "## Top-20 learned ranking", "",
              "| Rank | Gene | learned score | gold? |", "|---|---|---|---|"]
    for i, r in enumerate(payload["top_learned"], 1):
        lines.append(f"| {i} | {r['gene']} | {r['learned_score']} "
                     f"| {'✓' if r['is_gold'] else ''} |")
    lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
