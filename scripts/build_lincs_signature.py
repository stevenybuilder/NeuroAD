#!/usr/bin/env python3
"""
build_lincs_signature.py — build + validate the LINCS L1000 efficacy-proxy axis.

Adds a PERTURBATIONAL / efficacy axis to the discovery half, orthogonal to every
existing (association) signal. Queries SigCom LINCS (keyless) for genetic loss-of-
function perturbations whose transcriptional signature REVERSES a curated Alzheimer
brain signature, aggregates a per-gene reversal-efficacy proxy, and validates it the
same honest way the rest of the pipeline is validated: AUC + bootstrap CI + permutation
p + BH-FDR against the held-out gold sets, with the housekeeping DECOY as a negative
control.

Outputs:
  * data/../integrations/data/lincs_ad_reversal_snapshot.json  (committed; keeps the
    demo/offline path deterministic and network-free)
  * reports/lincs_efficacy.{json,md}

HONESTY: this is an efficacy PROXY, not efficacy — L1000 lines are (mostly) cancer, not
neurons. A hit is a hypothesis. See docs/LINCS_SPEC.md.

Usage:
  PYTHONPATH=src ./.venv/bin/python scripts/build_lincs_signature.py            # LIVE
  PYTHONPATH=src ./.venv/bin/python scripts/build_lincs_signature.py --offline  # snapshot
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neuroad.integrations import lincs as L  # noqa: E402
from neuroad.harness import validation as V  # noqa: E402


def _fmt_ci(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"


def build(prefer_offline: bool, limit: int):
    client = L.LincsClient(prefer_offline=prefer_offline)
    proxy = client.ad_reversal_efficacy(limit=limit)         # reversers -> ranking signal
    universe = client.reversal_universe(limit=limit)          # signed -> validation
    src = "offline_snapshot" if prefer_offline else (
        "live" if proxy or universe else "offline_snapshot")
    return client, proxy, universe, src


def validate_axis(universe: dict[str, float], src: str, n_perm: int, n_boot: int,
                  seed: int):
    """Validate the signed efficacy universe against every gold set + decoy control."""
    genes = list(universe.keys())
    scores = [universe[g] for g in genes]
    uni = V.RankingUniverse(genes=genes, scores=scores, source=src)
    battery = []
    for name, gold, kind in (
            ("efficacy_vs_gwas", V.GWAS_GOLD, "clean_orthogonal"),
            ("efficacy_vs_drugs", V.DRUG_GOLD, "clean_orthogonal"),
            ("efficacy_vs_novel2022", V.NOVEL_2022, "prospective"),
            ("efficacy_vs_DECOY", V.DECOY_GOLD, "negative_control")):
        r = V.validate(uni, gold, ranking_source="lincs_l1000",
                       evidence_mode="lof_reversal", n_perm=n_perm, seed=seed,
                       with_ci=True, n_boot=n_boot,
                       extra_caveat="Efficacy PROXY (L1000 cancer lines) — a "
                                    "perturbational axis orthogonal to association.")
        battery.append((name, kind, r))
    # BH-FDR across the non-control tests
    idx = [i for i, (_, k, _) in enumerate(battery) if k != "negative_control"]
    qs = V.benjamini_hochberg([battery[i][2].permutation_p for i in idx])
    for j, i in enumerate(idx):
        battery[i][2].q_value = qs[j]
    return uni, battery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--limit", type=int, default=1000,
                    help="top reverser/mimicker signatures per LoF database")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-snapshot", action="store_true",
                    help="do not (over)write the committed snapshot")
    args = ap.parse_args()
    prefer_offline = bool(args.offline)
    print(f"[lincs] prefer_offline={prefer_offline} limit={args.limit}", flush=True)

    client, proxy, universe, src = build(prefer_offline, args.limit)
    print(f"  reversal-efficacy proxy: {len(proxy)} genes (source={src})", flush=True)
    print(f"  signed validation universe: {len(universe)} genes", flush=True)
    if not universe:
        print("  [!] no LINCS data (live failure and no snapshot) — nothing to write.")
        sys.exit(1)

    uni, battery = validate_axis(universe, src, args.n_perm, args.n_boot, args.seed)
    for name, kind, r in battery:
        print(f"  [{kind}] {name}: n_gold={r.n_gold} AUC={r.roc_auc} "
              f"CI={_fmt_ci(r.roc_auc_ci)} p={r.permutation_p} q={r.q_value}", flush=True)

    # top reversal-efficacy genes (the hypotheses this axis surfaces), as dicts
    top = [p.to_dict() for p in
           sorted(proxy.values(), key=lambda p: p.reversal_score, reverse=True)[:20]]

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "prefer_offline": prefer_offline, "source": src,
        "params": {"limit": args.limit, "n_perm": args.n_perm,
                   "n_boot": args.n_boot, "seed": args.seed},
        "signature": {"up": list(L.AD_SIGNATURE_UP), "down": list(L.AD_SIGNATURE_DOWN),
                      "citation": L._SIGNATURE_CITATION},
        "databases": list(L._LOF_DATABASES),
        "n_proxy_genes": len(proxy), "n_universe_genes": len(universe),
        "validation": [{"name": n, "kind": k, **r.to_dict()} for n, k, r in battery],
        "top_reversal_efficacy": top,
    }
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports",
                                       "lincs_efficacy.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(payload, open(out, "w"), indent=2)
    _write_md(out.replace(".json", ".md"), payload, battery, top)
    print(f"[wrote] {out}\n[wrote] {out.replace('.json', '.md')}")

    # Commit the snapshot (per-gene proxy + signed validation universe) for a
    # deterministic, network-free offline/demo path.
    if not args.no_snapshot and not prefer_offline and (proxy or universe):
        snap = {
            "generated_utc": payload["generated_utc"],
            "signature_citation": L._SIGNATURE_CITATION,
            "databases": list(L._LOF_DATABASES),
            "genes": [p.to_dict() for p in
                      sorted(proxy.values(), key=lambda x: x.reversal_score,
                             reverse=True)],
            "validation_universe": universe,
        }
        json.dump(snap, open(L._SNAPSHOT_PATH, "w"), indent=1)
        print(f"[wrote] {L._SNAPSHOT_PATH}  ({len(proxy)} genes)")


def _write_md(path, payload, battery, top):
    p = payload["params"]
    L_ = ["# LINCS L1000 — perturbational efficacy-proxy axis", "",
          f"_Generated {payload['generated_utc']}; source={payload['source']}; "
          f"databases={', '.join(payload['databases'])}; limit={p['limit']}, "
          f"n_boot={p['n_boot']}, n_perm={p['n_perm']}._", "",
          "A **perturbational / efficacy** axis, orthogonal to every association signal "
          "(PI4AD, Open Targets, STRING). Queries SigCom LINCS (keyless) for genetic "
          "loss-of-function perturbations whose signature **reverses** a curated AD "
          "brain signature: a gene whose knockout reverses AD is an efficacy-relevant "
          "**inhibition** target.", "",
          "> **HONEST CAVEAT.** This is an efficacy *proxy*, not efficacy. L1000 lines "
          "are (mostly) cancer cell lines, not neurons/microglia, so 'reverses an AD "
          "transcriptomic signature in a cancer line' is a weak surrogate for neuronal "
          "AD efficacy. Every hit is a hypothesis. The AD signature is a curated "
          "consensus (approximation), not a single-study DE table.", "",
          "## Validation — does the efficacy proxy recover known AD biology?", "",
          "| Test | kind | n_gold | AUC | 95% CI | perm p | BH q |",
          "|---|---|---|---|---|---|---|"]
    for name, kind, r in battery:
        q = f"{r.q_value:.3f}" if r.q_value is not None else "—"
        pp = f"{r.permutation_p:.3f}" if r.permutation_p is not None else "—"
        auc = f"{r.roc_auc:.3f}" if r.roc_auc is not None else "—"
        L_.append(f"| {name} | {kind} | {r.n_gold} | {auc} | {_fmt_ci(r.roc_auc_ci)} "
                  f"| {pp} | {q} |")
    L_ += ["", "A *clean_orthogonal* test whose CI lower bound clears 0.5 after FDR "
           "would be genuine orthogonal recovery of AD biology; a null is the honest, "
           "expected outcome for a cancer-line proxy and is still useful as a weak, "
           "independent feature for the learned ranker (which can down-weight it). "
           "The *negative_control* (housekeeping decoys) must sit at chance.", "",
           "## Top reversal-efficacy hypotheses (strongest AD-signature reversers)", "",
           "| Gene | reversal score | # sigs | database | cell line |",
           "|---|---|---|---|---|"]
    for t in top[:15]:
        L_.append(f"| {t['gene']} | {t['reversal_score']} | {t['n_signatures']} "
                  f"| {t['best_database']} | {t['best_cell_line']} |")
    L_ += ["", f"_AD signature: {len(payload['signature']['up'])} up / "
           f"{len(payload['signature']['down'])} down genes. "
           f"{payload['signature']['citation']}_", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(L_))


if __name__ == "__main__":
    main()
