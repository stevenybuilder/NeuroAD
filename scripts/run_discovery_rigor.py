#!/usr/bin/env python3
"""
run_discovery_rigor.py — statistical-rigor hardening for the discovery half.

The target-prioritization validation reports point AUCs with permutation p-values.
On the tiny cited gold sets (15 GWAS, 9 drug genes) a point AUC is fragile and a
battery of p-values overstates significance. This side-artifact adds the four things
a reviewer would (rightly) demand before trusting "AUC 0.728":

  1. BOOTSTRAP 95% CIs on every AUC — so a 0.73 on n_gold=15 is read with its
     uncertainty, not as a hard number.
  2. BH-FDR q-values across the whole honest-test battery — multiple-testing control.
  3. NEGATIVE CONTROLS:
     (a) a housekeeping DECOY gold set that an honest ranker MUST score at chance
         (specificity — if decoys score high the signal is an artifact); and
     (b) a DEGREE-MATCHED null for the STRING-RWR network test (hubs score high on
         any centrality metric — does the ranking beat a degree-matched random set?).
  4. SHORTLIST RANK-STABILITY — bootstrap-resample the universe and leave-one-signal-
     out, and report how often the lead shortlist genes stay in the top-k. A robust
     shortlist is not an artifact of one signal or one weighting.

Read-only: touches nothing on the referee/demo path. Writes reports/discovery_rigor.
{json,md}. Live by default (network fetches, no GPU); --offline uses snapshots.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neuroad.harness import validation as V  # noqa: E402
from neuroad.harness import ranking as R  # noqa: E402
from neuroad.integrations import pi4ad as P  # noqa: E402


def _fmt_ci(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"


def honest_battery(prefer_offline: bool, n_perm: int, n_boot: int, seed: int):
    """The non-circular + control test battery, each with a bootstrap AUC CI."""
    tests = []

    # --- clean (non-circular) held-out tests ---
    ot_ng = V.opentargets_universe(prefer_offline=prefer_offline,
                                   evidence="non_genetic", top_n=2000)
    tests.append(("opentargets_vs_gwas_heldout_nongenetic", "clean",
                  V.validate(ot_ng, V.GWAS_GOLD, ranking_source="opentargets",
                             evidence_mode="non_genetic", n_perm=n_perm, seed=seed,
                             with_ci=True, n_boot=n_boot,
                             extra_caveat="Clean: GWAS gold from non-genetic evidence.")))

    ot_nc = V.opentargets_universe(prefer_offline=prefer_offline,
                                   evidence="non_clinical", top_n=2000)
    tests.append(("opentargets_vs_drugs_heldout_nonclinical", "clean",
                  V.validate(ot_nc, V.DRUG_GOLD, ranking_source="opentargets",
                             evidence_mode="non_clinical", n_perm=n_perm, seed=seed,
                             with_ci=True, n_boot=n_boot,
                             extra_caveat="Clean: drug gold from non-clinical evidence.")))

    # --- residually-circular comparator (flagged, not counted as clean) ---
    pit = V.pi4ad_universe(prefer_offline=prefer_offline)
    tests.append(("pi4ad_vs_gwas", "residually_circular",
                  V.validate(pit, V.GWAS_GOLD, ranking_source="pi4ad",
                             evidence_mode="priority", n_perm=n_perm, seed=seed,
                             optimistic=True, with_ci=True, n_boot=n_boot,
                             extra_caveat="PI4AD integrates genetic evidence — "
                                          "residually circular vs a GWAS gold set.")))

    # --- NEGATIVE CONTROLS: decoy gold must score at chance on both clean rankers ---
    tests.append(("opentargets_nongenetic_vs_DECOY", "negative_control",
                  V.validate(ot_ng, V.DECOY_GOLD, ranking_source="opentargets",
                             evidence_mode="non_genetic", n_perm=n_perm, seed=seed,
                             with_ci=True, n_boot=n_boot,
                             extra_caveat="NEGATIVE CONTROL: housekeeping decoys must "
                                          "score at chance (AUC~0.5).")))
    tests.append(("pi4ad_vs_DECOY", "negative_control",
                  V.validate(pit, V.DECOY_GOLD, ranking_source="pi4ad",
                             evidence_mode="priority", n_perm=n_perm, seed=seed,
                             with_ci=True, n_boot=n_boot,
                             extra_caveat="NEGATIVE CONTROL: housekeeping decoys must "
                                          "score at chance (AUC~0.5).")))
    return tests


def apply_fdr(tests):
    """Attach BH-FDR q-values across the CLEAN + circular tests (controls excluded
    from the correction's m — they are specificity checks, not discovery tests)."""
    idx = [i for i, (_, kind, _) in enumerate(tests) if kind != "negative_control"]
    pvals = [tests[i][2].permutation_p for i in idx]
    qs = V.benjamini_hochberg(pvals)
    for j, i in enumerate(idx):
        tests[i][2].q_value = qs[j]
    return tests


def network_degree_matched(prefer_offline: bool, add_nodes: int, n_draws: int,
                           seed: int):
    """Degree-matched specificity null for the STRING-RWR network test.

    Seeds RWR on KNOWN_2019, ranks non-seed nodes by propagated mass, and asks
    whether NOVEL_2022 genes rank above a set MATCHED to their STRING degree. If the
    observed AUC only equals the degree-matched null, the network 'signal' is just
    'novel genes are high-degree', not genuine propagation-based anticipation."""
    seeds = sorted(V.KNOWN_2019.symbols)
    nodes = P.propagate_hits(seeds, prefer_offline=prefer_offline, method="rwr",
                             add_nodes=0 if prefer_offline else add_nodes)
    non_seed = [n for n in nodes if not n.is_seed]
    if not non_seed:
        return None
    genes = [n.gene for n in non_seed]
    scores = [float(n.propagated_score) for n in non_seed]
    degrees = {n.gene.upper(): float(n.degree) for n in non_seed}
    res = V.degree_matched_null_auc(genes, scores, V.NOVEL_2022.symbols, degrees,
                                    n_draws=n_draws, seed=seed)
    if res:
        res["universe"] = len(genes)
        res["source"] = "live" if (nodes and nodes[0].source == "string_live") \
            else "offline_snapshot"
    return res


def shortlist_stability(prefer_offline: bool, n_boot: int, top_k: int, seed: int):
    """Bootstrap + leave-one-signal-out stability of the composite shortlist.

    One live signal-gather; all resampling is in-memory so it costs no extra
    network. Reports, per lead gene, how often it stays in the composite top-k
    under (a) universe bootstrap resampling and (b) dropping each signal in turn."""
    import numpy as np
    genes, sig = R.gather_signals(prefer_offline=prefer_offline)
    base_rows = R.composite(genes, sig)
    base_top = [r["gene"] for r in base_rows[:top_k]]

    # (a) universe bootstrap: resample the gene set with replacement, renormalize,
    #     re-rank, and count top-k membership for each base-top gene.
    rng = np.random.default_rng(seed)
    boot_hits: Counter = Counter()
    n = len(genes)
    for _ in range(max(1, int(n_boot))):
        idx = rng.integers(0, n, n)
        gb = [genes[i] for i in idx]
        # subset each signal dict to the resampled genes (dedup preserves ranking)
        uniq = list(dict.fromkeys(gb))
        sub = {k: {g: sig[k].get(g) for g in uniq} for k in
               ("pi4ad_priority", "pi4ad_rank", "ot_assoc_heldout",
                "net_centrality", "struct_confidence")}
        sub["net_degree"] = sig["net_degree"]
        rows = R.composite(uniq, sub)
        top = {r["gene"] for r in rows[:top_k]}
        for g in base_top:
            if g in top:
                boot_hits[g] += 1

    # (b) leave-one-signal-out: drop each signal, re-rank, record the top-k.
    loo = {}
    for drop in ("pi4ad_priority", "ot_assoc_heldout", "net_centrality",
                 "struct_confidence"):
        sig2 = {k: ({g: None for g in genes} if k == drop else v)
                for k, v in sig.items()}
        rows = R.composite(genes, sig2)
        loo[f"drop_{drop}"] = [r["gene"] for r in rows[:top_k]]

    denom = max(1, int(n_boot))
    return {
        "top_k": top_k,
        "baseline_top": base_top,
        "bootstrap_topk_frequency": {g: boot_hits[g] / denom for g in base_top},
        "leave_one_signal_out_topk": loo,
        "n_universe": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-draws", type=int, default=1000)
    ap.add_argument("--add-nodes", type=int, default=1000)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    prefer_offline = bool(args.offline)
    print(f"[rigor] prefer_offline={prefer_offline} n_perm={args.n_perm} "
          f"n_boot={args.n_boot} n_draws={args.n_draws}", flush=True)

    tests = apply_fdr(honest_battery(prefer_offline, args.n_perm, args.n_boot,
                                     args.seed))
    for name, kind, r in tests:
        print(f"  [{kind}] {name}: AUC={r.roc_auc} CI={_fmt_ci(r.roc_auc_ci)} "
              f"p={r.permutation_p} q={r.q_value} n_gold={r.n_gold}", flush=True)

    print("  [degree-matched null] computing...", flush=True)
    dmn = network_degree_matched(prefer_offline, args.add_nodes, args.n_draws,
                                 args.seed)
    if dmn:
        print(f"    observed={dmn['observed_auc']:.3f} null_mean={dmn['null_mean']:.3f} "
              f"null_ci={_fmt_ci(dmn['null_ci'])} emp_p={dmn['empirical_p']:.3f}",
              flush=True)

    print("  [shortlist stability] computing...", flush=True)
    stab = shortlist_stability(prefer_offline, args.n_boot, args.top_k, args.seed)
    print(f"    baseline top-{stab['top_k']}: {stab['baseline_top']}", flush=True)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "prefer_offline": prefer_offline,
        "params": {"n_perm": args.n_perm, "n_boot": args.n_boot,
                   "n_draws": args.n_draws, "add_nodes": args.add_nodes,
                   "top_k": args.top_k, "seed": args.seed},
        "battery": [{"name": n, "kind": k, **r.to_dict()} for n, k, r in tests],
        "degree_matched_null": dmn,
        "shortlist_stability": stab,
    }
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports",
                                       "discovery_rigor.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(payload, open(out, "w"), indent=2)
    _write_md(out.replace(".json", ".md"), payload, tests, dmn, stab)
    print(f"\n[wrote] {out}\n[wrote] {out.replace('.json', '.md')}")


def _write_md(path, payload, tests, dmn, stab):
    L = ["# Discovery half — statistical rigor report", "",
         f"_Generated {payload['generated_utc']}; prefer_offline="
         f"{payload['prefer_offline']}; "
         f"n_boot={payload['params']['n_boot']}, n_perm={payload['params']['n_perm']}, "
         f"n_draws={payload['params']['n_draws']}._", "",
         "Adds bootstrap AUC confidence intervals, BH-FDR multiple-testing control, "
         "negative controls (housekeeping decoys + a degree-matched network null), and "
         "shortlist rank-stability to the target-prioritization validation. Read-only; "
         "does not touch the referee/demo path.", "",
         "## 1. Test battery — AUC with 95% bootstrap CI and BH-FDR q-value", "",
         "| Test | kind | n_gold | AUC | 95% CI | perm p | BH q |",
         "|---|---|---|---|---|---|---|"]
    for name, kind, r in tests:
        q = f"{r.q_value:.3f}" if r.q_value is not None else "—"
        p = f"{r.permutation_p:.3f}" if r.permutation_p is not None else "—"
        auc = f"{r.roc_auc:.3f}" if r.roc_auc is not None else "—"
        L.append(f"| {name} | {kind} | {r.n_gold} | {auc} | {_fmt_ci(r.roc_auc_ci)} "
                 f"| {p} | {q} |")
    L += ["", "**How to read it:** a *clean* test with a CI whose lower bound stays "
          "above 0.5 after FDR is genuine signal; a *negative_control* test SHOULD "
          "straddle 0.5 (that is the pass condition — decoys must not score high); the "
          "*residually_circular* row is the optimistic ceiling, not evidence.", ""]

    L += ["## 2. Degree-matched network null (specificity of the STRING-RWR test)", ""]
    if dmn:
        L += [f"- Observed AUC (NOVEL_2022 via RWR from KNOWN_2019): "
              f"**{dmn['observed_auc']:.3f}**",
              f"- Degree-matched null AUC: mean **{dmn['null_mean']:.3f}**, "
              f"95% {_fmt_ci(dmn['null_ci'])} over {dmn['n_draws_effective']} draws",
              f"- Empirical p (observed vs degree-matched null): "
              f"**{dmn['empirical_p']:.3f}**",
              "",
              "This isolates propagation signal from the trivial 'novel genes are "
              "high-degree hubs' confound: if the observed AUC is not clearly above the "
              "degree-matched null, the network test carries little beyond degree.", ""]
    else:
        L += ["- (degree-matched null unavailable — universe degenerate/offline)", ""]

    L += ["## 3. Shortlist rank-stability", "",
          f"Baseline composite top-{stab['top_k']}: "
          f"**{', '.join(stab['baseline_top'])}**", "",
          "Bootstrap top-k membership frequency (universe resampled "
          f"{payload['params']['n_boot']}×):", ""]
    for g, f in stab["bootstrap_topk_frequency"].items():
        L.append(f"- {g}: **{f*100:.0f}%**")
    L += ["", "Leave-one-signal-out top-k (does the shortlist survive dropping each "
          "signal?):", ""]
    for k, v in stab["leave_one_signal_out_topk"].items():
        L.append(f"- {k}: {', '.join(v)}")
    L += ["", "A gene that stays top-k across most bootstraps AND most signal-drops is "
          "a robust recommendation, not an artifact of one signal or one weighting.", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    main()
