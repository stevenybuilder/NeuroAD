"""`neuroad` command-line entry point.

    neuroad demo                       run SURVIVOR + KILL on the synthetic
                                       harness, print verdict cards, write
                                       reports/ (the UI reads these).
    neuroad run <dataset> "<claim>"    run one claim on one dataset.
                                       dataset in:
                                         synthetic:SURVIVOR
                                         synthetic:KILL
                                         oasis

The CLI demo path is fully offline (synthetic harness + Claude template
fallbacks). The visual workbench (`app/index.html`) boots on real OASIS and
keeps the synthetic cases as a labeled harness.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Repo root = two levels up from this file (src/neuroad/cli.py).
_ROOT = Path(__file__).resolve().parents[2]
_REPORTS = _ROOT / "reports"

# A default hunch for the demo, phrased the way a scientist would type it.
_DEMO_CLAIM = (
    "MCI patients who convert to AD show a distinct structural-MRI signature "
    "in their frozen embeddings versus non-converters."
)


# ---------------------------------------------------------------------------
# Rendering helpers.
# ---------------------------------------------------------------------------

def _fmt_card(card) -> str:
    """Pretty terminal rendering of a ClaimCard."""
    lines: list[str] = []
    ne = card.naive_effect
    lines.append("=" * 66)
    lines.append(f"  CLAIM: {card.claim.claim_text}")
    lines.append(f"  substrate: {card.claim.substrate}  |  head: {card.claim.head}")
    lines.append("-" * 66)
    lines.append(f"  naive effect: {ne.get('metric','AUC')} = {ne.get('value','?')} "
                 f"(target={ne.get('target','?')}, n={ne.get('n','?')})")
    lines.append("  gauntlet:")
    for t in card.tests:
        dim = _dim_label(t.key)
        lines.append(f"    - {dim:<26} {t.result.value:<14} {t.detail}")
    lines.append("-" * 66)
    lines.append(f"  ROBUSTNESS SCORE: {card.score}/100")
    lines.append(f"  VERDICT: {card.verdict.value.upper()}"
                 f"   [{'PROMOTED' if card.promoted else 'not promoted'}]")
    badge = getattr(card, "claude", None)
    if isinstance(badge, dict):
        # Truthful badge: LIVE only when Claude actually produced the text
        # (last_call_live). A key can be configured yet every call fall through
        # to the deterministic template — that is OFFLINE, not LIVE. `live`
        # (== a key is present) only distinguishes 'configured but unused' from
        # 'not configured'.
        if badge.get("last_call_live"):
            tag = "LIVE CLAUDE"
        elif badge.get("live"):
            tag = "OFFLINE (configured, unused — all calls used template)"
        else:
            tag = "OFFLINE (template)"
        lines.append(f"  CLAUDE: {tag} — {badge.get('path', badge.get('model', '?'))}")
    if card.biology_hypothesis:
        lines.append(f"  biology: {card.biology_hypothesis}")
    if card.next_experiment:
        lines.append("  next experiment:")
        for step in card.next_experiment:
            lines.append(f"    * {step}")
    if card.caveats:
        lines.append("  caveats:")
        for c in card.caveats:
            lines.append(f"    ! {c}")
    narration = getattr(card, "narration", None)
    if narration:
        lines.append("-" * 66)
        lines.append(f"  {narration}")
    lines.append("=" * 66)
    return "\n".join(lines)


def _dim_label(key: str) -> str:
    from neuroad.contract import GAUNTLET_BY_KEY
    d = GAUNTLET_BY_KEY.get(key)
    return d.label if d else key


def _fmt_experiment_card(xcard) -> str:
    """Pretty terminal rendering of a harness ExperimentCard (the L5 artifact).

    Shows the frozen ClaimCard's gauntlet/verdict block, then the three harness
    annotations (novelty_class, honesty_rung, biomarker anchor) and the
    pre-registered next experiment."""
    card = xcard.card
    lines = _fmt_card(card).split("\n")
    # Splice the harness annotations in before the closing rule.
    if lines and lines[-1].startswith("="):
        lines.pop()
    lines.append("-" * 66)
    lines.append(f"  NOVELTY CLASS: {xcard.novelty_class}")
    lines.append(f"  HONESTY RUNG:  {xcard.honesty_rung}")
    atn = xcard.atn_profile or {}
    prov = xcard.discovery_provenance or {}
    gate = prov.get("anchor_gate", {}) if isinstance(prov, dict) else {}
    status = gate.get("status", atn.get("anchor_status", "?"))
    mech = gate.get("routed_mechanism", atn.get("routed_mechanism", "?"))
    lines.append(f"  BIOMARKER ANCHOR: {status}  (mechanism: {mech})")
    ptau = atn.get("T_ptau217_ci_lo")
    gfap = atn.get("I_gfap_ci_lo")
    if ptau is not None or gfap is not None:
        lines.append(f"    p-tau217 CI lo = {ptau}   GFAP CI lo = {gfap}")
    if gate.get("blocked_promotion"):
        lines.append("    !! anchor HARD GATE blocked promotion")
    if not card.next_experiment:
        ne = prov.get("kill_criterion") if isinstance(prov, dict) else None
        if ne:
            lines.append("  next experiment:")
            lines.append(f"    * kill criterion: {ne}")
    lines.append("=" * 66)
    return "\n".join(lines)


def _write_experiment_report(name: str, xcard) -> list[Path]:
    """Serialize an ExperimentCard to reports/ (mirrors _write_reports but keeps
    the harness annotations + the read-only Claude side-artifacts on card.card)."""
    _REPORTS.mkdir(parents=True, exist_ok=True)
    payload = xcard.to_dict()
    card = xcard.card
    for attr in ("narration", "adjudication", "reviewer", "biology"):
        v = getattr(card, attr, None)
        if v:
            payload[attr] = v
    badge = getattr(card, "claude", None)
    if isinstance(badge, dict):
        payload["claude"] = badge
    written: list[Path] = []
    slug = "investigate_" + name.replace(":", "_").lower()
    jp = _REPORTS / f"{slug}.json"
    jp.write_text(json.dumps(payload, indent=2, default=str))
    written.append(jp)
    try:
        import yaml
        yp = _REPORTS / f"{slug}.yaml"
        yp.write_text(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))
        written.append(yp)
    except Exception:
        pass
    return written


def _write_reports(name: str, card) -> list[Path]:
    _REPORTS.mkdir(parents=True, exist_ok=True)
    payload = card.to_dict()
    narration = getattr(card, "narration", None)
    if narration:
        payload["narration"] = narration
    adjudication = getattr(card, "adjudication", None)
    if adjudication:
        payload["adjudication"] = adjudication
    # Structured reviewer critique + biology blocks (set by pipeline.run_referee
    # as read-only side artifacts). Without these the reviewer's critique survives
    # only flattened into caveats and the biology dict is dropped entirely.
    reviewer = getattr(card, "reviewer", None)
    if reviewer:
        payload["reviewer"] = reviewer
    biology = getattr(card, "biology", None)
    if biology:
        payload["biology"] = biology
    badge = getattr(card, "claude", None)
    if isinstance(badge, dict):
        payload["claude"] = badge
    written: list[Path] = []
    slug = name.replace(":", "_").lower()
    jp = _REPORTS / f"{slug}.json"
    jp.write_text(json.dumps(payload, indent=2, default=str))
    written.append(jp)
    try:
        import yaml
        yp = _REPORTS / f"{slug}.yaml"
        yp.write_text(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))
        written.append(yp)
    except Exception:
        pass
    return written


# ---------------------------------------------------------------------------
# Data loading.
# ---------------------------------------------------------------------------

def _load(dataset: str, seed: int = 0):
    from neuroad.data import loaders
    return loaders.load(dataset, seed=seed)


# ---------------------------------------------------------------------------
# Subcommands.
# ---------------------------------------------------------------------------

def _cmd_demo(_args) -> int:
    from neuroad import pipeline
    # Pinned demo seeds (match app/build_demo_data.py): the KILL uses a seed whose
    # naive AUC is HIGHER than the survivor's yet is still refused — the punchline.
    presets = [("synthetic:SURVIVOR", 0), ("synthetic:KILL", 6)]
    print("\nNeuroAD Discovery Engine — demo (SURVIVOR vs KILL on the synthetic harness)\n")
    for name, seed in presets:
        try:
            df = _load(name, seed=seed)
            card = pipeline.run_referee(df, _DEMO_CLAIM)
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] could not run: {exc}", file=sys.stderr)
            continue
        print(_fmt_card(card))
        written = _write_reports(name, card)
        print(f"  wrote: {', '.join(str(p.relative_to(_ROOT)) for p in written)}\n")

    # Fold the workbench payload generation into `neuroad demo` so the UI always
    # animates the just-run engine, never a stale committed artifact. Guarded so a
    # build hiccup never fails the demo itself.
    if not getattr(_args, "no_demo_data", False):
        try:
            import importlib.util
            bd_path = _ROOT / "app" / "build_demo_data.py"
            spec = importlib.util.spec_from_file_location("build_demo_data", bd_path)
            bd = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(bd)
            bd.main([])
            print("  wrote: app/demo_data.json (the workbench reads this)")
        except Exception as exc:  # noqa: BLE001
            print(f"  [demo] app/demo_data.json build skipped ({exc})", file=sys.stderr)

    print("Serve over http (python -m http.server) and open app/index.html to watch "
          "the gauntlet tick through these reports.")
    return 0


def _cmd_reproduce_finding(args) -> int:
    """Regenerate the headline scanner-leakage AUC (with CI + permutation p) from
    a tiny checked-in PCA-reduced frozen-inference fixture — reproducible from a
    clean clone WITHOUT the git-ignored 768-d embedding tables or gated weights."""
    from neuroad import reproduce
    fixture = getattr(args, "fixture", None)
    try:
        out = reproduce.reproduce_finding(fixture_path=fixture)
    except FileNotFoundError as exc:
        print(f"fixture not found: {exc}", file=sys.stderr)
        return 2
    print("\nNeuroAD — reproduce-finding (frozen Neuro-JEPA scanner leakage)\n")
    print(f"  fixture:  {out['fixture']}")
    print(f"  provenance: {out['provenance']}")
    print(f"  n = {out['n']}   PCA components = {out['n_components']}   "
          f"scanner classes = {out['n_classes']}")
    ci = out.get("ci")
    ci_txt = "" if not ci else f"  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]"
    print(f"\n  scanner-leakage AUC = {out['auc']:.3f}{ci_txt}")
    if out.get("p_perm") is not None:
        print(f"  permutation-null p  = {out['p_perm']:.4f} "
              f"({'excludes chance' if out.get('ci_excludes_chance') else 'includes chance'})")
    print(f"\n  {out['message']}\n")
    return 0


def _honest_substrate(dataset: str) -> str:
    """The truthful substrate label per feeder — never mislabel real OASIS
    morphometry as Neuro-JEPA embeddings."""
    low = dataset.lower()
    if low.startswith("oasis") and ":neurojepa" not in low:
        return "OASIS structural-derived features (weight-free feeder; nWBV/eTIV/ASF)"
    if low.startswith("openbhb") and ":neurojepa" not in low:
        return "OpenBHB structural-derived features (weight-free feeder)"
    return "frozen Neuro-JEPA structural embeddings"


def _cmd_run(args) -> int:
    from neuroad import pipeline
    try:
        df = _load(args.dataset)
    except Exception as exc:  # noqa: BLE001
        print(f"could not load dataset '{args.dataset}': {exc}", file=sys.stderr)
        return 2
    # Parse the claim first so we can stamp the honest substrate BEFORE the naive
    # effect (which copies claim.substrate) is computed.
    claim = pipeline._parse_claim(args.claim, df)
    claim.substrate = _honest_substrate(args.dataset)
    card = pipeline.run_referee(df, claim)
    print(_fmt_card(card))
    written = _write_reports(args.dataset, card)
    print(f"\nwrote: {', '.join(str(p.relative_to(_ROOT)) for p in written)}")
    return 0


def _cmd_investigate(args) -> int:
    """L5 entry point: drive the whole instrument from a plain-language hypothesis.

    Chains parse -> route -> referee -> anchor gate -> honesty-stamped card via
    harness.orchestrator.investigate(). Offline-deterministic by default (no
    network unless --api is passed AND ANTHROPIC_API_KEY is set)."""
    from neuroad.harness import orchestrator
    try:
        xcard = orchestrator.investigate(
            args.hypothesis, args.dataset, api=args.api, seed=args.seed)
    except Exception as exc:  # noqa: BLE001
        print(f"could not investigate on '{args.dataset}': {exc}", file=sys.stderr)
        return 2
    print("\nNeuroAD Discovery Engine — investigate (hypothesis -> refereed ExperimentCard)\n")
    print(_fmt_experiment_card(xcard))
    written = _write_experiment_report(args.dataset, xcard)
    print(f"\nwrote: {', '.join(str(p.relative_to(_ROOT)) for p in written)}")
    return 0


def _cmd_scanner_leakage(_args) -> int:
    """Demonstrate the STAR batch effect on REAL healthy multi-scanner data."""
    from neuroad.data import openbhb
    print("\nNeuroAD Discovery Engine — REAL scanner leakage (OpenBHB healthy controls)\n")
    out = openbhb.real_scanner_leakage()
    if not out:
        return 1
    scan, site = out["detail"]["scanner"], out["detail"]["site"]
    print(f"  scanner AUC = {out['scanner_auc']:.4f}  "
          f"(n={scan['n']}, {scan['n_classes']} classes)")
    print(f"  site    AUC = {out['site_auc']:.4f}  "
          f"(n={site['n']}, {site['n_classes']} classes)")
    print(f"\n{out['message']}\n")
    return 0


def _cmd_discover(args) -> int:
    """Run the Detective (unsupervised phenotype discovery) + per-cluster gauntlet."""
    from neuroad import discovery
    from neuroad.data import synthetic, loaders
    if args.dataset == "phenotypes":
        df = synthetic.generate_phenotype_cohort(seed=0)
    else:
        df = loaders.load(args.dataset)
    res = discovery.discover_and_referee(df)
    print("\nNeuroAD Discovery Engine — the Detective (unsupervised phenotype discovery)\n")
    print(res.get("note", ""))
    if res.get("ari") is not None:
        print(f"ground-truth recovery: ARI={res['ari']}  AMI={res['ami']}")
    print("-" * 66)
    for c in res.get("clusters", []):
        gv = c["gauntlet"].get("verdict") if isinstance(c.get("gauntlet"), dict) else "?"
        print(f"  cluster {c['cluster']}: n={c['n']}  stability={c['stability']}  "
              f"verdict={gv}  -> {c['status']}")
    print("-" * 66)
    return 0


# ---------------------------------------------------------------------------
# Argument parsing.
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neuroad",
        description="An Alzheimer's structural-MRI referee: falsify a signal "
                    "against scanner leakage, demographics, brain-age and "
                    "replication; require a biomarker anchor when available or "
                    "leakage-clean replication when plasma is unavailable.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="run SURVIVOR + KILL on the synthetic harness "
                                    "and (re)build app/demo_data.json for the workbench")
    d.add_argument("--no-demo-data", action="store_true",
                   help="skip the app/demo_data.json rebuild (reports only)")
    d.set_defaults(func=_cmd_demo)

    rf = sub.add_parser(
        "reproduce-finding",
        help="regenerate the frozen Neuro-JEPA scanner-leakage AUC (+ CI + "
             "permutation p) from the checked-in PCA-10 fixture (no gated weights)")
    rf.add_argument("--fixture", default=None,
                    help="path to a PCA-reduced fixture CSV "
                         "(default: data/real/fixtures/openbhb_neurojepa_pca10.csv)")
    rf.set_defaults(func=_cmd_reproduce_finding)

    r = sub.add_parser("run", help="run one claim on one dataset")
    r.add_argument("dataset",
                   help="synthetic:SURVIVOR | synthetic:KILL | oasis | openbhb")
    r.add_argument("claim", help="the claim / hunch in plain language")
    r.set_defaults(func=_cmd_run)

    iv = sub.add_parser(
        "investigate",
        help="L5: drive the instrument from a plain-language hypothesis "
             "(parse -> route -> referee -> anchor gate -> honesty-stamped card)")
    iv.add_argument("hypothesis", help="the hypothesis / hunch in plain language")
    iv.add_argument("dataset",
                    help="synthetic:SURVIVOR | synthetic:KILL | oasis | openbhb")
    iv.add_argument("--seed", type=int, default=0,
                    help="seed forwarded to the (synthetic) feeder")
    iv.add_argument("--api", action="store_true",
                    help="allow a live Claude claim-parse (still falls back offline "
                         "on any failure); default is fully offline-deterministic")
    iv.set_defaults(func=_cmd_investigate)

    sl = sub.add_parser("scanner-leakage",
                        help="REAL batch effect on healthy multi-scanner OpenBHB data")
    sl.set_defaults(func=_cmd_scanner_leakage)

    dv = sub.add_parser("discover",
                        help="the Detective: unsupervised phenotype discovery + per-cluster gauntlet")
    dv.add_argument("dataset", nargs="?", default="phenotypes",
                    help="phenotypes (planted, default) | oasis | synthetic:SURVIVOR")
    dv.set_defaults(func=_cmd_discover)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
