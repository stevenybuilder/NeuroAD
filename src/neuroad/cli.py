"""`neuroad` command-line entry point.

    neuroad demo                       run SURVIVOR + KILL on the synthetic
                                       harness, print verdict cards, write
                                       reports/ (the UI reads these).
    neuroad run <dataset> "<claim>"    run one claim on one dataset.
                                       dataset in:
                                         synthetic:SURVIVOR
                                         synthetic:KILL
                                         oasis

The demo path is fully offline (synthetic harness + Claude template fallbacks).
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


def _write_reports(name: str, card) -> list[Path]:
    _REPORTS.mkdir(parents=True, exist_ok=True)
    payload = card.to_dict()
    narration = getattr(card, "narration", None)
    if narration:
        payload["narration"] = narration
    adjudication = getattr(card, "adjudication", None)
    if adjudication:
        payload["adjudication"] = adjudication
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
    print("Open app/index.html to watch the gauntlet tick through these reports.")
    return 0


def _cmd_run(args) -> int:
    from neuroad import pipeline
    try:
        df = _load(args.dataset)
    except Exception as exc:  # noqa: BLE001
        print(f"could not load dataset '{args.dataset}': {exc}", file=sys.stderr)
        return 2
    card = pipeline.run_referee(df, args.claim)
    print(_fmt_card(card))
    written = _write_reports(args.dataset, card)
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
                    "replication; gate survivors behind a biomarker anchor.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="run SURVIVOR + KILL on the synthetic harness")
    d.set_defaults(func=_cmd_demo)

    r = sub.add_parser("run", help="run one claim on one dataset")
    r.add_argument("dataset",
                   help="synthetic:SURVIVOR | synthetic:KILL | oasis | openbhb")
    r.add_argument("claim", help="the claim / hunch in plain language")
    r.set_defaults(func=_cmd_run)

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
