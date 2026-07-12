#!/usr/bin/env python3
"""
Real-ADNI MCI->AD CONVERSION card — the prognostic task.

Cross-sectional AD-vs-CN diagnosis (run_adni_survivor.py) asks "is this brain
already demented?". The clinically valuable, and much harder, question is
PROGNOSIS: given a cognitively-impaired-but-not-demented (MCI) subject today,
will they CONVERT to AD? ADNI is one of the only cohorts with the longitudinal
follow-up to label this — the contract's ``conversion`` column is derived from
each subject's full DXSUM trajectory (baseline MCI + >=1 follow-up -> 1 if any
later Dementia else 0). That yields ~1,199 labeled subjects (412 converters),
a far larger and more decision-relevant sample than the cross-sectional AD count.

This script points the SAME frozen referee head at ``target="conversion"`` (a
leakage-free out-of-fold probe with permutation testing, site-grouped), on the
ComBat-harmonized contract (scanner batch removed, label-blind). It writes:
  * reports/adni_conversion_card.json   — full ClaimCard.to_dict()
  * reports/ADNI_CONVERSION.md          — human-readable summary

Honest framing: predicting conversion from structural features is genuinely hard;
a modest-but-significant, cross-validated AUC here is a REAL prognostic signal,
and an at-chance result is reported as such (not hidden). Deterministic; reads the
real contract via loaders.load("adni:combat").

Usage:
    PYTHONPATH=src ./.venv/bin/python scripts/run_adni_conversion.py
"""
from __future__ import annotations

import json
from pathlib import Path

from neuroad import contract, pipeline
from neuroad.data import loaders

_ROOT = Path(__file__).resolve().parents[1]


def build_conversion_card(mode: str = "combat"):
    """Run the referee on the MCI->AD conversion phenotype."""
    if mode == "combat":
        sub = loaders.load("adni:combat")
        label = "ComBat-harmonized full cohort (scanner batch removed, label-blind)"
    else:
        sub = loaders.load("adni")
        label = "raw ADNI contract"

    if sub.attrs.get("is_stub"):
        raise SystemExit("loaders.load returned the STUB, not real data. "
                         "Build first: python scripts/build_adni_contract.py")

    claim = contract.Claim(
        claim_id="adni_mci_conversion",
        claim_text=("MCI->AD conversion is decodable from ADNI structural features "
                    "(prognosis: will an impaired-but-not-demented subject convert?)."),
        target="conversion")
    from neuroad.data.loaders import honest_substrate
    claim.substrate = honest_substrate("adni")
    card = pipeline.run_referee(sub, claim)
    return sub, card, label


def _summary_md(sub, card, label: str) -> str:
    import pandas as pd
    conv = pd.to_numeric(sub["conversion"], errors="coerce")
    n_lab = int(conv.notna().sum())
    n_pos = int((conv == 1).sum())
    ne = card.naive_effect
    lines = [
        "# Real-ADNI CONVERSION card — MCI->AD prognosis",
        "",
        f"**Cohort:** {n_lab} conversion-labeled subjects ({n_pos} converters / "
        f"{n_lab - n_pos} stable) — {label}, {sub['site'].nunique()} sites, "
        f"D={len(contract.embedding_columns(sub))} FreeSurfer features. Real ADNI.",
        "",
        f"**Naive effect (OOF probe):** {ne.get('metric','AUC')} = {ne.get('value')} "
        f"(n={ne.get('n')}).",
        "",
        f"**Verdict:** {card.verdict.value.upper()} — score {card.score}/100 — "
        f"{'PROMOTED' if card.promoted else 'not promoted'}.",
        "",
        "## Gauntlet",
        "",
        "| Test | Result | Headline |",
        "|---|---|---|",
    ]
    for t in card.tests:
        stat = ", ".join(
            f"{k}={round(v, 3) if isinstance(v, float) else v}"
            for k, v in list((t.stats or {}).items())[:3])
        lines.append(f"| {t.key} | {t.result.value} | {stat} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "Prognostic conversion prediction is a harder task than cross-sectional "
        "diagnosis: a cross-validated, permutation-significant AUC above chance is "
        "a REAL structural prognostic signal; an at-chance result means structure "
        "alone does not forecast conversion in this cohort (an honest negative).",
    ]
    return "\n".join(lines)


def main() -> int:
    sub, card, label = build_conversion_card("combat")
    out_dir = _ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "adni_conversion_card.json").write_text(json.dumps(card.to_dict(), indent=2))
    (out_dir / "ADNI_CONVERSION.md").write_text(_summary_md(sub, card, label))
    ne = card.naive_effect
    print(f"conversion cohort: naive OOF {ne.get('metric')}={ne.get('value')} "
          f"(n={ne.get('n')})")
    print(f"verdict: {card.verdict.value}  score {card.score}/100  "
          f"promoted={card.promoted}")
    print(f"wrote: {out_dir / 'adni_conversion_card.json'}")
    print(f"wrote: {out_dir / 'ADNI_CONVERSION.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
