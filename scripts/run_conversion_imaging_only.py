#!/usr/bin/env python3
"""
Imaging-only MCI->AD conversion prediction, split by plasma availability.

The multimodal fusion analysis (run_adni_conversion_multimodal.py) keeps only
subjects who have BOTH a conversion label AND a plasma p-tau217 draw (n=498, 142
converters) so every modality is scored on the same people. That discards 701
conversion-labeled subjects (270 converters) whose only omission is a blood draw.

This script asks the COMPLEMENTARY question the fusion analysis cannot: how well
does structural imaging alone predict conversion for the people plasma can't serve?
It scores imaging-only (structural morphometry, the contract embedding matrix)
through the same leakage-honest probe (site-disjoint CV, in-fold PCA, bootstrap CI,
permutation null) on three slices:

  * plasma-PRESENT  (n=498)  — head-to-head context vs the plasma workhorse
  * plasma-ABSENT   (n=701)  — the population with NO blood test at all
  * FULL cohort     (n=1199) — imaging conversion arm at 412 converters (2.9x the
                               142 the fusion arm is limited to)

The point is NOT that imaging beats plasma (it doesn't, on people who have plasma).
It is that imaging delivers a real, honestly-measured conversion signal for a large
population plasma leaves silent — complementary value, not competition.

Compliance: reads only derived contract features; writes only AUC-grade numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neuroad import contract, probe
from neuroad.data import loaders

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = _ROOT / "reports" / "conversion_imaging_only.json"


def _slice(name: str, X, y, groups, *, n_boot: int, n_perm: int,
           n_repeats: int) -> dict:
    r = probe.auc_ci_perm(X, y, groups=groups, n_boot=n_boot, n_perm=n_perm,
                          n_repeats=n_repeats)
    return {
        "slice": name,
        "n": int(len(y)),
        "converters": int(y.sum()),
        "auc": r["auc"],
        "ci": None if r["ci_lo"] is None else [r["ci_lo"], r["ci_hi"]],
        "p_perm": r["p_perm"],
        "ci_excludes_chance": r["ci_excludes_chance"],
    }


def main() -> int:
    df = loaders.load("adni:combat")
    if df.attrs.get("is_stub"):
        raise SystemExit("stub, not real data — run scripts/build_adni_contract.py")

    conv = pd.to_numeric(df["conversion"], errors="coerce")
    ptau = pd.to_numeric(df["p_tau217"], errors="coerce")
    site = df["site"].astype("string").fillna("__na__")

    X_all = contract.embedding_matrix(df)  # structural morphometry (imaging)
    labeled = conv.notna().to_numpy()
    have_plasma = ptau.notna().to_numpy()

    def _codes(mask):
        return np.unique(site.to_numpy()[mask], return_inverse=True)[1]

    slices = {
        "plasma_present": labeled & have_plasma,
        "plasma_absent": labeled & ~have_plasma,
        "full_cohort": labeled,
    }
    results = []
    for name, mask in slices.items():
        y = conv[mask].to_numpy(dtype=int)
        results.append(_slice(name, X_all[mask], y, _codes(mask),
                              n_boot=1000, n_perm=1000, n_repeats=5))

    report = {
        "analysis": "imaging-only MCI->AD conversion, split by plasma availability",
        "target": "conversion",
        "substrate": "structural morphometry (ComBat-harmonized contract embedding)",
        "method": ("probe.auc_ci_perm: site-disjoint CV, in-fold auto-PCA, "
                   "bootstrap CI, permutation null, repeated-CV ensembling"),
        "slices": results,
        "context": {
            "fusion_arm_plasma_present_n": 498,
            "fusion_arm_converters": 142,
            "plasma_alone_conversion_auc": 0.814,
            "note": ("The fusion arm is limited to plasma-present subjects. Imaging "
                     "alone extends conversion prediction to the plasma-absent "
                     "population (701 subjects / 270 converters) and to the full "
                     "412-converter cohort — the complementary value plasma cannot "
                     "provide, not a claim that imaging beats plasma."),
        },
        "compliance": "Frozen/derived features only; AUC-grade numbers only.",
    }

    out = DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[conv-imaging] wrote {out}")
    for r in results:
        print(f"[{r['slice']:>14}] n={r['n']:>4} converters={r['converters']:>3}"
              f"  imaging AUC {r['auc']}  CI {r['ci']}  p={r['p_perm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
