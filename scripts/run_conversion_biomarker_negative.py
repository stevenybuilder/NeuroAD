#!/usr/bin/env python3
"""
Does imaging catch converters the BLOOD TEST would reassure?

The plasma-unavailable analysis (run_conversion_imaging_only.py) covered people
with NO p-tau217 draw. This is the harder, complementary question: among people
who WERE tested and came back biomarker-NEGATIVE (low p-tau217 / amyloid-negative),
does structural imaging still flag the ones who convert?

Within a biomarker-negative subgroup, plasma's discriminative range is by
construction restricted, so plasma-alone should collapse toward chance there — the
honest test of whether imaging adds orthogonal signal exactly where the blood test
goes quiet. Imaging-only and plasma-only are scored through the same leakage-honest
probe (site-disjoint CV, in-fold PCA, bootstrap CI, permutation null).

Subgroups (all within conversion-labeled AND plasma-tested, n=498):
  * p-tau217 LOW  (below median)  — "blood test low"
  * p-tau217 HIGH (at/above median)
  * amyloid-negative              — canonical biomarker-negative
  * amyloid-positive

Compliance: derived contract features only; AUC-grade numbers only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neuroad import contract, probe
from neuroad.data import loaders

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = _ROOT / "reports" / "conversion_biomarker_negative.json"


def _auc(X, y, groups):
    if len(np.unique(y)) < 2 or y.sum() < 5 or (len(y) - y.sum()) < 5:
        return {"auc": None, "ci": None, "p_perm": None,
                "note": "too few in a class for a stable estimate"}
    r = probe.auc_ci_perm(X, y, groups=groups, n_boot=1000, n_perm=1000, n_repeats=5)
    return {"auc": r["auc"],
            "ci": None if r["ci_lo"] is None else [r["ci_lo"], r["ci_hi"]],
            "p_perm": r["p_perm"], "ci_excludes_chance": r["ci_excludes_chance"]}


def main() -> int:
    df = loaders.load("adni:combat")
    if df.attrs.get("is_stub"):
        raise SystemExit("stub, not real data — run scripts/build_adni_contract.py")

    conv = pd.to_numeric(df["conversion"], errors="coerce")
    ptau = pd.to_numeric(df["p_tau217"], errors="coerce")
    amy = pd.to_numeric(df["amyloid"], errors="coerce")
    site = df["site"].astype("string").fillna("__na__")
    X_img = contract.embedding_matrix(df)

    tested = conv.notna() & ptau.notna()
    med = np.nanmedian(ptau[tested])

    subgroups = {
        "ptau217_low": tested & (ptau < med),
        "ptau217_high": tested & (ptau >= med),
        "amyloid_negative": conv.notna() & (amy == 0),
        "amyloid_positive": conv.notna() & (amy == 1),
    }

    results = []
    for name, mask in subgroups.items():
        m = mask.fillna(False).to_numpy(dtype=bool)
        y = conv[m].to_numpy(dtype=int)
        groups = np.unique(site.to_numpy()[m], return_inverse=True)[1]
        X_plasma = ptau[m].to_numpy(dtype=float).reshape(-1, 1)
        X_plasma = np.nan_to_num(X_plasma, nan=float(np.nanmedian(X_plasma)))
        results.append({
            "subgroup": name,
            "n": int(m.sum()),
            "converters": int(y.sum()),
            "imaging_only": _auc(X_img[m], y, groups),
            "plasma_only": _auc(X_plasma, y, groups),
        })

    report = {
        "analysis": "imaging-only conversion within biomarker-NEGATIVE subgroups",
        "question": ("does structural imaging flag converters that the plasma test "
                     "calls low-risk / reassures?"),
        "cohort": "ADNI conversion-labeled + plasma-tested (n=498, 142 converters)",
        "method": ("probe.auc_ci_perm within each subgroup: site-disjoint CV, "
                   "in-fold auto-PCA, bootstrap CI, permutation null"),
        "ptau217_split": "median of the triangulated z-scored ensemble",
        "subgroups": results,
        "reading": ("Within a biomarker-negative subgroup plasma's range is "
                    "restricted, so plasma-only collapses toward chance there; "
                    "imaging-only retaining a CI-clears-chance signal is the "
                    "evidence that imaging adds orthogonal value exactly where the "
                    "blood test goes quiet."),
        "compliance": "Derived contract features only; AUC-grade numbers only.",
    }
    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[biomarker-neg] wrote {DEFAULT_OUT}\n")
    for r in results:
        io, po = r["imaging_only"], r["plasma_only"]
        print(f"[{r['subgroup']:>18}] n={r['n']:>3} conv={r['converters']:>3}"
              f"  imaging {io['auc']} {io.get('ci')}  |  plasma {po['auc']} {po.get('ci')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
