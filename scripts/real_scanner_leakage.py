#!/usr/bin/env python3
"""
Run the STAR scanner/site leakage star on REAL OpenBHB healthy brains.

OpenBHB is healthy controls ONLY — there is no disease signal to find. Yet the
same reused linear head, pointed at the scanner (field strength) and the site,
predicts which machine acquired the scan at AUC ~0.90 from structure alone. That
AUC is the acquisition batch effect the NeuroAD Discovery Engine gates against — measured
on real, published data, not injected by a synthetic KILL.

Prints the real AUCs and the framing, and writes reports/openbhb_scanner_leakage.json.

Usage
-----
    PYTHONPATH=src python scripts/real_scanner_leakage.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuroad import contract  # noqa: E402
from neuroad.data.openbhb import load_openbhb, real_scanner_leakage  # noqa: E402

REPORTS_DIR = REPO_ROOT / "reports"
OUT_JSON = REPORTS_DIR / "openbhb_scanner_leakage.json"


def main() -> int:
    df = load_openbhb()
    contract.validate_table(df)

    result = real_scanner_leakage(df)

    n = len(df)
    n_field_strengths = df["scanner"].nunique(dropna=True)
    n_sites = df["site"].nunique(dropna=True)

    print("=" * 72)
    print("REAL SCANNER-LEAKAGE STAR  —  OpenBHB (healthy controls only)")
    print("=" * 72)
    print(f"subjects            : {n}")
    print(f"diagnoses           : {dict(df['dx'].value_counts(dropna=False))}")
    print(f"field strengths     : {n_field_strengths}  "
          f"({sorted(df['scanner'].dropna().unique().tolist())})")
    print(f"acquisition sites   : {n_sites}")
    print("-" * 72)
    print(f"scanner (field-strength) leakage AUC : {result['scanner_auc']:.4f}"
          f"   (n={result['detail']['scanner']['n']})")
    print(f"site leakage AUC (macro OVR)         : {result['site_auc']:.4f}"
          f"   (n={result['detail']['site']['n']}, "
          f"{result['detail']['site']['n_classes']} sites)")
    print("-" * 72)
    print(result["message"])
    print("=" * 72)

    payload = {
        "n_subjects": int(n),
        "n_field_strengths": int(n_field_strengths),
        "n_sites": int(n_sites),
        "dx_counts": {str(k): int(v) for k, v in
                      df["dx"].value_counts(dropna=False).items()},
        **result,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
