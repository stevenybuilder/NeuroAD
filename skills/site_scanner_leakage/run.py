#!/usr/bin/env python3
"""
Runnable demo for gauntlet stage 2 (STAR) — Site / scanner leakage.

Run from this directory:

    PYTHONPATH=../../src ../../.venv/bin/python run.py

or from the repo root:

    PYTHONPATH=src ./.venv/bin/python skills/site_scanner_leakage/run.py

Loads a contract-valid cohort and runs `neuroad.gauntlet.test_site_scanner`,
which points the SAME probe head at the scanner/site label and reports the
subject-disjoint leakage margin = outcome_AUC - scanner_AUC.
"""
from __future__ import annotations

import sys

from neuroad.data.loaders import load
from neuroad.gauntlet import test_site_scanner

# `synthetic:KILL` is where this test earns its keep: scanner AUC >= outcome AUC.
COHORT = sys.argv[1] if len(sys.argv) > 1 else "synthetic:SURVIVOR"
TARGET = sys.argv[2] if len(sys.argv) > 2 else "conversion"


def main() -> None:
    df = load(COHORT)
    ev = test_site_scanner(df, TARGET)

    print(f"cohort   : {COHORT}  (n={len(df)})")
    print(f"target   : {TARGET}")
    print(f"test     : site_scanner  (gauntlet stage 2, STAR, weight 25/100)")
    print(f"result   : {ev.result.value.upper()}")
    print(f"detail   : {ev.detail}")
    print("stats    :")
    for k, v in ev.stats.items():
        print(f"    {k:12s} = {v}")


if __name__ == "__main__":
    main()
