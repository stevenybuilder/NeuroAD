#!/usr/bin/env python3
"""
Runnable demo for gauntlet stage 1 — Age / sex adjustment.

Run from this directory:

    PYTHONPATH=../../src ../../.venv/bin/python run.py

or from the repo root:

    PYTHONPATH=src ./.venv/bin/python skills/age_sex/run.py

Loads a contract-valid cohort and runs `neuroad.gauntlet.test_age_sex`,
printing the TestEvidence verdict and the statistics that justify it.
"""
from __future__ import annotations

import sys

from neuroad.data.loaders import load
from neuroad.gauntlet import test_age_sex

# `synthetic:SURVIVOR` is calibrated so a real (non-artifact) signal survives;
# swap for `synthetic:KILL` to watch the same test collapse the effect.
COHORT = sys.argv[1] if len(sys.argv) > 1 else "synthetic:SURVIVOR"
TARGET = sys.argv[2] if len(sys.argv) > 2 else "conversion"


def main() -> None:
    df = load(COHORT)
    ev = test_age_sex(df, TARGET)

    print(f"cohort   : {COHORT}  (n={len(df)})")
    print(f"target   : {TARGET}")
    print(f"test     : age_sex  (gauntlet stage 1, weight 15/100)")
    print(f"result   : {ev.result.value.upper()}")
    print(f"detail   : {ev.detail}")
    print("stats    :")
    for k, v in ev.stats.items():
        print(f"    {k:12s} = {v}")


if __name__ == "__main__":
    main()
