#!/usr/bin/env python3
"""
Runnable demo for gauntlet stage 4 (HARD GATE) — Biomarker anchor.

Run from this directory:

    PYTHONPATH=../../src ../../.venv/bin/python run.py

or from the repo root:

    PYTHONPATH=src ./.venv/bin/python skills/biomarker_anchor/run.py

Loads a contract-valid cohort and runs `neuroad.gauntlet.test_biomarker_anchor`,
which correlates the out-of-fold probe score with plasma p-tau217 / GFAP on the
complete-case subset. This is the promotion gate: imaging alone cannot pass.
"""
from __future__ import annotations

import sys

from neuroad.data.loaders import load
from neuroad.gauntlet import test_biomarker_anchor

COHORT = sys.argv[1] if len(sys.argv) > 1 else "synthetic:SURVIVOR"
TARGET = sys.argv[2] if len(sys.argv) > 2 else "conversion"


def main() -> None:
    df = load(COHORT)
    ev = test_biomarker_anchor(df, TARGET)

    print(f"cohort   : {COHORT}  (n={len(df)})")
    print(f"target   : {TARGET}")
    print(f"test     : biomarker_anchor  (gauntlet stage 4, HARD GATE, weight 20/100)")
    print(f"result   : {ev.result.value.upper()}")
    print(f"detail   : {ev.detail}")
    print("stats    :")
    for k, v in ev.stats.items():
        print(f"    {k:12s} = {v}")


if __name__ == "__main__":
    main()
