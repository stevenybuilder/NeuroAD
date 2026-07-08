#!/usr/bin/env python3
"""
Runnable demo for gauntlet stage 3 (STAR) — Brain-age control.

Run from this directory:

    PYTHONPATH=../../src ../../.venv/bin/python run.py

or from the repo root:

    PYTHONPATH=src ./.venv/bin/python skills/brain_age_control/run.py

Loads a contract-valid cohort and runs `neuroad.gauntlet.test_brain_age`, which
fits an embedding-derived brain-age model on cognitively-normal subjects,
regresses predicted brain age out of the embedding, and re-measures the outcome.
"""
from __future__ import annotations

import sys

from neuroad.data.loaders import load
from neuroad.gauntlet import test_brain_age

COHORT = sys.argv[1] if len(sys.argv) > 1 else "synthetic:SURVIVOR"
TARGET = sys.argv[2] if len(sys.argv) > 2 else "conversion"


def main() -> None:
    df = load(COHORT)
    ev = test_brain_age(df, TARGET)

    print(f"cohort   : {COHORT}  (n={len(df)})")
    print(f"target   : {TARGET}")
    print(f"test     : brain_age  (gauntlet stage 3, STAR, weight 25/100)")
    print(f"result   : {ev.result.value.upper()}")
    print(f"detail   : {ev.detail}")
    print("stats    :")
    for k, v in ev.stats.items():
        print(f"    {k:12s} = {v}")


if __name__ == "__main__":
    main()
