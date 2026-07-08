#!/usr/bin/env python3
"""
Runnable demo for gauntlet stage 5 — Replication split.

Run from this directory:

    PYTHONPATH=../../src ../../.venv/bin/python run.py

or from the repo root:

    PYTHONPATH=src ./.venv/bin/python skills/replication/run.py

Loads a contract-valid cohort and runs `neuroad.gauntlet.test_replication`,
which holds out one site/cohort, trains on the rest, and reports the held-out
AUC vs the in-cohort AUC.
"""
from __future__ import annotations

import sys

from neuroad.data.loaders import load
from neuroad.gauntlet import test_replication

COHORT = sys.argv[1] if len(sys.argv) > 1 else "synthetic:SURVIVOR"
TARGET = sys.argv[2] if len(sys.argv) > 2 else "conversion"


def main() -> None:
    df = load(COHORT)
    ev = test_replication(df, TARGET)

    print(f"cohort   : {COHORT}  (n={len(df)})")
    print(f"target   : {TARGET}")
    print(f"test     : replication  (gauntlet stage 5, weight 15/100)")
    print(f"result   : {ev.result.value.upper()}")
    print(f"detail   : {ev.detail}")
    print("stats    :")
    for k, v in ev.stats.items():
        print(f"    {k:12s} = {v}")


if __name__ == "__main__":
    main()
