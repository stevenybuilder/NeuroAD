"""
One-name dispatch over every contract feeder.

Names:
  'synthetic:SURVIVOR'  -> synthetic.generate_cohort('SURVIVOR')
  'synthetic:KILL'      -> synthetic.generate_cohort('KILL')
  'oasis'               -> real.load_oasis('both')
  'oasis:oasis1' / 'oasis:oasis2' -> single-cohort OASIS
  'openbhb'             -> openbhb.load_openbhb()  (real healthy-control scanner star)
"""
from __future__ import annotations

import pandas as pd

from neuroad.data import synthetic, real, openbhb


def load(name: str, *, seed: int = 0) -> pd.DataFrame:
    """Return a contract-valid table for a registered dataset ``name``."""
    key = name.strip()
    low = key.lower()

    if low.startswith("synthetic:"):
        preset = key.split(":", 1)[1]
        return synthetic.generate_cohort(preset, seed=seed)

    if low == "oasis":
        return real.load_oasis("both")
    if low.startswith("oasis:"):
        which = low.split(":", 1)[1]
        return real.load_oasis(which)

    if low == "openbhb":
        return openbhb.load_openbhb()

    raise ValueError(
        f"unknown dataset {name!r}; try "
        "'synthetic:SURVIVOR', 'synthetic:KILL', 'oasis', 'oasis:oasis1', "
        "'oasis:oasis2', 'openbhb'"
    )


#: Human-facing catalogue (kept in sync with data/registry.yaml notation).
AVAILABLE = [
    "synthetic:SURVIVOR",
    "synthetic:KILL",
    "oasis",
    "oasis:oasis1",
    "oasis:oasis2",
    "openbhb",
]
