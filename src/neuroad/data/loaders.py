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

from pathlib import Path

import pandas as pd

from neuroad.data import synthetic, real, openbhb, openbhb_jepa, oasis_jepa, gated

# Conventional drop location for a mapped gated export (see scripts/adni_to_contract.py).
_GATED_DIR = Path(__file__).resolve().parents[3] / "data" / "real" / "_gated"


def load(name: str, *, seed: int = 0) -> pd.DataFrame:
    """Return a contract-valid table for a registered dataset ``name``."""
    key = name.strip()
    low = key.lower()

    if low.startswith("synthetic:"):
        preset = key.split(":", 1)[1]
        return synthetic.generate_cohort(preset, seed=seed)

    if low in ("oasis:neurojepa", "oasis:jepa"):
        return oasis_jepa.load_oasis_neurojepa()
    if low == "oasis":
        return real.load_oasis("both")
    if low.startswith("oasis:"):
        which = low.split(":", 1)[1]
        return real.load_oasis(which)

    if low == "openbhb":
        return openbhb.load_openbhb()
    if low in ("openbhb:neurojepa", "openbhb:jepa"):
        return openbhb_jepa.load_openbhb_neurojepa()

    # Gated cohorts (ADNI/OASIS-3/NACC/EPAD): load a mapped export from
    # data/real/_gated/<name>.csv if present, else the clearly-marked stub.
    # 'adni' is shorthand for 'gated:adni'.
    if low in gated.GATED_NAMES or low.startswith("gated:"):
        gname = low.split(":", 1)[1] if low.startswith("gated:") else low
        path = _GATED_DIR / f"{gname}.csv"
        return gated.load_gated(str(path), gname)

    raise ValueError(
        f"unknown dataset {name!r}; try "
        "'synthetic:SURVIVOR', 'synthetic:KILL', 'oasis', 'oasis:oasis1', "
        "'oasis:oasis2', 'openbhb', 'openbhb:neurojepa', 'adni'"
    )


#: Human-facing catalogue (kept in sync with data/registry.yaml notation).
AVAILABLE = [
    "synthetic:SURVIVOR",
    "synthetic:KILL",
    "oasis",
    "oasis:oasis1",
    "oasis:oasis2",
    "openbhb",
    "openbhb:neurojepa",
    "oasis:neurojepa",
    "adni",
    "oasis3",
    "nacc",
    "epad",
]
