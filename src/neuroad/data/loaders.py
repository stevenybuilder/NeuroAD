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

from neuroad.data import (synthetic, real, openbhb, openbhb_jepa, oasis_jepa,
                          adni_jepa, gated)

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

    # ADNI with REAL frozen Neuro-JEPA embeddings: the 590-subject disease-signal
    # cohort (87 AD / 503 CN, multi-site, real plasma) on the foundation model's
    # own representation — the raw-MRI -> Neuro-JEPA stage output, made consumable.
    if low in ("adni:neurojepa", "adni:jepa"):
        return adni_jepa.load_adni_neurojepa()

    # ADNI ComBat-harmonized full cohort: 'adni:combat' removes the scanner
    # (field-strength) batch effect from the emb_* features while preserving the
    # whole cohort — a stronger de-confound than the 'adni:3t' slice, which just
    # drops every 1.5T scan. Label-blind (protects age/sex, not dx). Optional
    # 'adni:combat-site' harmonizes by acquisition site instead.
    if low == "adni:combat" or low.startswith("adni:combat"):
        from neuroad.data import harmonize as _harm
        batch = "site" if low in ("adni:combat-site", "adni:combat:site") else "scanner"
        base = gated.load_gated(str(_GATED_DIR / "adni.csv"), "adni")
        out = _harm.harmonize(base, batch=batch, covariates=("age", "sex"))
        return out

    # ADNI field-strength slice: 'adni:3t' / 'adni:1.5t' restrict the real ADNI
    # contract table to a single scanner field strength. This is the de-confound
    # that turns the scanner-dominated full-cohort KILL into a promotable
    # SURVIVOR (the 3T-only AD-vs-CN card): with one field strength the STAR
    # site/scanner test can no longer be dominated by the 3T-vs-1.5T split.
    if low.startswith("adni:") and low.split(":", 1)[1] not in ("", "neurojepa", "jepa"):
        fs = key.split(":", 1)[1].strip().lower()
        base = gated.load_gated(str(_GATED_DIR / "adni.csv"), "adni")
        mask = base["scanner"].astype("string").str.lower().eq(fs).fillna(False)
        sub = base[mask].copy()
        sub.attrs.update(base.attrs)
        sub.attrs["field_strength"] = key.split(":", 1)[1]
        return sub

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


def honest_substrate(name: str) -> str:
    """The TRUTHFUL substrate label for a dataset name — never mislabel one
    feeder's features as another's.

    The frozen-contract default (``Claim.substrate``) says "frozen Neuro-JEPA
    structural embeddings", which is only accurate for the ``*:neurojepa``
    feeders. This maps every registered name to what its ``emb_*`` columns
    ACTUALLY are, so a card/report can be stamped before the naive effect copies
    ``claim.substrate``. Anti-overclaim: ADNI/OASIS/OpenBHB tabular feeders are
    morphometry, not the foundation model, and must say so.
    """
    low = (name or "").strip().lower()
    if low.startswith("synthetic:"):
        return "synthetic contract embeddings (badged demo cohort)"
    if ":neurojepa" in low or ":jepa" in low:
        return "frozen Neuro-JEPA structural embeddings"
    if low.startswith("adni"):
        return ("ADNI z-standardized FreeSurfer morphometry "
                "(UCSFFSX7 ST regions; weight-free feeder)")
    if low.startswith(("oasis3", "nacc", "epad", "gated:")):
        return "gated cohort FreeSurfer morphometry (weight-free feeder)"
    if low.startswith("oasis"):
        return ("OASIS structural-derived features "
                "(weight-free feeder; nWBV/eTIV/ASF)")
    if low.startswith("openbhb"):
        return "OpenBHB structural-derived features (weight-free feeder)"
    return "structural-derived features (weight-free feeder)"


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
