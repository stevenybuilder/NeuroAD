"""ADNI FreeSurfer named-ROI feeder — the cohort where ``emb_i`` IS a real brain
region's volume, so a region parsed from a hypothesis conditions the whole probe.

Built offline by ``scripts/build_adni_roi_table.py`` into a de-identified table
(``data/real/_gated/adni_roi.csv``). ``REGION_ORDER`` is DYNAMIC: it is derived
from the ``roi_*`` columns actually present in that table (sorted deterministically),
so the probe covers the full Desikan-Killiany cortical + subcortical set the ETL
emitted, not a fixed 7. Each ROI volume is z-standardized into ``emb_0..emb_{k-1}``
in that order and ``df.attrs['region_columns']`` is stamped so
``contract.restrict_to_region`` can subset the feature matrix to the queried region.
Every value is a real FreeSurfer volume; no fabrication, no LLM in the numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import contract
from . import gated

#: Anatomical GROUPS -> member region slugs. A coarse hypothesis ("medial temporal
#: atrophy") restricts to every listed member that EXISTS in the cohort. Groups are
#: rebuilt against the present regions in ``_region_columns``; absent members are
#: silently skipped so the group is always a real subset of real columns.
_GROUPS: dict[str, list[str]] = {
    "medial_temporal": ["hippocampus", "entorhinal", "parahippocampal", "fusiform",
                         "amygdala"],
    "temporal": ["hippocampus", "entorhinal", "middletemporal", "inferiortemporal",
                 "superiortemporal", "fusiform", "temporalpole", "transversetemporal",
                 "bankssts"],
    "parietal": ["precuneus", "inferiorparietal", "superiorparietal", "supramarginal",
                 "postcentral"],
}


def _region_order_from_columns(columns) -> list[str]:
    """Region slugs derived from the ``roi_*`` columns present, sorted deterministically."""
    return sorted(c[len("roi_"):] for c in columns if str(c).startswith("roi_"))


#: Region key (and anatomical GROUP) -> the emb_* columns it spans, built against the
#: regions actually present in this cohort.
def _region_columns(region_order: list[str]) -> dict[str, list[str]]:
    idx = {r: f"emb_{i}" for i, r in enumerate(region_order)}
    present = set(region_order)
    groups = {
        g: [idx[m] for m in members if m in present]
        for g, members in _GROUPS.items()
    }
    groups = {g: cols for g, cols in groups.items() if cols}  # drop empty groups
    return {**{r: [c] for r, c in idx.items()}, **groups}


def load_adni_roi(csv_path, *, seed: int = 0) -> pd.DataFrame:
    """Contract table whose ``emb_i`` is the z-standardized volume of the i-th
    named ROI (``REGION_ORDER``, derived from the table's ``roi_*`` columns). Carries
    ``icv`` as a head-size covariate and ``df.attrs['region_columns']`` /
    ``df.attrs['icv_col']`` for region restriction."""
    raw = pd.read_csv(csv_path)
    region_order = _region_order_from_columns(raw.columns)
    roi_cols = [f"roi_{r}" for r in region_order]

    frame = pd.DataFrame({"subject_id": raw["subject_id"].astype("string")})
    for c in contract.METADATA_COLUMNS:
        if c == "subject_id":
            continue
        frame[c] = raw[c] if c in raw.columns else pd.NA
    # icv is a real covariate column (not an emb feature).
    frame["icv"] = pd.to_numeric(raw.get("icv"), errors="coerce") if "icv" in raw.columns else pd.NA

    # z-standardize each ROI volume into emb_0..emb_{k-1} in REGION_ORDER.
    for i, rc in enumerate(roi_cols):
        v = pd.to_numeric(raw[rc], errors="coerce")
        mu, sd = float(v.mean()), float(v.std(ddof=0))
        frame[f"emb_{i}"] = ((v - mu) / sd) if sd > 0 else 0.0

    frame = gated._coerce_contract_dtypes(frame.reset_index(drop=True))
    contract.validate_table(frame)
    frame.attrs["region_columns"] = _region_columns(region_order)
    frame.attrs["icv_col"] = "icv"
    frame.attrs["substrate"] = "adni:roi"
    return frame
