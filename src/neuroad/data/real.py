"""
Real-data feeder: OASIS-1 (cross-sectional) + OASIS-2 (longitudinal) -> ONE
contract table.

The weight-free "embedding" is the standardized structural-derived feature set
[nWBV, eTIV, ASF] plus two engineered structural ratios. We deliberately do NOT
feed MMSE or CDR into the embedding — those *define* the labels (dx / conversion)
and would leak the answer.

Honest caveats surfaced by this feeder:
  * OASIS-1 & OASIS-2 are each effectively single-scanner, so ``scanner`` is a
    single value. The real leakage ⭐ on this table is reframed as *cohort/batch*
    leakage: ``site`` is the pseudo-site OASIS1 vs OASIS2. The ground-truth
    scanner-confound KILL lives in the synthetic harness.
  * No open OASIS cohort has plasma p-tau217 / GFAP / NfL / amyloid / APOE ->
    those biomarker columns are all <NA> (route survivors to ADNI/EPAD).

Label mapping:
  dx:         CDR == 0 -> CN, CDR == 0.5 -> MCI, CDR >= 1 -> AD.
  conversion: OASIS-2 Group == 'Converted' -> 1, 'Nondemented' -> 0, else <NA>.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neuroad import contract

# Repo layout: .../src/neuroad/data/real.py -> repo root is parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_REAL_DIR = _REPO_ROOT / "data" / "real"
OASIS1_CSV = _REAL_DIR / "oasis_cross-sectional.csv"
OASIS2_CSV = _REAL_DIR / "oasis_longitudinal.csv"
#: OPTIONAL FastSurfer-derived volume table (subject_id + hippocampal/ventricle/
#: cortex/whole-brain/intracranial volumes), produced by
#: ``scripts/fastsurfer_volumes_colab.py`` on a GPU runtime. Git-ignored / kept
#: local by design; the feeder works fully WITHOUT it (see ``_join_volumes``).
OASIS_VOLUMES_CSV = _REAL_DIR / "oasis_volumes.csv"

#: structural-derived features used as the weight-free embedding.
_STRUCTURAL_FEATURES = ["nWBV", "eTIV", "ASF"]

#: FastSurfer volume columns additively joined onto the contract table when the
#: derived volume CSV is present (fusion-schema keys; see
#: ``integrations.structural_segmenter.VOLUME_KEYS``).
_VOLUME_COLS = [
    "hippocampal_volume", "ventricle_volume", "whole_brain_volume",
    "cortex_volume", "intracranial_volume",
]


def _join_volumes(frame: pd.DataFrame) -> pd.DataFrame:
    """Additively left-join FastSurfer volumes on ``subject_id`` when available.

    Honest degrade (mirrors the 'cache not found' behavior of the Neuro-JEPA
    feeders): if ``OASIS_VOLUMES_CSV`` is absent — or lacks a ``subject_id`` column
    — the frame is returned UNCHANGED, so the feeder is fully functional without any
    GPU-derived table. When present, it exposes ``hippocampal_volume`` (+ ventricle
    / cortex / whole-brain / intracranial) columns; rows without a volume row get
    NaN. Extra columns are allowed by the contract, so ``validate_table`` still
    passes either way. The join never drops or duplicates subject rows (the volume
    table is de-duplicated on ``subject_id`` first).
    """
    if not OASIS_VOLUMES_CSV.exists():
        return frame
    vols = pd.read_csv(OASIS_VOLUMES_CSV)
    if "subject_id" not in vols.columns:
        return frame
    keep = ["subject_id"] + [c for c in _VOLUME_COLS if c in vols.columns]
    vols = vols[keep].copy()
    vols["subject_id"] = vols["subject_id"].astype(str)
    vols = vols.drop_duplicates("subject_id", keep="first")
    return frame.merge(vols, on="subject_id", how="left")


def _dx_from_cdr(cdr: float) -> object:
    if pd.isna(cdr):
        return pd.NA
    if cdr == 0:
        return "CN"
    if cdr == 0.5:
        return "MCI"
    return "AD"  # CDR >= 1


def _load_oasis1() -> pd.DataFrame:
    """OASIS-1 cross-sectional. Keep only CDR-labeled rows (others have no dx)."""
    raw = pd.read_csv(OASIS1_CSV)
    raw = raw[raw["CDR"].notna()].copy()
    out = pd.DataFrame(index=raw.index)
    out["subject_id"] = "OAS1_" + raw["ID"].astype(str)
    out["cohort"] = "OASIS1"
    out["site"] = "OASIS1"
    out["scanner"] = "OASIS1_Siemens_1.5T"
    out["age"] = raw["Age"].astype(float)
    out["sex"] = raw["M/F"].astype(str)
    out["CDR"] = raw["CDR"].astype(float)
    out["group"] = pd.NA          # no conversion info in the cross-sectional set
    out["nWBV"] = raw["nWBV"].astype(float)
    out["eTIV"] = raw["eTIV"].astype(float)
    out["ASF"] = raw["ASF"].astype(float)
    return out


def _load_oasis2() -> pd.DataFrame:
    """OASIS-2 longitudinal -> one baseline (Visit==1) row per subject."""
    raw = pd.read_csv(OASIS2_CSV)
    raw = raw.sort_values(["Subject ID", "Visit"])
    base = raw[raw["Visit"] == 1].drop_duplicates("Subject ID", keep="first").copy()
    out = pd.DataFrame(index=base.index)
    out["subject_id"] = "OAS2_" + base["Subject ID"].astype(str)
    out["cohort"] = "OASIS2"
    out["site"] = "OASIS2"
    out["scanner"] = "OASIS2_Siemens_1.5T"
    out["age"] = base["Age"].astype(float)
    out["sex"] = base["M/F"].astype(str)
    out["CDR"] = base["CDR"].astype(float)
    out["group"] = base["Group"].astype(str)
    out["nWBV"] = base["nWBV"].astype(float)
    out["eTIV"] = base["eTIV"].astype(float)
    out["ASF"] = base["ASF"].astype(float)
    return out


def load_oasis(which: str = "both") -> pd.DataFrame:
    """Map the vendored OASIS CSVs into a single contract table.

    Parameters
    ----------
    which : {'both', 'oasis1', 'oasis2'}
        Which cohort(s) to include. 'both' stacks them (enabling the
        cohort/batch-leakage pseudo-site star + a real replication split).

    Returns
    -------
    pd.DataFrame  passing ``contract.validate_table``.
    """
    which = which.lower()
    parts: list[pd.DataFrame] = []
    if which in ("both", "oasis1"):
        parts.append(_load_oasis1())
    if which in ("both", "oasis2"):
        parts.append(_load_oasis2())
    if not parts:
        raise ValueError(f"unknown which={which!r}; choose both/oasis1/oasis2")
    raw = pd.concat(parts, ignore_index=True)

    # --- structural-derived embedding (standardized) --------------------
    feats = raw[_STRUCTURAL_FEATURES].astype(float).copy()
    # two engineered structural ratios (still weight-free, no label leakage)
    feats["nWBV_x_eTIV"] = raw["nWBV"] * raw["eTIV"]
    feats["brain_vol_proxy"] = raw["nWBV"] * raw["eTIV"] / raw["ASF"]
    Z = (feats - feats.mean()) / feats.std(ddof=0)
    emb = contract.make_embedding_frame(Z.to_numpy())

    # --- assemble ------------------------------------------------------
    frame = emb
    frame.insert(0, "subject_id", raw["subject_id"].to_numpy())
    dx = raw["CDR"].map(_dx_from_cdr)
    frame["dx"] = pd.Categorical(dx, categories=contract.DX_LEVELS)

    conv = raw["group"].map(
        {"Converted": 1, "Nondemented": 0}).astype("Int8")
    frame["conversion"] = pd.array(conv.to_numpy(), dtype="Int8")

    frame["age"] = raw["age"].to_numpy(dtype=float)
    sex = raw["sex"].where(raw["sex"].isin(["M", "F"]))
    frame["sex"] = pd.Categorical(sex, categories=contract.SEX_LEVELS)
    frame["site"] = pd.Categorical(raw["site"])
    frame["scanner"] = pd.Categorical(raw["scanner"])

    # honest longitudinal flag: OASIS-2 has follow-up; OASIS-1 does not.
    frame["longitudinal"] = (raw["cohort"] == "OASIS2").to_numpy()

    # No plasma markers / amyloid / APOE in open OASIS -> all <NA>.
    n = len(frame)
    na_i8 = pd.array([pd.NA] * n, dtype="Int8")
    frame["amyloid"] = na_i8
    frame["p_tau217"] = np.full(n, np.nan, dtype="float64")
    frame["gfap"] = np.full(n, np.nan, dtype="float64")
    frame["nfl"] = np.full(n, np.nan, dtype="float64")
    frame["apoe4"] = pd.array([pd.NA] * n, dtype="Int8")

    # subject_id must be unique; a handful of OASIS-2 IDs could collide only if
    # both cohorts included the same key — the OAS1_/OAS2_ prefixes prevent it.
    frame = frame.drop_duplicates("subject_id", keep="first").reset_index(drop=True)

    # Additively attach FastSurfer volumes if the derived CSV exists (no-op when
    # absent — the feeder is fully functional without any GPU-derived table).
    frame = _join_volumes(frame)

    contract.validate_table(frame)
    return frame
