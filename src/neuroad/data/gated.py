"""
Gated-dataset drop-in feeder — turn a USER-SUPPLIED real export into a contract
table with ZERO code change anywhere else.

The genuinely open cohorts (OASIS-1/2, OpenBHB) are vendored and loaded by
``real.py``. The richest cohorts — OASIS-3, ADNI, NACC, EPAD — sit behind a
registration / data-use agreement or a credentialed application, so we cannot
vendor them. This module is the seam that lets a scientist drop the real file in
once access is granted:

    from neuroad.data import gated
    df = gated.load_gated("~/Downloads/adni_ucsf_freesurfer.csv", "adni")
    # -> a contract-valid table; the whole referee runs on it unchanged.

Two source shapes are accepted, in priority order:

  1. **Already in contract shape** (the stubs, or a pre-mapped export): the file
     already carries ``subject_id, dx, ..., emb_0..emb_k``. We coerce dtypes and
     validate. This is why dropping a real file that matches the stub schema is a
     literal file swap.
  2. **A raw FreeSurfer + clinical export**: the file carries the source's own
     column names (e.g. ADNI ``DX_bl``, ``Hippocampus``; OASIS-3 ``cdr``,
     ``IntraCranialVol``). We map source columns -> contract columns using the
     per-dataset :data:`GATED_CONFIGS` (documented candidate names), standardize
     the structural features into ``emb_*``, and validate.

If no real file is supplied, :func:`load_gated` transparently falls back to the
hand-written stub at ``data/real/_stubs/<name>_stub.csv`` and marks the result
as a stub (``df.attrs["is_stub"] = True``) so nothing downstream mistakes a
placeholder for a result.

NOTE (wiring): these functions are dispatch-compatible — :func:`load_gated` takes
a dataset name and returns a contract table, exactly like ``loaders.load``. They
are intentionally NOT wired into ``loaders.py`` here (that file is owned
elsewhere); :func:`load_gated` is exposed for that later one-line wiring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from neuroad import contract

# Repo layout: .../src/neuroad/data/gated.py -> repo root is parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_STUB_DIR = _REPO_ROOT / "data" / "real" / "_stubs"


# ---------------------------------------------------------------------------
# Per-dataset mapping configs.
#
# ``field_map`` lists, for each contract metadata column, the ordered candidate
# SOURCE column names we look for (first match wins). ``structural_features`` are
# the SOURCE columns we standardize into emb_0..emb_k when the file is a raw
# export (ignored when the file already carries emb_* columns). These candidate
# names are documented, not exhaustive — a real export with different headers is
# handled by adding to (or hand-renaming into) these lists; still zero change to
# any downstream module.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GatedConfig:
    name: str                       # canonical dataset key
    stub_name: str                  # basename under _stubs (without _stub.csv)
    field_map: dict[str, list[str]] # contract col -> candidate source cols
    structural_features: list[str]  # source cols -> standardized emb_*
    cdr_columns: list[str]          # source cols carrying a global CDR (dx fallback)
    dx_value_map: dict[str, str]    # raw dx string (lowercased) -> DX level
    has_plasma: bool
    unlocks: str                    # the referee capability this cohort enables


# Common encodings shared across cohorts.
_SEX_MAP = {
    "m": "M", "male": "M", "1": "M",
    "f": "F", "female": "F", "2": "F", "0": "F",
}
_CONVERSION_MAP = {
    "1": 1, "0": 0, "converter": 1, "converted": 1, "convert": 1,
    "progressor": 1, "nonconverter": 0, "non-converter": 0,
    "stable": 0, "nondemented": 0,
}
_AMYLOID_MAP = {
    "1": 1, "0": 0, "pos": 1, "positive": 1, "a+": 1,
    "neg": 0, "negative": 0, "a-": 0,
}
# Clinical diagnosis strings seen across ADNI / OASIS-3 / NACC exports.
_DX_MAP = {
    "cn": "CN", "control": "CN", "nc": "CN", "normal": "CN",
    "nl": "CN", "hc": "CN", "cognitively normal": "CN",
    "mci": "MCI", "emci": "MCI", "lmci": "MCI", "smc": "MCI",
    "mci to dementia": "MCI", "impaired not mci": "MCI",
    "ad": "AD", "dementia": "AD", "dat": "AD", "demented": "AD",
    "alzheimer": "AD", "alzheimers disease": "AD",
}
# AIBL codes current diagnosis numerically (DXCURREN 1=HC, 2=MCI, 3=AD); keep the
# shared text mappings and add the numeric codes (`_map_str` round-trips "1.0"->"1").
_AIBL_DX_MAP = {**_DX_MAP, "1": "CN", "2": "MCI", "3": "AD"}


GATED_CONFIGS: dict[str, GatedConfig] = {
    # ADNI — the reference AD cohort WITH plasma p-tau217/GFAP. Columns follow
    # ADNIMERGE + the UCSF Cross-Sectional FreeSurfer table + plasma biomarker
    # tables (LONI IDA).
    "adni": GatedConfig(
        name="ADNI",
        stub_name="adni",
        field_map={
            "subject_id": ["subject_id", "PTID", "RID", "IMAGEUID"],
            "dx": ["dx", "DX", "DX_bl", "DIAGNOSIS", "diagnosis"],
            "conversion": ["conversion", "CONVERT", "Conversion", "DXCONV"],
            "age": ["age", "AGE", "Age"],
            "sex": ["sex", "PTGENDER", "GENDER", "M/F"],
            "site": ["site", "SITE", "ORIGPROT", "COLPROT"],
            "scanner": ["scanner", "SCANNER", "FLDSTRENG", "MAGSTRENGTH"],
            "amyloid": ["amyloid", "AMYLOID", "AMYLOID_STATUS", "AV45_pos"],
            "p_tau217": ["p_tau217", "PLASMA_PTAU217", "PTAU217", "plasma_ptau217"],
            "gfap": ["gfap", "GFAP", "PLASMA_GFAP"],
            "nfl": ["nfl", "NfL", "NEFL", "PLASMA_NFL"],
            "apoe4": ["apoe4", "APOE4", "APOE_e4_count", "APGEN"],
        },
        structural_features=[
            "Hippocampus", "WholeBrain", "Ventricles",
            "Entorhinal", "MidTemp", "Fusiform", "ICV",
        ],
        cdr_columns=["CDGLOBAL", "CDRSB", "cdr"],
        dx_value_map=_DX_MAP,
        has_plasma=True,
        unlocks="biomarker_anchor (plasma p-tau217/GFAP gate) on real AD data",
    ),
    # OASIS-3 — multi-scanner MR + PET with AD labels but NO plasma markers.
    # Columns follow the FreeSurfer volumetric spreadsheet + clinical CDR CSV
    # exported from XNAT Central.
    "oasis3": GatedConfig(
        name="OASIS-3",
        stub_name="oasis3",
        field_map={
            "subject_id": ["subject_id", "OASISID", "MR ID", "Subject", "MRI_ID"],
            "dx": ["dx", "dx1", "cdx", "diagnosis", "DIAGNOSIS"],
            "conversion": ["conversion", "CONVERT"],
            "age": ["age", "ageAtEntry", "Age", "AgeAtEntry"],
            "sex": ["sex", "M/F", "GENDER", "Sex"],
            "site": ["site", "Site"],
            "scanner": ["scanner", "Scanner", "ScannerModel", "MagneticField"],
            "amyloid": ["amyloid", "AMYLOID", "pib_pos", "av45_pos"],
            "p_tau217": ["p_tau217"],
            "gfap": ["gfap"],
            "nfl": ["nfl"],
            "apoe4": ["apoe4", "APOE", "apoe"],
        },
        structural_features=[
            "IntraCranialVol", "TotalGrayVol", "SubCortGrayVol",
            "lhCortexVol", "rhCortexVol",
            "Left-Hippocampus", "Right-Hippocampus", "CortexVol",
        ],
        cdr_columns=["cdr", "CDR", "CDRTOT"],
        dx_value_map=_DX_MAP,
        has_plasma=False,
        unlocks="real scanner-leakage star WITH AD labels + FreeSurfer clustering",
    ),
    # NACC — multi-center UDS + optional MRI; strongest real site/scanner spread.
    "nacc": GatedConfig(
        name="NACC",
        stub_name="nacc",
        field_map={
            "subject_id": ["subject_id", "NACCID", "naccid"],
            "dx": ["dx", "NACCUDSD", "NACCALZD", "diagnosis"],
            "conversion": ["conversion", "NACCCONV"],
            "age": ["age", "NACCAGE", "Age"],
            "sex": ["sex", "SEX", "M/F"],
            "site": ["site", "NACCADC", "ADC", "CENTER"],
            "scanner": ["scanner", "SCANNER", "NACCMRFI"],
            "amyloid": ["amyloid", "NACCAMY", "amyloid_status"],
            "p_tau217": ["p_tau217", "PLASMA_PTAU217", "PTAU217"],
            "gfap": ["gfap", "GFAP"],
            "nfl": ["nfl", "NfL", "NEFL"],
            "apoe4": ["apoe4", "NACCNE4S", "APOE4"],
        },
        structural_features=[
            "HIPPOVOL", "WHOLEBRAIN", "VENTRICLES",
            "ENTORHINAL", "MIDTEMP", "INTRACRANIALVOL",
        ],
        cdr_columns=["CDRGLOB", "CDRSUM", "cdr"],
        dx_value_map=_DX_MAP,
        has_plasma=True,
        unlocks="ground-truth site-leakage star on real multi-center data",
    ),
    # EPAD — preclinical/prodromal cohort WITH CSF+plasma markers (via ADDI).
    "epad": GatedConfig(
        name="EPAD",
        stub_name="epad",
        field_map={
            "subject_id": ["subject_id", "patient_id", "SubjectID"],
            "dx": ["dx", "clinical_status", "diagnosis", "DIAGNOSIS"],
            "conversion": ["conversion", "progression"],
            "age": ["age", "Age", "AGE"],
            "sex": ["sex", "gender", "M/F"],
            "site": ["site", "centre", "center", "Site"],
            "scanner": ["scanner", "Scanner", "field_strength"],
            "amyloid": ["amyloid", "amyloid_status", "csf_abeta_pos"],
            "p_tau217": ["p_tau217", "plasma_ptau217", "PTAU217"],
            "gfap": ["gfap", "GFAP", "plasma_gfap"],
            "nfl": ["nfl", "NfL", "plasma_nfl"],
            "apoe4": ["apoe4", "APOE4", "apoe_e4_count"],
        },
        structural_features=[
            "hippocampus", "whole_brain", "ventricles",
            "entorhinal", "total_gray", "icv",
        ],
        cdr_columns=["cdr_global", "CDR", "cdr"],
        dx_value_map=_DX_MAP,
        has_plasma=True,
        unlocks="biomarker anchor for early-stage (preclinical/prodromal) survivors",
    ),
    # AIBL — Australian Imaging, Biomarkers & Lifestyle. On the SAME LONI/IDA portal
    # as ADNI (low-effort acquisition), carries plasma p-tau + longitudinal
    # conversion, and mirrors ADNI's LONI column naming. Highest converter-with-plasma
    # yield per the power analysis (docs/DATA_EXPANSION_SPEC.md) -> the #1 expansion.
    "aibl": GatedConfig(
        name="AIBL",
        stub_name="aibl",
        field_map={
            "subject_id": ["subject_id", "RID", "PTID", "IMAGEUID"],
            "dx": ["dx", "DXCURREN", "Simple_Group", "DX", "diagnosis"],
            "conversion": ["conversion", "CONVERT", "DXCONV"],
            "age": ["age", "AGE", "Age"],
            "sex": ["sex", "PTGENDER", "Gender", "GENDER", "M/F"],
            "site": ["site", "SITEID", "SITE"],
            "scanner": ["scanner", "FLDSTRENG", "MAGSTRENGTH", "FieldStrength"],
            "amyloid": ["amyloid", "AMYLOID", "PIB_STATUS", "av45_pos"],
            "p_tau217": ["p_tau217", "PLASMA_PTAU217", "PTAU217", "plasma_ptau217"],
            "gfap": ["gfap", "GFAP", "PLASMA_GFAP"],
            "nfl": ["nfl", "NfL", "NEFL", "PLASMA_NFL"],
            "apoe4": ["apoe4", "APOE4", "APGEN1", "APGEN2"],
        },
        structural_features=[
            "Hippocampus", "WholeBrain", "Ventricles",
            "Entorhinal", "MidTemp", "Fusiform", "ICV",
        ],
        cdr_columns=["CDGLOBAL", "CDR", "cdr"],
        dx_value_map=_AIBL_DX_MAP,
        has_plasma=True,
        unlocks="converters-with-plasma to power fusion-vs-plasma (LONI, same DUA as ADNI)",
    ),
}


# ---------------------------------------------------------------------------
# Small resolution / coercion helpers.
# ---------------------------------------------------------------------------
def _first_present(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """First candidate source column actually present in ``df`` (else None)."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _map_str(value: object, table: dict[str, object]) -> object:
    """Case-insensitive string map; passthrough of already-numeric codes."""
    if pd.isna(value):
        return pd.NA
    key = str(value).strip().lower()
    if key in table:
        return table[key]
    # Numeric round-trips (CSV/float) render 1 as "1.0"; match the int form too.
    try:
        f = float(key)
        if f.is_integer() and str(int(f)) in table:
            return table[str(int(f))]
    except (ValueError, TypeError):
        pass
    return pd.NA


def _dx_from_cdr(cdr: float) -> object:
    """CDR 0 -> CN, 0.5 -> MCI, >=1 -> AD (mirrors real.py)."""
    if pd.isna(cdr):
        return pd.NA
    if cdr == 0:
        return "CN"
    if cdr == 0.5:
        return "MCI"
    return "AD"


def _to_int8(series: pd.Series, mapping: Optional[dict] = None) -> pd.array:
    """Coerce a source series to the contract's Int8 (1/0/<NA>)."""
    if mapping is not None:
        # NOTE: .map() over a nullable Int8 series nulls every entry, so an
        # already-clean 1/0/<NA> column (e.g. a derived `conversion`) would be
        # silently erased. Coerce to object first — a no-op for str/object input.
        vals = series.astype(object).map(lambda v: _map_str(v, mapping))
    else:
        vals = pd.to_numeric(series, errors="coerce")
    return pd.array(pd.Series(vals).astype("Float64").round().astype("Int8"),
                    dtype="Int8")


def _is_contract_shaped(df: pd.DataFrame) -> bool:
    """True if the file already carries the contract columns + emb_* columns."""
    has_meta = all(c in df.columns for c in contract.METADATA_COLUMNS)
    has_emb = bool(contract.embedding_columns(df))
    return has_meta and has_emb


def _coerce_contract_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce a raw-but-contract-shaped frame to the exact contract dtypes."""
    out = frame.copy()
    out["subject_id"] = out["subject_id"].astype(str)

    dx = out["dx"].map(lambda v: _map_str(v, _DX_MAP)
                       if str(v).strip().lower() not in contract.DX_LEVELS
                       else str(v).strip())
    # keep already-valid DX levels verbatim (stub carries CN/MCI/AD directly)
    dx = out["dx"].where(out["dx"].isin(contract.DX_LEVELS), dx)
    out["dx"] = pd.Categorical(dx, categories=contract.DX_LEVELS)

    out["conversion"] = _to_int8(out["conversion"], _CONVERSION_MAP)
    out["age"] = pd.to_numeric(out["age"], errors="coerce").astype("float64")

    sex = out["sex"].map(lambda v: _map_str(v, _SEX_MAP)
                         if str(v).strip().upper() not in contract.SEX_LEVELS
                         else str(v).strip().upper())
    out["sex"] = pd.Categorical(sex, categories=contract.SEX_LEVELS)

    out["site"] = pd.Categorical(out["site"].astype("string"))
    out["scanner"] = pd.Categorical(out["scanner"].astype("string"))

    out["amyloid"] = _to_int8(out["amyloid"], _AMYLOID_MAP)
    for m in ("p_tau217", "gfap", "nfl"):
        out[m] = pd.to_numeric(out[m], errors="coerce").astype("float64")
    out["apoe4"] = _to_int8(out["apoe4"])
    # OPTIONAL triangulated plasma signals (from data.plasma_ensemble): coerce to
    # float when a richer feeder carried them in, so the biomarker anchor + Bridge
    # routing can read them. Absent -> untouched (a plain export still validates).
    for m in contract.EXTENDED_BIOMARKER_COLUMNS:
        if m in out.columns:
            out[m] = pd.to_numeric(out[m], errors="coerce").astype("float64")
    return out


def _build_embeddings(df: pd.DataFrame, cfg: GatedConfig) -> pd.DataFrame:
    """emb_* frame: reuse existing emb_* columns, else standardize the config's
    structural features that are present in the export."""
    existing = contract.embedding_columns(df)
    if existing:
        return df[existing].astype(float).reset_index(drop=True)

    feats = [c for c in cfg.structural_features if c in df.columns]
    if not feats:
        raise contract.ContractError(
            f"{cfg.name}: no emb_* columns and none of the expected structural "
            f"features present. Looked for {cfg.structural_features}. Either "
            "pre-map the export into emb_0..emb_k or rename the FreeSurfer "
            "volume columns to one of those names."
        )
    Z = df[feats].astype(float)
    Z = (Z - Z.mean()) / Z.std(ddof=0).replace(0.0, 1.0)
    return contract.make_embedding_frame(Z.to_numpy())


# ---------------------------------------------------------------------------
# The mapper: raw/contract-shaped export -> contract table.
# ---------------------------------------------------------------------------
def map_export(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Map a USER-SUPPLIED export ``df`` for ``dataset`` into a contract table.

    Accepts both a file already in contract shape (fast path: coerce + validate)
    and a raw FreeSurfer + clinical export (map source columns via the dataset's
    :class:`GatedConfig`). Always ends by calling ``contract.validate_table``.
    """
    cfg = _config_for(dataset)

    # Fast path: already contract-shaped (the stub, or a pre-mapped export).
    if _is_contract_shaped(df):
        frame = _coerce_contract_dtypes(df.reset_index(drop=True))
        frame = frame.drop_duplicates("subject_id", keep="first").reset_index(drop=True)
        contract.validate_table(frame)
        return frame

    # Raw export path: resolve each contract column from candidate source names.
    n = len(df)
    resolved: dict[str, Optional[str]] = {
        col: _first_present(df, cands) for col, cands in cfg.field_map.items()
    }

    emb = _build_embeddings(df, cfg)
    frame = emb

    sid_col = resolved["subject_id"]
    if sid_col is not None:
        subject_id = df[sid_col].astype(str).to_numpy()
    else:
        subject_id = np.array(
            [f"{cfg.name.replace('-', '')}_{i:04d}" for i in range(n)])
    frame.insert(0, "subject_id", subject_id)

    # dx: prefer an explicit diagnosis column; fall back to CDR banding.
    dx_col = resolved["dx"]
    dx: pd.Series
    if dx_col is not None:
        dx = df[dx_col].map(lambda v: _map_str(v, cfg.dx_value_map))
    else:
        dx = pd.Series([pd.NA] * n)
    if dx.isna().all():
        cdr_col = _first_present(df, cfg.cdr_columns)
        if cdr_col is not None:
            dx = pd.to_numeric(df[cdr_col], errors="coerce").map(_dx_from_cdr)
    frame["dx"] = pd.Categorical(list(dx), categories=contract.DX_LEVELS)

    conv_col = resolved["conversion"]
    if conv_col is not None:
        frame["conversion"] = _to_int8(df[conv_col], _CONVERSION_MAP)
    else:
        frame["conversion"] = pd.array([pd.NA] * n, dtype="Int8")

    age_col = resolved["age"]
    frame["age"] = (pd.to_numeric(df[age_col], errors="coerce").astype("float64")
                    if age_col else np.full(n, np.nan))

    sex_col = resolved["sex"]
    sex = (df[sex_col].map(lambda v: _map_str(v, _SEX_MAP))
           if sex_col else pd.Series([pd.NA] * n))
    frame["sex"] = pd.Categorical(list(sex), categories=contract.SEX_LEVELS)

    site_col = resolved["site"]
    frame["site"] = pd.Categorical(
        df[site_col].astype("string") if site_col else pd.Series([cfg.name] * n))
    scanner_col = resolved["scanner"]
    frame["scanner"] = pd.Categorical(
        df[scanner_col].astype("string") if scanner_col
        else pd.Series([f"{cfg.name}_unknown"] * n))

    amy_col = resolved["amyloid"]
    frame["amyloid"] = (_to_int8(df[amy_col], _AMYLOID_MAP) if amy_col
                        else pd.array([pd.NA] * n, dtype="Int8"))
    for m in ("p_tau217", "gfap", "nfl"):
        col = resolved[m]
        frame[m] = (pd.to_numeric(df[col], errors="coerce").astype("float64")
                    if col else np.full(n, np.nan, dtype="float64"))
    apoe_col = resolved["apoe4"]
    frame["apoe4"] = (_to_int8(df[apoe_col]) if apoe_col
                      else pd.array([pd.NA] * n, dtype="Int8"))

    frame = frame.drop_duplicates("subject_id", keep="first").reset_index(drop=True)
    contract.validate_table(frame)
    return frame


# ---------------------------------------------------------------------------
# Public entry points.
# ---------------------------------------------------------------------------
def _config_for(dataset: str) -> GatedConfig:
    key = dataset.strip().lower()
    if key not in GATED_CONFIGS:
        raise ValueError(
            f"unknown gated dataset {dataset!r}; "
            f"choose from {sorted(GATED_CONFIGS)}")
    return GATED_CONFIGS[key]


def _stub_path(dataset: str) -> Path:
    return _STUB_DIR / f"{_config_for(dataset).stub_name}_stub.csv"


def load_gated_stub(dataset: str) -> pd.DataFrame:
    """Load the hand-written placeholder stub for ``dataset`` as a contract
    table, clearly marked (``df.attrs['is_stub'] = True``). Raises if missing."""
    path = _stub_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"no stub for {dataset!r} at {path}")
    raw = pd.read_csv(path, comment="#")
    frame = map_export(raw, dataset)
    frame.attrs.update(is_stub=True, source="stub", dataset=_config_for(dataset).name)
    return frame


def load_gated(csv_path: Optional[str] = None, dataset: str = "") -> pd.DataFrame:
    """Load a gated ``dataset`` into a contract table.

    Parameters
    ----------
    csv_path : str | None
        Path to a USER-SUPPLIED real export. If None or the file does not exist,
        transparently falls back to the placeholder stub and marks the result
        (``df.attrs['is_stub'] = True``).
    dataset : {'adni', 'oasis3', 'nacc', 'epad'}
        Which gated cohort's mapping config to apply.

    Returns
    -------
    pd.DataFrame passing ``contract.validate_table``. ``df.attrs`` carries
    ``is_stub`` (bool), ``source`` ('real'|'stub') and ``dataset``.
    """
    cfg = _config_for(dataset)
    if csv_path is not None and Path(csv_path).expanduser().exists():
        raw = pd.read_csv(Path(csv_path).expanduser(), comment="#")
        frame = map_export(raw, dataset)
        frame.attrs.update(is_stub=False, source="real", dataset=cfg.name)
        return frame
    # No real file -> fall back to the clearly-marked stub.
    return load_gated_stub(dataset)


#: Human-facing catalogue of the gated names this feeder handles.
GATED_NAMES = sorted(GATED_CONFIGS)
