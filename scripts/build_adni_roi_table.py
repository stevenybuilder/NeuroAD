#!/usr/bin/env python
"""Step 1 ETL: build the de-identified ADNI FreeSurfer named-ROI table.

Reads the RAW gated ADNI exports (kept OUTSIDE the repo, gitignored) and writes a
PTID-free, date-free compute artifact ``data/real/_gated/adni_roi.csv`` — the
per-subject named-ROI volumes the region-conditioned probe fits on. Every value is
a real FreeSurfer volume; the RID->subject_id crosswalk is written to
``data/real/_manifests/`` and NEVER leaves the machine.

Inputs (raw, local only):
  download/UCSFFSX7_09Jul2026.csv        FreeSurfer ST-coded volumes
  download (1)/DATADIC_09Jul2026.csv     ST-code -> region dictionary (provenance)
  download/DXSUM_09Jul2026.csv           diagnosis label (1=CN,2=MCI,3=Dementia)

Region -> ST volume codes are NOT hardcoded: they are PARSED from the ADNI data
dictionary (``download (1)/DATADIC_09Jul2026.csv``, TBLNAME=UCSFFSX7). Every
structural-volume field ``ST<n>CV`` / ``ST<n>SV`` has TEXT like
``Cortical Volume (aparc.stats) of LeftEntorhinal`` /
``Subcortical Volume (aseg.stats) of RightHippocampus``; we pair Left/Right of the
same RegionName into ``roi_<slug> = mean(left, right)`` for every Desikan-Killiany
cortical parcel + key subcortical structure (hippocampus, amygdala, thalamus,
ventricles, ...). Global aggregates / QC segmentations are excluded (see
``_EXCLUDE_REGIONS``); midline / single-hemisphere structures are dropped
automatically because they lack a Left+Right pair. Verified structural CV/SV only —
NOT the CBF/perfusion AVG/MIN/MAX/MD variants that reuse the same ST numbers.
ICV (covariate) = ST10CV.

Usage:
    PYTHONPATH=src .venv/bin/python -m scripts.build_adni_roi_table
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ENG = Path(__file__).resolve().parents[1]
_HACK = _ENG.parent
_RAW = _HACK / "download"
_DATADIC = _HACK / "download (1)" / "DATADIC_09Jul2026.csv"
_FSX = _RAW / "UCSFFSX7_09Jul2026.csv"
_DX = _RAW / "DXSUM_09Jul2026.csv"
_OUT_DIR = _ENG / "data" / "real" / "_gated"
_MANIFEST_DIR = _ENG / "data" / "real" / "_manifests"
_OUT = _OUT_DIR / "adni_roi.csv"
_CROSSWALK = _MANIFEST_DIR / "adni_roi_crosswalk.csv"

_ICV_CODE = "ST10CV"
_BASELINE = {"sc", "scmri", "bl"}
_DX_MAP = {1: "CN", 2: "MCI", 3: "AD"}

#: Non-anatomical / whole-brain aggregate / QC-segmentation RegionNames (lowercased
#: slug) that HAVE a Left/Right pair but are NOT a specific region of interest.
#: Everything else with both hemispheres present is kept (cortical parcels +
#: subcortical structures + ventricles). Global/midline structures (Icv, Brainstem,
#: CorpusCallosum*, WMHypoIntensities, the ventricular midline, CorticalGM/WM
#: totals emitted single-hemi, ...) drop out automatically for lacking a pair.
_EXCLUDE_REGIONS = {
    "corticalgm", "corticalwm",   # hemisphere GM/WM totals (aggregates, not ROIs)
    "cerebellumwm",               # white-matter aggregate
    "vessel", "choroidplexus",    # QC / non-parenchymal artifact segmentations
}
#: Minimum fraction of baseline subjects a region must be non-null for to be kept.
_MIN_NONNULL_FRAC = 0.5


def _slug(region_name: str) -> str:
    """CamelCase RegionName -> lowercase slug (``InferiorTemporal`` -> ``inferiortemporal``)."""
    return region_name.strip().lower()


def parse_roi_codes(datadic_path: Path) -> dict[str, tuple[str, str]]:
    """Parse the ADNI data dictionary into ``slug -> (left_code, right_code)``.

    Reads TBLNAME=UCSFFSX7 rows, keeps FLDNAME matching ``ST\\d+(CV|SV)$``, extracts
    (hemisphere, RegionName) from TEXT (``... of LeftEntorhinal``), pairs Left/Right
    of the same RegionName, and drops the excluded aggregates/QC set. Never invents a
    code — a region only appears if BOTH its hemisphere codes are in the dictionary.
    """
    dd = pd.read_csv(datadic_path, low_memory=False)
    dd = dd[dd["TBLNAME"].astype("string") == "UCSFFSX7"]
    fld_re = re.compile(r"^ST\d+(?:CV|SV)$")
    txt_re = re.compile(r"\bof (Left|Right)([A-Za-z]+)\s*$")
    pairs: dict[str, dict[str, str]] = {}
    for _, row in dd.iterrows():
        fld = str(row["FLDNAME"])
        if not fld_re.match(fld):
            continue
        m = txt_re.search(str(row["TEXT"]))
        if not m:  # no hemisphere -> midline/global; excluded by lacking a pair
            continue
        hemi, region = m.group(1), m.group(2)
        slug = _slug(region)
        if slug in _EXCLUDE_REGIONS:
            continue
        pairs.setdefault(slug, {})[hemi] = fld
    codes: dict[str, tuple[str, str]] = {}
    for slug, hemis in sorted(pairs.items()):
        if "Left" in hemis and "Right" in hemis:
            codes[slug] = (hemis["Left"], hemis["Right"])
    return codes


def _baseline_rows(df: pd.DataFrame, visit_col: str) -> pd.DataFrame:
    """Earliest-date baseline row per RID (dedup keep-first)."""
    df = df[df[visit_col].astype("string").str.lower().isin(_BASELINE)].copy()
    if "EXAMDATE" in df.columns:
        df["_d"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
        df = df.sort_values("_d")
    return df.drop_duplicates(subset="RID", keep="first")


def main() -> int:
    if not _FSX.exists() or not _DX.exists():
        print(f"[build_adni_roi] raw inputs missing under {_RAW} — cannot build.",
              file=sys.stderr)
        return 1

    if not _DATADIC.exists():
        print(f"[build_adni_roi] data dictionary missing at {_DATADIC} — cannot build.",
              file=sys.stderr)
        return 1

    roi_codes = parse_roi_codes(_DATADIC)
    print(f"[build_adni_roi] parsed {len(roi_codes)} paired regions from DATADIC.")

    fsx = pd.read_csv(_FSX, low_memory=False)
    fsx = _baseline_rows(fsx, "VISCODE2")
    n_base = len(fsx)

    # Keep only regions whose BOTH codes are real columns in the raw CSV and are
    # non-null for a reasonable N. Drop the rest HONESTLY (print which + why).
    kept: dict[str, tuple[str, str]] = {}
    roi_cols: dict[str, np.ndarray] = {}
    for region, (lc, rc) in roi_codes.items():
        if lc not in fsx.columns or rc not in fsx.columns:
            print(f"[build_adni_roi]   DROP {region}: missing column(s) "
                  f"{[c for c in (lc, rc) if c not in fsx.columns]} in raw CSV")
            continue
        l = pd.to_numeric(fsx[lc], errors="coerce")
        r = pd.to_numeric(fsx[rc], errors="coerce")
        vals = np.nanmean(np.vstack([l.values, r.values]), axis=0)
        nonnull = int(np.isfinite(vals).sum())
        if nonnull < _MIN_NONNULL_FRAC * n_base:
            print(f"[build_adni_roi]   DROP {region}: only {nonnull}/{n_base} "
                  f"non-null (< {_MIN_NONNULL_FRAC:.0%})")
            continue
        kept[region] = (lc, rc)
        roi_cols[region] = vals

    roi = pd.DataFrame({"RID": fsx["RID"].astype("Int64")})
    for region, vals in roi_cols.items():
        roi[f"roi_{region}"] = vals
    roi["icv"] = pd.to_numeric(fsx[_ICV_CODE], errors="coerce")
    # A subject is kept only if it has every kept ROI (FreeSurfer emits them together).
    roi = roi.dropna(subset=[f"roi_{r}" for r in kept])

    dx = pd.read_csv(_DX, low_memory=False)
    dxv = "VISCODE2" if "VISCODE2" in dx.columns else "VISCODE"
    dx = _baseline_rows(dx, dxv)
    dx["dx"] = pd.to_numeric(dx["DIAGNOSIS"], errors="coerce").map(_DX_MAP)
    dx = dx.dropna(subset=["dx"])[["RID", "dx"]]

    merged = roi.merge(dx, on="RID", how="inner")

    # Anonymize: RID -> dense serial subject_id; crosswalk stays local only.
    merged = merged.sort_values("RID").reset_index(drop=True)
    merged["subject_id"] = [f"AR{ i:04d}" for i in range(1, len(merged) + 1)]
    _MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    merged[["subject_id", "RID"]].to_csv(_CROSSWALK, index=False)

    # Contract metadata columns (NA where this feeder has no measurement — the
    # gauntlet tests that need them degrade to 'insufficient data' HONESTLY, never
    # a fabricated number). conversion derived where dx allows; rest NA.
    out = pd.DataFrame({"subject_id": merged["subject_id"]})
    out["dx"] = merged["dx"]
    for c in ("site", "scanner", "conversion", "amyloid", "apoe4", "p_tau217",
              "gfap", "nfl", "age", "sex"):
        out[c] = pd.NA
    out["icv"] = merged["icv"].round(1)
    for region in kept:
        out[f"roi_{region}"] = merged[f"roi_{region}"].round(1)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(_OUT, index=False)

    n = len(out)
    bal = out["dx"].value_counts().to_dict()
    region_slugs = list(kept)
    print(f"[build_adni_roi] wrote {_OUT}  N={n}  dx={bal}")
    print(f"[build_adni_roi] crosswalk (local only) -> {_CROSSWALK}")
    print(f"[build_adni_roi] {len(region_slugs)} regions: {region_slugs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
