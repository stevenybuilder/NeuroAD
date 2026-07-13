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
import tarfile
import tempfile
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

# --- RAW metadata feeders (local only, joined by RID at baseline) -----------
_ADNIMERGE_TGZ = _RAW / "ADNIMERGE2.tar.gz"        # -> PTDEMOG.rda (age, sex)
_MRIMETA = _RAW / "MRIMETA_09Jul2026.csv"          # 1.5T scans: SITEID, FIELD_STRENGTH
_MRI3META = _RAW / "MRI3META_09Jul2026.csv"        # 3T scans: SITEID, FIELD_STRENGTH
_PTAU217 = _RAW / "LILLY_PTAU217_MSD600_09Jul2026.csv"       # long-format assay
_AMYLOID = _RAW / "ADSP_PHC_PET_Amyloid_Simple_09Jul2026.csv"  # PHC_AMYLOID_STATUS
_C2N = _RAW / "C2N_PRECIVITYAD2_PLASMA_09Jul2026.csv"          # AB42_AB40_C2N
_APOE = _RAW / "APOERES_09Jul2026.csv"                          # GENOTYPE
_UPENN = _RAW / "UPENN_PLASMA_FUJIREBIO_QUANTERIX_09Jul2026.csv"  # GFAP_Q / NfL_Q

_ICV_CODE = "ST10CV"
_BASELINE = {"sc", "scmri", "bl"}
_DX_MAP = {1: "CN", 2: "MCI", 3: "AD"}
#: ADNI numeric sentinel for "not applicable / not done" — must NEVER become a value.
_SENTINELS = {-4.0, -1.0}

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


# ---------------------------------------------------------------------------
# Real per-subject metadata joined by RID at BASELINE. Every feeder returns a
# one-row-per-RID frame; a subject with no measurement stays absent (-> NA after
# the left-join). ADNI sentinels (-4/-1) are dropped, NEVER kept as a value.
# ---------------------------------------------------------------------------
def _num(series: pd.Series, *, positive: bool = False) -> pd.Series:
    """Coerce to numeric, blanking ADNI sentinels; optionally require > 0."""
    v = pd.to_numeric(series, errors="coerce")
    v = v.mask(v.isin(_SENTINELS))
    if positive:
        v = v.mask(v <= 0)
    return v


def _demographics(base_dates: pd.DataFrame, tmp: Path) -> pd.DataFrame:
    """age (at the subject's baseline scan) + sex, from ADNIMERGE2 PTDEMOG.rda.

    ``base_dates`` is RID -> baseline EXAMDATE (the FreeSurfer scan date). PTDOB is
    'MM/YYYY'; age = years between DOB and that scan date. Sex from PTGENDER."""
    import pyreadr  # local raw dep; only needed when raw inputs are present

    with tarfile.open(_ADNIMERGE_TGZ) as tf:
        member = "ADNIMERGE2/data/PTDEMOG.rda"
        tf.extract(member, path=tmp)
    dem = list(pyreadr.read_r(str(tmp / member)).values())[0]

    # One demographics row per RID (DOB/sex are static): prefer a screening row,
    # keep the first with a usable DOB + gender.
    dem = dem.copy()
    dem["_isbl"] = dem["VISCODE2"].astype("string").str.lower().isin(_BASELINE)
    dem = dem.sort_values("_isbl", ascending=False)
    dem = dem.dropna(subset=["PTGENDER"]).drop_duplicates(subset="RID", keep="first")

    sex_map = {"male": "M", "1": "M", "female": "F", "2": "F"}
    sex = dem["PTGENDER"].astype("string").str.strip().str.lower().map(sex_map)

    # DOB year/month from 'MM/YYYY' (fall back to PTDOBYY for the year).
    dob = dem["PTDOB"].astype("string").str.extract(r"(?P<mo>\d{1,2})/(?P<yr>\d{4})")
    byear = pd.to_numeric(dob["yr"], errors="coerce")
    byear = byear.fillna(pd.to_numeric(dem["PTDOBYY"], errors="coerce"))
    bmonth = pd.to_numeric(dob["mo"], errors="coerce").fillna(6)

    d = pd.DataFrame({"RID": dem["RID"].astype("Int64"), "sex": sex.values,
                      "_byear": byear.values, "_bmonth": bmonth.values})
    d = d.merge(base_dates, on="RID", how="left")
    exam = pd.to_datetime(d["EXAMDATE"], errors="coerce")
    d["age"] = (exam.dt.year - d["_byear"]) + (exam.dt.month - d["_bmonth"]) / 12.0
    d.loc[d["_byear"].isna() | exam.isna(), "age"] = np.nan
    return d[["RID", "age", "sex"]]


def _mri_meta() -> pd.DataFrame:
    """site (SITEID) + scanner (field strength label) from the 1.5T + 3T MRI meta
    tables, baseline scan per RID."""
    frames = []
    for path in (_MRIMETA, _MRI3META):
        if not path.exists():
            continue
        m = pd.read_csv(path, low_memory=False)
        vc = "VISCODE2" if "VISCODE2" in m.columns else "VISCODE"
        m = _baseline_rows(m, vc)
        fs = m["FIELD_STRENGTH"].astype("string") if "FIELD_STRENGTH" in m.columns else pd.Series(pd.NA, index=m.index, dtype="string")
        # Fall back to the coded field strength (1 -> 1.5T, 2/3 -> 3T) when the
        # string label is missing, rounding the actual field strength.
        if "FLDSTRNGTH" in m.columns:
            code = pd.to_numeric(m["FLDSTRNGTH"], errors="coerce")
            derived = code.map(lambda c: "1.5T" if c == 1 else ("3T" if c in (2, 3) else pd.NA))
            fs = fs.fillna(pd.Series(derived.values, index=m.index).astype("string"))
        frames.append(pd.DataFrame({
            "RID": m["RID"].astype("Int64"),
            "site": m["SITEID"].astype("Int64").astype("string") if "SITEID" in m.columns else pd.NA,
            "scanner": fs.values,
            "_d": pd.to_datetime(m["EXAMDATE"], errors="coerce").values,
        }))
    if not frames:
        return pd.DataFrame(columns=["RID", "site", "scanner"])
    allm = pd.concat(frames, ignore_index=True).sort_values("_d")
    allm = allm.drop_duplicates(subset="RID", keep="first")
    return allm[["RID", "site", "scanner"]]


def _ptau217() -> pd.DataFrame:
    """plasma p_tau217 from the UPenn Fujirebio/Quanterix assay column ``pT217_F``
    (~1,593 subjects) — the SAME broad, real plasma p-tau217 source the adni:combat
    contract uses (scripts/build_adni_contract.py: ``p_tau217 = pT217_F``), NOT the
    thin Lilly MSD sub-study (~278). Real plasma pg/mL, baseline per RID."""
    u = pd.read_csv(_UPENN, low_memory=False)
    if "pT217_F" not in u.columns:
        return pd.DataFrame(columns=["RID", "p_tau217"])
    u = u.assign(p_tau217=_num(u["pT217_F"], positive=True)).dropna(subset=["p_tau217"])
    vc = "VISCODE2" if "VISCODE2" in u.columns else "VISCODE"
    u = _baseline_rows(u, vc)
    return u[["RID", "p_tau217"]]


def _amyloid() -> pd.DataFrame:
    """amyloid positivity (1/0) from ADSP PHC PET, baseline scan per RID.
    Status 9 (indeterminate) and NA are dropped."""
    a = pd.read_csv(_AMYLOID, low_memory=False)
    if "EXAMDATE" not in a.columns and "PHC_SCANDATE" in a.columns:
        a = a.rename(columns={"PHC_SCANDATE": "EXAMDATE"})
    a["amyloid"] = pd.to_numeric(a["PHC_AMYLOID_STATUS"], errors="coerce")
    a = a[a["amyloid"].isin([0.0, 1.0])].copy()
    vc = "VISCODE2" if "VISCODE2" in a.columns else "VISCODE"
    a = _baseline_rows(a, vc)
    a["amyloid"] = a["amyloid"].astype("Int8")
    return a[["RID", "amyloid"]]


def _ab42_40() -> pd.DataFrame:
    """plasma Aβ42/40 ratio from C2N PrecivityAD2, baseline per RID."""
    c = pd.read_csv(_C2N, low_memory=False)
    c["ab42_40"] = _num(c["AB42_AB40_C2N"], positive=True)
    c = c.dropna(subset=["ab42_40"])
    vc = "VISCODE2" if "VISCODE2" in c.columns else "VISCODE"
    c = _baseline_rows(c, vc)
    return c[["RID", "ab42_40"]]


def _apoe4() -> pd.DataFrame:
    """APOE e4 allele count (0/1/2) from the GENOTYPE string, one row per RID."""
    ap = pd.read_csv(_APOE, low_memory=False)
    g = ap["GENOTYPE"].astype("string")
    count = g.str.count("4").where(g.notna())
    ap = ap.assign(apoe4=count).dropna(subset=["apoe4"])
    ap = ap.drop_duplicates(subset="RID", keep="first")
    ap["apoe4"] = ap["apoe4"].astype("Int8")
    return ap[["RID", "apoe4"]]


def _gfap_nfl() -> pd.DataFrame:
    """plasma GFAP + NfL (Quanterix Simoa) from the UPenn panel, baseline per RID."""
    u = pd.read_csv(_UPENN, low_memory=False)
    out = pd.DataFrame({"RID": u["RID"]})
    out["gfap"] = _num(u["GFAP_Q"], positive=True) if "GFAP_Q" in u.columns else np.nan
    out["nfl"] = _num(u["NfL_Q"], positive=True) if "NfL_Q" in u.columns else np.nan
    u = u.assign(gfap=out["gfap"].values, nfl=out["nfl"].values)
    u = u.dropna(subset=["gfap", "nfl"], how="all")
    vc = "VISCODE2" if "VISCODE2" in u.columns else "VISCODE"
    u = _baseline_rows(u, vc)
    return u[["RID", "gfap", "nfl"]]


def _conversion(dx_all: pd.DataFrame) -> pd.DataFrame:
    """MCI->AD conversion (1/0/<NA>) from longitudinal DXSUM.

    Baseline MCI who later reach Dementia -> 1; baseline MCI with follow-up but never
    Dementia -> 0; no follow-up or non-MCI baseline -> NA (left absent -> honest NA)."""
    d = dx_all.copy()
    d["_d"] = pd.to_datetime(d["EXAMDATE"], errors="coerce")
    d["dx"] = pd.to_numeric(d["DIAGNOSIS"], errors="coerce").map(_DX_MAP)
    d = d.dropna(subset=["dx", "_d"]).sort_values("_d")
    rows = []
    for rid, g in d.groupby("RID"):
        seq = list(g["dx"])
        if not seq or seq[0] != "MCI":
            continue  # non-MCI baseline -> NA
        follow = seq[1:]
        if not follow:
            continue  # no follow-up -> NA
        conv = 1 if "AD" in follow else 0
        rows.append((rid, conv))
    if not rows:
        return pd.DataFrame(columns=["RID", "conversion"])
    out = pd.DataFrame(rows, columns=["RID", "conversion"])
    out["RID"] = out["RID"].astype("Int64")
    out["conversion"] = out["conversion"].astype("Int8")
    return out


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
    # Baseline scan date per RID (for computing age at scan) — stays LOCAL, never
    # written to the de-identified output.
    base_dates = pd.DataFrame({"RID": fsx["RID"].astype("Int64"),
                               "EXAMDATE": fsx["EXAMDATE"]}).dropna(subset=["RID"])

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

    dx_all = pd.read_csv(_DX, low_memory=False)  # longitudinal (for conversion)
    dxv = "VISCODE2" if "VISCODE2" in dx_all.columns else "VISCODE"
    dx = _baseline_rows(dx_all, dxv)
    dx["dx"] = pd.to_numeric(dx["DIAGNOSIS"], errors="coerce").map(_DX_MAP)
    dx = dx.dropna(subset=["dx"])[["RID", "dx"]]

    merged = roi.merge(dx, on="RID", how="inner")

    # Anonymize: RID -> dense serial subject_id; crosswalk stays local only.
    merged = merged.sort_values("RID").reset_index(drop=True)
    merged["subject_id"] = [f"AR{ i:04d}" for i in range(1, len(merged) + 1)]
    _MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    merged[["subject_id", "RID"]].to_csv(_CROSSWALK, index=False)

    # --- REAL metadata joined by RID at baseline -------------------------
    # Each feeder is a one-row-per-RID frame; left-join preserves the ROI cohort N
    # and leaves NA where a subject has no measurement (honest coverage gaps, never
    # imputed or fabricated). Raw tarball -> temp dir, cleaned up automatically.
    keys = merged[["RID"]].copy()
    with tempfile.TemporaryDirectory() as _td:
        tmp = Path(_td)
        feeders: list[tuple[str, pd.DataFrame]] = []
        for label, fn in (
            ("demographics", lambda: _demographics(base_dates, tmp)),
            ("mri_meta", _mri_meta),
            ("p_tau217", _ptau217),
            ("amyloid", _amyloid),
            ("ab42_40", _ab42_40),
            ("apoe4", _apoe4),
            ("gfap/nfl", _gfap_nfl),
            ("conversion", lambda: _conversion(dx_all)),
        ):
            try:
                fr = fn()
                fr["RID"] = fr["RID"].astype("Int64")
                keys = keys.merge(fr, on="RID", how="left")
            except Exception as exc:  # noqa: BLE001 — a missing feeder stays NA, honestly
                print(f"[build_adni_roi]   metadata feeder '{label}' skipped: {exc}")

    meta = keys.set_index("RID")

    def _col(name: str) -> pd.Series:
        s = meta[name] if name in meta.columns else pd.Series(pd.NA, index=meta.index)
        return s.reindex(merged["RID"].values).reset_index(drop=True)

    # Contract metadata columns. NA where a feeder has no measurement — the
    # gauntlet tests degrade to 'insufficient data' HONESTLY, never a fabricated
    # number.
    out = pd.DataFrame({"subject_id": merged["subject_id"]})
    out["dx"] = merged["dx"]
    out["conversion"] = pd.array(_col("conversion"), dtype="Int8")
    out["age"] = pd.to_numeric(_col("age"), errors="coerce").round(1)
    out["sex"] = _col("sex")
    out["site"] = _col("site")
    out["scanner"] = _col("scanner")
    out["amyloid"] = pd.array(_col("amyloid"), dtype="Int8")
    out["apoe4"] = pd.array(_col("apoe4"), dtype="Int8")
    out["p_tau217"] = pd.to_numeric(_col("p_tau217"), errors="coerce")
    out["gfap"] = pd.to_numeric(_col("gfap"), errors="coerce")
    out["nfl"] = pd.to_numeric(_col("nfl"), errors="coerce")
    out["ab42_40"] = pd.to_numeric(_col("ab42_40"), errors="coerce")
    out["icv"] = merged["icv"].round(1)
    for region in kept:
        out[f"roi_{region}"] = merged[f"roi_{region}"].round(1)

    _cov_cols = ["age", "sex", "site", "scanner", "amyloid", "apoe4",
                 "p_tau217", "ab42_40", "gfap", "nfl", "conversion"]
    print("[build_adni_roi] metadata coverage (fraction non-null at baseline):")
    for c in _cov_cols:
        nn = int(out[c].notna().sum())
        print(f"[build_adni_roi]   {c:<10} {nn:>4}/{len(out)}  ({nn/len(out):.1%})")

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
