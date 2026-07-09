"""
ADNI multi-file export -> one contract table (the ADNI swap-in helper).

Why this exists
---------------
``neuroad.data.gated.load_gated(path, "adni")`` already maps a SINGLE ADNI file
into the contract, and for a plain ADNIMERGE pull that is enough to light up the
diagnosis + site/scanner-leakage + brain-age beats on REAL data. But the three
things ADNI is *uniquely* worth downloading for are NOT ready columns in
ADNIMERGE:

  * ``conversion``  -> must be DERIVED from the longitudinal DX trajectory
                       (baseline MCI that later becomes Dementia).
  * ``p_tau217 / gfap / nfl`` -> live in SEPARATE plasma biomarker tables and
                       must be JOINED onto the subject by RID.
  * ``amyloid``     -> ADNIMERGE ships AV45 SUVR / CSF ABETA, not a 0/1 status;
                       it needs a THRESHOLD.

This helper does exactly those three derivations, then hands the assembled frame
(still carrying ADNI's own column names) to ``gated.map_export(df, "adni")`` so
ALL the existing, tested mapping (dx banding, structural emb_*, dtypes,
validation) is reused unchanged. Output is a contract CSV that ``loaders`` /
``gated.load_gated`` load with zero further work.

Usage
-----
    python scripts/adni_to_contract.py \
        --adnimerge ~/Downloads/ADNIMERGE.csv \
        --plasma ~/Downloads/UGOT_PTAU217.csv \
        --plasma ~/Downloads/ADNI_PLASMA_SIMOA.csv \
        --out data/real/_gated/adni.csv

    # then, anywhere downstream:
    from neuroad.data import gated
    df = gated.load_gated("data/real/_gated/adni.csv", "adni")   # real=True

``--plasma`` is optional and repeatable; each file is auto-scanned for an RID
key and any of the p-tau217 / GFAP / NfL markers by column-name pattern. Nothing
here is ADNI-account-specific: it runs entirely on files you have already
downloaded.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make ``src`` importable when run as a plain script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from neuroad.data import gated  # noqa: E402
from neuroad import contract  # noqa: E402

# Baseline visit codes used across ADNI phases.
_BASELINE_VISCODES = {"bl", "sc", "scmri", "m00", "v01", "init"}

# DX strings that count as an MCI-ish baseline eligible to convert.
_MCI_BASELINE = {"mci", "emci", "lmci", "smc"}
# DX strings that count as having converted to AD/dementia.
_AD_DX = {"ad", "dementia", "dat", "demented", "alzheimer", "alzheimers disease"}


def _norm(s: object) -> str:
    return str(s).strip().lower()


def _rid_col(df: pd.DataFrame) -> str | None:
    for c in ("RID", "rid", "PTID", "ptid", "subject_id", "Subject"):
        if c in df.columns:
            return c
    return None


def _viscode_col(df: pd.DataFrame) -> str | None:
    for c in ("VISCODE2", "VISCODE", "viscode", "Visit", "VISIT"):
        if c in df.columns:
            return c
    return None


def _dx_series(df: pd.DataFrame) -> pd.Series:
    """Best available per-row diagnosis string (DX preferred, DX_bl fallback)."""
    for c in ("DX", "DIAGNOSIS", "diagnosis"):
        if c in df.columns:
            return df[c]
    for c in ("DX_bl", "DXBL"):
        if c in df.columns:
            return df[c]
    return pd.Series([pd.NA] * len(df))


def derive_conversion(adnimerge: pd.DataFrame) -> pd.Series:
    """Per-RID conversion label from the longitudinal DX trajectory.

    1 = baseline MCI-ish AND any later visit is AD/dementia.
    0 = baseline MCI-ish with follow-up, never converts.
    <NA> = not MCI at baseline, or no usable follow-up.

    Returned Series is indexed by RID (as string).
    """
    rid = _rid_col(adnimerge)
    vis = _viscode_col(adnimerge)
    if rid is None:
        return pd.Series(dtype="Int8")
    dx = _dx_series(adnimerge).map(_norm)
    work = pd.DataFrame({"rid": adnimerge[rid].astype(str), "dx": dx})
    if vis is not None:
        work["vis"] = adnimerge[vis].map(_norm)
        work["is_bl"] = work["vis"].isin(_BASELINE_VISCODES)
    else:
        work["is_bl"] = False

    out: dict[str, object] = {}
    for r, g in work.groupby("rid"):
        bl_rows = g[g["is_bl"]]
        baseline_dx = (bl_rows["dx"].iloc[0] if len(bl_rows)
                       else g["dx"].iloc[0])
        mci_base = any(m in baseline_dx for m in _MCI_BASELINE)
        if not mci_base:
            out[r] = pd.NA
            continue
        ever_ad = g["dx"].apply(lambda d: any(a in d for a in _AD_DX)).any()
        has_followup = len(g) > 1
        out[r] = 1 if ever_ad else (0 if has_followup else pd.NA)
    return pd.Series(out, dtype="Int8")


def derive_amyloid(adnimerge: pd.DataFrame,
                   av45_cut: float = 1.11,
                   abeta_cut: float = 980.0) -> pd.Series:
    """Binary amyloid status from AV45 florbetapir SUVR (>= cut -> positive) or,
    failing that, CSF ABETA (< cut -> positive). <NA> where neither present."""
    n = len(adnimerge)
    if "AV45" in adnimerge.columns:
        v = pd.to_numeric(adnimerge["AV45"], errors="coerce")
        return (v >= av45_cut).astype("Float64").astype("Int8").where(v.notna())
    for c in ("ABETA", "ABETA42", "abeta"):
        if c in adnimerge.columns:
            v = pd.to_numeric(adnimerge[c], errors="coerce")
            return (v < abeta_cut).astype("Float64").astype("Int8").where(v.notna())
    return pd.Series([pd.NA] * n, dtype="Int8")


_MARKER_PATTERNS = {
    "p_tau217": re.compile(r"(p[\W_]?tau[\W_]?217|ptau217|plasma.*217)", re.I),
    "gfap": re.compile(r"gfap", re.I),
    "nfl": re.compile(r"(\bnfl\b|nefl|neurofil)", re.I),
}


def merge_plasma(subject_rids: pd.Series,
                 plasma_paths: list[str]) -> pd.DataFrame:
    """Return a frame indexed by RID(str) with any p_tau217/gfap/nfl columns
    found across the plasma files, taking each subject's baseline (else first)
    value. Missing markers are simply absent (filled NA by the caller)."""
    acc = pd.DataFrame(index=subject_rids.astype(str).unique())
    for p in plasma_paths:
        path = Path(p).expanduser()
        if not path.exists():
            print(f"  [plasma] SKIP (not found): {path}", file=sys.stderr)
            continue
        df = pd.read_csv(path, comment="#", low_memory=False)
        rid = _rid_col(df)
        if rid is None:
            print(f"  [plasma] SKIP (no RID column): {path.name}", file=sys.stderr)
            continue
        vis = _viscode_col(df)
        df = df.copy()
        df["_rid"] = df[rid].astype(str)
        if vis is not None:
            df["_bl"] = df[vis].map(_norm).isin(_BASELINE_VISCODES)
            df = df.sort_values("_bl", ascending=False)  # baseline rows first
        picked = df.drop_duplicates("_rid", keep="first").set_index("_rid")
        for marker, pat in _MARKER_PATTERNS.items():
            hit = next((c for c in df.columns if pat.search(str(c))), None)
            if hit is not None:
                acc.loc[picked.index, marker] = pd.to_numeric(
                    picked[hit], errors="coerce")
                print(f"  [plasma] {path.name}: '{hit}' -> {marker} "
                      f"({acc[marker].notna().sum()} values)")
    return acc


def build(adnimerge_path: str, plasma_paths: list[str]) -> pd.DataFrame:
    am = pd.read_csv(Path(adnimerge_path).expanduser(), comment="#",
                     low_memory=False)
    rid = _rid_col(am)
    vis = _viscode_col(am)

    # Subject table = one baseline row per RID (fall back to first seen row).
    if vis is not None:
        am["_bl"] = am[vis].map(_norm).isin(_BASELINE_VISCODES)
        base = am.sort_values("_bl", ascending=False)
    else:
        base = am
    subj = base.drop_duplicates(rid, keep="first").reset_index(drop=True) \
        if rid else am.reset_index(drop=True)

    # 1) conversion from the FULL longitudinal frame, joined back by RID.
    conv = derive_conversion(am)
    if rid and len(conv):
        subj["conversion"] = subj[rid].astype(str).map(conv).astype("Int8")

    # 2) amyloid from AV45 / ABETA threshold.
    subj["amyloid"] = derive_amyloid(subj).to_numpy()

    # 3) plasma markers joined by RID.
    if plasma_paths and rid:
        pl = merge_plasma(subj[rid], plasma_paths)
        for marker in ("p_tau217", "gfap", "nfl"):
            if marker in pl.columns:
                subj[marker] = subj[rid].astype(str).map(pl[marker])

    # Hand off to the tested gated mapper (dx banding, emb_*, dtypes, validate).
    table = gated.map_export(subj, "adni")
    table.attrs.update(is_stub=False, source="real", dataset="ADNI")
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adnimerge", required=True,
                    help="Path to ADNIMERGE.csv (longitudinal is best; carries "
                         "DX, DX_bl, AGE, PTGENDER, SITE, FLDSTRENG, APOE4, and "
                         "the FreeSurfer summary columns).")
    ap.add_argument("--plasma", action="append", default=[],
                    help="Optional plasma biomarker CSV (repeatable): UGOT "
                         "p-tau217, ADNI Simoa GFAP/NfL, etc. Auto-joined by RID.")
    ap.add_argument("--out", default="data/real/_gated/adni.csv",
                    help="Output contract CSV path.")
    args = ap.parse_args()

    print(f"[adni] reading {args.adnimerge}")
    table = build(args.adnimerge, args.plasma)
    contract.validate_table(table)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    n = len(table)
    print(f"\n[adni] wrote {out}  ({n} subjects, contract-valid)")
    print("[adni] coverage:")
    for c in ("dx", "conversion", "amyloid", "p_tau217", "gfap", "nfl", "apoe4"):
        print(f"        {c:11s}: {table[c].notna().sum():4d}/{n}")
    dxc = table["dx"].value_counts().to_dict()
    print(f"[adni] dx: {dxc}")
    print("\nNext: from neuroad.data import gated; "
          f"df = gated.load_gated({str(out)!r}, 'adni')")


if __name__ == "__main__":
    main()
