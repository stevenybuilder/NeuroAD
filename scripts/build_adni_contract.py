"""
Assemble the 11 raw ADNI LONI tables into ONE contract-shaped CSV.

Why this exists (and why it is separate from ``adni_to_contract.py``)
---------------------------------------------------------------------
``adni_to_contract.py`` maps a single ADNIMERGE-style export through
``gated.map_export``'s candidate-name field map. The raw LONI tables downloaded
here (DXSUM / UCSFFSX7 / UPENN plasma / ADSP PET / APOERES / ADSL.rda) do NOT
share those column names and each carries per-VISIT longitudinal rows, while the
contract wants ONE ROW PER SUBJECT. Forcing them through the candidate-name
mapper would silently drop most columns.

So this assembler does all the ADNI-specific work itself and writes a
CONTRACT-SHAPED CSV (all metadata columns + emb_0..emb_{D-1}). That CSV is loaded
by ``gated.load_gated`` / ``loaders.load("adni")`` through the map_export FAST
PATH (dtype coercion + validation only) — ZERO code change to contract.py /
gated.py / loaders.py.

The three derivations ADNI is uniquely worth downloading for
-----------------------------------------------------------
  * conversion  -> DERIVED from the FULL longitudinal DXSUM trajectory per RID:
                   baseline (earliest) DIAGNOSIS==2 (MCI) with >=1 follow-up ->
                   1 if any later DIAGNOSIS==3 (Dementia) else 0; everything else
                   <NA>. DIAGNOSIS is harmonized 1=CN/2=MCI/3=Dementia.
  * p_tau217/gfap/nfl -> JOINED from the UPENN plasma table onto each subject's
                   imaging-anchor visit by NEAREST EXAMDATE (nearest-ever; the
                   imaging anchor is often a screening scan years before the
                   plasma draw). Negative sentinels (-4/-5) masked to NaN first.
  * amyloid     -> ADSP PET PHC_AMYLOID_STATUS (0/1; 9/NaN -> <NA>), the 24/25
                   Centiloid cutoff already baked in, taken at the earliest scan.

Two-track anchoring
-------------------
  1. IMAGING ANCHOR (defines the subject row + emb_* + scanner + site): per RID in
     UCSFFSX7, keep rows with >0.90 non-null ST-column fraction, sort by EXAMDATE
     asc, take the FIRST. These RIDs ARE the cohort (contract requires emb_*).
  2. Every other table left-joins by RID at its own per-RID earliest-date anchor;
     plasma uses nearest-EXAMDATE; conversion uses the whole DXSUM trajectory.

Embeddings = the weight-free structural feeder: z-standardized UCSFFSX7 ST region
columns (coverage>0.90 across the cohort), NaN filled with 0 (= standardized
mean) for a dense matrix. map_export passes existing emb_* through unchanged, so
this script owns standardization.

age/sex come only from ADSL.rda (ADNIMERGE2 R package). pyreadr and
rdata.conversion.convert both fail on it ("unsupported features" / AssertionError)
so a small custom rdata.parser tree-walk decodes the SUBJID/AGE/SEX columns.

Usage
-----
    python scripts/build_adni_contract.py            # all defaults
    python scripts/build_adni_contract.py --out data/real/_gated/adni.csv

then anywhere downstream:
    from neuroad.data import loaders
    df = loaders.load("adni")        # is_stub == False, real contract table
"""
from __future__ import annotations

import argparse
import re
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Make ``src`` importable when run as a plain script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from neuroad.data import gated  # noqa: E402
from neuroad import contract  # noqa: E402

# ---------------------------------------------------------------------------
# Default raw-file locations (the two download dirs; note the space in dir B).
# ---------------------------------------------------------------------------
_DL_A = Path("/Users/stevenyang/Documents/claude-life-sciences-hack/download")
_DL_B = Path("/Users/stevenyang/Documents/claude-life-sciences-hack/download (1)")

_DEFAULTS = {
    "dxsum": _DL_A / "DXSUM_09Jul2026.csv",
    "fs": _DL_A / "UCSFFSX7_09Jul2026.csv",
    "apoe": _DL_A / "APOERES_09Jul2026.csv",
    "plasma": _DL_A / "UPENN_PLASMA_FUJIREBIO_QUANTERIX_09Jul2026.csv",
    "pet": _DL_A / "ADSP_PHC_PET_Amyloid_Simple_09Jul2026.csv",
    "adnimerge_tar": _DL_A / "ADNIMERGE2.tar.gz",
    "out": _REPO_ROOT / "data" / "real" / "_gated" / "adni.csv",
}

# Regex for the FreeSurfer region columns that become emb_* (headers uppercased
# first so 'ST87sa' -> 'ST87SA').
_ST_RE = re.compile(r"^ST\d+(SV|CV|SA|TA|TS)$")

# DIAGNOSIS is harmonized across all ADNI phases: 1=CN, 2=MCI, 3=Dementia.
_DIAG_TO_DX = {1: "CN", 2: "MCI", 3: "AD"}

# Structural-feature coverage floor (fraction non-null across the cohort).
_COVERAGE_FLOOR = 0.90


# ---------------------------------------------------------------------------
# ADSL.rda tree-walk parser (pyreadr / rdata high-level both fail on this file).
# ---------------------------------------------------------------------------
def _load_adsl_age_sex(tar_path: Path) -> pd.DataFrame:
    """Return a DataFrame[SUBJID(str), AGE(float), SEX('M'/'F')] from ADSL.rda.

    ADSL.rda is a bzip2 R-data serialization of a single data.frame. Its layout:
    root LIST node -> car is a VEC (generic vector = the columns), whose
    ``names`` attribute is the STR vector of column names. Each column is an INT
    (factor if it carries a ``levels`` attr), REAL, or STR vector. R's integer-NA
    sentinel is INT_MIN (-2147483648).
    """
    import rdata
    from rdata.parser import RObjectType, parse_file

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(tar_path, "r:gz") as tf:
            member = next(m for m in tf.getmembers()
                          if m.name.endswith("data/ADSL.rda"))
            tf.extract(member, tmp)
            rda = Path(tmp) / member.name
            root = parse_file(rda).object

            def symname(sym) -> str | None:
                if sym is None:
                    return None
                ro = sym.value if sym.info.type == RObjectType.SYM else sym
                v = ro.value
                return v.decode() if isinstance(v, bytes) else v

            def get_attr(obj, name):
                a = obj.attributes
                while a is not None and a.info.type == RObjectType.LIST:
                    if symname(a.tag) == name:
                        return a.value[0]
                    a = a.value[1]
                return None

            int_min = -2147483648

            def decode(col) -> list:
                t = col.info.type
                levels = get_attr(col, "levels")
                if t == RObjectType.INT:
                    arr = np.asarray(col.value)
                    if levels is not None:  # factor: map codes -> labels
                        labels = [x.value.decode() if isinstance(x.value, bytes)
                                  else x.value for x in levels.value]
                        return [None if (v == int_min or v < 1)
                                else labels[int(v) - 1] for v in arr]
                    return [None if v == int_min else int(v) for v in arr]
                if t == RObjectType.REAL:
                    arr = np.asarray(col.value, dtype=float)
                    return [None if np.isnan(v) else float(v) for v in arr]
                if t == RObjectType.STR:
                    return [x.value.decode() if isinstance(x.value, bytes)
                            else x.value for x in col.value]
                raise ValueError(f"unhandled ADSL column type {t}")

            df_obj = root.value[0]
            names = [x.value.decode() for x in get_attr(df_obj, "names").value]
            cols = df_obj.value
            data = {c: decode(cols[names.index(c)]) for c in ("SUBJID", "AGE", "SEX")}

    adsl = pd.DataFrame(data)
    adsl["SUBJID"] = adsl["SUBJID"].astype(str)
    adsl["AGE"] = pd.to_numeric(adsl["AGE"], errors="coerce")
    adsl["sex"] = adsl["SEX"].map({"Male": "M", "Female": "F"})
    return adsl.rename(columns={"AGE": "age"})[["SUBJID", "age", "sex"]]


# ---------------------------------------------------------------------------
# Imaging anchor + embeddings (UCSFFSX7).
# ---------------------------------------------------------------------------
def _imaging_anchor(fs: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Pick one imaging-anchor row per RID and return (anchor_df, st_cols).

    Anchor = first (earliest EXAMDATE) row per RID among rows with >0.90 non-null
    ST-column coverage. NOT OVERALLQC=='Pass' and NOT VISCODE2=='bl' (screening
    'sc'/'scmri' rows precede 'bl').
    """
    fs = fs.copy()
    fs.columns = [c.upper() for c in fs.columns]
    st_cols = sorted(c for c in fs.columns if _ST_RE.match(c))

    st_num = fs[st_cols].apply(pd.to_numeric, errors="coerce")
    row_cov = st_num.notna().mean(axis=1)
    usable = fs[row_cov > _COVERAGE_FLOOR].copy()
    usable[st_cols] = st_num.loc[usable.index]

    usable["_examdate"] = pd.to_datetime(usable["EXAMDATE"], errors="coerce")
    # Earliest usable scan per RID; NaT dates sort last so a dated scan wins.
    usable = usable.sort_values(["RID", "_examdate"], na_position="last")
    anchor = usable.drop_duplicates("RID", keep="first").reset_index(drop=True)
    return anchor, st_cols


def _build_embeddings(anchor: pd.DataFrame, st_cols: list[str]) -> pd.DataFrame:
    """z-standardize the ST columns with cohort coverage>0.90 into emb_*.

    Per-column: drop coverage<=0.90, z-score (ddof=0, zero-std->1.0), fill
    residual NaNs with 0.0 (= standardized mean) for a dense matrix.
    """
    Z = anchor[st_cols]
    keep = [c for c in st_cols if Z[c].notna().mean() > _COVERAGE_FLOOR]
    Z = Z[keep].astype(float)
    std = Z.std(ddof=0).replace(0.0, 1.0)
    Z = (Z - Z.mean()) / std
    Z = Z.fillna(0.0)
    emb = contract.make_embedding_frame(Z.to_numpy())
    return emb


# ---------------------------------------------------------------------------
# dx + conversion (DXSUM).
# ---------------------------------------------------------------------------
def _prep_dxsum(dxsum: pd.DataFrame) -> pd.DataFrame:
    """Dedup (RID,VISCODE2) keeping latest update_stamp, drop null DIAGNOSIS,
    sort per RID by EXAMDATE ascending."""
    dx = dxsum.copy()
    dx["_stamp"] = pd.to_datetime(dx["update_stamp"], errors="coerce")
    dx = dx.sort_values("_stamp").drop_duplicates(
        ["RID", "VISCODE2"], keep="last")
    dx = dx[dx["DIAGNOSIS"].notna()].copy()
    dx["_examdate"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce")
    dx = dx.sort_values(["RID", "_examdate"], na_position="last")
    dx["DIAGNOSIS"] = pd.to_numeric(dx["DIAGNOSIS"], errors="coerce")
    return dx


def _baseline_dx(dx_prepped: pd.DataFrame) -> pd.DataFrame:
    """Per-RID earliest-EXAMDATE DIAGNOSIS mapped to CN/MCI/AD."""
    base = dx_prepped.drop_duplicates("RID", keep="first")
    out = pd.DataFrame({
        "RID": base["RID"].to_numpy(),
        "dx": base["DIAGNOSIS"].map(_DIAG_TO_DX).to_numpy(),
    })
    return out


def _conversion(dx_prepped: pd.DataFrame) -> pd.DataFrame:
    """MCI->AD conversion from the full trajectory: baseline MCI + >=1 follow-up
    -> 1 if any later Dementia else 0; else <NA>."""
    rows = []
    for rid, g in dx_prepped.groupby("RID", sort=False):
        diags = g["DIAGNOSIS"].to_numpy()
        if len(diags) < 2 or diags[0] != 2:  # need baseline MCI + a follow-up
            rows.append((rid, pd.NA))
            continue
        later = diags[1:]
        rows.append((rid, 1 if (later == 3).any() else 0))
    return pd.DataFrame(rows, columns=["RID", "conversion"])


# ---------------------------------------------------------------------------
# amyloid (ADSP PET), apoe4 (APOERES), plasma (UPENN).
# ---------------------------------------------------------------------------
def _amyloid(pet: pd.DataFrame) -> pd.DataFrame:
    """Earliest PHC_SCANDATE per RID; PHC_AMYLOID_STATUS 0/1, 9/NaN -> <NA>."""
    p = pet.copy()
    p["_scandate"] = pd.to_datetime(p["PHC_SCANDATE"], errors="coerce")
    p = p.sort_values(["RID", "_scandate"], na_position="last")
    p = p.drop_duplicates("RID", keep="first")
    status = pd.to_numeric(p["PHC_AMYLOID_STATUS"], errors="coerce")
    status = status.where(status.isin([0, 1]))  # 9 / NaN -> NaN
    return pd.DataFrame({"RID": p["RID"].to_numpy(), "amyloid": status.to_numpy()})


def _apoe4(apoe: pd.DataFrame) -> pd.DataFrame:
    """APOE GENOTYPE 'a/b' -> count of e4 alleles (0/1/2), one row per RID."""
    a = apoe.dropna(subset=["GENOTYPE"]).drop_duplicates("RID", keep="first")
    count = a["GENOTYPE"].astype(str).str.count("4")
    return pd.DataFrame({"RID": a["RID"].to_numpy(), "apoe4": count.to_numpy()})


def _plasma_nearest(plasma: pd.DataFrame,
                    anchor_dates: pd.Series) -> pd.DataFrame:
    """Nearest-EXAMDATE (nearest-ever) plasma draw per cohort RID.

    Sentinel masking (<0 -> NaN) precedes combine_first:
      p_tau217 <- pT217_F ; gfap <- GFAP_Q.combine_first(GFAP_F) ;
      nfl <- NfL_Q.combine_first(NfL_F).
    ``anchor_dates`` is a RID->imaging-EXAMDATE (datetime) map; a null anchor date
    falls back to that RID's earliest plasma row.
    """
    p = plasma.copy()
    for col in ("pT217_F", "GFAP_Q", "GFAP_F", "NfL_Q", "NfL_F"):
        p[col] = pd.to_numeric(p[col], errors="coerce")
        p[col] = p[col].where(p[col] >= 0)  # mask -4/-5 sentinels
    p["p_tau217"] = p["pT217_F"]
    p["gfap"] = p["GFAP_Q"].combine_first(p["GFAP_F"])
    p["nfl"] = p["NfL_Q"].combine_first(p["NfL_F"])
    p["_date"] = pd.to_datetime(p["EXAMDATE"], errors="coerce")

    rows = []
    for rid, g in p.groupby("RID", sort=False):
        if rid not in anchor_dates.index:
            continue
        anchor_date = anchor_dates[rid]
        g = g.sort_values("_date", na_position="last")
        gap_days = np.nan
        if pd.isna(anchor_date):
            pick = g.iloc[0]  # earliest plasma row; gap undefined (no imaging date)
        else:
            gaps = (g["_date"] - anchor_date).abs()
            if gaps.notna().any():
                pick = g.loc[gaps.idxmin()]
                gap_days = float(gaps.loc[gaps.idxmin()].days)
            else:
                pick = g.iloc[0]
        rows.append((rid, pick["p_tau217"], pick["gfap"], pick["nfl"], gap_days))
    # p_tau217_gap_days: |imaging anchor EXAMDATE - picked plasma EXAMDATE| in days.
    # A QC column (not a contract metadata/emb column, so probe/validate ignore it)
    # so the biomarker-anchor test can report or filter on the scan<->blood gap.
    return pd.DataFrame(
        rows, columns=["RID", "p_tau217", "gfap", "nfl", "p_tau217_gap_days"])


# ---------------------------------------------------------------------------
# Assembly.
# ---------------------------------------------------------------------------
def assemble(paths: dict[str, Path]) -> pd.DataFrame:
    """Read the raw files and build the contract-shaped frame (pre-coercion)."""
    fs = pd.read_csv(paths["fs"], low_memory=False)
    dxsum = pd.read_csv(paths["dxsum"], low_memory=False)
    apoe = pd.read_csv(paths["apoe"], low_memory=False)
    plasma = pd.read_csv(paths["plasma"], low_memory=False)
    pet = pd.read_csv(paths["pet"], low_memory=False)

    anchor, st_cols = _imaging_anchor(fs)
    cohort_rids = anchor["RID"].to_numpy()
    n = len(cohort_rids)
    print(f"[anchor] imaging-anchored cohort RIDs: {n}")

    emb = _build_embeddings(anchor, st_cols)
    print(f"[emb] embedding_dim D = {emb.shape[1]}")

    # site (PTID prefix '###_S_####') + scanner (FIELD_STRENGTH) from anchor row.
    site = anchor["PTID"].astype(str).str.split("_").str[0]
    scanner = anchor["FIELD_STRENGTH"].astype(str)
    anchor_dates = pd.to_datetime(anchor["EXAMDATE"], errors="coerce")
    anchor_dates.index = cohort_rids

    dx_prepped = _prep_dxsum(dxsum)
    dx_base = _baseline_dx(dx_prepped)
    conv = _conversion(dx_prepped)
    adsl = _load_adsl_age_sex(paths["adnimerge_tar"])
    adsl_by_rid = adsl.rename(columns={"SUBJID": "RID"})
    adsl_by_rid["RID"] = pd.to_numeric(adsl_by_rid["RID"], errors="coerce")
    amy = _amyloid(pet)
    ap = _apoe4(apoe)
    plas = _plasma_nearest(plasma, anchor_dates)

    # Assemble one row per cohort RID via left joins on RID.
    frame = pd.DataFrame({"subject_id": cohort_rids.astype(str),
                          "_rid": cohort_rids})
    frame["site"] = site.to_numpy()
    frame["scanner"] = scanner.to_numpy()

    for src in (dx_base, conv, amy, ap, plas):
        merged = frame.merge(src, left_on="_rid", right_on="RID", how="left")
        for col in src.columns:
            if col == "RID":
                continue
            frame[col] = merged[col].to_numpy()

    demo = frame.merge(adsl_by_rid, left_on="_rid", right_on="RID", how="left")
    frame["age"] = demo["age"].to_numpy()
    frame["sex"] = demo["sex"].to_numpy()

    frame = frame.drop(columns=["_rid"])
    out = pd.concat([frame.reset_index(drop=True),
                     emb.reset_index(drop=True)], axis=1)

    # Report coverage before handing to the seam.
    for col in ("dx", "conversion", "age", "sex", "amyloid",
                "p_tau217", "gfap", "nfl", "apoe4"):
        print(f"[coverage] {col:>10}: {int(out[col].notna().sum())}/{len(out)}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dxsum", type=Path, default=_DEFAULTS["dxsum"])
    ap.add_argument("--fs", type=Path, default=_DEFAULTS["fs"])
    ap.add_argument("--apoe", type=Path, default=_DEFAULTS["apoe"])
    ap.add_argument("--plasma", type=Path, default=_DEFAULTS["plasma"])
    ap.add_argument("--pet", type=Path, default=_DEFAULTS["pet"])
    ap.add_argument("--adnimerge-tar", type=Path,
                    default=_DEFAULTS["adnimerge_tar"], dest="adnimerge_tar")
    ap.add_argument("--out", type=Path, default=_DEFAULTS["out"])
    args = ap.parse_args(argv)

    paths = {
        "dxsum": args.dxsum, "fs": args.fs, "apoe": args.apoe,
        "plasma": args.plasma, "pet": args.pet,
        "adnimerge_tar": args.adnimerge_tar, "out": args.out,
    }
    missing = [str(p) for k, p in paths.items()
               if k != "out" and not Path(p).exists()]
    if missing:
        print("ERROR: missing raw files:\n  " + "\n  ".join(missing),
              file=sys.stderr)
        return 2

    raw = assemble(paths)

    # Round-trip through the seam's fast path to guarantee contract validity.
    frame = gated.map_export(raw, "adni")
    contract.validate_table(frame)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = contract.cohort_summary(frame)
    print(f"[done] wrote {args.out}")
    print(f"[done] n_subjects={summary['n_subjects']} "
          f"embedding_dim={summary['embedding_dim']}")
    print(f"[done] dx_counts={summary['dx_counts']} "
          f"n_sites={summary['n_sites']} n_scanners={summary['n_scanners']}")
    print(f"[done] age_mean={summary['age_mean']} "
          f"pct_female={summary['pct_female']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
