"""
plasma_ensemble — triangulate ADNI plasma across the three assays for a stronger,
better-covered biomarker anchor.

The current ADNI contract reads ONE assay (UPenn Fujirebio/Quanterix). ADNI ships
two more that the engine ignores: C2N PrecivityAD2 (a second p-tau217, its
%p-tau217 ratio — one of the best-validated plasma markers — and Aβ42/40) and
Lilly MSD600 (a third p-tau217). This module fuses them so the biomarker anchor
that gates promotion, routes mechanism, and seeds the molecule-side target priors
rests on more subjects AND, where assays overlap, an averaged (noise-reduced)
measurement rather than a single draw.

Assays are on different scales, so each is **z-scored within-assay** before
combining; the ensemble p-tau217 is the mean of a subject's available z-scores.
``p_tau217_n_assays`` records how many independent assays backed each subject (1
= single draw, 2+ = triangulated). New columns the contract lacks: plasma
``ab42_40`` and ``pct_ptau217`` (C2N %p-tau217).

LOCAL-ONLY / GATED: reads the raw LONI CSVs from the download dir (outside the
repo, gitignored like adni.csv). The engine runs fine without it; this is a power
upgrade of the anchor, not a dependency.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_log = logging.getLogger("neuroad.data.plasma_ensemble")

#: Default location of the raw LONI plasma tables (repo-root/../download).
_DEFAULT_DOWNLOAD = Path(__file__).resolve().parents[3].parent / "download"

_FILES = {
    "upenn": "UPENN_PLASMA_FUJIREBIO_QUANTERIX_09Jul2026.csv",
    "c2n": "C2N_PRECIVITYAD2_PLASMA_09Jul2026.csv",
    "lilly": "LILLY_PTAU217_MSD600_09Jul2026.csv",
}

#: Per-assay p-tau217 column name(s), tried in order.
_PTAU_COLS = {
    "upenn": ["pT217_F"],
    "c2n": ["pT217_C2N"],
    "lilly": ["PTAU217", "pTau217", "PLASMAPTAU217", "RESULT"],
}


@dataclass
class EnsembleStats:
    """Coverage summary for the ensembled anchor (vs the single-assay baseline)."""
    n_subjects: int = 0
    ptau217_union: int = 0
    ptau217_triangulated: int = 0     # subjects with >=2 independent assays
    ab42_40_coverage: int = 0
    pct_ptau217_coverage: int = 0
    assays_present: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_subjects": self.n_subjects,
            "ptau217_union": self.ptau217_union,
            "ptau217_triangulated": self.ptau217_triangulated,
            "ab42_40_coverage": self.ab42_40_coverage,
            "pct_ptau217_coverage": self.pct_ptau217_coverage,
            "assays_present": list(self.assays_present),
        }


def _mask_sentinels(s: pd.Series) -> pd.Series:
    """ADNI codes missing as negative sentinels (-4/-5); to NaN, keep positives."""
    v = pd.to_numeric(s, errors="coerce")
    return v.mask(v < 0)


def _zscore(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    mu, sd = v.mean(), v.std()
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (v - mu) / sd


def _per_subject(df: pd.DataFrame, col: str, agg: str = "mean") -> pd.Series:
    """One value per RID for ``col`` (subject-level, sentinel-masked)."""
    if "RID" not in df.columns or col not in df.columns:
        return pd.Series(dtype=float)
    v = _mask_sentinels(df[col])
    g = pd.DataFrame({"RID": pd.to_numeric(df["RID"], errors="coerce"), "v": v}).dropna(subset=["RID"])
    g["RID"] = g["RID"].astype(int)
    return getattr(g.groupby("RID")["v"], agg)()


def build_plasma_ensemble(download_dir: Optional[Path] = None
                          ) -> tuple[pd.DataFrame, EnsembleStats]:
    """Return a per-RID ensembled plasma table + coverage stats.

    Columns: RID, p_tau217 (z-harmonized ensemble), p_tau217_n_assays,
    ab42_40 (z-harmonized), pct_ptau217 (C2N %p-tau217, z), gfap, nfl.
    Empty frame (no error) if the download dir is absent — the engine degrades to
    the single-assay contract."""
    ddir = Path(download_dir) if download_dir else _DEFAULT_DOWNLOAD
    tables: dict[str, pd.DataFrame] = {}
    for key, fname in _FILES.items():
        p = ddir / fname
        if p.exists():
            try:
                tables[key] = pd.read_csv(p, low_memory=False)
            except Exception as exc:  # noqa: BLE001
                _log.warning("could not read %s: %r", fname, exc)

    stats = EnsembleStats(assays_present=sorted(tables.keys()))
    if not tables:
        return pd.DataFrame(columns=["RID"]), stats

    # --- p-tau217: z-score each assay's subject-level value, then average ---
    ptau_z: dict[str, pd.Series] = {}
    for key, df in tables.items():
        col = next((c for c in _PTAU_COLS.get(key, []) if c in df.columns), None)
        if col is None:
            continue
        subj = _per_subject(df, col, "mean")
        if not subj.empty:
            ptau_z[key] = _zscore(subj)

    ptau_df = pd.DataFrame(ptau_z)
    ens = pd.DataFrame(index=ptau_df.index)
    if not ptau_df.empty:
        ens["p_tau217"] = ptau_df.mean(axis=1, skipna=True)
        ens["p_tau217_n_assays"] = ptau_df.notna().sum(axis=1).astype(int)

    # --- extra markers the contract lacks: plasma Aβ42/40 + C2N %p-tau217 ---
    ab_parts = []
    for key, sub_col in (("upenn", "AB42_AB40_F"), ("c2n", "AB42_AB40_C2N")):
        if key in tables:
            s = _per_subject(tables[key], sub_col, "mean")
            if not s.empty:
                ab_parts.append(_zscore(s).rename(key))
    if ab_parts:
        ab = pd.concat(ab_parts, axis=1)
        ens = ens.join(ab.mean(axis=1, skipna=True).rename("ab42_40"), how="outer")

    if "c2n" in tables:
        pct = _per_subject(tables["c2n"], "pT217_npT217_C2N", "mean")
        if not pct.empty:
            ens = ens.join(_zscore(pct).rename("pct_ptau217"), how="outer")

    # --- gfap / nfl (UPenn Quanterix, kept in native units) ---
    if "upenn" in tables:
        for out, col in (("gfap", "GFAP_Q"), ("nfl", "NfL_Q")):
            s = _per_subject(tables["upenn"], col, "mean")
            if not s.empty:
                ens = ens.join(s.rename(out), how="outer")

    ens = ens.reset_index().rename(columns={"index": "RID"})
    if "RID" in ens.columns:
        ens["RID"] = pd.to_numeric(ens["RID"], errors="coerce").astype("Int64")

    stats.n_subjects = int(ens["RID"].nunique()) if "RID" in ens else 0
    if "p_tau217" in ens:
        stats.ptau217_union = int(ens["p_tau217"].notna().sum())
    if "p_tau217_n_assays" in ens:
        stats.ptau217_triangulated = int((ens["p_tau217_n_assays"] >= 2).sum())
    if "ab42_40" in ens:
        stats.ab42_40_coverage = int(ens["ab42_40"].notna().sum())
    if "pct_ptau217" in ens:
        stats.pct_ptau217_coverage = int(ens["pct_ptau217"].notna().sum())
    return ens, stats
