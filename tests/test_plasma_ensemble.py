"""Tests for the plasma biomarker ensemble (data/plasma_ensemble.py).

Deterministic, offline, synthetic — builds tiny stand-in LONI CSVs in a temp dir
so the z-harmonization + triangulation logic is verified without the gated raw
tables. A real-data smoke is skipped when the download folder is absent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neuroad import contract
from neuroad.data import gated
from neuroad.data.plasma_ensemble import (
    build_plasma_ensemble, merge_into_contract, _zscore, _DEFAULT_DOWNLOAD,
)


def test_zscore_centers_and_scales():
    z = _zscore(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert abs(z.mean()) < 1e-9
    assert abs(z.std() - 1.0) < 1e-9


def _write_assays(d):
    # UPenn: RIDs 1-4 have p-tau217 + Aβ42/40 + gfap/nfl
    pd.DataFrame({
        "RID": [1, 2, 3, 4], "VISCODE": ["bl"] * 4,
        "pT217_F": [0.1, 0.2, 0.3, -4], "AB42_AB40_F": [0.05, 0.06, 0.07, 0.08],
        "GFAP_Q": [100, 120, 140, 160], "NfL_Q": [10, 12, 14, 16],
    }).to_csv(d / "UPENN_PLASMA_FUJIREBIO_QUANTERIX_09Jul2026.csv", index=False)
    # C2N: RIDs 1-2 have a SECOND p-tau217 + %p-tau217 (triangulation on 1,2)
    pd.DataFrame({
        "RID": [1, 2], "VISCODE": ["bl"] * 2,
        "pT217_C2N": [5.0, 6.0], "pT217_npT217_C2N": [3.1, 3.4],
        "AB42_AB40_C2N": [0.051, 0.061],
    }).to_csv(d / "C2N_PRECIVITYAD2_PLASMA_09Jul2026.csv", index=False)


def test_ensemble_triangulates_and_adds_markers(tmp_path):
    _write_assays(tmp_path)
    ens, stats = build_plasma_ensemble(download_dir=tmp_path)
    ens = ens.set_index("RID")

    # RIDs 1,2 were measured by TWO assays -> triangulated; 3 by one; 4 masked (-4)
    assert ens.loc[1, "p_tau217_n_assays"] == 2
    assert ens.loc[2, "p_tau217_n_assays"] == 2
    assert ens.loc[3, "p_tau217_n_assays"] == 1
    assert stats.ptau217_triangulated == 2

    # New markers the single-assay contract lacks
    assert "ab42_40" in ens.columns
    assert "pct_ptau217" in ens.columns
    assert stats.pct_ptau217_coverage == 2         # C2N %p-tau217 for RIDs 1,2
    assert set(stats.assays_present) >= {"upenn", "c2n"}


def test_lilly_long_format_triangulates_to_three(tmp_path):
    """Regression: Lilly MSD600 ships LONG-format (TESTCD/ORRES), not a wide
    PTAU217 column. It was silently dropped, capping triangulation at 2 assays.
    With normalization it must contribute, letting a subject reach depth 3."""
    _write_assays(tmp_path)  # UPenn + C2N (RIDs 1,2 overlap)
    # Lilly long-format: RID 1 gets a THIRD p-tau217; -4 is the missing sentinel.
    pd.DataFrame({
        "RID": [1, 2, 5], "VISCODE2": ["bl"] * 3,
        "TESTCD": ["PTAU217"] * 3, "ORRES": [0.44, -4, 0.30],
    }).to_csv(tmp_path / "LILLY_PTAU217_MSD600_09Jul2026.csv", index=False)

    ens, stats = build_plasma_ensemble(download_dir=tmp_path)
    ens = ens.set_index("RID")

    assert "lilly" in stats.assays_present
    assert ens.loc[1, "p_tau217_n_assays"] == 3        # UPenn + C2N + Lilly
    assert ens.loc[2, "p_tau217_n_assays"] == 2        # Lilly value was -4 (masked)
    assert ens.loc[5, "p_tau217_n_assays"] == 1        # Lilly-only subject


def test_absent_download_degrades_to_empty(tmp_path):
    ens, stats = build_plasma_ensemble(download_dir=tmp_path / "nope")
    assert ens.empty or list(ens.columns) == ["RID"]
    assert stats.ptau217_union == 0


@pytest.mark.skipif(not _DEFAULT_DOWNLOAD.exists(),
                    reason="gated raw LONI download folder not present")
def test_real_data_smoke():
    ens, stats = build_plasma_ensemble()
    assert stats.ptau217_union > 1000            # real ADNI plasma coverage
    assert stats.ptau217_triangulated > 0        # >=2 independent assays exist
    assert set(stats.assays_present) == {"upenn", "c2n", "lilly"}   # all three read
    assert int(ens["p_tau217_n_assays"].max()) == 3  # Lilly not silently dropped


# ---------------------------------------------------------------------------
# Wiring: the ensemble triangulated INTO the ADNI contract anchor.
# ---------------------------------------------------------------------------
def _contract_frame(subject_ids, *, native_ptau=None):
    """A minimal contract-valid ADNI-shaped frame (subject_id == RID string)."""
    n = len(subject_ids)
    rng = np.random.default_rng(0)
    ptau = [np.nan] * n if native_ptau is None else list(native_ptau)
    meta = pd.DataFrame({
        "subject_id": [str(s) for s in subject_ids],
        "dx": (["CN", "MCI", "AD"] * n)[:n],
        "conversion": pd.array([pd.NA] * n, dtype="Int8"),
        "age": rng.normal(72, 6, n),
        "sex": (["M", "F"] * n)[:n],
        "site": ["s1"] * n,
        "scanner": ["3T"] * n,
        "amyloid": pd.array([pd.NA] * n, dtype="Int8"),
        "p_tau217": ptau,
        "gfap": [np.nan] * n,
        "nfl": [np.nan] * n,
        "apoe4": pd.array([pd.NA] * n, dtype="Int8"),
    })
    emb = contract.make_embedding_frame(rng.normal(size=(n, 4)))
    frame = pd.concat([meta, emb], axis=1)
    contract.validate_table(frame)   # sanity: the base frame is contract-valid
    return frame


def test_merge_triangulates_ptau_and_adds_markers(tmp_path):
    _write_assays(tmp_path)
    ens, _ = build_plasma_ensemble(download_dir=tmp_path)
    # native p_tau217 only for subject 1 (single-assay baseline coverage == 1)
    frame = _contract_frame([1, 2, 3, 4, 5], native_ptau=[99.0, np.nan, np.nan,
                                                           np.nan, np.nan])
    base_cov = int(frame["p_tau217"].notna().sum())

    merged, stats = merge_into_contract(frame, ens)

    # Higher coverage: ensemble covers RIDs 1,2,3 -> strictly more than baseline.
    assert int(merged["p_tau217"].notna().sum()) > base_cov
    assert int(merged["p_tau217"].notna().sum()) == 3
    # Single-assay draw preserved (no column removed) and p_tau217 replaced by the
    # z-harmonized ensemble value (differs from the native 99.0 sentinel).
    assert merged.loc[0, "p_tau217_native"] == 99.0
    assert merged.loc[0, "p_tau217"] != 99.0
    # New contract biomarker signals present + triangulation depth recorded.
    for col in ("ab42_40", "pct_ptau217", "p_tau217_n_assays"):
        assert col in merged.columns
    assert set(contract.EXTENDED_BIOMARKER_COLUMNS) <= set(merged.columns)
    assert merged.loc[0, "p_tau217_n_assays"] == 2   # RID 1: upenn + c2n
    # Still contract-valid + coverage surfaced through cohort_summary.
    contract.validate_table(merged)
    summ = contract.cohort_summary(merged)
    assert "pct_ptau217" in summ["extended_biomarker_coverage"]
    assert summ["extended_biomarker_coverage"]["ab42_40"] > 0


def test_merge_survives_gated_fast_path(tmp_path):
    """Merged frame round-trips through gated.map_export (the loader seam) with the
    extended markers preserved + coerced to float, and validates."""
    _write_assays(tmp_path)
    ens, _ = build_plasma_ensemble(download_dir=tmp_path)
    frame = _contract_frame([1, 2, 3, 4])
    merged, _ = merge_into_contract(frame, ens)

    out = gated.map_export(merged, "adni")     # ends in contract.validate_table
    for col in ("ab42_40", "pct_ptau217"):
        assert col in out.columns
        assert out[col].dtype == np.dtype("float64")


def test_merge_without_download_is_a_noop(tmp_path):
    """No gated plasma tables -> frame returned unchanged (graceful degradation)."""
    frame = _contract_frame([1, 2, 3], native_ptau=[1.0, 2.0, 3.0])
    merged, stats = merge_into_contract(frame, download_dir=tmp_path / "nope")
    assert "ab42_40" not in merged.columns
    assert list(merged["p_tau217"]) == [1.0, 2.0, 3.0]
    assert stats.ptau217_union == 0


def test_routing_reads_extended_markers():
    """Bridge mechanism routing folds Aβ42/40 + %p-tau217 into the amyloid/tau
    pole: markers the ensemble adds actually change which mechanism a survivor
    routes to."""
    from neuroad.claude import bridge

    n = 60
    rng = np.random.default_rng(1)
    disease = np.array([1, 0] * (n // 2))
    df = pd.DataFrame({
        "subject_id": [str(i) for i in range(n)],
        "dx": np.where(disease == 1, "AD", "CN"),
        "conversion": pd.array([pd.NA] * n, dtype="Int8"),
        # native fluid markers: gfap carries a MODERATE separation, others flat.
        "p_tau217": rng.normal(0, 1, n),
        "gfap": disease * 0.8 + rng.normal(0, 0.3, n),
        "nfl": rng.normal(0, 1, n),
    })
    # Without the ensemble markers, GFAP dominates -> glial.
    assert bridge._route(df) == "glial"
    # Add strong Aβ42/40 + %p-tau217 separation (the ensemble's contribution):
    # the amyloid/tau pole now wins -> amyloid_cascade.
    df["ab42_40"] = disease * 1.5 + rng.normal(0, 0.2, n)
    df["pct_ptau217"] = disease * 1.5 + rng.normal(0, 0.2, n)
    assert bridge._route(df) == "amyloid_cascade"


def test_anchor_reports_pct_ptau217():
    """The referee biomarker anchor computes + reports a %p-tau217 correlation
    when the ensemble triangulated it in (contract.EXTENDED_BIOMARKER_COLUMNS)."""
    from neuroad.data import synthetic
    from neuroad.gauntlet import test_biomarker_anchor

    df = synthetic.generate_cohort("SURVIVOR", seed=0)
    disease = df["dx"].astype("string").map({"CN": 0, "MCI": 0, "AD": 1}).fillna(0)
    rng = np.random.default_rng(3)
    df["pct_ptau217"] = disease.to_numpy() * 1.0 + rng.normal(0, 0.5, len(df))

    ev = test_biomarker_anchor(df, "dx_binary")
    assert "pct_ptau217_r" in ev.stats
    assert ev.stats["pct_ptau217_r"] is not None       # correlation actually computed
    assert ev.stats["pct_ptau217_n"] >= 20
