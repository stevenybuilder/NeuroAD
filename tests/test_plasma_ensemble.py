"""Tests for the plasma biomarker ensemble (data/plasma_ensemble.py).

Deterministic, offline, synthetic — builds tiny stand-in LONI CSVs in a temp dir
so the z-harmonization + triangulation logic is verified without the gated raw
tables. A real-data smoke is skipped when the download folder is absent.
"""
from __future__ import annotations

import pandas as pd
import pytest

from neuroad.data.plasma_ensemble import (
    build_plasma_ensemble, _zscore, _DEFAULT_DOWNLOAD,
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
