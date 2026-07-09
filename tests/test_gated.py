"""
Tests for the gated-dataset drop-in feeder (neuroad.data.gated).

Two guarantees:
  1. With no real file supplied, ``load_gated`` falls back to the clearly-marked
     stub and yields a contract-valid table for every gated name.
  2. A synthetic in-memory "real export" (source column names, raw FreeSurfer +
     clinical fields) maps into a valid contract table — the zero-code drop-in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neuroad import contract
from neuroad.data import gated


# --------------------------------------------------------------------------- #
# 1. stub fallback: schema-valid + clearly marked
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dataset", gated.GATED_NAMES)
def test_load_gated_falls_back_to_stub(dataset):
    df = gated.load_gated(csv_path=None, dataset=dataset)
    contract.validate_table(df)                 # raises on any violation
    assert df.attrs["is_stub"] is True
    assert df.attrs["source"] == "stub"
    assert len(df) > 0
    assert set(df["dx"].dropna().unique()) <= set(contract.DX_LEVELS)


@pytest.mark.parametrize("dataset", gated.GATED_NAMES)
def test_missing_real_path_uses_stub(dataset, tmp_path):
    # A path that does not exist must not raise — it falls back to the stub.
    df = gated.load_gated(str(tmp_path / "not_there.csv"), dataset)
    contract.validate_table(df)
    assert df.attrs["is_stub"] is True


def test_load_gated_stub_matches_expected_plasma_coverage():
    # OASIS-3 has no plasma markers; ADNI does — the stubs must reflect that.
    oasis3 = gated.load_gated_stub("oasis3")
    assert oasis3["p_tau217"].notna().sum() == 0
    adni = gated.load_gated_stub("adni")
    assert adni["p_tau217"].notna().sum() > 0


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        gated.load_gated(dataset="does-not-exist")


# --------------------------------------------------------------------------- #
# 2. synthetic real-export mapping (raw source column names -> contract)
# --------------------------------------------------------------------------- #
def _fake_adni_export() -> pd.DataFrame:
    """An in-memory stand-in for a real ADNI FreeSurfer + plasma export, using
    the source's own column names (never the contract names)."""
    return pd.DataFrame({
        "PTID": ["S1", "S2", "S3", "S4", "S5"],
        "DX_bl": ["CN", "LMCI", "AD", "EMCI", "CN"],
        "DXCONV": [0, 1, np.nan, 0, np.nan],
        "AGE": [70.1, 74.3, 80.0, 68.9, 72.2],
        "PTGENDER": ["Female", "Male", "Female", "Male", "Female"],
        "ORIGPROT": ["ADNI2", "ADNI3", "ADNI2", "ADNI3", "ADNI2"],
        "FLDSTRENG": ["3T", "3T", "1.5T", "3T", "3T"],
        "AV45_pos": [0, 1, 1, 1, 0],
        "PLASMA_PTAU217": [0.11, 0.34, 0.61, 0.25, 0.18],
        "GFAP": [80.0, 142.0, 210.0, 120.0, 90.0],
        "NEFL": [10.0, 20.0, 33.0, 15.0, 12.0],
        "APOE4": [0, 1, 2, 1, 0],
        "Hippocampus": [7000, 6200, 5100, 6500, 6900],
        "WholeBrain": [1_100_000, 1_050_000, 980_000, 1_080_000, 1_095_000],
        "Ventricles": [30_000, 42_000, 60_000, 35_000, 31_000],
    })


def test_real_export_maps_to_contract():
    out = gated.map_export(_fake_adni_export(), "adni")
    contract.validate_table(out)
    # diagnosis strings normalized into the three contract levels
    assert set(out["dx"].dropna().unique()) == {"CN", "MCI", "AD"}
    # EMCI/LMCI both collapse to MCI
    assert (out["dx"] == "MCI").sum() == 2
    # sex normalized to M/F
    assert set(out["sex"].dropna().unique()) <= set(contract.SEX_LEVELS)
    # structural features standardized into emb_* (3 features here)
    assert len(contract.embedding_columns(out)) == 3
    # plasma markers carried through as floats
    assert out["p_tau217"].notna().all()
    assert out["p_tau217"].dtype == np.float64


def test_real_export_via_csv_path_marks_real(tmp_path):
    p = tmp_path / "fake_adni.csv"
    _fake_adni_export().to_csv(p, index=False)
    out = gated.load_gated(str(p), "adni")
    contract.validate_table(out)
    assert out.attrs["is_stub"] is False
    assert out.attrs["source"] == "real"
    assert out.attrs["dataset"] == "ADNI"


def test_oasis3_export_derives_dx_from_cdr():
    """OASIS-3 exports often ship CDR, not a dx string -> band it (0/0.5/>=1)."""
    raw = pd.DataFrame({
        "OASISID": ["OAS30001", "OAS30002", "OAS30003", "OAS30004"],
        "cdr": [0.0, 0.5, 1.0, 0.0],
        "ageAtEntry": [69, 78, 84, 71],
        "M/F": ["F", "M", "F", "M"],
        "Scanner": ["Siemens_TrioTim_3T", "Siemens_BioGraph_3T",
                    "Siemens_TrioTim_3T", "Siemens_BioGraph_3T"],
        "IntraCranialVol": [1_500_000, 1_450_000, 1_400_000, 1_520_000],
        "TotalGrayVol": [600_000, 560_000, 520_000, 610_000],
        "Left-Hippocampus": [3600, 3200, 2800, 3700],
    })
    out = gated.map_export(raw, "oasis3")
    contract.validate_table(out)
    assert out["dx"].tolist() == ["CN", "MCI", "AD", "CN"]
    # OASIS-3 has no plasma -> those columns stay <NA>
    assert out["p_tau217"].notna().sum() == 0
    # real multi-scanner heterogeneity is preserved (the leakage-star substrate)
    assert out["scanner"].nunique() == 2


def test_missing_structural_features_raises_clearly():
    raw = pd.DataFrame({"PTID": ["S1"], "AGE": [70], "PTGENDER": ["F"]})
    with pytest.raises(contract.ContractError):
        gated.map_export(raw, "adni")


def test_precleaned_int8_conversion_survives_map_and_csv_roundtrip(tmp_path):
    """Regression: a already-clean Int8 1/0/<NA> column (as the ADNI helper
    derives for `conversion`) must survive both map_export and a CSV round-trip.

    Guards two latent bugs in the Int8/string coercion path:
      * ``Series.map`` over a nullable Int8 column nulled every value.
      * float round-trips ("1.0") failed to match the "1" mapping key.
    """
    raw = _fake_adni_export().copy()           # 5 rows
    # inject a pre-derived, nullable-Int8 conversion column (1/0/<NA>)
    raw["conversion"] = pd.array([1, 0, pd.NA, 1, 0], dtype="Int8")

    mapped = gated.map_export(raw, "adni")
    assert mapped["conversion"].notna().sum() == 4       # <-- was 0 before fix
    assert int((mapped["conversion"] == 1).sum()) == 2

    # CSV round-trip (float rendering of ints) must not erase the labels
    p = tmp_path / "adni_contract.csv"
    mapped.to_csv(p, index=False)
    reloaded = gated.load_gated(str(p), "adni")
    contract.validate_table(reloaded)
    assert reloaded["conversion"].notna().sum() == 4
    assert int((reloaded["conversion"] == 1).sum()) == 2
