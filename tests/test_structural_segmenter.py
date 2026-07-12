"""
Deterministic, offline tests for the FastSurfer structural-volume extractor.

No torch, no GPU, no network, no FastSurfer install is required: the aseg.stats
parser is exercised against a bundled fixture, and the GPU path is proven to
HONESTLY DEGRADE to None (never fabricate volumes) when torch / GPU / FastSurfer
are absent — all via monkeypatched seams so the suite is deterministic regardless
of what is installed on the runner.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from neuroad import contract
from neuroad.data import real
from neuroad.integrations import structural_segmenter as ss

_FIXTURE = Path(__file__).parent / "fixtures" / "aseg.stats"


# --------------------------------------------------------------------------- #
# Layer 1 — pure aseg.stats parser
# --------------------------------------------------------------------------- #
def test_parse_aseg_stats_expected_dict():
    out = ss.parse_aseg_stats(_FIXTURE)
    # bilateral hippocampus = Left(3450.0) + Right(3350.0)
    assert out["hippocampal_volume"] == 6800.0
    # ventricles = LLV(7600.5)+LILV(210)+3rd(1050)+4th(950)+RLV(7100)+RILV(190)
    assert out["ventricle_volume"] == 17100.5
    # whole brain = BrainSegVolNotVent
    assert out["whole_brain_volume"] == 1130000.0
    # cortex = CortexVol
    assert out["cortex_volume"] == 480000.0
    # intracranial = eTIV
    assert out["intracranial_volume"] == 1500000.0


def test_parse_aseg_stats_provenance_and_subject():
    out = ss.parse_aseg_stats(_FIXTURE)
    assert out["source"] == ss.SOURCE_ASEG
    assert out["aseg_path"] == str(_FIXTURE)
    # subjectname header is recovered
    assert out["subject_id"] == "OAS1_0001_MR1"


def test_parse_aseg_stats_all_volume_keys_present():
    out = ss.parse_aseg_stats(_FIXTURE)
    for k in ss.VOLUME_KEYS:
        assert k in out and out[k] is not None


def test_parse_aseg_stats_missing_structures_are_none(tmp_path):
    # a stats file with the header block but NO table rows -> subcortical keys None,
    # measure-derived keys still parsed (never fabricated).
    stub = tmp_path / "aseg.stats"
    stub.write_text(
        "# Measure Cortex, CortexVol, Total cortical gray matter volume, 400000.0, mm^3\n"
        "# ColHeaders Index SegId NVoxels Volume_mm3 StructName\n",
        encoding="utf-8",
    )
    out = ss.parse_aseg_stats(stub)
    assert out["hippocampal_volume"] is None       # no table rows -> honestly None
    assert out["ventricle_volume"] is None
    assert out["cortex_volume"] == 400000.0        # from the Measure line
    assert out["whole_brain_volume"] is None


def test_parse_aseg_stats_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ss.parse_aseg_stats(tmp_path / "does_not_exist.stats")


# --------------------------------------------------------------------------- #
# Layer 2 — GPU path degrades honestly to None
# --------------------------------------------------------------------------- #
def _touch_nifti(tmp_path) -> Path:
    p = tmp_path / "sub.nii.gz"
    p.write_bytes(b"\x00")   # existence is all the GPU-guard checks care about
    return p


def test_segment_volume_none_when_no_torch_gpu(tmp_path, monkeypatch):
    nifti = _touch_nifti(tmp_path)
    monkeypatch.setattr(ss, "_torch_gpu_available",
                        lambda: (False, "torch not importable (ModuleNotFoundError)"))
    # runner should never be reached; if it were, it must not fabricate anything.
    assert ss.segment_volume(nifti, _runner=lambda *a: _FIXTURE) is None


def test_segment_volume_none_when_fastsurfer_absent(tmp_path, monkeypatch):
    # simulate a GPU being present but FastSurfer / its weights NOT installed.
    nifti = _touch_nifti(tmp_path)
    monkeypatch.setattr(ss, "_torch_gpu_available", lambda: (True, ""))
    monkeypatch.setattr(ss, "_locate_fastsurfer", lambda: None)
    assert ss.segment_volume(nifti, _runner=lambda *a: _FIXTURE) is None


def test_segment_volume_none_when_input_missing(tmp_path):
    # a nonexistent volume never fabricates volumes, regardless of deps.
    assert ss.segment_volume(tmp_path / "nope.nii.gz") is None


def test_segment_volume_none_when_segmentation_fails(tmp_path, monkeypatch):
    nifti = _touch_nifti(tmp_path)
    monkeypatch.setattr(ss, "_torch_gpu_available", lambda: (True, ""))
    monkeypatch.setattr(ss, "_locate_fastsurfer", lambda: "/fake/run_fastsurfer.sh")
    # runner returns None (non-zero exit / missing output) -> honest None.
    assert ss.segment_volume(nifti, _runner=lambda *a: None) is None


def test_segment_volume_success_parses_and_restamps(tmp_path, monkeypatch):
    # Full GPU path SIMULATED: torch+GPU present, FastSurfer present, and the runner
    # points at a REAL aseg.stats (the fixture). The result must be the parsed
    # volumes, re-stamped source=SOURCE_SEGMENT (proving it only ever reads a real
    # stats file — it does not invent numbers).
    nifti = _touch_nifti(tmp_path)
    monkeypatch.setattr(ss, "_torch_gpu_available", lambda: (True, ""))
    monkeypatch.setattr(ss, "_locate_fastsurfer", lambda: "/fake/run_fastsurfer.sh")
    out = ss.segment_volume(nifti, subject_id="OAS1_0001_MR1",
                            _runner=lambda *a: _FIXTURE)
    assert out is not None
    assert out["source"] == ss.SOURCE_SEGMENT
    assert out["hippocampal_volume"] == 6800.0
    assert out["subject_id"] == "OAS1_0001_MR1"


# --------------------------------------------------------------------------- #
# real.py feeder — additive volume join, and full contract WITHOUT the CSV
# --------------------------------------------------------------------------- #
def test_feeder_without_volumes_csv_passes_contract(monkeypatch, tmp_path):
    # point the feeder at a nonexistent volumes CSV -> must behave exactly as the
    # shipping 'no cache' case: full table, contract-valid, no volume columns.
    monkeypatch.setattr(real, "OASIS_VOLUMES_CSV", tmp_path / "absent_volumes.csv")
    df = real.load_oasis("both")
    contract.validate_table(df)          # the honest 'cache absent' contract holds
    assert "hippocampal_volume" not in df.columns


def test_feeder_joins_volumes_csv_when_present(monkeypatch, tmp_path):
    # baseline table (no CSV) to grab real subject_ids.
    monkeypatch.setattr(real, "OASIS_VOLUMES_CSV", tmp_path / "absent.csv")
    base = real.load_oasis("both")
    sids = list(base["subject_id"].astype(str).head(3))

    vols_csv = tmp_path / "oasis_volumes.csv"
    pd.DataFrame({
        "subject_id": sids,
        "hippocampal_volume": [6800.0, 6500.0, 7000.0],
        "ventricle_volume": [17100.5, 20000.0, 15000.0],
        "whole_brain_volume": [1130000.0, 1100000.0, 1150000.0],
        "cortex_volume": [480000.0, 470000.0, 490000.0],
        "intracranial_volume": [1500000.0, 1480000.0, 1520000.0],
        "source": [ss.SOURCE_SEGMENT] * 3,
    }).to_csv(vols_csv, index=False)

    monkeypatch.setattr(real, "OASIS_VOLUMES_CSV", vols_csv)
    df = real.load_oasis("both")

    contract.validate_table(df)                         # extra cols are contract-legal
    assert len(df) == len(base)                         # no rows dropped/duplicated
    assert "hippocampal_volume" in df.columns
    joined = df.set_index("subject_id")
    assert joined.loc[sids[0], "hippocampal_volume"] == 6800.0
    assert joined.loc[sids[1], "ventricle_volume"] == 20000.0
    # subjects without a volume row get NaN (not a fabricated number).
    non_joined = df[~df["subject_id"].isin(sids)]
    assert non_joined["hippocampal_volume"].isna().all()


def test_feeder_join_ignores_csv_without_subject_id(monkeypatch, tmp_path):
    # a malformed volume CSV (no subject_id col) is ignored, not fatal.
    bad = tmp_path / "oasis_volumes.csv"
    pd.DataFrame({"hippocampal_volume": [6800.0]}).to_csv(bad, index=False)
    monkeypatch.setattr(real, "OASIS_VOLUMES_CSV", bad)
    df = real.load_oasis("both")
    contract.validate_table(df)
    assert "hippocampal_volume" not in df.columns
