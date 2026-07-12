"""Each check stays quiet on clean data and fires on the planted defect."""

import pytest

from sfg.checks.intensity import IntensityConsistencyCheck
from sfg.checks.orientation import OrientationCheck
from sfg.checks.volume_sanity import VolumeSanityCheck
from sfg.gallery import build_gallery_flags


def test_volume_sanity_quiet_on_clean_scan(brats, store):
    assert VolumeSanityCheck().run(brats, store) == []


def test_volume_sanity_catches_corrupt_fixture(fixture_scans, store):
    if "corrupt" not in fixture_scans:
        pytest.skip("no corrupt fixture")
    flags = VolumeSanityCheck().run(fixture_scans["corrupt"], store)
    sevs = {f.severity.value for f in flags}
    assert "error" in sevs  # NaN/Inf voxels
    assert any("non-finite" in f.explanation.lower() for f in flags)


def test_orientation_quiet_on_consistent_scan(brats, store):
    assert OrientationCheck().run(brats, store) == []


def test_orientation_catches_lrflip_fixture(fixture_scans, store):
    if "lrflip" not in fixture_scans:
        pytest.skip("no lrflip fixture")
    flags = OrientationCheck().run(fixture_scans["lrflip"], store)
    assert flags and flags[0].severity.value == "error"
    assert "different orientations" in flags[0].explanation


def test_gallery_renders_every_payload_kind(registry, store):
    flags = build_gallery_flags(registry, store)
    if not flags:
        pytest.skip("no BraTS seg staged")
    kinds = {f.payload.kind for f in flags}
    assert {"mask", "mesh", "heatmap", "point", "bbox", "none"} <= kinds


def test_intensity_flags_cross_scanner_divergence(registry, store):
    ixi = registry.by_source("ixi")
    if len({s.site for s in ixi}) < 2:
        pytest.skip("need multiple IXI scanners")
    flags = IntensityConsistencyCheck().run_cohort(registry.scans(), store)
    assert len(flags) == 1
    assert flags[0].extra["median_cov_raw_pct"] > 0
    assert len(flags[0].extra["histograms"]) == 2
