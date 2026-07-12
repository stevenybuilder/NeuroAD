"""The Flag envelope and its open, discriminated payload union."""

from sfg.flags import (
    Flag,
    HeatmapPayload,
    Marker,
    MaskPayload,
    NonePayload,
    PointsPayload,
    Severity,
)


def test_severity_ranks_order_correctly():
    assert Severity.critical.rank > Severity.error.rank > Severity.warn.rank > Severity.info.rank


def test_default_payload_is_none():
    f = Flag(check_id="c", scan_id="s", severity="info", explanation="x")
    assert isinstance(f.payload, NonePayload)


def test_payload_union_discriminates_by_kind():
    # A dict payload is parsed into the right variant by its "kind" tag.
    f = Flag(check_id="c", scan_id="s", severity="warn", explanation="x",
             payload={"kind": "mask", "resource": "m.nii.gz"})
    assert isinstance(f.payload, MaskPayload)
    assert f.payload.resource == "m.nii.gz"


def test_points_payload_roundtrips():
    p = PointsPayload(markers=[Marker(coord_mm=[1, 2, 3], text="R")])
    f = Flag(check_id="c", scan_id="s", severity="info", explanation="x", payload=p)
    reloaded = Flag(**f.model_dump())
    assert isinstance(reloaded.payload, PointsPayload)
    assert reloaded.payload.markers[0].text == "R"


def test_sort_key_orders_severe_first():
    a = Flag(check_id="a", scan_id="s", severity="info", explanation="x")
    b = Flag(check_id="b", scan_id="s", severity="critical", explanation="x")
    assert sorted([a, b], key=lambda f: f.sort_key())[0] is b


def test_heatmap_payload_carries_range():
    h = HeatmapPayload(resource="h.nii.gz", cal_min=0.1, cal_max=1.0)
    assert h.kind == "heatmap"
