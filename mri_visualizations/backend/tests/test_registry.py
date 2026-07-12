"""Registry discovery and the resource store."""

import numpy as np

from sfg.resources import make_key


def test_registry_finds_brats_with_seg(brats):
    assert brats.source == "brats"
    assert brats.seg is not None
    meta = brats.header_meta()
    assert len(meta["shape"]) == 3
    assert meta["orientation"]  # e.g. "LPS"


def test_make_key_is_filesystem_safe():
    key = make_key("IXI002-Guys/0828", "weak strip")
    assert "/" not in key and " " not in key


def test_store_put_volume_and_resolve(store, brats):
    data = np.zeros((6, 6, 6), dtype=np.uint8)
    data[2:4, 2:4, 2:4] = 1
    affine = np.eye(4)
    name = store.put_volume(make_key("t", "mask"), data, affine, np.uint8)
    assert name.endswith(".nii.gz")
    assert store.path(name) is not None


def test_store_mesh_from_mask(store):
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[3:7, 3:7, 3:7] = True
    name = store.put_mesh_from_mask(make_key("t", "mesh"), mask, np.eye(4), step_size=1)
    assert name.endswith(".obj")
    text = store.path(name).read_text()
    assert text.startswith("v ")  # OBJ vertices


def test_store_rejects_path_traversal(store):
    assert store.path("../../etc/passwd") is None
