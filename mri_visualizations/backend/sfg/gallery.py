"""Annotation-framework self-test: one real Flag of every payload kind.

This is not a check - it is the producer-agnostic renderer's smoke test and the
Phase-0 done-criterion ("render each payload type at least once", using the real
BraTS seg as the 3D mask). It builds mask / mesh / heatmap / point / bbox /
plaintext flags from an actual tumour segmentation so every branch of the viewer
renderer is exercised on real geometry.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

from .flags import (
    BBoxPayload,
    Flag,
    HeatmapPayload,
    Location,
    MaskPayload,
    MeshPayload,
    NonePayload,
    PointPayload,
)
from .registry import Registry
from .resources import ResourceStore, make_key


def _corners_world(affine: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[list, list]:
    """World-space min/max of the box spanned by voxel corners lo..hi."""
    combos = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    world = (affine @ np.column_stack([combos, np.ones(len(combos))]).T).T[:, :3]
    return world.min(0).tolist(), world.max(0).tolist()


def build_gallery_flags(registry: Registry, store: ResourceStore) -> list[Flag]:
    scan = next((s for s in registry.by_source("brats") if s.seg), None)
    if scan is None:
        return []

    seg_img = nib.load(str(scan.seg))
    affine = np.asarray(seg_img.affine, dtype=float)
    seg = np.asarray(seg_img.dataobj)
    mask = seg > 0
    if not mask.any():
        return []

    # Geometry from the real tumour.
    idx = np.argwhere(mask)
    centroid_vox = idx.mean(0)
    centroid_mm = (affine @ np.append(centroid_vox, 1.0))[:3].tolist()
    lo, hi = idx.min(0), idx.max(0)
    box_min, box_max = _corners_world(affine, lo, hi)
    loc = Location(world_mm=centroid_mm, voxel=[int(v) for v in centroid_vox.round()])

    # 1) Binarized whole-tumour mask overlay.
    mask_key = store.put_volume(
        make_key(scan.scan_id, "gallery", "tumor-mask"), mask.astype(np.uint8), affine, dtype=np.uint8
    )
    # 2) Marching-cubes surface of the same mask.
    mesh_key = store.put_mesh_from_mask(
        make_key(scan.scan_id, "gallery", "tumor-mesh"), mask, affine, step_size=2
    )
    # 3) Synthesized smooth "anomaly score" field centred on the tumour.
    heat = gaussian_filter(mask.astype(np.float32), sigma=8.0)
    if heat.max() > 0:
        heat /= heat.max()
    heat_key = store.put_volume(
        make_key(scan.scan_id, "gallery", "anomaly-heat"), heat, affine, dtype=np.float32
    )

    return [
        Flag(
            check_id="gallery.mask",
            scan_id=scan.scan_id,
            severity="warn",
            explanation="Volumetric mask overlay: whole-tumour segmentation drawn on the scan.",
            location=loc,
            payload=MaskPayload(resource=mask_key, colormap="red", opacity=0.5, label="tumor"),
        ),
        Flag(
            check_id="gallery.mesh",
            scan_id=scan.scan_id,
            severity="warn",
            explanation="Surface mesh: marching-cubes of the tumour mask in the 3D render.",
            location=loc,
            payload=MeshPayload(resource=mesh_key, rgba=[1.0, 0.35, 0.2, 1.0]),
        ),
        Flag(
            check_id="gallery.heatmap",
            scan_id=scan.scan_id,
            severity="info",
            explanation="Continuous heatmap: a synthesized anomaly-score field peaking at the lesion.",
            location=loc,
            payload=HeatmapPayload(resource=heat_key, colormap="warm", opacity=0.6, cal_min=0.05, cal_max=1.0),
        ),
        Flag(
            check_id="gallery.point",
            scan_id=scan.scan_id,
            severity="info",
            explanation="Labelled point marker at the tumour centroid (world/mm).",
            location=loc,
            payload=PointPayload(coord_mm=centroid_mm, text="tumor centroid", rgba=[1.0, 1.0, 0.0, 1.0]),
        ),
        Flag(
            check_id="gallery.bbox",
            scan_id=scan.scan_id,
            severity="info",
            explanation="Axis-aligned bounding box around the lesion extent (world/mm).",
            location=loc,
            payload=BBoxPayload(min_mm=box_min, max_mm=box_max, text="tumor bbox"),
        ),
        Flag(
            check_id="gallery.plaintext",
            scan_id=scan.scan_id,
            severity="info",
            explanation=(
                "Plaintext-only flag: no geometry, still first-class. "
                f"Tumour spans {int(mask.sum())} voxels; centroid at "
                f"({centroid_mm[0]:.1f}, {centroid_mm[1]:.1f}, {centroid_mm[2]:.1f}) mm."
            ),
            location=loc,
            payload=NonePayload(),
        ),
    ]
