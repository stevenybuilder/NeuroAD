"""Small imaging helpers shared by check modules.

Kept deliberately thin - just the operations that recur across checks (loading,
voxel<->world, a robust brain bounding box). Anything used by only one check
stays in that check until a third caller appears (rule of three).
"""

from __future__ import annotations

import nibabel as nib
import numpy as np

from .registry import Scan


def load(scan: Scan, modality: str | None = None):
    """Return (data float32, affine, nibabel image) for a scan modality."""
    img = nib.load(str(scan.modality_path(modality)))
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    return data, np.asarray(img.affine, dtype=float), img


def vox_to_world(affine: np.ndarray, ijk) -> list[float]:
    return (affine @ np.append(np.asarray(ijk, dtype=float), 1.0))[:3].tolist()


def foreground_mask(data: np.ndarray, frac: float = 0.02) -> np.ndarray:
    """Coarse foreground: voxels above a small fraction of the robust max.

    Uses the 99th percentile as the "max" so a few hot voxels do not shrink the
    threshold. Good enough to bound the brain and to separate signal from air.
    """
    finite = np.isfinite(data)
    hi = np.percentile(data[finite], 99) if finite.any() else 0.0
    return finite & (data > frac * hi)


def bbox(mask: np.ndarray):
    """Inclusive (min, max) voxel indices of a boolean mask, or None if empty."""
    if not mask.any():
        return None
    idx = np.argwhere(mask)
    return idx.min(0), idx.max(0)
