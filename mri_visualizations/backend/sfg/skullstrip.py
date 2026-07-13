"""Skull-strippers and mask verification, used by check 1.2.

Two strippers on purpose:
- ``weak_strip``: a naive intensity-threshold + largest-component stripper. On a
  T1 with an intact skull it characteristically over-includes (bright skull, fat
  and eyes stay connected to brain) - the classic weak classical result.
- ``synthstrip_mask``: SynthStrip (nipreps), contrast-agnostic and robust, used
  as the reference.

The verification metrics (plausible brain volume, connectivity, holes, and
disagreement with the reference) are stripper-agnostic: give them any mask and
they tell you where it is untrustworthy.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu

from . import config

# Plausible adult intracranial/brain-mask volume window (cc). Outside -> suspect.
BRAIN_CC_MIN, BRAIN_CC_MAX = 900.0, 1950.0


def weak_strip(data: np.ndarray) -> np.ndarray:
    """Naive threshold + largest connected component + hole fill."""
    finite = np.nan_to_num(data, nan=0.0)
    thr = threshold_otsu(finite[finite > 0]) if (finite > 0).any() else 0.0
    fg = finite > thr
    lbl, n = ndimage.label(fg)
    if n == 0:
        return fg
    largest = 1 + int(np.argmax(ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))))
    mask = lbl == largest
    return ndimage.binary_fill_holes(mask)


def synthstrip_mask(scan_path: Path, cache_path: Path) -> np.ndarray | None:
    """Run SynthStrip (cached). Returns a boolean brain mask, or None if the
    tool is unavailable."""
    if cache_path.exists():
        return np.asanyarray(nib.load(str(cache_path)).dataobj) > 0
    if not config.SYNTHSTRIP_MODEL.exists():
        return None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Resolve the CLI next to the running interpreter (the server is launched via
    # <env>/bin/python, so its console script sits in the same bin dir) — do not
    # rely on the server process inheriting the env's bin on PATH.
    bin_path = Path(sys.executable).with_name("nipreps-synthstrip")
    synthstrip = str(bin_path) if bin_path.exists() else "nipreps-synthstrip"
    with tempfile.TemporaryDirectory() as td:
        out_mask = Path(td) / "mask.nii.gz"
        try:
            subprocess.run(
                [synthstrip, "-i", str(scan_path), "-m", str(out_mask),
                 "--model", str(config.SYNTHSTRIP_MODEL)],
                check=True, capture_output=True, timeout=600,
            )
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
        img = nib.load(str(out_mask))
        nib.save(img, str(cache_path))
        return np.asanyarray(img.dataobj) > 0


# --- verification -------------------------------------------------------------


def brain_volume_cc(mask: np.ndarray, zooms_mm) -> float:
    voxel_cc = float(np.prod(zooms_mm[:3])) / 1000.0
    return float(mask.sum()) * voxel_cc


def n_components(mask: np.ndarray) -> int:
    _lbl, n = ndimage.label(mask)
    return int(n)


def hole_volume_cc(mask: np.ndarray, zooms_mm) -> float:
    filled = ndimage.binary_fill_holes(mask)
    voxel_cc = float(np.prod(zooms_mm[:3])) / 1000.0
    return float((filled & ~mask).sum()) * voxel_cc


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    denom = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / denom) if denom else 1.0
