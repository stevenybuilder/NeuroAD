"""Induced-failure fixtures: real scans deliberately broken in one way each.

A clean cohort should not trip the deterministic checks, so to *show* a check
firing we synthesize a variant with exactly one planted defect. Each fixture is
written to data/fixtures/ with a .json sidecar naming its source and defect, and
is discovered by the registry like any other scan (source="fixture").

Defects:
- corrupt:  a NaN block (coil dropout) + one zeroed interior slice (1.4).
- lrflip:   sform and qform made to encode opposite L/R, with the data mirrored
            to match the sform - the classic silently-flipped scan (1.1).
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from . import config
from .imaging import bbox, foreground_mask


def _sidecar(out: Path, meta: dict) -> None:
    stem = out.name[: -len(".nii.gz")]
    (out.parent / f"{stem}.json").write_text(json.dumps(meta, indent=2))


def make_corrupt(src: Path, out: Path, site: str) -> None:
    img = nib.load(str(src))
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    fg = foreground_mask(data)
    bb = bbox(fg)
    lo, hi = bb
    c = ((lo + hi) / 2).astype(int)

    # NaN block: a small cube just off-centre, as a coil/reconstruction dropout.
    r = 12
    data[c[0]:c[0] + r, c[1]:c[1] + r, c[2] - r:c[2]] = np.nan
    # Dropped slice: zero one interior axial (k) slice within the brain.
    kdrop = int((lo[2] + hi[2]) / 2) + 5
    data[:, :, kdrop] = 0.0

    # Preserve geometry but store float32 so the injected NaNs survive (the
    # source int16 header would cast them away).
    hdr = img.header.copy()
    hdr.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(data, img.affine, hdr), str(out))
    _sidecar(out, {"source": src.stem, "defect": "corrupt", "site": site,
                   "note": f"NaN block near {c.tolist()}; zeroed axial slice k={kdrop}"})


def make_lrflip(src: Path, out: Path, site: str) -> None:
    img = nib.load(str(src))
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    affine = np.asarray(img.affine, dtype=float)

    # sform reflects the L/R-flipped world (negate the world-X row); qform keeps
    # the original. The two now disagree about left vs right.
    sform = affine.copy()
    sform[0, :] *= -1.0
    qform = affine

    flipped = np.ascontiguousarray(data[::-1, :, :])  # mirror data to match sform
    out_img = nib.Nifti1Image(flipped, sform)
    out_img.set_sform(sform, code=1)
    out_img.set_qform(qform, code=1)
    nib.save(out_img, str(out))
    _sidecar(out, {"source": src.stem, "defect": "lrflip", "site": site,
                   "note": "sform and qform encode opposite L/R; data mirrored to sform"})


def ensure_fixtures() -> None:
    """Generate the fixture set once (idempotent) from a Guy's IXI T1 source."""
    config.FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    src = next(iter(sorted(config.IXI_DIR.glob("IXI*-Guys-*-T1.nii.gz"))), None)
    if src is None:
        return  # no IXI yet; nothing to base fixtures on
    stem = src.name[: -len(".nii.gz")]
    jobs = [
        (f"{stem}-CORRUPT.nii.gz", make_corrupt),
        (f"{stem}-LRFLIP.nii.gz", make_lrflip),
    ]
    for name, fn in jobs:
        out = config.FIXTURES_DIR / name
        if not out.exists():
            fn(src, out, site="IXI/Guy's")
