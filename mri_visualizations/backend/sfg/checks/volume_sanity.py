"""Check 1.4 - Volume sanity linter.

Nearly free, catches dumb-but-real corruption that silently poisons a cohort:
non-finite voxels, interior dead/dropout slices, implausible spacing or
dimensions, and constant volumes. Tuned to stay quiet on clean scans (all-zero
*background* slices at the volume border are normal and are not flagged - only
dead slices strictly inside the brain's extent are).
"""

from __future__ import annotations

import numpy as np

from ..flags import Flag, Location, NonePayload, PointPayload
from ..imaging import bbox, foreground_mask, load, vox_to_world
from ..registry import Scan
from ..resources import ResourceStore
from .base import register

_AXES = ["i", "j", "k"]


class VolumeSanityCheck:
    check_id = "1.4.volume_sanity"
    description = "Flags non-finite voxels, interior dead slices, and implausible spacing/dimensions."

    def run(self, scan: Scan, store: ResourceStore) -> list[Flag]:
        data, affine, img = load(scan)
        flags: list[Flag] = []
        zooms = [float(z) for z in img.header.get_zooms()[:3]]

        # --- dimensions ------------------------------------------------------
        if data.ndim != 3 or min(data.shape) < 8:
            flags.append(self._flag(
                scan, "error", f"Implausible dimensions {tuple(data.shape)}.",
                extra={"shape": list(data.shape)}))
            return flags  # further checks assume a sane 3D grid

        # --- spacing ---------------------------------------------------------
        if any(not (0.3 <= z <= 5.0) for z in zooms) or (max(zooms) / max(min(zooms), 1e-6)) > 8:
            flags.append(self._flag(
                scan, "warn", f"Implausible or highly anisotropic voxel spacing {zooms} mm.",
                extra={"zooms_mm": zooms}))

        # --- non-finite ------------------------------------------------------
        nonfinite = ~np.isfinite(data)
        n_nf = int(nonfinite.sum())
        if n_nf:
            ijk = np.argwhere(nonfinite)[0]
            world = vox_to_world(affine, ijk)
            flags.append(self._flag(
                scan, "error",
                f"{n_nf} non-finite (NaN/Inf) voxels - these corrupt any downstream statistic.",
                world=world, ijk=ijk,
                payload=PointPayload(coord_mm=world, text="NaN/Inf", rgba=[1, 0.2, 0.2, 1]),
                extra={"n_nonfinite": n_nf, "fraction": round(n_nf / data.size, 6)}))

        finite = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # --- constant volume -------------------------------------------------
        if float(finite.std()) < 1e-6:
            flags.append(self._flag(scan, "error", "Volume is constant (no signal variation)."))
            return flags

        # --- interior dead slices -------------------------------------------
        fg = foreground_mask(finite)
        bb = bbox(fg)
        if bb is not None:
            gmax = float(finite.max())
            for axis in range(3):
                lo, hi = int(bb[0][axis]), int(bb[1][axis])
                # Vectorized per-slice reductions over the other two axes.
                other = tuple(a for a in range(3) if a != axis)
                smax = finite.max(axis=other)
                smin = finite.min(axis=other)
                is_dead = (smax <= 1e-6 * gmax) | (smax - smin == 0)
                interior = np.arange(lo + 1, hi)
                dead = [int(k) for k in interior if is_dead[k]]
                if dead:
                    mid = dead[len(dead) // 2]
                    center = list(bb[0].astype(float) + (bb[1] - bb[0]) / 2)
                    center[axis] = float(mid)
                    world = vox_to_world(affine, center)
                    flags.append(self._flag(
                        scan, "warn",
                        f"{len(dead)} dead/dropout slice(s) inside the brain along {_AXES[axis]} "
                        f"(e.g. index {mid}) - likely acquisition or export corruption.",
                        world=world, ijk=[int(c) for c in center],
                        payload=PointPayload(coord_mm=world, text=f"dead {_AXES[axis]}={mid}",
                                             rgba=[1, 0.6, 0.1, 1]),
                        extra={"axis": _AXES[axis], "dead_indices": dead[:50]}))
        return flags

    def _flag(self, scan, severity, explanation, world=None, ijk=None, payload=None, extra=None):
        return Flag(
            check_id=self.check_id, scan_id=scan.scan_id, severity=severity,
            explanation=explanation,
            location=Location(world_mm=world, voxel=[int(v) for v in ijk]) if world else None,
            payload=payload or NonePayload(),
            extra=extra or {},
        )


register(VolumeSanityCheck())
