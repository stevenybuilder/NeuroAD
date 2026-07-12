"""Check 1.1 - Orientation / affine consistency + LR-flip (Problem E).

Validates the two spatial transforms a NIfTI can carry (sform and qform). The
classic silent failure is a scan whose sform and qform encode *opposite* left/right
- tools pick one or the other, so half a pipeline sees the brain mirrored and
nobody notices. We flag that disagreement deterministically, report the stored
orientation vs canonical RAS, and drop L/R laterality markers into the viewer so a
human can confirm sidedness in seconds.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np

from ..flags import Flag, Location, Marker, PointsPayload
from ..registry import Scan
from ..resources import ResourceStore
from .base import register

RAS = ("R", "A", "S")


class OrientationCheck:
    check_id = "1.1.orientation"
    description = "Validates sform/qform agreement and L/R orientation; flags silently mirrored scans."

    def run(self, scan: Scan, store: ResourceStore) -> list[Flag]:
        img = nib.load(str(scan.modality_path()))
        sform, s_code = img.get_sform(coded=True)
        qform, q_code = img.get_qform(coded=True)
        primary = np.asarray(img.affine, dtype=float)

        stored = "".join(nib.aff2axcodes(primary))
        det = float(np.linalg.det(primary[:3, :3]))
        handed = "left-handed (radiological)" if det < 0 else "right-handed (neurological)"

        readout = {
            "stored_orientation": stored,
            "sform_code": int(s_code), "qform_code": int(q_code),
            "sform_axcodes": "".join(nib.aff2axcodes(sform)) if s_code else None,
            "qform_axcodes": "".join(nib.aff2axcodes(qform)) if q_code else None,
            "determinant": round(det, 4),
            "handedness": handed,
            "canonical_ras": stored == "RAS",
            "affine": primary.round(3).tolist(),
        }
        markers = self._laterality_markers(primary, img.shape)
        payload = PointsPayload(markers=markers)
        loc = Location(world_mm=markers[0].coord_mm)

        # --- decide severity + message --------------------------------------
        if s_code == 0 and q_code == 0:
            return [self._flag(scan, "error",
                "No valid spatial transform (sform_code=qform_code=0); orientation is undefined.",
                payload, loc, readout)]

        s_ax = nib.aff2axcodes(sform) if s_code else None
        q_ax = nib.aff2axcodes(qform) if q_code else None
        if s_code and q_code and s_ax != q_ax:
            return [self._flag(scan, "error",
                f"sform ({''.join(s_ax)}) and qform ({''.join(q_ax)}) encode different orientations - "
                "L/R may be silently mirrored between tools. Confirm sidedness against the L/R markers.",
                payload, loc, readout)]

        if s_code == 0 or q_code == 0:
            return [self._flag(scan, "warn",
                f"Only {'sform' if s_code else 'qform'} is set; the other is absent, so orientation "
                "cannot be cross-validated.", payload, loc, readout)]

        # Consistent scan: no flag (a guard should stay quiet on clean data). The
        # affine readout and canonicalization target are still available on demand.
        return []

    def _laterality_markers(self, affine: np.ndarray, shape) -> list[Marker]:
        center_vox = (np.asarray(shape[:3], dtype=float) - 1) / 2
        center_mm = (affine @ np.append(center_vox, 1.0))[:3]
        # World +X is anatomical Right in RAS-mm world space (NiiVue's frame).
        r = (center_mm + np.array([55.0, 0, 0])).tolist()
        left = (center_mm - np.array([55.0, 0, 0])).tolist()
        return [
            Marker(coord_mm=r, text="R (header +X)", rgba=[0.3, 0.7, 1.0, 1.0]),
            Marker(coord_mm=left, text="L (header -X)", rgba=[1.0, 0.55, 0.2, 1.0]),
        ]

    def _flag(self, scan, severity, explanation, payload, loc, readout):
        return Flag(check_id=self.check_id, scan_id=scan.scan_id, severity=severity,
                    explanation=explanation, location=loc, payload=payload,
                    extra={"affine_readout": readout})


register(OrientationCheck())
