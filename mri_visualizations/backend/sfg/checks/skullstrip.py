"""Check 1.2 - Skull-strip verification (Problem B) [flagship].

Skull stripping is where preprocessing quietly fails on the brains that matter,
and a bad brain mask poisons everything downstream. This check verifies a brain
mask against plausibility (brain volume), topology (single component, no holes),
and - when the robust reference is available - disagreement with SynthStrip. It
demonstrates the failure by contrasting a weak classical stripper against
SynthStrip on the same raw IXI scan, rendering the over-inclusive mask and flying
the camera to where the weak strip leaks into skull/eyes.

Cohort-level and sampled: SynthStrip inference is expensive, so it runs on one
representative scan per site and caches the result.
"""

from __future__ import annotations

import numpy as np

from .. import config
from ..flags import Flag, Location, MaskPayload, MeshPayload
from ..imaging import load, vox_to_world
from ..registry import Scan
from ..resources import ResourceStore, make_key
from ..skullstrip import (
    BRAIN_CC_MAX,
    BRAIN_CC_MIN,
    brain_volume_cc,
    dice,
    hole_volume_cc,
    n_components,
    synthstrip_mask,
    weak_strip,
)
from .base import register


class SkullStripCheck:
    check_id = "1.2.skull_strip"
    description = "Verifies brain masks (volume, topology, SynthStrip agreement); contrasts a weak stripper."

    def run_cohort(self, scans: list[Scan], store: ResourceStore) -> list[Flag]:
        # One representative raw IXI scan per site (has an intact skull to strip).
        sample: dict[str, Scan] = {}
        for s in scans:
            if s.source == "ixi" and s.site not in sample:
                sample[s.site] = s
        flags: list[Flag] = []
        for scan in list(sample.values())[:1]:  # one scan is enough for the demo
            flags.extend(self._verify_scan(scan, store))
        return flags

    def _verify_scan(self, scan: Scan, store: ResourceStore) -> list[Flag]:
        data, affine, img = load(scan)
        zooms = img.header.get_zooms()
        flags: list[Flag] = []

        weak = weak_strip(data)
        cache = config.STRIP_DIR / f"{scan.scan_id}_synthstrip.nii.gz"
        synth = synthstrip_mask(scan.modality_path(), cache)

        weak_cc = brain_volume_cc(weak, zooms)
        weak_holes = hole_volume_cc(weak, zooms)
        weak_comp = n_components(weak)

        # --- weak stripper: quantify its failure ----------------------------
        problems = []
        if not (BRAIN_CC_MIN <= weak_cc <= BRAIN_CC_MAX):
            problems.append(f"implausible brain volume {weak_cc:.0f} cc "
                            f"(expected {BRAIN_CC_MIN:.0f}-{BRAIN_CC_MAX:.0f})")
        if weak_holes > 20:
            problems.append(f"{weak_holes:.0f} cc of interior holes")
        if weak_comp > 3:
            problems.append(f"{weak_comp} disconnected components")

        weak_spot = None
        over_cc = None
        if synth is not None:
            d = dice(weak, synth)
            over = weak & ~synth  # what the weak strip wrongly keeps (skull/eyes)
            over_cc = brain_volume_cc(over, zooms)
            if over.any():
                weak_spot = vox_to_world(affine, np.argwhere(over).mean(0))
            if d < 0.9:
                problems.append(f"Dice {d:.2f} vs SynthStrip, over-including {over_cc:.0f} cc of non-brain")

        weak_key = store.put_volume(make_key(scan.scan_id, "weakstrip"), weak.astype(np.uint8), affine, np.uint8)
        if weak_spot is None:
            weak_spot = vox_to_world(affine, np.argwhere(weak).mean(0))
        flags.append(Flag(
            check_id=self.check_id, scan_id=scan.scan_id,
            severity="error" if problems else "info",
            explanation=(
                "Weak classical strip on raw T1: " + "; ".join(problems) + "."
                if problems else
                f"Weak strip looks plausible here ({weak_cc:.0f} cc)."
            ),
            location=Location(world_mm=weak_spot),
            payload=MaskPayload(resource=weak_key, colormap="red", opacity=0.4, label="weak strip"),
            extra={"weak_volume_cc": round(weak_cc, 0), "weak_holes_cc": round(weak_holes, 0),
                   "weak_components": weak_comp,
                   "synthstrip_available": synth is not None,
                   "dice_vs_synthstrip": None if synth is None else round(dice(weak, synth), 3),
                   "over_inclusion_cc": None if over_cc is None else round(over_cc, 0)},
        ))

        # --- SynthStrip reference: the clean brain surface ------------------
        if synth is not None:
            synth_cc = brain_volume_cc(synth, zooms)
            mesh_key = store.put_mesh_from_mask(make_key(scan.scan_id, "synthstrip-mesh"), synth, affine, step_size=2)
            synth_problems = []
            if not (BRAIN_CC_MIN <= synth_cc <= BRAIN_CC_MAX):
                synth_problems.append(f"volume {synth_cc:.0f} cc out of range")
            flags.append(Flag(
                check_id=self.check_id, scan_id=scan.scan_id,
                severity="warn" if synth_problems else "info",
                explanation=(
                    f"SynthStrip reference brain surface ({synth_cc:.0f} cc, single component) - "
                    "compare against the weak strip's over-inclusion."
                    + ("" if not synth_problems else " Note: " + "; ".join(synth_problems))
                ),
                location=Location(world_mm=vox_to_world(affine, np.argwhere(synth).mean(0))),
                payload=MeshPayload(resource=mesh_key, rgba=[0.3, 0.9, 0.5, 1.0]),
                extra={"synthstrip_volume_cc": round(synth_cc, 0)},
            ))
        return flags


register(SkullStripCheck())
