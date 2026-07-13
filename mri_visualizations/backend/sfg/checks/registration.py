"""Check 1.5 - Registration verification (Problem B) [stretch].

Registration silently fails on exactly the brains that need it, and a
misaligned brain quietly corrupts every voxelwise or atlas-based analysis. This
check verifies an affine registration by post-alignment brain overlap (Dice) and
renders the residual as a mismatch heatmap so a human can see *where* alignment
broke down. It demonstrates both outcomes on the reference brain: a well-posed
alignment that registration recovers, and an ill-posed large-rotation case it
cannot - the low-overlap failure the check is meant to catch.

Self-contained: registers against synthetically transformed copies of the cohort
reference brain, so it needs no atlas download and reuses the cached SynthStrip
mask from check 1.2.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.ndimage import rotate

from .. import config
from ..flags import Flag, HeatmapPayload, Location
from ..imaging import load, vox_to_world
from ..registration import affine_register, conform_ras, dice, resample_to_grid
from ..registry import Scan
from ..resources import ResourceStore, make_key
from ..skullstrip import synthstrip_mask
from .base import register

DICE_OK = 0.85


class RegistrationCheck:
    check_id = "1.5.registration"
    description = "Verifies affine registration via post-alignment Dice; renders the residual mismatch heatmap."

    def run_cohort(self, scans: list[Scan], store: ResourceStore) -> list[Flag]:
        ref = next((s for s in scans if s.source == "ixi" and s.site == "Guy's"), None)
        if ref is None:
            ref = next((s for s in scans if s.source == "ixi"), None)
        if ref is None:
            return []

        # Registration is deterministic and expensive; cache the flags so reruns
        # are instant as long as the residual resources still exist.
        cache = config.STRIP_DIR / f"{ref.scan_id}_registration.json"
        cached = self._load_cache(cache, store)
        if cached is not None:
            return cached

        data, affine, _img = load(ref)
        mask = synthstrip_mask(ref.modality_path(), config.STRIP_DIR / f"{ref.scan_id}_synthstrip.nii.gz")
        if mask is None:
            return []  # SynthStrip unavailable; registration demo needs a clean brain
        brain = data * mask
        fixed, caffine = conform_ras(brain, affine)
        fixed = self._norm(fixed)

        grid = (data.shape, affine)
        flags = []
        # Verified: a misaligned brain that registration successfully recovers.
        flags.append(self._case(ref, store, caffine, fixed, grid, angle=18, shift=8,
                                 do_register=True, label="registration verified"))
        # Missing/failed: the same misalignment left unregistered - the brains a
        # broken or skipped registration would silently leave mismatched.
        flags.append(self._case(ref, store, caffine, fixed, grid, angle=18, shift=8,
                                 do_register=False, label="registration missing"))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([f.model_dump() for f in flags]))
        return flags

    def _load_cache(self, cache, store) -> list[Flag] | None:
        if not cache.exists():
            return None
        try:
            dumped = json.loads(cache.read_text())
            flags = [Flag(**d) for d in dumped]
        except Exception:
            return None
        # Only trust the cache if every referenced residual is still on disk.
        if all(store.path(f.payload.resource) for f in flags if hasattr(f.payload, "resource")):
            return flags
        return None

    def _case(self, ref, store, caffine, fixed, grid, angle, shift, do_register, label) -> Flag:
        moving = rotate(fixed, angle, axes=(1, 2), reshape=False, order=1)
        if shift:
            moving = np.roll(moving, shift, axis=0)
        registered = self._norm(affine_register(fixed, moving) if do_register else moving)

        d = dice(fixed > 0.1, registered > 0.1)
        residual = np.abs(fixed - registered) * (fixed > 0.05)
        if residual.max() > 0:
            residual = residual / residual.max()

        # Resample residual onto the reference scan's own grid so it overlays
        # cleanly (shared grid) rather than tinting the whole FOV.
        orig_shape, orig_affine = grid
        residual_grid = resample_to_grid(residual, caffine, orig_shape, orig_affine)
        # Background -> 0 (NOT NaN): this NiiVue build renders NaN voxels as an
        # opaque wash, so zero the low-residual background instead. The viewer pairs
        # this with colormapType ZERO_TO_MAX_TRANSPARENT_BELOW_MIN + cal_min 0.15 so
        # everything below threshold is fully transparent and only the mismatch
        # hot-spots tint.
        residual_grid = np.where(residual_grid < 0.02, 0.0, residual_grid).astype(np.float32)
        key = store.put_volume(make_key(ref.scan_id, "reg-residual", label), residual_grid, orig_affine, np.float32)
        peak = np.unravel_index(int(np.argmax(residual_grid)), residual_grid.shape)
        world = vox_to_world(orig_affine, peak)

        ok = d >= DICE_OK
        return Flag(
            check_id=self.check_id, scan_id=ref.scan_id,
            severity="info" if ok else "error",
            explanation=(
                f"Registration {label}: post-alignment Dice {d:.2f} "
                + ("- verified, residual mismatch is low." if ok else
                   "- FAILED to align (Dice below "
                   f"{DICE_OK}); the heatmap shows large residual mismatch. A pipeline that trusted "
                   "this registration would compare mismatched anatomy.")
            ),
            location=Location(world_mm=world),
            payload=HeatmapPayload(resource=key, colormap="warm", opacity=0.7, cal_min=0.15, cal_max=1.0),
            extra={"post_registration_dice": round(d, 3), "case": label},
        )

    def _norm(self, arr: np.ndarray) -> np.ndarray:
        p99 = np.percentile(arr[arr > 0], 99) if (arr > 0).any() else 1.0
        return np.clip(arr / max(p99, 1e-6), 0, 1)


register(RegistrationCheck())
