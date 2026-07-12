"""Where derived overlays live.

A check produces geometry (a mask, a heatmap field, a surface) but a Flag only
carries a *key*. The ResourceStore writes the geometry to a cache dir and the
server streams it back on demand at ``/api/resource/{key}``. Base scans are NOT
stored here - they are served straight from the registry.
"""

from __future__ import annotations

import re
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage import measure

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def make_key(*parts: str) -> str:
    """Filesystem-safe, human-legible resource key from arbitrary parts."""
    return _SAFE.sub("-", "__".join(str(p) for p in parts)).strip("-")


class ResourceStore:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path | None:
        """Resolve a resource *filename* (extension included) to a file."""
        p = self.cache_dir / name
        return p if p.exists() and p.parent == self.cache_dir else None

    def put_volume(
        self, key: str, data: np.ndarray, affine: np.ndarray, dtype=np.float32
    ) -> str:
        """Persist a derived volume (mask or heatmap) as NIfTI, return its filename.

        The returned value carries the extension so it doubles as a URL segment
        the viewer can hand to NiiVue (which infers format from the extension).
        """
        name = f"{key}.nii.gz"
        img = nib.Nifti1Image(np.asarray(data, dtype=dtype), affine)
        nib.save(img, self.cache_dir / name)
        return name

    def put_mesh_from_mask(
        self,
        key: str,
        mask: np.ndarray,
        affine: np.ndarray,
        level: float = 0.5,
        step_size: int = 1,
    ) -> str:
        """Marching-cubes a binary mask into a world/mm OBJ surface, return filename.

        Vertices are pushed through the volume affine so the mesh lands in the
        same scanner/world space NiiVue places the volume in - no manual
        alignment needed in the viewer.
        """
        name = f"{key}.obj"
        mask = np.asarray(mask) > 0
        if not mask.any():
            # Degenerate: write an empty mesh so callers still get a valid file.
            self._write_obj(name, np.zeros((0, 3)), np.zeros((0, 3), int))
            return name
        verts, faces, _normals, _values = measure.marching_cubes(
            mask.astype(np.float32), level=level, step_size=step_size
        )
        # verts are (i, j, k) voxel coords -> homogeneous -> world mm.
        vox_h = np.column_stack([verts, np.ones(len(verts))])
        world = (affine @ vox_h.T).T[:, :3]
        self._write_obj(name, world, faces)
        return name

    def _write_obj(self, name: str, verts: np.ndarray, faces: np.ndarray) -> None:
        lines = [f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in verts]
        # OBJ face indices are 1-based.
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
        (self.cache_dir / name).write_text("\n".join(lines) + "\n")
