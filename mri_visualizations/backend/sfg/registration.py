"""Affine registration helpers for check 1.5.

Atlas-free by design: we register a subject's (skull-stripped) brain to a cohort
reference brain rather than downloading a template, which keeps the check
self-contained. Everything happens in a shared RAS 1.6 mm conformed grid so the
registration runs in voxel space without RAS/LPS bookkeeping, and the resulting
residual overlays back onto the reference in world coordinates.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from nibabel.processing import conform, resample_from_to

CONFORM_SHAPE = (128, 128, 128)
CONFORM_MM = (1.6, 1.6, 1.6)


def conform_ras(data: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resample to a fixed RAS iso grid; returns (array, affine)."""
    img = conform(nib.Nifti1Image(data.astype(np.float32), affine),
                  out_shape=CONFORM_SHAPE, voxel_size=CONFORM_MM, order=1)
    return np.asanyarray(img.dataobj, dtype=np.float32), np.asarray(img.affine, float)


def _sitk(arr: np.ndarray) -> sitk.Image:
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr, dtype=np.float32))
    img.SetSpacing(CONFORM_MM)
    return img


def affine_register(fixed: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """Affinely register moving onto fixed (both in the conformed grid). Returns
    the resampled moving array."""
    f, m = _sitk(fixed), _sitk(moving)
    init = sitk.CenteredTransformInitializer(
        f, m, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, seed=1234)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=200,
        gradientMagnitudeTolerance=1e-6)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(init, inPlace=False)
    transform = reg.Execute(f, m)
    out = sitk.Resample(m, f, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    return sitk.GetArrayFromImage(out)


def resample_to_grid(arr: np.ndarray, from_affine: np.ndarray, shape, to_affine: np.ndarray) -> np.ndarray:
    """Resample a conformed-space array onto a reference scan's grid so it
    overlays cleanly (shared grid) in the viewer instead of tinting the whole FOV."""
    src = nib.Nifti1Image(arr.astype(np.float32), from_affine)
    out = resample_from_to(src, (tuple(int(s) for s in shape), to_affine), order=1)
    return np.asanyarray(out.dataobj, dtype=np.float32)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    denom = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / denom) if denom else 1.0
