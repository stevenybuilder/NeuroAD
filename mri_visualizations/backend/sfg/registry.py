"""Discovers the scans on disk and exposes them by ``scan_id``.

Two sources this run:
- BraTS-GLI cases (viewer/annotation test material; 4 modalities + a real seg).
- IXI cases (the Phase-1 workhorse; raw T1/T2 with intact skulls, real affines,
  three scanners encoded in the filename).

Header metadata (shape, zooms, affine, orientation) is read lazily via nibabel
so listing the cohort never loads voxel data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

from . import config

# BraTS modality suffix -> friendly channel name.
_BRATS_MODALITIES = {"t1n": "t1n", "t1c": "t1c", "t2w": "t2w", "t2f": "t2f"}

# IXI site code (from filename) -> scanner/site label.
_IXI_SITES = {"Guys": "Guy's", "HH": "Hammersmith", "IOP": "IOP"}


@dataclass
class Scan:
    scan_id: str
    source: str  # "brats" | "ixi"
    modalities: dict[str, Path]
    default_modality: str
    site: Optional[str] = None
    seg: Optional[Path] = None
    extra: dict = field(default_factory=dict)

    def modality_path(self, modality: Optional[str] = None) -> Path:
        return self.modalities[modality or self.default_modality]

    def header_meta(self, modality: Optional[str] = None) -> dict:
        img = nib.load(str(self.modality_path(modality)))
        affine = np.asarray(img.affine, dtype=float)
        return {
            "shape": [int(x) for x in img.shape[:3]],
            "zooms_mm": [round(float(z), 4) for z in img.header.get_zooms()[:3]],
            "orientation": "".join(nib.aff2axcodes(affine)),
            "affine": affine.round(4).tolist(),
            "dtype": str(img.get_data_dtype()),
        }

    def summary(self) -> dict:
        return {
            "scan_id": self.scan_id,
            "source": self.source,
            "site": self.site,
            "modalities": sorted(self.modalities),
            "default_modality": self.default_modality,
            "has_seg": self.seg is not None,
        }


class Registry:
    def __init__(self, roots: Optional[dict[str, Path]] = None):
        roots = roots or {}
        self.brats_root = roots.get("brats", config.BRATS_DIR)
        self.ixi_root = roots.get("ixi", config.IXI_DIR)
        self.adni_root = roots.get("adni", config.ADNI_DIR)
        self._scans: dict[str, Scan] = {}
        self.refresh()

    def refresh(self) -> None:
        self._scans = {}
        self._discover_brats()
        self._discover_ixi()
        self._discover_adni()
        self._discover_fixtures()

    def _discover_brats(self) -> None:
        """Inline layout: one dir per case with modality NIfTIs + <case>-seg.nii.gz."""
        if not self.brats_root.is_dir():
            return
        for case_dir in sorted(p for p in self.brats_root.iterdir() if p.is_dir()):
            case = case_dir.name
            modalities: dict[str, Path] = {}
            for suffix, name in _BRATS_MODALITIES.items():
                f = case_dir / f"{case}-{suffix}.nii.gz"
                if f.exists():
                    modalities[name] = f
            if not modalities:
                continue
            seg = case_dir / f"{case}-seg.nii.gz"
            default = "t1c" if "t1c" in modalities else sorted(modalities)[0]
            self._scans[case] = Scan(
                scan_id=case,
                source="brats",
                modalities=modalities,
                default_modality=default,
                site="BraTS-GLI",
                seg=seg if seg.exists() else None,
            )

    def _discover_ixi(self) -> None:
        if not self.ixi_root.is_dir():
            return
        for f in sorted(self.ixi_root.glob("*.nii.gz")):
            # e.g. IXI002-Guys-0828-T1.nii.gz -> id "IXI002-Guys-0828", mod "T1".
            stem = f.name[: -len(".nii.gz")]
            parts = stem.split("-")
            if len(parts) < 4:
                continue
            scan_id, modality = "-".join(parts[:-1]), parts[-1]
            site_code = parts[1]
            scan = self._scans.get(scan_id)
            if scan is None:
                scan = Scan(
                    scan_id=scan_id,
                    source="ixi",
                    modalities={},
                    default_modality=modality,
                    site=_IXI_SITES.get(site_code, site_code),
                )
                self._scans[scan_id] = scan
            scan.modalities[modality] = f
            if modality == "T1":
                scan.default_modality = "T1"

    def _discover_adni(self) -> None:
        if not self.adni_root.is_dir():
            return
        for f in sorted(self.adni_root.glob("*.nii.gz")):
            stem = f.name[: -len(".nii.gz")]           # e.g. DEMO_0001_T1
            subj = stem[: -len("_T1")] if stem.endswith("_T1") else stem
            site_code = subj.split("_")[0]              # ADNI site number, e.g. 003
            self._scans[f"ADNI_{subj}"] = Scan(
                scan_id=f"ADNI_{subj}",
                source="adni",
                modalities={"T1": f},
                default_modality="T1",
                site=f"ADNI-{site_code}",
            )

    def _discover_fixtures(self) -> None:
        if not config.FIXTURES_DIR.is_dir():
            return
        import json

        for f in sorted(config.FIXTURES_DIR.glob("*.nii.gz")):
            stem = f.name[: -len(".nii.gz")]
            sidecar = f.parent / f"{stem}.json"
            meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
            self._scans[stem] = Scan(
                scan_id=stem,
                source="fixture",
                modalities={"T1": f},
                default_modality="T1",
                site=meta.get("site", "fixture"),
                extra=meta,
            )

    def scans(self) -> list[Scan]:
        return list(self._scans.values())

    def get(self, scan_id: str) -> Scan:
        if scan_id not in self._scans:
            raise KeyError(scan_id)
        return self._scans[scan_id]

    def by_source(self, source: str) -> list[Scan]:
        return [s for s in self._scans.values() if s.source == source]
