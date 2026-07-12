#!/usr/bin/env python3
"""
Organize an IDA/LONI ADNI DICOM download into the RID-keyed NIfTI layout the
NeuroJEPA embed step expects.

IDA delivers DICOM nested as
    <dicom_root>/**/<PTID>/<Description>/<datetime>/I<IMAGEUID>/*.dcm
but ``scripts/build_adni_image_manifest.py --image-root`` and the embed loop key
on the numeric **RID** (``<out_root>/<RID>/T1.nii.gz``). This script bridges the
two using the PTID↔RID↔IMAGEUID crosswalk built from UCSFFSX7
(``data/real/_manifests/adni_ptid_rid_crosswalk.csv``).

For each crosswalk row it locates the ``I<IMAGEUID>`` series folder (the EXACT
FreeSurfer-anchor scan — not just any of the subject's scans), runs ``dcm2niix``
on it, and writes the single T1 volume to ``<out_root>/<RID>/T1.nii.gz``. If the
IMAGEUID folder is absent it falls back to the first MPRAGE/IR-FSPGR series under
that PTID and warns.

Idempotent: skips a subject whose ``T1.nii.gz`` already exists (``--force`` to
redo). Needs ``dcm2niix`` on PATH (``brew install dcm2niix`` /
``conda install -c conda-forge dcm2niix``).

Usage:
    ./.venv/bin/python scripts/organize_adni_downloads.py \
        --dicom-root /path/to/ADNI            # the unzipped IDA download
    # then (default out_root = data/real/ADNI_MRI):
    ./.venv/bin/python scripts/build_adni_image_manifest.py --image-root data/real/ADNI_MRI
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_XWALK = _ROOT / "data" / "real" / "_manifests" / "adni_ptid_rid_crosswalk.csv"
_OUT = _ROOT / "data" / "real" / "ADNI_MRI"


def _find_series(dicom_root: Path, imageuid, ptid: str) -> Path | None:
    """Locate the DICOM series dir. Prefer the exact I<IMAGEUID> folder; else the
    first MPRAGE/IR-FSPGR series under the PTID subtree."""
    if pd.notna(imageuid):
        hits = list(dicom_root.rglob(f"I{int(imageuid)}"))
        hits = [h for h in hits if h.is_dir() and any(h.glob("*.dcm"))]
        if hits:
            return hits[0]
    # Fallback: any structural series under this subject.
    for pdir in dicom_root.rglob(ptid):
        if not pdir.is_dir():
            continue
        for desc in pdir.iterdir():
            if not desc.is_dir():
                continue
            name = desc.name.upper()
            if "MPRAGE" in name or "IR-FSPGR" in name or "IR_FSPGR" in name:
                leaves = [d for d in desc.rglob("*") if d.is_dir() and any(d.glob("*.dcm"))]
                if leaves:
                    return leaves[0]
    return None


def _convert(series: Path, rid: int, out_root: Path) -> bool:
    """dcm2niix a series dir -> <out_root>/<rid>/T1.nii.gz (largest volume wins)."""
    subj_out = out_root / str(rid)
    subj_out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(
            ["dcm2niix", "-z", "y", "-f", "%p_%s", "-o", tmp, str(series)],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  [rid {rid}] dcm2niix failed: {r.stderr.strip().splitlines()[-1:]}")
            return False
        niis = sorted(Path(tmp).glob("*.nii.gz"), key=lambda p: p.stat().st_size, reverse=True)
        if not niis:
            print(f"  [rid {rid}] no NIfTI produced from {series}")
            return False
        shutil.copy2(niis[0], subj_out / "T1.nii.gz")
        return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom-root", type=Path, required=True,
                    help="Unzipped IDA ADNI DICOM download folder")
    ap.add_argument("--out-root", type=Path, default=_OUT)
    ap.add_argument("--crosswalk", type=Path, default=_XWALK)
    ap.add_argument("--force", action="store_true", help="reconvert even if T1 exists")
    args = ap.parse_args()

    if shutil.which("dcm2niix") is None:
        raise SystemExit("dcm2niix not found on PATH — install it "
                         "(brew install dcm2niix / conda install -c conda-forge dcm2niix).")
    if not args.dicom_root.exists():
        raise SystemExit(f"--dicom-root {args.dicom_root} does not exist.")
    xw = pd.read_csv(args.crosswalk)

    done = skipped = exact = fell_back = missing = failed = 0
    for _, row in xw.iterrows():
        rid = int(row["RID"])
        t1 = args.out_root / str(rid) / "T1.nii.gz"
        if t1.exists() and not args.force:
            skipped += 1
            continue
        series = _find_series(args.dicom_root, row.get("IMAGEUID"), str(row["PTID"]))
        if series is None:
            missing += 1
            print(f"  [rid {rid} / {row['PTID']}] no series found")
            continue
        exact += int(series.name == f"I{int(row['IMAGEUID'])}") if pd.notna(row.get("IMAGEUID")) else 0
        fell_back += int(series.name != f"I{int(row['IMAGEUID'])}") if pd.notna(row.get("IMAGEUID")) else 1
        if _convert(series, rid, args.out_root):
            done += 1
        else:
            failed += 1

    print(f"\n[organize] converted={done} skipped(existing)={skipped} "
          f"missing={missing} failed={failed}")
    print(f"[organize] of converted: exact-IMAGEUID={exact} fell-back-to-MPRAGE={fell_back}")
    print(f"[organize] output root: {args.out_root}")
    print(f"[organize] next: ./.venv/bin/python scripts/build_adni_image_manifest.py "
          f"--image-root {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
