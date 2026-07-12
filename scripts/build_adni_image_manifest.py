#!/usr/bin/env python3
"""
Build the ADNI raw-MRI embedding manifest from the tabular ADNI export.

ADNI on disk here is TABULAR ONLY: ``data/real/_gated/adni.csv`` carries the
FreeSurfer-derived ``emb_*`` columns plus the REAL plasma panel
(p_tau217/gfap/nfl), amyloid status, apoe4, dx, conversion, site, scanner, age,
sex for 2,951 subjects. There is **no raw ADNI MRI locally** — pulling it needs
the user's own LONI/IDA credentials + the ADNI Data Use Agreement (manual:
ida.loni.usc.edu -> Download -> Image Collections -> Advanced Search ->
MRI / MPRAGE -> download DICOM -> dcm2niix -> T1w NIfTI).

This script emits a join manifest so that, once the user drops their downloaded
T1w NIfTIs into a folder, the NeuroJEPA image embeddings computed from those
volumes can be joined back to each subject's biomarkers with ZERO ambiguity:

    subject_id, image_path, dx, age, sex, site, scanner,
    p_tau217, gfap, nfl, amyloid

``image_path`` is written as a PLACEHOLDER (``ADNI_MRI/<subject>/T1.nii.gz``).
It is intentionally NOT a real path yet — it is filled in after the IDA
download, once the user knows where their NIfTIs live. Two ways to fill it:

  * name each downloaded volume ``ADNI_MRI/<subject_id>/T1.nii.gz`` (mirror the
    placeholder layout) and pass ``--image-root`` so the manifest points at the
    real files, or
  * leave the placeholder and let the embed step's ``--image-root`` /
    per-subject glob resolve them at run time.

The manifest carries every biomarker column so the downstream embedding CSV
(``scripts/neurojepa_embed_colab.py`` copies all non-image columns through) lands
already joined to plasma p-tau217/GFAP/NfL + amyloid — which is exactly what the
biomarker-anchoring analysis in ``scripts/run_adni_crosscohort.py`` consumes.

Compliance: reads only the derived tabular table; writes only a small manifest
of ids + placeholder paths + numbers. No raw MRI is touched here.

Usage:
    ./.venv/bin/python scripts/build_adni_image_manifest.py
    # optionally point image_path at a real download root + naming pattern:
    ./.venv/bin/python scripts/build_adni_image_manifest.py \
        --image-root /data/ADNI_MRI --image-pattern '{subject}/T1.nii.gz'
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
ADNI_CSV = _ROOT / "data" / "real" / "_gated" / "adni.csv"
OUT = _ROOT / "data" / "real" / "_manifests" / "adni_image_manifest.csv"

# Columns carried through so the eventual image-embedding CSV is join-ready to
# biomarkers without a second merge. Order is deliberate (id, then the labels the
# analysis contrasts on, then the plasma/amyloid panel the anchoring probe uses).
CARRY = ["subject_id", "image_path", "dx", "age", "sex", "site", "scanner",
         "p_tau217", "gfap", "nfl", "amyloid"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adni-csv", default=str(ADNI_CSV),
                    help="tabular ADNI export (FreeSurfer emb_* + real plasma panel)")
    ap.add_argument("--out", default=str(OUT), help="manifest output path")
    ap.add_argument("--image-root", default="ADNI_MRI",
                    help="root the placeholder image_path is built under; set this to your "
                         "real IDA-download folder AFTER the download to point at actual NIfTIs")
    ap.add_argument("--image-pattern", default="{subject}/T1.nii.gz",
                    help="per-subject path template under --image-root; '{subject}' -> subject_id")
    args = ap.parse_args()

    src = Path(args.adni_csv)
    if not src.exists():
        raise SystemExit(f"ADNI tabular export not found: {src}")

    df = pd.read_csv(src)
    missing = [c for c in ("subject_id", "dx", "age", "sex", "site", "scanner",
                           "p_tau217", "gfap", "nfl", "amyloid") if c not in df.columns]
    if missing:
        raise SystemExit(f"{src} missing expected columns: {missing}")

    df["subject_id"] = df["subject_id"].astype(str)
    # PLACEHOLDER image_path — filled in after the IDA/LONI download (see module docstring).
    root = args.image_root.rstrip("/")
    image_path = df["subject_id"].map(
        lambda s: f"{root}/{args.image_pattern.format(subject=s)}")

    # Assemble the narrow manifest directly (avoids fragmenting the wide emb_* frame).
    man = df[[c for c in CARRY if c != "image_path"]].copy()
    man.insert(1, "image_path", image_path.to_numpy())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    man.to_csv(args.out, index=False)

    n_plasma = int(man["p_tau217"].notna().sum())
    n_amy = int(man["amyloid"].notna().sum())
    dxs = man["dx"].value_counts().to_dict()
    print(f"[adni-manifest] wrote {args.out}")
    print(f"[adni-manifest] {len(man)} subjects | dx {dxs}")
    print(f"[adni-manifest] plasma p-tau217 present: {n_plasma} | amyloid status present: {n_amy}")
    print(f"[adni-manifest] image_path is a PLACEHOLDER under '{root}/' "
          f"({args.image_pattern}) — fill it after the IDA download.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
