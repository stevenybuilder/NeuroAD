#!/usr/bin/env python3
"""
Build the ID crosswalk + label table for an IDA imaging cohort download.

The Colab embedding driver (``scripts/adni_colab_dicom_to_embed.py``) needs a
``crosswalk.csv`` with columns **RID, PTID, IMAGEUID** — it converts each
``I<IMAGEUID>`` anchor series in the raw DICOM zip into
``ADNI_MRI/<RID>/T1.nii.gz`` (see docs/DATA_INGESTION_ETL.md §4/§6). This script
produces that crosswalk (and a labels table) from an IDA ``*_IDA_Metadata.zip``,
so the "see the exact join in git history" step is now a reusable, tested tool.

Handles BOTH naming conventions the IDA metadata zip mixes:
  * nested pointer stubs:  ADNI/<PTID>/<desc>/<date>/I<IMAGEUID>/<file>.xml
  * flat rich metadata:    ADNI/ADNI_<PTID>_<desc>_S<series>_I<IMAGEUID>.xml
The flat files are the namespaced ``idaxs`` schema carrying researchGroup / sex /
age; the nested ones are 210-byte pointers. We take the (PTID, IMAGEUID) pairs
from both (deduped) and enrich with researchGroup/sex/age from the flat files.

RID is derived from PTID (``<site>_S_<RID>``). Conversion status (sMCI vs pMCI)
is NOT in the imaging metadata — join DXSUM_*.csv on RID afterwards for that.

Pure stdlib, offline, no GPU. Writes:
  data/real/_manifests/adni_<cohort>_crosswalk.csv   (RID,PTID,IMAGEUID)
  data/real/_manifests/adni_<cohort>_labels.csv      (+ group,sex,age)
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_MANIFEST_DIR = _REPO / "data" / "real" / "_manifests"

# ADNI/<PTID>/<desc>/<date>/I<IMAGEUID>/<file>.xml   (nested pointer stub)
_NESTED = re.compile(r"ADNI/(\d+_S_\d+)/.*/I(\d+)/[^/]+\.xml$")
# ADNI/ADNI_<PTID>_<...>_I<IMAGEUID>.xml             (flat rich idaxs metadata)
_FLAT = re.compile(r"ADNI/ADNI_(\d+_S_\d+)_.*_I(\d+)\.xml$")


def _localname_text(root: ET.Element, tag: str) -> str:
    """Find the first element whose local name == ``tag`` (namespace-agnostic)."""
    for el in root.iter():
        if el.tag.rsplit("}", 1)[-1] == tag and (el.text or "").strip():
            return el.text.strip()
    return ""


def _rid_from_ptid(ptid: str) -> str:
    return ptid.split("_S_")[-1]


def parse_metadata_zip(zip_path: Path) -> dict[str, dict]:
    """Return {IMAGEUID: {rid, ptid, image_uid, group, sex, age}} for one cohort zip."""
    z = zipfile.ZipFile(zip_path)
    records: dict[str, dict] = {}
    for name in z.namelist():
        if not name.endswith(".xml"):
            continue
        m = _FLAT.match(name) or _NESTED.match(name)
        if not m:
            continue
        ptid, iuid = m.group(1), m.group(2)
        rec = records.setdefault(iuid, {
            "rid": _rid_from_ptid(ptid), "ptid": ptid, "image_uid": iuid,
            "group": "", "sex": "", "age": ""})
        # Enrich from the flat rich (namespaced idaxs) files only.
        if _FLAT.match(name):
            try:
                root = ET.fromstring(z.read(name))
            except ET.ParseError:
                continue
            rec["group"] = rec["group"] or _localname_text(root, "researchGroup")
            rec["sex"] = rec["sex"] or _localname_text(root, "subjectSex")
            rec["age"] = rec["age"] or _localname_text(root, "subjectAge")
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("metadata_zip", help="path to an IDA *_IDA_Metadata.zip")
    ap.add_argument("--cohort", required=True,
                    help="short cohort slug, e.g. 'mci_conversion' or 'adcn_expand'")
    args, _ = ap.parse_known_args()

    zip_path = Path(args.metadata_zip).expanduser()
    if not zip_path.exists():
        print(f"ERROR: metadata zip not found: {zip_path}", file=sys.stderr)
        return 2

    records = parse_metadata_zip(zip_path)
    if not records:
        print("ERROR: no (PTID, IMAGEUID) pairs parsed — unexpected zip layout",
              file=sys.stderr)
        return 1

    _MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    xwalk = _MANIFEST_DIR / f"adni_{args.cohort}_crosswalk.csv"
    labels = _MANIFEST_DIR / f"adni_{args.cohort}_labels.csv"

    rows = sorted(records.values(), key=lambda r: int(r["image_uid"]))
    with open(xwalk, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["RID", "PTID", "IMAGEUID"])
        for r in rows:
            w.writerow([r["rid"], r["ptid"], r["image_uid"]])
    with open(labels, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["rid", "ptid", "image_uid",
                                           "group", "sex", "age"])
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    grp = Counter(r["group"] or "UNKNOWN" for r in rows)
    print(f"cohort '{args.cohort}': {len(rows)} images, "
          f"{len({r['rid'] for r in rows})} unique subjects")
    print("researchGroup:", dict(grp))
    print("wrote:", xwalk)
    print("wrote:", labels)
    print("IMAGEUID worklist (comma-separated, for IDA Advanced Image Search):")
    print(",".join(r["image_uid"] for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
