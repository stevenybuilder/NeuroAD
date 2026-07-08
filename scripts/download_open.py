#!/usr/bin/env python3
"""
Idempotent re-download of the two genuinely-open OASIS tabular CSVs into
``data/real/``. No login, no credentials.

These two files are already VENDORED in the repo — this script exists so the
open data is reproducible from scratch (e.g. a fresh clone that pruned them, or
to re-verify integrity). It is a utility, not part of the offline demo path
(the synthetic harness guarantees the demo runs with zero network access).

Behaviour
---------
* If a target file already exists and looks valid (right columns, >100 rows),
  it is left untouched unless ``--force`` is given.
* Otherwise each candidate mirror URL is tried in order until one yields a CSV
  that parses with the expected columns.

Usage
-----
    python scripts/download_open.py            # fill only what's missing
    python scripts/download_open.py --force     # re-download both
    python scripts/download_open.py --check     # validate, do not download
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_DIR = REPO_ROOT / "data" / "real"

# Each target: (filename, expected required columns, candidate mirror URLs).
# Multiple mirrors are tried in order; the first that parses with the expected
# columns wins. Kaggle's "MRI and Alzheimer's" (jboysen) is the canonical source
# but needs auth, so we prefer no-login raw mirrors of the identical CSVs.
TARGETS = [
    {
        "filename": "oasis_longitudinal.csv",
        "required": ["Subject ID", "MRI ID", "Group", "Visit", "CDR",
                     "eTIV", "nWBV", "ASF"],
        "min_rows": 300,
        # Verified live 2026-07-08 (HTTP 200, byte-identical to the vendored copy,
        # sha256 d2f0a15f…). These no-login raw mirrors are the actual provenance
        # of the vendored files.
        "urls": [
            "https://raw.githubusercontent.com/stnava/RMI/master/tomfletcher/oasis_longitudinal.csv",
        ],
    },
    {
        "filename": "oasis_cross-sectional.csv",
        "required": ["ID", "M/F", "Age", "CDR", "eTIV", "nWBV", "ASF"],
        "min_rows": 400,
        # Verified live 2026-07-08 (HTTP 200, byte-identical, sha256 2091ee6b…).
        "urls": [
            "https://raw.githubusercontent.com/jddunn/dementia-progression-analysis/master/oasis_cross-sectional.csv",
        ],
    },
]


def _looks_valid(path: Path, required: list[str], min_rows: int) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if len(df) < min_rows:
        return False
    return all(c in df.columns for c in required)


def _fetch(url: str, required: list[str], min_rows: int) -> str | None:
    """Return CSV text if the URL yields a valid table, else None."""
    try:
        import requests  # local import so --check works without requests
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        text = resp.text
    except Exception as exc:  # noqa: BLE001
        print(f"    ! {url}\n      {exc}")
        return None
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:  # noqa: BLE001
        print(f"    ! parsed-fail {url}: {exc}")
        return None
    if len(df) < min_rows or not all(c in df.columns for c in required):
        print(f"    ! wrong schema/size {url}")
        return None
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="re-download even if a valid file already exists")
    ap.add_argument("--check", action="store_true",
                    help="only validate existing files; never download")
    args = ap.parse_args(argv)

    REAL_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for t in TARGETS:
        path = REAL_DIR / t["filename"]
        valid = _looks_valid(path, t["required"], t["min_rows"])

        if args.check:
            status = "OK" if valid else "MISSING/INVALID"
            print(f"[{status}] {path.relative_to(REPO_ROOT)}")
            all_ok = all_ok and valid
            continue

        if valid and not args.force:
            print(f"[skip] {t['filename']} already present and valid")
            continue

        print(f"[fetch] {t['filename']}")
        wrote = False
        for url in t["urls"]:
            print(f"    trying {url}")
            text = _fetch(url, t["required"], t["min_rows"])
            if text is not None:
                path.write_text(text)
                print(f"    -> wrote {path.relative_to(REPO_ROOT)} "
                      f"({len(text)} bytes)")
                wrote = True
                break
        if not wrote:
            all_ok = False
            print(f"    FAILED to fetch {t['filename']} from any mirror.\n"
                  f"    The file is already vendored in the repo; if this is a "
                  f"fresh checkout, restore it from git or Kaggle "
                  f"(jboysen/mri-and-alzheimers).")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
