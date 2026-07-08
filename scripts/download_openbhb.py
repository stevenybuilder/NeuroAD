#!/usr/bin/env python3
"""
Idempotent re-download of the OpenBHB participants derivative table into
``data/real/openbhb_participants.tsv``. No login, no credentials.

The file is already VENDORED in the repo — this script exists so the open data is
reproducible from scratch (e.g. a fresh clone that pruned it, or to re-verify
integrity). It is a utility, not part of the offline demo path.

OpenBHB is a large multi-site cohort of **healthy controls only**. We use it for
the STAR leakage star on REAL data: with no disease present, a structural probe
that still predicts 3T-vs-1.5T at AUC ~0.90 is measuring pure acquisition physics.

Behaviour
---------
* If the target file already exists and looks valid (right columns, >=3000 rows,
  >=2 field strengths), it is left untouched unless ``--force`` is given.
* Otherwise each candidate mirror URL is tried in order until one yields a TSV
  that parses with the expected columns and row count.

Usage
-----
    python scripts/download_openbhb.py            # fill only if missing/invalid
    python scripts/download_openbhb.py --force    # re-download
    python scripts/download_openbhb.py --check    # validate, do not download
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_DIR = REPO_ROOT / "data" / "real"

FILENAME = "openbhb_participants.tsv"
REQUIRED = [
    "participant_id", "study", "sex", "age", "site", "diagnosis",
    "tiv", "csfv", "gmv", "wmv", "magnetic_field_strength",
    "acquisition_setting", "siteXacq", "split",
]
MIN_ROWS = 3000

# Verified live 2026-07-08 (HTTP 200, TAB-separated, 3984 healthy-control rows,
# Apache-2.0). No-login HuggingFace mirror of the OpenBHB participants table.
URLS = [
    "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/participants.tsv",
]


def _read_tsv(source) -> pd.DataFrame:
    return pd.read_csv(source, sep="\t")


def _looks_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        df = _read_tsv(path)
    except Exception:
        return False
    return _table_ok(df)


def _table_ok(df: pd.DataFrame) -> bool:
    if len(df) < MIN_ROWS:
        return False
    if not all(c in df.columns for c in REQUIRED):
        return False
    # Need at least two distinct field strengths for the scanner-leakage star.
    if df["magnetic_field_strength"].nunique(dropna=True) < 2:
        return False
    return True


def _fetch(url: str) -> str | None:
    """Return TSV text if the URL yields a valid table, else None."""
    try:
        import requests  # local import so --check works without requests
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        text = resp.text
    except Exception as exc:  # noqa: BLE001
        print(f"    ! {url}\n      {exc}")
        return None
    try:
        df = _read_tsv(io.StringIO(text))
    except Exception as exc:  # noqa: BLE001
        print(f"    ! parsed-fail {url}: {exc}")
        return None
    if not _table_ok(df):
        print(f"    ! wrong schema/size/field-strengths {url}")
        return None
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="re-download even if a valid file already exists")
    ap.add_argument("--check", action="store_true",
                    help="only validate the existing file; never download")
    args = ap.parse_args(argv)

    REAL_DIR.mkdir(parents=True, exist_ok=True)
    path = REAL_DIR / FILENAME
    valid = _looks_valid(path)

    if args.check:
        status = "OK" if valid else "MISSING/INVALID"
        print(f"[{status}] {path.relative_to(REPO_ROOT)}")
        return 0 if valid else 1

    if valid and not args.force:
        print(f"[skip] {FILENAME} already present and valid")
        return 0

    print(f"[fetch] {FILENAME}")
    for url in URLS:
        print(f"    trying {url}")
        text = _fetch(url)
        if text is not None:
            path.write_text(text)
            print(f"    -> wrote {path.relative_to(REPO_ROOT)} ({len(text)} bytes)")
            return 0

    print(f"    FAILED to fetch {FILENAME} from any mirror.\n"
          f"    The file is already vendored in the repo; if this is a fresh "
          f"checkout, restore it from git or the HuggingFace mirror "
          f"(benoit-dufumier/openBHB).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
