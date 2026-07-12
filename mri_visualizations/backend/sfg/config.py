"""Repo-relative paths. Everything else imports locations from here."""

from __future__ import annotations

from pathlib import Path

# backend/sfg/config.py -> parents[2] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
# BraTS cases in the inline layout: one dir per case holding the four modality
# NIfTIs plus <case>-seg.nii.gz (the authoritative 4-label tumour mask).
BRATS_DIR = DATA_DIR / "brats" / "cases"
IXI_DIR = DATA_DIR / "ixi"
# Induced-failure variants of real scans (mirrored headers, injected corruption)
# so every check has a positive demo on demand. Each has a .json sidecar.
FIXTURES_DIR = DATA_DIR / "fixtures"

# Derived overlays/meshes and the adjudication log live under a gitignored cache.
CACHE_DIR = REPO_ROOT / "backend" / ".cache"
RESOURCE_DIR = CACHE_DIR / "resources"
STRIP_DIR = CACHE_DIR / "strip"  # cached SynthStrip masks (expensive to recompute)
ADJUDICATION_LOG = CACHE_DIR / "adjudications.jsonl"

# SynthStrip model weights (fetched from a HuggingFace mirror; see README).
SYNTHSTRIP_MODEL = REPO_ROOT / "backend" / "models" / "synthstrip.1.pt"
