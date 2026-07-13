"""FastAPI app: serves scans, derived overlays, runs checks, logs adjudications.

The frontend (NiiVue) talks only to this API. Volumes and overlays are streamed
as files (NiiVue parses NIfTI/OBJ client-side); flags and adjudications are JSON.
"""

from __future__ import annotations

import json
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import nibabel as nib

from . import config
from .checks.base import all_checks
from .checks.loader import load_builtin_checks
from .fixtures import ensure_fixtures
from .gallery import build_gallery_flags
from .pipeline import run_pipeline
from .registry import Registry
from .resources import ResourceStore

app = FastAPI(title="Silent-Failure Guard")

# Vite proxies /api, but allow direct localhost access too (headless testing).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ensure_fixtures()
REGISTRY = Registry()
STORE = ResourceStore(config.RESOURCE_DIR)
load_builtin_checks()


# --- Scans & volumes ----------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "scans": len(REGISTRY.scans()), "checks": len(all_checks())}


@app.post("/api/refresh")
def refresh() -> dict:
    """Re-scan data dirs (e.g. after IXI is fetched at runtime)."""
    ensure_fixtures()
    REGISTRY.refresh()
    return {"ok": True, "scans": len(REGISTRY.scans())}


@app.get("/api/scans")
def list_scans() -> list[dict]:
    return [s.summary() for s in REGISTRY.scans()]


@app.get("/api/scans/{scan_id}")
def scan_detail(scan_id: str) -> dict:
    try:
        scan = REGISTRY.get(scan_id)
    except KeyError:
        raise HTTPException(404, f"unknown scan {scan_id}")
    out = scan.summary()
    out["meta"] = {m: scan.header_meta(m) for m in scan.modalities}
    return out


_DS_DIR = config.CACHE_DIR / "downsampled"


def _preview_volume(path):
    """Return a cached factor-2 downsampled copy of a NIfTI so the in-browser NiiVue
    viewer decodes/renders a QC preview fast — full-res T1s are ~15 MB and slow to
    texture + ray-cast on integrated GPUs. nibabel's slicer preserves world space, so
    the L/R markers and overlays still align. Generated once per scan, reused after;
    falls back to the full-res original on any error."""
    try:
        _DS_DIR.mkdir(parents=True, exist_ok=True)
        stem = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
        out = _DS_DIR / f"{stem}__ds2.nii.gz"
        if not (out.exists() and out.stat().st_mtime >= path.stat().st_mtime):
            small = nib.load(str(path)).slicer[::2, ::2, ::2]
            nib.save(small, str(out))
        return out
    except Exception:  # noqa: BLE001 - preview is best-effort; serve full-res on failure
        return path


@app.get("/api/volume/{scan_id}/{modality}")
def get_volume(scan_id: str, modality: str) -> FileResponse:
    try:
        scan = REGISTRY.get(scan_id)
    except KeyError:
        raise HTTPException(404, f"unknown scan {scan_id}")
    if modality == "seg":
        if not scan.seg:
            raise HTTPException(404, f"{scan_id} has no seg")
        path = scan.seg
    elif modality in scan.modalities:
        path = scan.modality_path(modality)
    else:
        raise HTTPException(404, f"{scan_id} has no modality {modality}")
    return FileResponse(_preview_volume(path), media_type="application/gzip", filename=path.name)


@app.get("/api/resource/{key}")
def get_resource(key: str) -> FileResponse:
    path = STORE.path(key)
    if path is None:
        raise HTTPException(404, f"unknown resource {key}")
    media = "model/obj" if path.suffix == ".obj" else "application/gzip"
    return FileResponse(path, media_type=media, filename=path.name)


# --- Checks & flags -----------------------------------------------------------


@app.get("/api/checks")
def list_checks() -> list[dict]:
    return [{"check_id": c.check_id, "description": c.description} for c in all_checks()]


class RunRequest(BaseModel):
    scan_ids: Optional[list[str]] = None
    check_ids: Optional[list[str]] = None


@app.post("/api/run")
def run(req: RunRequest) -> dict:
    flags = run_pipeline(REGISTRY, STORE, req.scan_ids, req.check_ids)
    return {"flags": [f.model_dump() for f in flags]}


@app.get("/api/gallery")
def gallery() -> dict:
    """Annotation-framework self-test: one flag of every payload kind."""
    flags = build_gallery_flags(REGISTRY, STORE)
    return {"flags": [f.model_dump() for f in flags]}


# --- Adjudication log ---------------------------------------------------------


class Adjudication(BaseModel):
    scan_id: str
    check_id: str
    decision: str  # confirm | reject | relabel
    relabel: Optional[str] = None
    note: Optional[str] = None


@app.post("/api/adjudications")
def add_adjudication(rec: Adjudication) -> dict:
    config.ADJUDICATION_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = rec.model_dump() | {"ts": time.time()}
    with config.ADJUDICATION_LOG.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return {"ok": True}


@app.get("/api/adjudications")
def list_adjudications() -> list[dict]:
    if not config.ADJUDICATION_LOG.exists():
        return []
    return [
        json.loads(line)
        for line in config.ADJUDICATION_LOG.read_text().splitlines()
        if line.strip()
    ]
