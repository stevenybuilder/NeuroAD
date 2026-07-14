"""Precomputed-grid cache for /api/investigate.

The engine collapses any free-text hypothesis to a small COORDINATE —
``(dataset, target, anchor)`` — where ``target`` is one of
{conversion, dx_binary, site, scanner} (keyword-inferred) and ``anchor`` is one
of {amyloid, p_tau217, gfap, nfl} or none. Because the routing is enum-based, the
grid of coordinates is FINITE and the real engine result for each cell is
deterministic. So we precompute the grid offline (``scripts/warm_investigate_cache.py``),
ship the small ``investigate_cache.json`` inside the (private) image, and the live
server returns a real cell by LOOKUP instead of a ~25s recompute.

Honesty: every cached value is a REAL engine output (same pipeline, full rigor),
not a fabricated number — the frozen-seam pattern already used by demo_data.json,
generalised from one cell to the grid. On a hit we personalise only the DISPLAYED
hypothesis text; the computed numbers are the genuine result for that coordinate.
A cache MISS still computes live and back-fills the cell (self-warming).
"""
from __future__ import annotations

import copy
import json
import logging
import threading
from pathlib import Path
from typing import Optional

_log = logging.getLogger("neuroad.investigate_cache")
_APP_DIR = Path(__file__).resolve().parent
_CACHE_FILE = _APP_DIR / "investigate_cache.json"
_ANCHORS = ("amyloid", "p_tau217", "gfap", "nfl")

_lock = threading.Lock()
_cache: Optional[dict] = None
_cache_mtime: float = -1.0


def _load() -> dict:
    """Load the cache, re-reading the file if it changed on disk (so a warm job
    running in a separate process while the server is up is picked up, not just
    the build-time preload)."""
    global _cache, _cache_mtime
    try:
        mtime = _CACHE_FILE.stat().st_mtime if _CACHE_FILE.exists() else -1.0
    except OSError:
        mtime = -1.0
    if _cache is None or mtime != _cache_mtime:
        try:
            _cache = json.loads(_CACHE_FILE.read_text()) if _CACHE_FILE.exists() else {}
            _cache_mtime = mtime
            if _cache:
                _log.info("investigate cache loaded: %d cells", len(_cache))
        except Exception as exc:  # noqa: BLE001
            _log.debug("investigate cache read failed: %r", exc)
            if _cache is None:
                _cache = {}
    return _cache


#: Cohorts whose emb_i are named ROI volumes, so a region conditions the probe.
#: Only these carry a region axis in the cache key; every other cohort keys
#: region="" (no region conditioning) and never pays a df load on a cache hit.
_REGION_DATASETS = ("adni:roi", "adni:freesurfer", "adni:fsx")


def _infer_target(hypothesis: str) -> str:
    """The SAME target the engine routes to — via the one canonical router
    (``claude.router.route_target``: routing-cache hit -> LLM-on-miss -> keyword
    backstop). Because ``claim_parser._fallback`` routes through the same function,
    the cache key's target can never diverge from the engine's routed target. A
    cache-hit is pure Python (<1ms), so the hot path stays fast; only a novel typed
    hypothesis pays one classify call, once. Any failure -> keyword -> "conversion",
    a miss, never a wrong number."""
    try:
        from neuroad.claude.router import route_target
        return route_target(hypothesis or "", None)
    except Exception:  # noqa: BLE001
        try:
            from neuroad.claude.claim_parser import _infer_target as _it
            return _it(hypothesis or "", None)
        except Exception:  # noqa: BLE001
            return "conversion"


def _region_for_key(dataset: str, hypothesis: str) -> str:
    """The region slug the engine would resolve, or "" for non-ROI cohorts. Loads
    the (lru-cached) cohort only for region-capable datasets, so adni:combat hits
    stay df-free. A mismatch only ever causes a miss, never a wrong number."""
    if (dataset or "").lower() not in _REGION_DATASETS:
        return ""
    try:
        from neuroad.data import loaders
        from neuroad.harness import region as _region
        slug, _cols = _region.extract_region(hypothesis or "", loaders.load(dataset))
        return slug or ""
    except Exception:  # noqa: BLE001
        return ""


def key(dataset: str, hypothesis: str, anchor: Optional[str], want_api: bool) -> str:
    a = anchor if anchor in _ANCHORS else "none"
    reg = _region_for_key(dataset, hypothesis)
    return f"{dataset}|{_infer_target(hypothesis)}|{reg}|{a}|{int(bool(want_api))}"


def get(dataset: str, hypothesis: str, anchor: Optional[str], want_api: bool):
    return _load().get(key(dataset, hypothesis, anchor, want_api))


def put(dataset: str, hypothesis: str, anchor: Optional[str], want_api: bool,
        result: dict, *, persist: bool = True) -> None:
    global _cache_mtime
    c = _load()
    k = key(dataset, hypothesis, anchor, want_api)
    with _lock:
        c[k] = result
        if persist:
            try:
                tmp = _CACHE_FILE.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(c))
                tmp.replace(_CACHE_FILE)
                # Adopt our own write as the current version so the next _load()
                # doesn't re-parse the whole file.
                _cache_mtime = _CACHE_FILE.stat().st_mtime
            except Exception as exc:  # noqa: BLE001
                _log.debug("investigate cache write failed: %r", exc)


def personalize(result: dict, hypothesis: str, dataset: str) -> dict:
    """Return a copy of a cached cell with the user's typed hypothesis on the
    display fields; the real computed numbers are untouched."""
    r = copy.deepcopy(result)
    meta = r.setdefault("_meta", {})
    meta["hypothesis"] = hypothesis
    meta["dataset"] = dataset
    meta["cached"] = True
    case = r.get("case")
    if isinstance(case, dict):
        claim = case.get("claim")
        if isinstance(claim, dict):
            claim["claim_text"] = hypothesis
    return r
