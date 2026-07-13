#!/usr/bin/env python
"""Preload the /api/investigate grid cache.

Runs the REAL engine once per coordinate (dataset x anchor; target is inferred
from a canonical hypothesis) and writes app/investigate_cache.json, which the
server then serves by lookup (<1s) instead of recomputing (~25s). Ship the file
inside the (private) image so the deployed demo is instant from the first click.

Every cell is a genuine engine output at full rigor — this is the frozen-seam
pattern (demo_data.json) generalised from one cell to the grid, not fabrication.

Usage (from repo root, a shell WITHOUT NEUROAD_N_BOOT so cells are full-rigor):
    PYTHONPATH=src .venv/bin/python -m scripts.warm_investigate_cache
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from app import investigate_cache  # noqa: E402
from app.server import compute_investigate  # noqa: E402

#: (dataset, hypothesis) -> the demo's real coordinates. Anchors are swept per
#: dataset. Keep hypotheses canonical per target so the inferred target is stable.
_DATASETS = [
    ("adni:combat", "p-tau217 predicts hippocampal atrophy in Alzheimer's disease"),
    ("adni:combat", "structural embeddings predict MCI to AD conversion"),
    ("oasis", "structural atrophy separates dementia from controls"),
]
_ANCHORS = [None, "amyloid", "p_tau217", "gfap", "nfl"]


def _region_cells() -> list[tuple[str, str, object]]:
    """(dataset, hypothesis, anchor) for every named ROI region on adni:roi.
    Regions are read from the cohort's own region map, so this tracks whatever the
    ETL emitted. Only dx_binary computes on this cohort (no conversion label / no
    plasma), so we sweep regions x that one outcome with the volume (none) anchor."""
    try:
        from neuroad.data import loaders
        df = loaders.load("adni:roi")
        region_map = df.attrs.get("region_columns", {}) or {}
        singles = [r for r, cols in region_map.items() if len(cols) == 1]
    except Exception as exc:  # noqa: BLE001
        print(f"[warm] adni:roi unavailable, skipping region grid: {exc}")
        return []
    # Both computable outcomes per region now that the cohort carries metadata:
    #   dx_binary  <- "...separates Alzheimer's dementia from controls"
    #   conversion <- "baseline ... predicts MCI to AD conversion"
    cells = []
    for slug in sorted(singles):
        cells.append(("adni:roi",
                      f"{slug} atrophy separates Alzheimer's dementia from controls",
                      None))
        cells.append(("adni:roi",
                      f"Baseline {slug} atrophy predicts MCI to AD conversion",
                      None))
    return cells


def main() -> int:
    base = [(d, h, a) for d, h in _DATASETS for a in _ANCHORS]
    cells = base + _region_cells()
    total = len(cells)
    done = 0
    t_all = time.time()
    for dataset, hyp, anchor in cells:
        done += 1
        if investigate_cache.get(dataset, hyp, anchor, False) is not None:
            print(f"[{done}/{total}] cached  {dataset} anchor={anchor} :: {hyp[:40]}")
            continue
        t0 = time.time()
        try:
            result = compute_investigate(
                hyp, dataset, seed=0, anchor=anchor, want_api=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[{done}/{total}] SKIP    {dataset} anchor={anchor}: {exc}")
            continue
        investigate_cache.put(dataset, hyp, anchor, False, result)
        reg = (result.get("case", {}).get("naive_effect", {}) or {}).get("region", "")
        print(f"[{done}/{total}] warmed  {dataset} anchor={anchor} region={reg} "
              f"({time.time()-t0:.1f}s)")
    print(f"\nDone: {done} cells in {time.time()-t_all:.1f}s -> "
          f"{investigate_cache._CACHE_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
