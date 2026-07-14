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

#: The EXACT display hypotheses the UI sends for the flagship (neuroad.html
#: entryValue + the /start prefill), so the router cache carries them and the demo
#: happy path is a deterministic router-cache hit — never a live classify call. The
#: grid cell is hit via the inferred target regardless of wording, but the router
#: cache is keyed by normalized text, so the flagship text must be pre-routed here.
_FLAGSHIP_SEEDS = [
    ("adni:combat",
     "p-tau217-anchored hippocampal atrophy separates Alzheimer's disease from cognitively normal"),
]


def _region_outcomes() -> list[tuple[str, str]]:
    """(hypothesis, outcome-tag) per named ROI region on adni:roi, for BOTH
    computable outcomes. Regions are read from the cohort's own region map, so
    this tracks whatever the ETL emitted:
        dx_binary  <- "...separates Alzheimer's dementia from controls"
        conversion <- "baseline ... predicts MCI to AD conversion"
    """
    try:
        from neuroad.data import loaders
        df = loaders.load("adni:roi")
        region_map = df.attrs.get("region_columns", {}) or {}
        singles = [r for r, cols in region_map.items() if len(cols) == 1]
    except Exception as exc:  # noqa: BLE001
        print(f"[warm] adni:roi unavailable, skipping region grid: {exc}")
        return []
    hyps = []
    for slug in sorted(singles):
        hyps.append((f"{slug} atrophy separates Alzheimer's dementia from controls", "dx"))
        hyps.append((f"Baseline {slug} atrophy predicts MCI to AD conversion", "conv"))
    return hyps


#: The four fluid-biomarker anchors swept ON TOP of the volume (None) base — but
#: ONLY for a promoted (region, outcome), because a fragile finding has no
#: translation and the anchor cannot change its (identical) output. Because the
#: orchestrator memoizes the anchor-invariant base per (dataset, target, region,
#: seed, api), these extra cells each cost only the ~0.5s translation re-apply,
#: not another full referee — the region×anchor×outcome grid is now affordable
#: in one pass instead of the ~2.5h a from-scratch sweep once implied.
_REGION_ANCHORS = ["amyloid", "p_tau217", "gfap", "nfl"]


def _is_promoted(result: dict) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("promoted") is True:
        return True
    case = result.get("case")
    return bool(isinstance(case, dict) and case.get("promoted"))


def _warm_cell(dataset, hyp, anchor, done, total) -> object:
    """Compute + persist ONE cell (idempotent on an already-warm cell). Returns
    the result dict (or None on skip) so the caller can gate the anchor sweep."""
    if investigate_cache.get(dataset, hyp, anchor, False) is not None:
        print(f"[{done}/{total}] cached  {dataset} anchor={anchor} :: {hyp[:44]}")
        return None
    t0 = time.time()
    try:
        result = compute_investigate(hyp, dataset, seed=0, anchor=anchor, want_api=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[{done}/{total}] SKIP    {dataset} anchor={anchor}: {exc}")
        return None
    investigate_cache.put(dataset, hyp, anchor, False, result)
    reg = (result.get("case", {}).get("naive_effect", {}) or {}).get("region", "")
    print(f"[{done}/{total}] warmed  {dataset} anchor={anchor} region={reg} "
          f"({time.time()-t0:.1f}s)")
    return result


def _warm_router(hypotheses) -> None:
    """Pre-route every seed hypothesis through the canonical router BEFORE the grid
    warm, so ``app/router_cache.json`` and ``app/investigate_cache.json`` ship
    together and are mutually consistent: at request time every seed is a
    deterministic router-cache hit (no live classify call), and the grid cells were
    keyed by the SAME routed target. A novel typed hypothesis is the only thing that
    ever pays a live Sonnet call, once, then it too is frozen. Offline (no key) this
    is a no-op-equivalent: the router falls back to the keyword target and nothing is
    persisted, so the file stays empty and the demo path is unchanged."""
    try:
        from neuroad.claude.router import route_target
    except Exception as exc:  # noqa: BLE001
        print(f"[warm-router] router unavailable, skipping: {exc}")
        return
    seen = set()
    for hyp in hypotheses:
        h = (hyp or "").strip()
        if not h or h in seen:
            continue
        seen.add(h)
        try:
            tgt = route_target(h, None)
            print(f"[warm-router] {tgt:11s} :: {h[:56]}")
        except Exception as exc:  # noqa: BLE001
            print(f"[warm-router] SKIP :: {h[:40]}: {exc}")


def main() -> int:
    t_all = time.time()
    # 1. Base datasets (plasma/embedding cohorts) x every anchor.
    base_cells = [(d, h, a) for d, h in _DATASETS for a in _ANCHORS]
    # 2. Region grid: every ROI x both outcomes, anchor=None first (the base);
    #    the 4 fluid anchors are appended only when the base PROMOTES (below).
    region_pairs = _region_outcomes()

    # 0. Pre-warm the routing cache over EVERY seed hypothesis so the router cache
    #    ships consistent with the grid (see _warm_router). Must precede grid warm.
    _warm_router([h for _d, h in _DATASETS]
                 + [h for h, _t in region_pairs]
                 + [e[1] for e in _FLAGSHIP_SEEDS])

    total = len(base_cells) + len(region_pairs)  # lower bound; grows per promotion
    done = 0

    for dataset, hyp, anchor in base_cells:
        done += 1
        _warm_cell(dataset, hyp, anchor, done, total)

    warmed_anchor = 0
    for hyp, _tag in region_pairs:
        done += 1
        base = _warm_cell("adni:roi", hyp, None, done, total)
        # Reuse the memoized base: sweep the fluid anchors ONLY where the finding
        # promoted (else the anchor cannot change the output).
        promoted = _is_promoted(base) if base is not None else \
            _is_promoted(investigate_cache.get("adni:roi", hyp, None, False))
        if not promoted:
            continue
        for anc in _REGION_ANCHORS:
            total += 1
            warmed_anchor += 1
            _warm_cell("adni:roi", hyp, anc, done, total)

    print(f"\nDone: base+region+{warmed_anchor} anchor cells in "
          f"{time.time()-t_all:.1f}s -> {investigate_cache._CACHE_FILE} "
          f"({len(investigate_cache._load())} total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
