#!/usr/bin/env python
"""Pre-warm app/router_cache.json over every demo happy-path hypothesis.

Run WITH ANTHROPIC_API_KEY set (source .env) so the flagship + one-click chips +
the canonical seeds resolve via the live Sonnet-5 router ONCE, here, and ship
frozen — so no demo click ever pays a live classify call at request time. Offline
(no key) this degrades to keyword routing and persists nothing, leaving the file
empty (the request-time keyword backstop is instant anyway).

    set -a; source .env; set +a
    PYTHONPATH=src:. .venv/bin/python -m scripts.prewarm_router
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from neuroad.claude import _client, router  # noqa: E402

# The exact texts the UI sends on a one-click demo path (flagship entryValue +
# the four chips in neuroad.html) plus the canonical warm seeds + a couple of
# adversarial cases the router exists to fix.
_HYPS = [
    # flagship (neuroad.html entryValue + /start prefill)
    "p-tau217-anchored hippocampal atrophy separates Alzheimer's disease from cognitively normal",
    # neuroad.html chips
    "p-tau217 predicts hippocampal atrophy",
    "GFAP astrocyte reactivity drives cortical thinning",
    "Amyloid PET burden precedes default-mode disruption",
    "Scanner field strength inflates atrophy estimates",
    # canonical warm seeds
    "p-tau217 predicts hippocampal atrophy in Alzheimer's disease",
    "structural embeddings predict MCI to AD conversion",
    "structural atrophy separates dementia from controls",
    # adversarial cases the router is meant to correct
    "p-tau217 predicts hippocampal atrophy in preclinical AD",
    "site-adjusted structural signal still separates AD from CN",
]


def main() -> int:
    print(f"live API: {_client.USING_LIVE_API}")
    for h in _HYPS:
        tgt = router.route_target(h, None)
        src = router.routing_source(h)
        print(f"  {tgt:11s} [{src.get('source')}] :: {h[:56]}")
    print(f"router cache -> {router._CACHE_FILE} ({len(router._load())} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
