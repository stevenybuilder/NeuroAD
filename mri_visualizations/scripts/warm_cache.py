#!/usr/bin/env python
"""Precompute the expensive check caches (SynthStrip mask, registration residuals)
so the first 'Run checks' in the UI is instant.

Run in the sfg env:  micromamba run -n sfg python scripts/warm_cache.py
"""

import sys
import time

sys.path.insert(0, "backend")

from sfg import config  # noqa: E402
from sfg.checks.loader import load_builtin_checks  # noqa: E402
from sfg.fixtures import ensure_fixtures  # noqa: E402
from sfg.pipeline import run_pipeline  # noqa: E402
from sfg.registry import Registry  # noqa: E402
from sfg.resources import ResourceStore  # noqa: E402


def main() -> int:
    ensure_fixtures()
    load_builtin_checks()
    registry = Registry()
    store = ResourceStore(config.RESOURCE_DIR)
    t = time.time()
    flags = run_pipeline(registry, store)
    print(f"warmed caches by running {len(flags)} flags in {time.time() - t:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
