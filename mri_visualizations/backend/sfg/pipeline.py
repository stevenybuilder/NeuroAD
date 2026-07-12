"""Runs checks over a cohort and returns flags ranked for adjudication."""

from __future__ import annotations

from typing import Optional

from .checks.base import Check, all_checks, get_check
from .flags import Flag
from .registry import Registry, Scan
from .resources import ResourceStore


def run_checks(
    scans: list[Scan],
    store: ResourceStore,
    checks: Optional[list[Check]] = None,
) -> list[Flag]:
    checks = checks if checks is not None else all_checks()
    flags: list[Flag] = []
    for check in checks:
        try:
            # Cohort-level checks (e.g. cross-scanner comparisons) see every scan
            # at once; per-scan checks are called once per scan.
            if hasattr(check, "run_cohort"):
                flags.extend(check.run_cohort(scans, store))
            else:
                for scan in scans:
                    flags.extend(check.run(scan, store))
        except Exception as exc:  # a broken check must not sink the cohort
            flags.append(
                Flag(
                    check_id=check.check_id,
                    scan_id=scans[0].scan_id if scans else "?",
                    severity="error",
                    explanation=f"check '{check.check_id}' raised: {exc}",
                )
            )
    flags.sort(key=lambda f: f.sort_key())
    return flags


def run_pipeline(
    registry: Registry,
    store: ResourceStore,
    scan_ids: Optional[list[str]] = None,
    check_ids: Optional[list[str]] = None,
) -> list[Flag]:
    scans = (
        [registry.get(s) for s in scan_ids] if scan_ids else registry.scans()
    )
    checks = [get_check(c) for c in check_ids] if check_ids else None
    return run_checks(scans, store, checks)
