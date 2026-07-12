"""Tool-shaped check interface + a registry.

Every check is an individually-callable object with a typed ``run`` and a
one-line ``description`` - so the eventual agent layer can enumerate and invoke
them with no glue. A check reads a Scan, may write derived overlays to the
ResourceStore, and returns zero or more Flags in the loose envelope.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..flags import Flag
from ..registry import Scan
from ..resources import ResourceStore


@runtime_checkable
class Check(Protocol):
    check_id: str
    description: str  # one line; shown in UI and readable by the future agent

    def run(self, scan: Scan, store: ResourceStore) -> list[Flag]: ...


# Global registry. Checks self-register via @register so the pipeline and the
# server can list them without importing each module by hand.
_REGISTRY: dict[str, Check] = {}


def register(check: Check) -> Check:
    if check.check_id in _REGISTRY:
        raise ValueError(f"duplicate check_id: {check.check_id}")
    _REGISTRY[check.check_id] = check
    return check


def all_checks() -> list[Check]:
    return list(_REGISTRY.values())


def get_check(check_id: str) -> Check:
    return _REGISTRY[check_id]
