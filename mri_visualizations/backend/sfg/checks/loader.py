"""Imports built-in check modules so they self-register via @register.

Add a line here as each check module is built; nothing else needs to know the
module names. Kept separate from base.py to avoid import cycles.
"""

from __future__ import annotations


def load_builtin_checks() -> None:
    # Each import triggers @register at module load. Add checks here as built.
    from . import (
        intensity,  # noqa: F401
        orientation,  # noqa: F401
        registration,  # noqa: F401
        skullstrip,  # noqa: F401
        volume_sanity,  # noqa: F401
    )
