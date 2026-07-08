"""
NeuroAD Discovery Engine — Claude reasoning layer (M3).

Claude is the ADJUDICATOR, not decoration. Each module below makes a
*consequential* decision in the referee loop:

  - claim_parser : natural-language hunch  -> structured, testable Claim
  - courtroom    : Prosecution / Defense / Judge personas render the verdict
  - narrator     : plain-language verdict + the assumption that would break it
  - bridge       : promoted survivors -> ONE biomarker-routed mechanism + experiment
  - reviewer     : argues AGAINST the tool's own verdict (self-referee)

Every module runs live against the Anthropic Messages API when
``ANTHROPIC_API_KEY`` is set (model ``claude-fable-5``, falling back to
``claude-opus-4-8`` / ``claude-sonnet-5``), and otherwise returns a
deterministic, literature-grounded TEMPLATE result so the demo runs fully
offline. Nothing here is random; nothing here crashes.
"""
from __future__ import annotations

from . import _client, claim_parser, courtroom, narrator, bridge, reviewer

__all__ = [
    "_client",
    "claim_parser",
    "courtroom",
    "narrator",
    "bridge",
    "reviewer",
]
