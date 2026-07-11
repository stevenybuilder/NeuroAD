"""
NeuroAD Discovery Engine — deterministic explanation layer.

These modules turn the referee's evidence into text — WITHOUT calling Claude.
Claude's ONLY role in the engine is the ORCHESTRATOR (``harness/agent.py``),
which sequences the engine's capabilities as tools; the referee itself is fully
deterministic, so every verdict/score/critique/narration below is pure Python:

  - claim_parser : natural-language hunch  -> structured, testable Claim (keywords)
  - courtroom    : Prosecution / Defense / Judge text from the gauntlet evidence
  - narrator     : plain-language verdict + the assumption that would break it
  - bridge       : promoted survivors -> ONE biomarker-routed mechanism + experiment
  - reviewer     : an adversarial critique of the card's own verdict

``_client`` now only holds the ORCHESTRATOR's model config + an honest
``model_badge``; it no longer drives these modules. Nothing here is random;
nothing here calls the network; nothing here crashes.
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
