"""
NeuroAD harness — the L5 layer that turns the referee ENGINE into a discovery
INSTRUMENT a researcher drives from a hypothesis.

Pieces (each self-contained, offline-deterministic, no network):
  * policy           — the L3 policy LOADER: read policy/ docs as tables /
                       thresholds / briefs, each with a hardcoded fallback,
  * discovery_router — decide supervised probe vs unsupervised Detective,
  * experiment_card  — wrap a ClaimCard into the researcher-facing artifact,
  * orchestrator     — the L5 entry point: investigate("<hypothesis>", dataset)
                       chains parse -> route -> referee -> anchor gate -> card,
                       and enforces the HONESTY GUARD before returning.

Everything here reads today's hardcoded constants (contract.VERDICT_BANDS,
calibration.CAL, bridge._MECHANISMS) directly, so the package works with or
without the optional L3 `policy/` documents.
"""
from __future__ import annotations

__all__ = ["policy", "discovery_router", "experiment_card", "orchestrator"]
