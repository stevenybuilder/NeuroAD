"""neuroad.integrations — external tool adapters for the translation layer.

Four offline-first adapters that turn a promoted imaging survivor into the
molecule / wet-lab side of the plan (AlphaFold, PI4AD, a multimodal biomarker
predictor, and a GNN/LLM repurposing engine). Each adapter follows the house
contract: a real access path PLUS a deterministic, clearly-labeled fallback, so
the engine never hard-depends on network or gated credentials. Every returned
record stamps its own provenance (``source`` = "live" / "offline_snapshot" /
"surrogate") — a fallback is never dressed up as live/published data.

Access at a glance (verified 2026-07-10):
  * AlphaFold  — EBI AlphaFold DB REST API, keyless & live; AlphaFold-3 folding
                 needs gated non-commercial weights (documented, not wired).
  * PI4AD      — genetictargets.com portal (live HTTP) + bundled priority
                 snapshot; the R package is not run (no R, by design).
  * fusion     — vkola-lab/ncomms2025 checkpoint (torch + adrd, gated) with a
                 transparent logistic SURROGATE as the shipping default.
  * gnn_llm    — curated repurposing snapshot + optional live TxGNN scaffold
                 (torch + PyG, heavy); LLM rationale reuses claude._client.

Import is lazy-safe: importing this package pulls in only the adapter modules,
which themselves import torch/adrd/PyG lazily inside their live paths.
"""
from __future__ import annotations

from .alphafold import (
    AlphaFoldClient,
    AlphaFoldStructure,
    structural_confidence,
)
from .pi4ad import (
    GenePriority,
    PI4AD,
    gene_priority,
    rank_ad_targets,
)
from .multimodal_transformer import (
    BiomarkerFusionPredictor,
    FusionPrediction,
    predict_biomarker_status,
)
from .gnn_llm import (
    RepurposingCandidate,
    RepurposingEngine,
)

__all__ = [
    # AlphaFold — structural layer
    "AlphaFoldClient",
    "AlphaFoldStructure",
    "structural_confidence",
    # PI4AD — target/gene prioritization
    "PI4AD",
    "GenePriority",
    "rank_ad_targets",
    "gene_priority",
    # Multimodal transformer — Aβ/τ biomarker prediction
    "BiomarkerFusionPredictor",
    "FusionPrediction",
    "predict_biomarker_status",
    # GNN/LLM — drug repurposing
    "RepurposingEngine",
    "RepurposingCandidate",
]
