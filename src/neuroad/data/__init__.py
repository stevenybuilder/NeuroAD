"""NeuroAD Discovery Engine — data layer (M2).

Three interchangeable feeders satisfy the embedding-table CONTRACT:
  * ``synthetic.generate_cohort`` — offline, deterministic, ground-truth
    scanner confound + biomarker anchor (the guaranteed live path),
  * ``real.load_oasis`` — real OASIS-1/2 structural-derived features,
  * ``loaders.load`` — a one-name dispatch over both.
"""
from neuroad.data import synthetic, real, loaders  # noqa: F401
