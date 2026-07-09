"""
Real-data feeder: OpenBHB with REAL frozen Neuro-JEPA embeddings.

This is the ``openbhb`` feeder's twin, but the "embedding" is no longer the 4
weight-free structural morphometry features (tiv/csfv/gmv/wmv) — it is the actual
768-dimensional frozen Neuro-JEPA representation of each subject's MNI152 T1w
volume, produced off-box on a GPU (see ``scripts/neurojepa_embed.py`` /
``scripts/openbhb_embed.py`` run on Colab) and cached to
``data/real/openbhb_neurojepa_embeddings.csv``.

Why this matters
----------------
It closes the gap between the pitch and what runs: the referee is deliberately
encoder-agnostic (a table swap), so pointing the SAME reused head at ``scanner``
on these REAL Neuro-JEPA embeddings measures whether the foundation model's
representation itself carries the batch effect the referee gates against — on
real, healthy, multi-site brains, with no disease signal to confound it.

Compliance
----------
Frozen inference only (no fine-tuning -> not a derivative of the CC-BY-NC-ND
weights). The weights are never stored in this repo; only the small derived
embedding table is cached, and that table is git-ignored (never redistributed).
If the cache is absent (e.g. a fresh clone with no GPU/HF access), this feeder
raises a clear error pointing at the reproduction script rather than failing
cryptically — the referee still runs fully on the ``openbhb`` / ``synthetic``
feeders without it.

Provenance
----------
Embeddings: frozen ``NYUMedML/Neuro-JEPA`` (JEPA + Mixture-of-Experts) over OpenBHB quasi-raw
MNI152 volumes (``huggingface.co/datasets/benoit-dufumier/openBHB``, Apache-2.0).
Metadata (age/sex/site/field-strength): OpenBHB ``participants.tsv``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neuroad import contract

_REPO_ROOT = Path(__file__).resolve().parents[3]
EMBEDDINGS_CSV = _REPO_ROOT / "data" / "real" / "openbhb_neurojepa_embeddings.csv"

_SEX_MAP = {"female": "F", "male": "M"}


def _scanner_label(field_strength: float) -> str:
    if pd.isna(field_strength):
        return pd.NA
    return f"{float(field_strength):g}T"


def load_openbhb_neurojepa() -> pd.DataFrame:
    """Map the cached real Neuro-JEPA embedding table into a contract table.

    Returns
    -------
    pd.DataFrame  passing ``contract.validate_table`` (dx='CN' throughout;
    healthy-control cohort, so the value is the scanner/site leakage star on a
    real foundation-model representation).

    Raises
    ------
    FileNotFoundError  if the embedding cache is missing, with instructions to
    regenerate it via the GPU embedding script.
    """
    if not EMBEDDINGS_CSV.exists():
        raise FileNotFoundError(
            f"Neuro-JEPA embedding cache not found at {EMBEDDINGS_CSV}.\n"
            "It is git-ignored by design (the weights are gated CC-BY-NC-ND and "
            "the derived table is kept local). To regenerate on a GPU runtime "
            "with gated HF access:\n"
            "  export HF_TOKEN=...   # your own gated Neuro-JEPA grant\n"
            "  python scripts/openbhb_embed.py   # (Colab T4 via `colab exec`)\n"
            "The referee runs fully without it via the 'openbhb' / 'synthetic' feeders."
        )

    raw = pd.read_csv(EMBEDDINGS_CSV)
    emb_cols = [c for c in raw.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError(f"{EMBEDDINGS_CSV} has no emb_* columns")

    # Standardize each embedding dimension (z-score), matching the openbhb feeder.
    E = raw[emb_cols].astype(float)
    Z = (E - E.mean()) / E.std(ddof=0).replace(0.0, 1.0)
    frame = contract.make_embedding_frame(Z.to_numpy())

    frame.insert(0, "subject_id",
                 ("BHB_" + raw["participant_id"].astype(str)).to_numpy())
    frame["dx"] = pd.Categorical(["CN"] * len(raw), categories=contract.DX_LEVELS)
    frame["age"] = raw["age"].to_numpy(dtype=float)
    frame["sex"] = pd.Categorical(raw["sex"].map(_SEX_MAP), categories=contract.SEX_LEVELS)
    frame["site"] = pd.Categorical("BHB_" + raw["site"].astype(float).astype("Int64").astype(str))
    frame["scanner"] = pd.Categorical(raw["field_strength"].map(_scanner_label))

    n = len(frame)
    frame["conversion"] = pd.array([pd.NA] * n, dtype="Int8")
    frame["amyloid"] = pd.array([pd.NA] * n, dtype="Int8")
    frame["p_tau217"] = np.full(n, np.nan, dtype="float64")
    frame["gfap"] = np.full(n, np.nan, dtype="float64")
    frame["nfl"] = np.full(n, np.nan, dtype="float64")
    frame["apoe4"] = pd.array([pd.NA] * n, dtype="Int8")

    frame = frame.drop_duplicates("subject_id", keep="first").reset_index(drop=True)
    contract.validate_table(frame)
    return frame
