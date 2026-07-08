"""
Real-data feeder: OASIS-1 with REAL frozen Neuro-JEPA embeddings + AD labels.

The disease-signal counterpart to ``openbhb_jepa`` (which carries the scanner-leakage
star). Here the 768-d frozen Neuro-JEPA embedding of each OASIS-1 subject's
atlas-registered, brain-masked T1w volume is paired with the real CDR-derived
diagnosis, so the ONE reused probe can be pointed at AD-vs-CN on the foundation
model's own representation.

Real result (n=61, see reports/oasis_neurojepa_ad.json): on the clean clinical
contrast (CDR>=1 AD vs CDR=0 CN) the frozen embedding separates AD from CN at
AUC ~0.81 — matching direct structural morphometry (~0.82) and confirming the
Neuro-JEPA representation carries the real disease signal. The very-mild/questionable
CDR=0.5 cases are structurally subtle and dilute the contrast (~0.61), which is an
honest, expected finding rather than a failure.

dx mapping: CDR 0 -> CN, CDR 0.5 -> MCI (questionable), CDR >=1 -> AD. So the
referee's ``dx_binary`` (AD vs CN) uses the clean clinical contrast and excludes
the questionable MCI band.

Compliance: frozen inference only (no fine-tuning; not a derivative of the gated
CC-BY-NC-ND weights). Weights never stored in-repo; the embedding table is
git-ignored (kept local). Single-cohort/single-scanner, so the scanner-leakage and
cross-cohort replication tests are NA here — this feeder demonstrates the real
disease signal; promotion/replication use the combined ``oasis`` feeder.

Provenance: frozen ``NYUMedML/Neuro-JEPA`` over OASIS-1 PROCESSED
``t88_masked_gfc`` images (download.nrg.wustl.edu, no-login); labels from OASIS-1
``oasis_cross-sectional.csv`` (CDR).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neuroad import contract

_REPO_ROOT = Path(__file__).resolve().parents[3]
EMBEDDINGS_CSV = _REPO_ROOT / "data" / "real" / "oasis1_neurojepa_embeddings.csv"

_SEX_MAP = {"F": "F", "M": "M", "female": "F", "male": "M"}


def _dx_from_cdr(cdr: float) -> str:
    if cdr == 0:
        return "CN"
    if cdr >= 1:
        return "AD"
    return "MCI"  # CDR 0.5 = very mild / questionable


def load_oasis_neurojepa() -> pd.DataFrame:
    """Map the cached real OASIS-1 Neuro-JEPA embedding table into a contract table.

    Raises FileNotFoundError with regeneration instructions if the cache is absent
    (git-ignored by design). The referee runs fully without it via other feeders.
    """
    if not EMBEDDINGS_CSV.exists():
        raise FileNotFoundError(
            f"OASIS-1 Neuro-JEPA embedding cache not found at {EMBEDDINGS_CSV}.\n"
            "Git-ignored by design (gated CC-BY-NC-ND weights; derived table kept "
            "local). Regenerate on a GPU runtime with your own HF_TOKEN:\n"
            "  export HF_TOKEN=...\n"
            "  # run scripts/openbhb_embed.py's OASIS analogue on Colab (see docs/HF_ACCESS.md)\n"
            "The referee runs without it via the 'oasis' / 'openbhb' / 'synthetic' feeders."
        )

    raw = pd.read_csv(EMBEDDINGS_CSV)
    emb_cols = [c for c in raw.columns if c.startswith("emb_")]
    E = raw[emb_cols].astype(float)
    Z = (E - E.mean()) / E.std(ddof=0).replace(0.0, 1.0)
    frame = contract.make_embedding_frame(Z.to_numpy())

    frame.insert(0, "subject_id", raw["participant_id"].astype(str).to_numpy())
    dx = raw["cdr"].astype(float).map(_dx_from_cdr)
    frame["dx"] = pd.Categorical(dx, categories=contract.DX_LEVELS)
    frame["age"] = raw["age"].to_numpy(dtype=float)
    frame["sex"] = pd.Categorical(raw["sex"].map(_SEX_MAP), categories=contract.SEX_LEVELS)
    # single cohort / single scanner -> site & scanner uninformative (NA-like constants)
    frame["site"] = pd.Categorical(["OASIS1"] * len(raw))
    frame["scanner"] = pd.Categorical(["OASIS1_MR1"] * len(raw))

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
