"""
Real-data feeder: ADNI MCI-conversion cohort with REAL frozen Neuro-JEPA
embeddings + plasma + the sMCI/pMCI conversion label.

The 334-subject prognostic counterpart to ``adni_jepa`` (the 590-subject AD-vs-CN
disease-signal feeder). Every row is a *baseline-MCI* subject whose T1w MPRAGE was
embedded by the frozen Neuro-JEPA encoder on Colab
(``scripts/run_conversion_embed_colab.py`` -> ``adni_conversion_neurojepa_embeddings.csv``),
paired with the subject's REAL plasma panel (p-tau217, GFAP, NfL, amyloid) and,
crucially, the ``conversion`` outcome (pMCI converter=1 / sMCI stable=0) joined from
the gated LONI contract export.

Why a SEPARATE feeder from ``adni:neurojepa``: that cohort is cross-sectional AD vs
CN, where three independent methods agree plasma p-tau217 dominates (~0.93) and the
MRI embedding adds little on top. The prognostic MCI->AD **conversion** arm is the
regime where structural imaging is expected to earn its keep — so this feeder points
the ONE reused probe at ``target="conversion"`` on the foundation model's own
representation, with the plasma anchor present for an honest imaging-vs-plasma
contrast. It is multi-site (58 sites), so the site-leakage test is INFORMATIVE and a
site-disjoint (leave-one-site-out) split is the honest single-cohort generalization
test. The cohort is small (58 converters) — genuinely underpowered — and any result
is reported as such.

Compliance: frozen inference only (no fine-tuning; not a derivative of the gated
CC-BY-NC-ND Neuro-JEPA weights). Weights never stored in-repo; the embedding table
and the gated contract are git-ignored (kept local, per the ADNI DUA). Provenance:
frozen ``NYUMedML/Neuro-JEPA`` over real ADNI T1w MPRAGE volumes; labels + plasma
from the gated LONI ADNI export.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neuroad import contract

_REPO_ROOT = Path(__file__).resolve().parents[3]
EMBEDDINGS_CSV = (_REPO_ROOT / "data" / "real"
                  / "adni_conversion_neurojepa_embeddings.csv")
#: The gated LONI contract carries the authoritative conversion label + APOE.
_GATED_CSV = _REPO_ROOT / "data" / "real" / "_gated" / "adni.csv"

_SEX_MAP = {"F": "F", "M": "M", "female": "F", "male": "M", "Female": "F",
            "Male": "M"}
_DX_MAP = {"CN": "CN", "AD": "AD", "MCI": "MCI", "Dementia": "AD",
           "LMCI": "MCI", "EMCI": "MCI", "SMC": "CN"}


def _load_gated_labels() -> tuple[dict[str, int], dict[str, int]]:
    """Return (subject_id -> conversion, subject_id -> apoe4) from the gated export.

    The conversion outcome (pMCI=1 / sMCI=0) is NOT in the embedding CSV — it lives
    in the gated contract (``data/real/_gated/adni.csv``), keyed by ``subject_id``.
    APOE e4 count is joined from the same source. Returns ``({}, {})`` when the gated
    file is absent (CI / fresh clone), so ``conversion``/``apoe4`` simply stay NA —
    never fabricated.
    """
    if not _GATED_CSV.exists():
        return {}, {}
    try:
        g = pd.read_csv(_GATED_CSV, low_memory=False,
                        usecols=lambda c: c in ("subject_id", "conversion", "apoe4"))
    except Exception:
        return {}, {}
    g = g.dropna(subset=["subject_id"]).copy()
    g["subject_id"] = g["subject_id"].astype(str)
    g = g.drop_duplicates("subject_id", keep="first").set_index("subject_id")
    conv = (pd.to_numeric(g["conversion"], errors="coerce")
            if "conversion" in g.columns else pd.Series(dtype=float))
    apoe = (pd.to_numeric(g["apoe4"], errors="coerce")
            if "apoe4" in g.columns else pd.Series(dtype=float))
    conv_map = {k: int(v) for k, v in conv.dropna().items()}
    apoe_map = {k: int(v) for k, v in apoe.dropna().items()}
    return conv_map, apoe_map


def load_adni_conversion_neurojepa() -> pd.DataFrame:
    """Map the cached ADNI MCI-conversion Neuro-JEPA table into a contract table.

    Raises FileNotFoundError with regeneration instructions if the embedding cache
    is absent (git-ignored by design; gated ADNI + CC-BY-NC-ND weights). The referee
    runs fully without it via the other feeders.
    """
    if not EMBEDDINGS_CSV.exists():
        raise FileNotFoundError(
            f"ADNI conversion Neuro-JEPA embedding cache not found at "
            f"{EMBEDDINGS_CSV}.\n"
            "Git-ignored by design (gated ADNI export + CC-BY-NC-ND weights; the "
            "derived table is kept local). Regenerate on a GPU runtime with your own "
            "HF_TOKEN + the LONI-downloaded MCI-conversion T1w volumes:\n"
            "  colab start --gpu a100\n"
            "  colab exec --session <id> scripts/run_conversion_embed_colab.py\n"
            "The referee runs without it via the 'oasis' / 'openbhb' / 'adni' feeders."
        )

    raw = pd.read_csv(EMBEDDINGS_CSV)
    emb_cols = [c for c in raw.columns if c.startswith("emb_")]
    E = raw[emb_cols].astype(float)
    Z = (E - E.mean()) / E.std(ddof=0).replace(0.0, 1.0)
    frame = contract.make_embedding_frame(Z.to_numpy())

    sid = raw["subject_id"].astype(str)
    frame.insert(0, "subject_id", sid.to_numpy())
    # These are all baseline-MCI subjects; normalize whatever casing/coding arrived.
    dx = raw["dx"].astype(str).map(lambda v: _DX_MAP.get(v, v))
    frame["dx"] = pd.Categorical(dx, categories=contract.DX_LEVELS)
    frame["age"] = pd.to_numeric(raw["age"], errors="coerce").to_numpy(dtype=float)
    frame["sex"] = pd.Categorical(raw["sex"].astype(str).map(_SEX_MAP),
                                  categories=contract.SEX_LEVELS)
    # Multi-site (site-leakage test INFORMATIVE / site-disjoint LOSO is the honest
    # generalization split); uniform 3T scanner (scanner-leakage test NA).
    frame["site"] = pd.Categorical(raw["site"].astype(str))
    frame["scanner"] = pd.Categorical(
        raw["scanner"].astype(str) if "scanner" in raw.columns
        else ["3T"] * len(raw))

    n = len(frame)
    # REAL plasma from the gated ADNI export, carried on the embedding CSV.
    for col in ("p_tau217", "gfap", "nfl"):
        frame[col] = (pd.to_numeric(raw[col], errors="coerce").to_numpy(dtype=float)
                      if col in raw.columns else np.full(n, np.nan))
    if "amyloid" in raw.columns:
        amy = pd.to_numeric(raw["amyloid"], errors="coerce").astype("Int8")
        frame["amyloid"] = pd.array(amy, dtype="Int8")
    else:
        frame["amyloid"] = pd.array([pd.NA] * n, dtype="Int8")

    # The prognostic label + APOE come from the gated contract (keyed by subject_id).
    # This is the whole point of the feeder: conversion (pMCI=1 / sMCI=0) is NOT in
    # the embedding CSV. Absent gated file -> NA (never faked).
    conv_map, apoe_map = _load_gated_labels()
    conv_vals = frame["subject_id"].map(lambda s: conv_map.get(str(s)))
    frame["conversion"] = pd.array(pd.to_numeric(conv_vals, errors="coerce"),
                                   dtype="Int8")
    apoe_vals = frame["subject_id"].map(lambda s: apoe_map.get(str(s)))
    frame["apoe4"] = pd.array(pd.to_numeric(apoe_vals, errors="coerce"),
                              dtype="Int8")

    frame = frame.drop_duplicates("subject_id", keep="first").reset_index(drop=True)
    contract.validate_table(frame)
    return frame
