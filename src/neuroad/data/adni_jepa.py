"""
Real-data feeder: ADNI with REAL frozen Neuro-JEPA embeddings + plasma + labels.

The 590-subject ADNI counterpart to ``oasis_jepa`` (the OASIS-1 disease-signal
feeder, n=61). Each row pairs the 768-d frozen Neuro-JEPA embedding of a real
ADNI T1w MPRAGE with the subject's diagnosis (CN/AD), acquisition site, and REAL
plasma biomarkers (p-tau217, GFAP, NfL) from the gated LONI export — so the ONE
reused probe can be pointed at AD-vs-CN on the foundation model's representation
over an order of magnitude more subjects than OASIS, WITH a molecular anchor.

Why this feeder matters (vs the thin OASIS n=61): 590 subjects (87 AD / 503 CN),
multi-site (so the site-leakage gauntlet test is INFORMATIVE here, unlike the
single-scanner OASIS/OpenBHB feeders), and complete real plasma on every row (so
the biomarker-anchor HARD GATE actually has data). Scanner is uniform 3T, so the
scanner-leakage test is NA; site leakage is the live confound here.

Compliance: frozen inference only (no fine-tuning; not a derivative of the gated
CC-BY-NC-ND Neuro-JEPA weights). Weights never stored in-repo; the embedding
table is git-ignored (kept local, per the ADNI DUA). Provenance: frozen
``NYUMedML/Neuro-JEPA`` over real ADNI T1w MPRAGE volumes; labels + plasma from
the gated LONI ADNI export.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neuroad import contract

_REPO_ROOT = Path(__file__).resolve().parents[3]
EMBEDDINGS_CSV = _REPO_ROOT / "data" / "real" / "adni_neurojepa_embeddings.csv"

_SEX_MAP = {"F": "F", "M": "M", "female": "F", "male": "M", "Female": "F",
            "Male": "M"}
_DX_MAP = {"CN": "CN", "AD": "AD", "MCI": "MCI", "Dementia": "AD",
           "LMCI": "MCI", "EMCI": "MCI", "SMC": "CN"}


def _load_apoe4_map() -> dict[str, int]:
    """RID -> APOE ε4 allele count from a gated APOERES export, if present.

    Optional + offline-safe: searches a few candidate download locations for an
    ``APOERES*.csv`` (gitignored gated file). apoe4 = number of '4' alleles in the
    ``GENOTYPE`` string (e.g. '3/4' -> 1, '4/4' -> 2). Returns ``{}`` when the file
    is absent (CI / fresh clone), so ``apoe4`` simply stays NA — never fabricated.
    """
    candidates = list(_REPO_ROOT.parent.glob("download*/APOERES*.csv"))
    candidates += list((_REPO_ROOT / "data" / "real" / "_gated").glob("APOERES*.csv"))
    for path in candidates:
        try:
            a = pd.read_csv(path, dtype=str, usecols=["RID", "GENOTYPE"])
            a = a.dropna(subset=["RID"])
            a["apoe4"] = a["GENOTYPE"].fillna("").str.count("4").astype(int)
            # one value per RID (genotype is stable across visits); keep first.
            return dict(a.drop_duplicates("RID").set_index("RID")["apoe4"])
        except Exception:
            continue
    return {}


def load_adni_neurojepa() -> pd.DataFrame:
    """Map the cached real ADNI Neuro-JEPA embedding table into a contract table.

    Raises FileNotFoundError with regeneration instructions if the cache is absent
    (git-ignored by design; gated ADNI + CC-BY-NC-ND weights). The referee runs
    fully without it via the other feeders.
    """
    if not EMBEDDINGS_CSV.exists():
        raise FileNotFoundError(
            f"ADNI Neuro-JEPA embedding cache not found at {EMBEDDINGS_CSV}.\n"
            "Git-ignored by design (gated ADNI export + CC-BY-NC-ND weights; the "
            "derived table is kept local). Regenerate on a GPU runtime with your "
            "own HF_TOKEN + LONI-downloaded ADNI T1w volumes:\n"
            "  colab start --gpu t4\n"
            "  colab exec --session <id> scripts/adni_colab_dicom_to_embed.py\n"
            "The referee runs without it via the 'oasis' / 'openbhb' / 'synthetic' "
            "feeders."
        )

    raw = pd.read_csv(EMBEDDINGS_CSV)
    emb_cols = [c for c in raw.columns if c.startswith("emb_")]
    E = raw[emb_cols].astype(float)
    Z = (E - E.mean()) / E.std(ddof=0).replace(0.0, 1.0)
    frame = contract.make_embedding_frame(Z.to_numpy())

    frame.insert(0, "subject_id", raw["subject_id"].astype(str).to_numpy())
    dx = raw["dx"].astype(str).map(lambda v: _DX_MAP.get(v, v))
    frame["dx"] = pd.Categorical(dx, categories=contract.DX_LEVELS)
    frame["age"] = pd.to_numeric(raw["age"], errors="coerce").to_numpy(dtype=float)
    frame["sex"] = pd.Categorical(raw["sex"].astype(str).map(_SEX_MAP),
                                  categories=contract.SEX_LEVELS)
    # Multi-site (site-leakage test is INFORMATIVE); single 3T scanner (scanner
    # test NA). Site codes are numeric in the export — carry them as string labels.
    frame["site"] = pd.Categorical(raw["site"].astype(str))
    frame["scanner"] = pd.Categorical(
        raw["scanner"].astype(str) if "scanner" in raw.columns
        else ["3T"] * len(raw))

    n = len(frame)
    frame["conversion"] = pd.array([pd.NA] * n, dtype="Int8")
    # REAL plasma from the gated ADNI export — every row populated.
    for col in ("p_tau217", "gfap", "nfl"):
        frame[col] = (pd.to_numeric(raw[col], errors="coerce").to_numpy(dtype=float)
                      if col in raw.columns else np.full(n, np.nan))
    if "amyloid" in raw.columns:
        amy = pd.to_numeric(raw["amyloid"], errors="coerce").astype("Int8")
        frame["amyloid"] = pd.array(amy, dtype="Int8")
    else:
        frame["amyloid"] = pd.array([pd.NA] * n, dtype="Int8")
    # apoe4 from the gated APOERES export when available (else NA — never faked).
    # Completes the 6-feature plasma/tabular block so the fusion + cross-attention
    # views can run on this cohort rather than degrading to emb-only.
    apoe4_map = _load_apoe4_map()
    if apoe4_map:
        vals = frame["subject_id"].map(lambda r: apoe4_map.get(str(r)))
        frame["apoe4"] = pd.array(pd.to_numeric(vals, errors="coerce"), dtype="Int8")
    else:
        frame["apoe4"] = pd.array([pd.NA] * n, dtype="Int8")

    frame = frame.drop_duplicates("subject_id", keep="first").reset_index(drop=True)
    contract.validate_table(frame)
    return frame
