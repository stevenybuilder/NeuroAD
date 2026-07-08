"""
Real-data feeder: OpenBHB (Open Big Healthy Brains) -> ONE contract table.

Why this dataset is here
------------------------
OpenBHB is a large multi-site cohort of **healthy controls only** (no dementia,
no AD labels). That sounds useless for an AD referee — until you realize it is
the cleanest possible demonstration of the STAR leakage mechanic on REAL data.

Because *every* subject is a healthy control, there is by construction NO disease
signal in the structural morphometry. So if a linear probe on those structural
features can still tell 3T from 1.5T scanners at AUC ~0.90, that AUC is pure
acquisition physics — the exact batch effect the Referee gates against. On the
synthetic KILL cohort we *inject* this confound; here it falls out of real,
published, healthy-brain data with nothing injected.

The weight-free "embedding" is the standardized structural-derived feature set
[tiv, csfv, gmv, wmv] (total intracranial / CSF / grey-matter / white-matter
volume). We deliberately do NOT put ``age`` in the embedding — age is a covariate
the gauntlet adjusts for, not a structural feature, and folding it in would blur
the "structure alone leaks the scanner" point.

Honest caveats surfaced by this feeder:
  * Healthy controls ONLY -> ``dx`` is 'CN' for every subject, so the disease /
    conversion probe cannot run on this table (that is expected). The point of
    this feeder is ``real_scanner_leakage()``, not a diagnosis.
  * No plasma p-tau217 / GFAP / NfL / amyloid / APOE -> those biomarker columns
    are all <NA>.

Label mapping:
  dx:       'CN' for ALL (healthy-controls cohort).
  sex:      'female' -> 'F', 'male' -> 'M'.
  site:     'BHB_' + integer site code (62 acquisition sites).
  scanner:  magnetic field strength label, e.g. '1.5T' / '3.0T' — the REAL
            multi-scanner label the STAR (site/scanner leakage) test needs.

Provenance
----------
Vendored derivative table ``data/real/openbhb_participants.tsv`` (TAB-separated,
3984 healthy-control rows). No-login HuggingFace mirror
``huggingface.co/datasets/benoit-dufumier/openBHB`` (participants.tsv),
Apache-2.0. Verified live 2026-07-08.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neuroad import contract
from neuroad import calibration as cal
from neuroad.probe import cross_val_auc, point_head

# Repo layout: .../src/neuroad/data/openbhb.py -> repo root is parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_REAL_DIR = _REPO_ROOT / "data" / "real"
OPENBHB_TSV = _REAL_DIR / "openbhb_participants.tsv"

#: Structural-derived features used as the weight-free embedding (emb_0..emb_3).
#: NOTE: age is intentionally EXCLUDED — it is a covariate, not a structural
#: feature, and the whole point is that structure ALONE leaks the scanner.
_STRUCTURAL_FEATURES = ["tiv", "csfv", "gmv", "wmv"]

_SEX_MAP = {"female": "F", "male": "M"}


def _scanner_label(field_strength: float) -> str:
    """Field-strength (Tesla) -> a stable scanner label like '1.5T' / '3.0T'."""
    if pd.isna(field_strength):
        return pd.NA
    return f"{float(field_strength):g}T"


def load_openbhb() -> pd.DataFrame:
    """Map the vendored OpenBHB participants table into a contract table.

    Every subject is a healthy control, so ``dx`` is 'CN' throughout and the
    disease probe cannot run — by design. The value of this table is the real
    scanner/site leakage star exposed by :func:`real_scanner_leakage`.

    Returns
    -------
    pd.DataFrame  passing ``contract.validate_table``.
    """
    raw = pd.read_csv(OPENBHB_TSV, sep="\t")

    # --- structural-derived embedding (standardized), age deliberately absent -
    feats = raw[_STRUCTURAL_FEATURES].astype(float)
    Z = (feats - feats.mean()) / feats.std(ddof=0)
    frame = contract.make_embedding_frame(Z.to_numpy())

    # --- identity + labels ---------------------------------------------------
    frame.insert(0, "subject_id",
                 ("BHB_" + raw["participant_id"].astype(str)).to_numpy())

    # Healthy-controls cohort -> every subject is CN.
    frame["dx"] = pd.Categorical(["CN"] * len(raw), categories=contract.DX_LEVELS)

    frame["age"] = raw["age"].to_numpy(dtype=float)
    sex = raw["sex"].map(_SEX_MAP)
    frame["sex"] = pd.Categorical(sex, categories=contract.SEX_LEVELS)

    # site: 'BHB_' + integer site code (62 acquisition sites).
    site = "BHB_" + raw["site"].astype(float).astype("Int64").astype(str)
    frame["site"] = pd.Categorical(site)

    # scanner: field-strength label ('1.5T' / '3.0T') — the REAL multi-scanner
    # label the STAR test points the probe at.
    scanner = raw["magnetic_field_strength"].map(_scanner_label)
    frame["scanner"] = pd.Categorical(scanner)

    # --- no disease/conversion or molecular markers in a healthy-control set --
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


def real_scanner_leakage(df: pd.DataFrame | None = None) -> dict:
    """Point the ONE reused head at ``scanner`` (field strength) and ``site`` on
    real OpenBHB healthy brains and report the leakage AUCs.

    This is the STAR mechanic on REAL data: the subjects are ALL healthy controls,
    so there is no disease signal to find — yet the structural embedding still
    predicts which machine acquired the scan. That AUC is the batch effect the
    Referee gates against, measured (not injected).

    Returns a dict with the scanner/site AUCs, sample size, cited prior art, and
    a plain-language message.
    """
    if df is None:
        df = load_openbhb()

    results: dict[str, dict] = {}
    for target in ("scanner", "site"):
        X, y, _ = point_head(df, target)
        # No group-aware CV here — holding out the very group you predict is
        # degenerate; we WANT to see the machine/site signal (see probe.point_head).
        auc = cross_val_auc(X, y, groups=None)
        results[target] = {
            "auc": round(float(auc), 4),
            "n": int(len(y)),
            "n_classes": int(len(np.unique(y))),
        }

    scanner_auc = results["scanner"]["auc"]
    message = (
        f"The structural embedding predicts the scanner (field strength) at "
        f"AUC {scanner_auc:.3f} in {results['scanner']['n']} healthy subjects "
        f"with NO disease — this is the batch effect the referee gates against, "
        f"on REAL data (not synthetic). Prior art: "
        f"'{cal.PRIOR_ART[0][0]}' ({cal.PRIOR_ART[0][1]})."
    )

    return {
        "dataset": "openbhb",
        "healthy_controls_only": True,
        "scanner_auc": scanner_auc,
        "site_auc": results["site"]["auc"],
        "detail": results,
        "prior_art": [
            {"title": t, "cite": c, "note": note} for (t, c, note) in cal.PRIOR_ART
        ],
        "message": message,
    }
