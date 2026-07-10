"""
multimodal_transformer — per-subject Abeta & tau PET status from fused features.

Adapter over the Jasodanand et al. multimodal fusion transformer (Nat Commun
2025, doi 10.1038/s41467-025-62590-4; code vkola-lab/ncomms2025), which predicts
individual amyloid (amy_label) and tau (tau_label) PET status/burden from fused
MRI volumetrics, cognitive scores, demographics, and plasma biomarkers. The REAL
path lazily git-clones/points at that repo and loads its committed torch
checkpoint (dev/ckpt/model_stage_1.ckpt, Abeta+meta-tau, ungated, no creds) via
``adrd.model.ADRDModel.from_ckpt`` — see ``from_pretrained``.

OFFLINE / DETERMINISTIC CONTRACT: imports and runs with NO network, NO
credentials, and NO torch. The shipping DEFAULT is a transparent, HAND-SET
logistic SURROGATE over the ~5 plasma/volumetric features the existing ADNI
contract already carries (p_tau217, gfap, hippocampal_volume, age, apoe4). It is
honestly labelled ``model="surrogate_logistic"``, ``source="offline_surrogate"``
and is a STAND-IN for — never a reproduction of — the published fusion
transformer: its coefficients encode biomarker directionality (domain knowledge),
are not fitted to data, and are not calibrated to any published AUROC. This
mirrors ``neuroad.probe``'s honest-labelling philosophy. Heavy real-path deps
(torch, monai, adrd) are imported LAZILY inside ``from_pretrained``/``_predict_real``
and any absence degrades to the surrogate — never an ImportError at module import.

Credentials: none required (weights are public in-repo). Optional env vars:
``NCOMMS2025_REPO`` (path to a local clone of vkola-lab/ncomms2025) and
``NCOMMS2025_CKPT`` (path to model_stage_1.ckpt) let the real path skip cloning.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import pandas as pd

# ---------------------------------------------------------------------------
# Bundled, versioned surrogate spec (hand-set coefficients + standardization).
# ---------------------------------------------------------------------------
_SPEC_PATH = Path(__file__).with_name("data") / "multimodal_transformer_surrogate.json"
_HTTP_TIMEOUT = 15  # seconds — real-path fetch/clone must never hang the engine

#: Model + provenance tags (single source of truth for the honesty stamps).
MODEL_REAL = "jasodanand2025"
MODEL_SURROGATE = "surrogate_logistic"
SOURCE_LIVE = "live"
SOURCE_SURROGATE = "offline_surrogate"

#: Canonical repo + target checkpoint for the REAL path (recon-verified).
NCOMMS2025_REPO_URL = "https://github.com/vkola-lab/ncomms2025"
NCOMMS2025_CKPT_RELPATH = "dev/ckpt/model_stage_1.ckpt"

#: Alias map: harmonized / FreeSurfer-ish keys a caller might pass -> the
#: surrogate's canonical feature names. Lets a sparse contract row feed straight
#: in without the caller knowing the surrogate's exact key spelling.
_FEATURE_ALIASES: dict[str, str] = {
    "ptau217": "p_tau217",
    "p-tau217": "p_tau217",
    "plasma_ptau217": "p_tau217",
    "blood_ptau217": "p_tau217",
    "plasma_gfap": "gfap",
    "blood_gfap": "gfap",
    "plasma_nfl": "nfl",
    "blood_nfl": "nfl",
    "hippocampus_volume": "hippocampal_volume",
    "hippocampal_vol": "hippocampal_volume",
    "fs_mtl_volume": "hippocampal_volume",
    "hv": "hippocampal_volume",
    "apoe4_count": "apoe4",
    "apoe_e4": "apoe4",
    "age_years": "age",
}


def _load_spec() -> dict:
    """Load the bundled hand-set surrogate spec (coefficients + standardization)."""
    with open(_SPEC_PATH, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


_SPEC = _load_spec()


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# ---------------------------------------------------------------------------
# Structured return
# ---------------------------------------------------------------------------


@dataclass
class FusionPrediction:
    """Predicted Abeta & tau PET status/burden for one subject.

    ``source`` is the provenance stamp: ``"live"`` (real Jasodanand2025 fusion
    transformer) or ``"offline_surrogate"`` (the hand-set logistic stand-in).
    ``model`` is ``"jasodanand2025"`` or ``"surrogate_logistic"`` to match. Probs
    are calibrated only in the trivial sigmoid sense on the surrogate path — they
    are NOT the published model's probabilities and carry no validated accuracy.
    """
    abeta_status: bool
    abeta_prob: float
    tau_status: bool
    tau_prob: float
    model: str                                # "jasodanand2025" | "surrogate_logistic"
    source: str                               # "live" | "offline_surrogate"
    features_used: list[str] = field(default_factory=list)
    missing_features: list[str] = field(default_factory=list)
    threshold: float = 0.5
    error: str = ""                           # non-fatal note (e.g. why fell back)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "abeta_status": bool(self.abeta_status),
            "abeta_prob": self.abeta_prob,
            "tau_status": bool(self.tau_status),
            "tau_prob": self.tau_prob,
            "model": self.model,
            "source": self.source,
            "features_used": list(self.features_used),
            "missing_features": list(self.missing_features),
            "threshold": self.threshold,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# The predictor
# ---------------------------------------------------------------------------


class BiomarkerFusionPredictor:
    """Predict individual Abeta & tau PET status from fused multimodal features.

    Offline-first: the default instance runs the transparent hand-set logistic
    SURROGATE (no network, no torch, no creds) and stamps every prediction
    ``model="surrogate_logistic"``, ``source="offline_surrogate"``. Use the
    :meth:`from_pretrained` classmethod to attempt the REAL fusion transformer
    (vkola-lab/ncomms2025 model_stage_1.ckpt via ``adrd.model.ADRDModel``); if
    torch/adrd/monai or the checkpoint are unavailable, that classmethod degrades
    to the surrogate rather than raising, and ``predict`` still works.
    """

    def __init__(self, *, threshold: float = 0.5, spec: Optional[dict] = None) -> None:
        self.threshold = float(threshold)
        self._spec = spec or _SPEC
        #: Populated only when from_pretrained successfully loads real weights.
        self._real_model = None
        self._real_note = ""

    # -- input schema advertisement ---------------------------------------

    @property
    def expected_features(self) -> dict[str, str]:
        """The surrogate's input schema: feature name -> unit/meaning.

        These are the features the existing ADNI contract already carries, so the
        harness knows exactly what to pass. Any subset is accepted — a missing
        feature is masked (contributes nothing, mirroring the real transformer's
        native missing-feature masking) and reported in ``missing_features``.
        """
        std = self._spec.get("standardization", {})
        return {name: meta.get("note", "") for name, meta in std.items()}

    # -- normalization -----------------------------------------------------

    @staticmethod
    def _as_mapping(features: Union[dict, pd.Series]) -> dict:
        if isinstance(features, pd.Series):
            return {str(k): features[k] for k in features.index}
        return dict(features)

    def _canonicalize(self, raw: dict) -> dict[str, float]:
        """Map alias/casing variants onto canonical feature names, drop NaN/None."""
        out: dict[str, float] = {}
        for key, val in raw.items():
            if val is None:
                continue
            name = str(key).strip()
            canon = _FEATURE_ALIASES.get(name.lower(), name)
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if math.isnan(fval):
                continue
            # Last write wins, but prefer a canonical key over an alias if both.
            out[canon] = fval
        return out

    def _logit(self, target: str, feats: dict[str, float]) -> tuple[float, list[str], list[str]]:
        """Standardize + accumulate one target's logit. Missing feats are masked.

        Returns (logit, used, missing) where ``missing`` lists the target's
        expected features not supplied by the caller."""
        std = self._spec.get("standardization", {})
        block = self._spec.get(target, {})
        logit = float(block.get("bias", 0.0))
        used: list[str] = []
        missing: list[str] = []
        for name, coef in block.get("coefficients", {}).items():
            if name not in feats:
                missing.append(name)
                continue
            s = std.get(name, {})
            mean = float(s.get("mean", 0.0))
            scale = float(s.get("scale", 1.0)) or 1.0
            z = (feats[name] - mean) / scale
            logit += float(coef) * z
            used.append(name)
        return logit, used, missing

    # -- main entry --------------------------------------------------------

    def predict(self, features: Union[dict, pd.Series]) -> FusionPrediction:
        """Predict Abeta & tau status/burden for one subject's feature row.

        Tries the REAL fusion transformer first only if :meth:`from_pretrained`
        loaded it; on ANY real-path failure (and by default) falls back to the
        hand-set surrogate. Never raises on missing features/creds/torch — the
        result is always provenance-stamped."""
        raw = self._as_mapping(features)
        if self._real_model is not None:
            real = self._predict_real(raw)
            if real is not None:
                return real
        return self._predict_surrogate(raw)

    # -- surrogate path ----------------------------------------------------

    def _predict_surrogate(self, raw: dict, note: str = "") -> FusionPrediction:
        feats = self._canonicalize(raw)
        a_logit, a_used, a_missing = self._logit("abeta", feats)
        t_logit, t_used, t_missing = self._logit("tau", feats)
        a_prob = round(_sigmoid(a_logit), 4)
        t_prob = round(_sigmoid(t_logit), 4)
        used = sorted(set(a_used) | set(t_used))
        missing = sorted(set(a_missing) | set(t_missing))
        return FusionPrediction(
            abeta_status=a_prob >= self.threshold,
            abeta_prob=a_prob,
            tau_status=t_prob >= self.threshold,
            tau_prob=t_prob,
            model=MODEL_SURROGATE,
            source=SOURCE_SURROGATE,
            features_used=used,
            missing_features=missing,
            threshold=self.threshold,
            error=note or self._real_note,
            extra={
                "abeta_logit": round(a_logit, 4),
                "tau_logit": round(t_logit, 4),
                "spec_version": self._spec.get("version"),
                "disclaimer": (
                    "hand-set logistic stand-in for the Jasodanand2025 fusion "
                    "transformer; coefficients encode biomarker directionality, "
                    "are NOT fitted and carry NO validated accuracy"
                ),
            },
        )

    # -- real path (lazy torch/adrd) --------------------------------------

    def _predict_real(self, raw: dict) -> Optional[FusionPrediction]:
        """Run the loaded ADRDModel; None on any failure so predict() degrades.

        The real model expects a harmonized-UDS3 feature dict + its TOML config;
        it natively masks missing keys, so a sparse row is architecturally
        feedable. Wrapped end-to-end: any exception returns None (-> surrogate)."""
        try:
            model = self._real_model
            x = [dict(raw)]  # ADRDModel.predict_proba takes a list[dict]
            _, probas, _ = model.predict_proba(x, skip_embedding=True)
            rec = probas[0] if isinstance(probas, (list, tuple)) else probas
            a_prob = float(rec["amy_label"])
            t_prob = float(rec["tau_label"])
        except Exception as exc:  # noqa: BLE001 — degrade, never crash
            self._real_note = f"real fusion transformer inference failed: {exc!r}"
            return None
        return FusionPrediction(
            abeta_status=a_prob >= self.threshold,
            abeta_prob=round(a_prob, 4),
            tau_status=t_prob >= self.threshold,
            tau_prob=round(t_prob, 4),
            model=MODEL_REAL,
            source=SOURCE_LIVE,
            features_used=sorted(raw.keys()),
            missing_features=[],
            threshold=self.threshold,
            extra={"note": self._real_note},
        )

    # -- real-model loader -------------------------------------------------

    @classmethod
    def from_pretrained(cls, weights_path: Optional[str] = None, *,
                        repo_dir: Optional[str] = None,
                        hf_id: Optional[str] = None,
                        threshold: float = 0.5,
                        allow_clone: bool = False) -> "BiomarkerFusionPredictor":
        """Attempt to load the REAL Jasodanand2025 fusion transformer, else surrogate.

        Resolution order for the checkpoint:
          1. ``weights_path`` (an explicit model_stage_1.ckpt path), else
          2. ``NCOMMS2025_CKPT`` env var, else
          3. ``repo_dir`` / ``NCOMMS2025_REPO`` env var + ``dev/ckpt/model_stage_1.ckpt``,
          4. if ``allow_clone`` and none of the above resolve, best-effort shallow
             clone of vkola-lab/ncomms2025 into a temp dir (network; wrapped).

        torch + adrd (+ monai) are imported LAZILY here. On ANY failure — missing
        dep, missing checkpoint, no network — this returns a surrogate-configured
        predictor (with the reason recorded in ``error`` of later predictions)
        rather than raising. ``hf_id`` is accepted for signature symmetry but the
        weights live on GitHub, not HuggingFace, so it is unused today.
        """
        self = cls(threshold=threshold)
        ckpt = cls._resolve_ckpt(weights_path, repo_dir, allow_clone)
        if ckpt is None:
            self._real_note = (
                "real weights unavailable (no checkpoint resolved); using "
                "offline surrogate"
            )
            return self
        try:
            import torch  # noqa: F401  — lazy; absence => surrogate
            from adrd.model import ADRDModel  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self._real_note = (
                f"torch/adrd unavailable ({exc!r}); using offline surrogate"
            )
            return self
        try:
            self._real_model = ADRDModel.from_ckpt(str(ckpt), device="cpu",
                                                   img_dict=None)
        except Exception as exc:  # noqa: BLE001
            self._real_model = None
            self._real_note = (
                f"ADRDModel.from_ckpt failed ({exc!r}); using offline surrogate"
            )
        return self

    @staticmethod
    def _resolve_ckpt(weights_path: Optional[str], repo_dir: Optional[str],
                      allow_clone: bool) -> Optional[Path]:
        """Locate model_stage_1.ckpt from arg/env/repo, optionally cloning."""
        if weights_path:
            p = Path(weights_path)
            if p.exists():
                return p
        env_ckpt = os.environ.get("NCOMMS2025_CKPT")
        if env_ckpt and Path(env_ckpt).exists():
            return Path(env_ckpt)
        repo = repo_dir or os.environ.get("NCOMMS2025_REPO")
        if repo:
            cand = Path(repo) / NCOMMS2025_CKPT_RELPATH
            if cand.exists():
                return cand
        if allow_clone:
            return BiomarkerFusionPredictor._try_clone()
        return None

    @staticmethod
    def _try_clone() -> Optional[Path]:
        """Best-effort shallow clone of the repo; None on any failure."""
        import subprocess
        import tempfile
        try:
            dest = Path(tempfile.mkdtemp(prefix="ncomms2025_"))
            subprocess.run(
                ["git", "clone", "--depth", "1", NCOMMS2025_REPO_URL, str(dest)],
                check=True, capture_output=True, timeout=_HTTP_TIMEOUT * 20,
            )
            cand = dest / NCOMMS2025_CKPT_RELPATH
            return cand if cand.exists() else None
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def predict_biomarker_status(features: Union[dict, pd.Series], *,
                             threshold: float = 0.5) -> FusionPrediction:
    """One-shot surrogate prediction for a subject row (harness convenience).

    Thin wrapper over ``BiomarkerFusionPredictor().predict`` for callers that just
    want the offline Abeta/tau status estimate for one promoted target without
    managing a predictor instance."""
    return BiomarkerFusionPredictor(threshold=threshold).predict(features)
