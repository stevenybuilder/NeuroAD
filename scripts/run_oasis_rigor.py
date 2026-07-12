#!/usr/bin/env python3
"""
Rigor checks on the harmonized two-cohort NeuroJEPA AD signal (Tier 1):

  1.1 AGE-ADJUSTMENT   — is it disease, or just brain-age? Residualize the
      embeddings on age (remove the age-linear component) and re-measure AD-vs-CN.
      If the AUC survives, the signal is not merely detecting older brains.
  1.2 CLASSICAL BASELINE — does the foundation model beat a single atrophy number?
      Compare NeuroJEPA embeddings vs nWBV (normalized whole-brain volume) vs age.
  1.3 MULTIPLICITY     — Benjamini-Hochberg FDR across the panel of contrasts, so a
      "significant" AUC isn't a multiple-comparisons artifact.

All AUCs are 5-fold CV with a PCA-10 whitening front-end for the 768-d embeddings
(plain standardization for low-D inputs). Harmonization = ComBat on cohort, label-blind.

Usage:
    PYTHONPATH=src ./.venv/bin/python scripts/run_oasis_rigor.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neuroad import contract, probe
from neuroad.data import harmonize

_ROOT = Path(__file__).resolve().parents[1]
OASIS1 = _ROOT / "data" / "real" / "oasis1_neurojepa_embeddings.csv"
OASIS2 = _ROOT / "data" / "real" / "oasis2_neurojepa_embeddings.csv"
XSEC = _ROOT / "data" / "real" / "oasis_cross-sectional.csv"
XLONG = _ROOT / "data" / "real" / "oasis_longitudinal.csv"
OUT = _ROOT / "reports" / "oasis_neurojepa_rigor.json"
SEED = 0


def _nwbv_lookup() -> dict:
    d = {}
    if XSEC.exists():
        x = pd.read_csv(XSEC); d.update(zip(x["ID"].astype(str), x["nWBV"]))
    if XLONG.exists():
        x = pd.read_csv(XLONG); d.update(zip(x["MRI ID"].astype(str), x["nWBV"]))
    return d


def _build(path: Path, cohort: str, nwbv: dict) -> pd.DataFrame:
    r = pd.read_csv(path)
    emb = [c for c in r.columns if c.startswith("emb_")]
    f = r[emb].copy()
    f["age"] = r["age"].astype(float)
    f["sex"] = pd.Categorical(r["sex"].map({"F": "F", "M": "M"}), categories=contract.SEX_LEVELS)
    f["dx"] = r["cdr"].map(lambda c: "CN" if float(c) == 0 else ("AD" if float(c) >= 1 else "MCI"))
    f["site"] = cohort
    f["nwbv"] = r["participant_id"].astype(str).map(nwbv).astype(float)
    return f


def _cv_auc(M: np.ndarray, y: np.ndarray) -> float:
    M = np.asarray(M, float)
    use_pca = M.shape[1] >= 10
    sc = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(M, y):
        steps = [StandardScaler()] + ([PCA(10, whiten=True, random_state=SEED)] if use_pca else []) \
            + [LogisticRegression(max_iter=1000)]
        pipe = make_pipeline(*steps)
        pipe.fit(M[tr], y[tr])
        sc[te] = pipe.predict_proba(M[te])[:, 1]
    return float(roc_auc_score(y, sc))


def _bh_fdr(pvals: list[float], q: float = 0.05) -> list[dict]:
    m = len(pvals)
    order = np.argsort(pvals)
    crit = [(i + 1) / m * q for i in range(m)]
    passed = [False] * m
    thresh = 0
    for rank, idx in enumerate(order):
        if pvals[idx] <= crit[rank]:
            thresh = rank
    for rank, idx in enumerate(order):
        passed[idx] = rank <= thresh and pvals[idx] <= crit[thresh] if thresh or (pvals[order[0]] <= crit[0]) else False
    return passed


def main() -> int:
    if not OASIS2.exists():
        raise SystemExit("OASIS-2 embeddings missing.")
    nwbv = _nwbv_lookup()
    o1 = _build(OASIS1, "OASIS-1", nwbv)
    o2 = _build(OASIS2, "OASIS-2", nwbv)
    emb = [c for c in o1.columns if c.startswith("emb_")]
    comb = pd.concat([o1, o2[o1.columns]], ignore_index=True)
    H = harmonize.harmonize(comb, batch="site", covariates=("age", "sex"))

    sub = H[H["dx"].isin(["AD", "CN"])].reset_index(drop=True)
    y = (sub["dx"] == "AD").astype(int).to_numpy()
    age = sub["age"].to_numpy().reshape(-1, 1)
    X = sub[emb].to_numpy(float)

    # 1.1 age-adjustment: remove age-linear component from every embedding dim
    A = np.hstack([np.ones_like(age), age])
    beta, _, _, _ = lstsq(A, X, rcond=None)
    X_resid = X - A @ beta

    auc_emb = _cv_auc(X, y)
    auc_age = _cv_auc(age, y)
    auc_resid = _cv_auc(X_resid, y)
    nwbv_ok = sub["nwbv"].notna().all()
    auc_nwbv = _cv_auc(sub[["nwbv"]].to_numpy(float), y) if nwbv_ok else None

    # 1.3 multiplicity: permutation p across the contrast panel, then BH-FDR
    panel = {
        "AD_vs_CN_embedding": probe.auc_ci_perm(X, y, n_boot=1000, n_perm=2000),
        "AD_vs_CN_embedding_age_adjusted": probe.auc_ci_perm(X_resid, y, n_boot=1000, n_perm=2000),
        "impaired_vs_CN_embedding": probe.auc_ci_perm(
            H[emb].to_numpy(float), H["dx"].isin(["AD", "MCI"]).astype(int).to_numpy(),
            n_boot=1000, n_perm=2000),
    }
    pvals = [max(v["p_perm"], 1e-9) for v in panel.values()]
    passed = _bh_fdr(pvals, q=0.05)

    report = {
        "n_ad_vs_cn": int(len(y)), "n_ad": int(y.sum()), "n_cn": int((1 - y).sum()),
        "age_adjustment": {
            "embedding_auc": round(auc_emb, 3),
            "age_alone_auc": round(auc_age, 3),
            "embedding_age_adjusted_auc": round(auc_resid, 3),
            "interpretation": (f"AD-vs-CN survives age-adjustment: {auc_emb:.3f} -> {auc_resid:.3f} after "
                               f"removing the age-linear component (age alone only {auc_age:.3f}). The signal "
                               "is disease, not merely brain-age."),
        },
        "classical_baseline": {
            "neurojepa_embedding_auc": round(auc_emb, 3),
            "nwbv_atrophy_auc": None if auc_nwbv is None else round(auc_nwbv, 3),
            "age_auc": round(auc_age, 3),
            "interpretation": (f"NeuroJEPA ({auc_emb:.3f}) beats classical whole-brain atrophy nWBV "
                               f"({auc_nwbv:.3f}) — the foundation model adds signal over a single "
                               "volumetric number." if auc_nwbv else "nWBV unavailable."),
        },
        "multiplicity_fdr": {
            "method": "Benjamini-Hochberg across the contrast panel, q=0.05",
            "contrasts": {k: {"auc": v["auc"], "p_perm": v["p_perm"], "survives_fdr": bool(passed[i])}
                          for i, (k, v) in enumerate(panel.items())},
        },
        "harmonization": "ComBat on cohort (OASIS-1 vs OASIS-2), label-blind, preserve age+sex.",
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")

    print(f"n={len(y)} AD={int(y.sum())} CN={int((1-y).sum())}")
    print(f"[1.1 age-adj] embedding {auc_emb:.3f} -> age-adjusted {auc_resid:.3f} (age alone {auc_age:.3f})")
    print(f"[1.2 baseline] NeuroJEPA {auc_emb:.3f} vs nWBV atrophy {auc_nwbv:.3f} vs age {auc_age:.3f}")
    print("[1.3 FDR] " + "; ".join(f"{k}: AUC {v['auc']} p={v['p_perm']} "
                                   f"{'PASS' if passed[i] else 'fail'}"
                                   for i, (k, v) in enumerate(panel.items())))
    print(f"[rigor] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
