"""
Tests for ComBat harmonization (neuroad.data.harmonize).

The scientific guarantees the referee relies on:
  * the harmonized table is still contract-valid, one row per subject, no NaN
    introduced into the embeddings;
  * the batch (scanner/site) signal is REDUCED — a probe can no longer decode
    the batch as well after harmonization;
  * a biological signal correlated with a PRESERVED covariate survives.

We build a small synthetic cohort with a planted batch (location+scale) shift
and a planted age signal so the assertions are deterministic and fast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

from neuroad import contract
from neuroad.data.harmonize import combat_cv_auc, harmonize


def _planted_cohort(n=240, d=12, seed=0):
    rng = np.random.default_rng(seed)
    batch = rng.integers(0, 3, size=n)                     # 3 scanners
    age = rng.uniform(60, 85, size=n)
    # Biology: a direction that tracks age (must survive). Batch: a large
    # location + scale shift per scanner (must be removed).
    Z = rng.standard_normal((n, d))
    Z[:, 0] += (age - age.mean()) / 5.0                    # age-linked signal
    for b in range(3):
        m = batch == b
        Z[m] = Z[m] * (1.0 + 0.6 * b) + (3.0 * b)          # scale + location shift
    frame = contract.make_embedding_frame(Z)
    frame.insert(0, "subject_id", [f"S{i:04d}" for i in range(n)])
    frame["dx"] = pd.Categorical(
        np.where(age > np.median(age), "AD", "CN"), categories=contract.DX_LEVELS)
    frame["conversion"] = pd.array([pd.NA] * n, dtype="Int8")
    frame["age"] = age
    frame["sex"] = pd.Categorical(rng.choice(["M", "F"], size=n), categories=contract.SEX_LEVELS)
    frame["site"] = pd.Categorical([f"site{b}" for b in batch])
    frame["scanner"] = pd.Categorical([f"scan{b}" for b in batch])
    frame["amyloid"] = pd.array([pd.NA] * n, dtype="Int8")
    for m in ("p_tau217", "gfap", "nfl"):
        frame[m] = np.nan
    frame["apoe4"] = pd.array([pd.NA] * n, dtype="Int8")
    return frame


def _auc(X, y):
    p = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=5,
                          method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def test_harmonize_stays_contract_valid_and_finite():
    df = _planted_cohort()
    out = harmonize(df, batch="scanner", covariates=("age", "sex"))
    contract.validate_table(out)                            # raises if invalid
    X = contract.embedding_matrix(out)
    assert np.isfinite(X).all(), "harmonization introduced NaN/inf into emb_*"
    assert len(out) == len(df)
    assert out.attrs.get("harmonized")


def test_harmonize_reduces_batch_signal_preserves_biology():
    df = _planted_cohort()
    Xr = contract.embedding_matrix(df)
    Xh = contract.embedding_matrix(harmonize(df, batch="scanner", covariates=("age", "sex")))
    batch = df["scanner"].astype("string").str.replace("scan", "").astype(int).to_numpy()
    y_ad = (df["dx"].astype("string") == "AD").astype(int).to_numpy()

    # Batch becomes materially harder to decode (3-class -> use one-vs-rest AUC on batch 2).
    b2 = (batch == 2).astype(int)
    assert _auc(Xh, b2) < _auc(Xr, b2) - 0.05, "batch signal not reduced"
    # The age-linked biology (AD label tracks age) is preserved.
    assert _auc(Xh, y_ad) > 0.65, "biological signal destroyed by harmonization"


def test_combat_cv_auc_is_fold_honest_and_not_inflated():
    # The fold-honest ComBat (fit inside each fold on train rows only) must run,
    # return a finite AUC, and sit AT OR BELOW the whole-cohort-harmonized AUC —
    # whole-cohort ComBat peeks at the held-out rows, which inflates the naive
    # number. Relational assert (seed-robust), not a fixed value.
    from neuroad import probe
    df = _planted_cohort()
    honest = combat_cv_auc(df, "dx_binary", batch="scanner", covariates=("age", "sex"))
    assert np.isfinite(honest)
    Xh, yh, gh = probe.point_head(harmonize(df, batch="scanner",
                                            covariates=("age", "sex")), "dx_binary")
    whole = probe.cross_val_auc(Xh, yh, groups=gh)
    assert honest <= whole + 1e-9, (
        f"fold-honest AUC {honest} should not exceed whole-cohort {whole}")


def test_harmonize_handles_singleton_batch():
    df = _planted_cohort()
    # Force a singleton site: reassign one subject to a unique site.
    sites = df["site"].astype("string").to_numpy()
    sites[0] = "site_singleton"
    df["site"] = pd.Categorical(sites)
    out = harmonize(df, batch="site", covariates=("age", "sex"))
    assert np.isfinite(contract.embedding_matrix(out)).all()
