"""
The five-test gauntlet — every way a structural-MRI finding can be an artifact.

Each test returns a `contract.TestEvidence(key, result, detail, stats)`:
  1. age_sex          — does it survive demographic covariates?
  2. site_scanner ⭐   — disease signal, or which machine acquired the scan?
  3. brain_age   ⭐   — more than generic aging / atrophy?
  4. biomarker_anchor — backed by molecular pathology (p-tau217 / GFAP)? HARD GATE.
  5. replication      — reproduces on a held-out site / cohort?

Thresholds come from `calibration.CAL` (retained-fraction bands), never from
free-floating constants. Where the data cannot support a test, its result is
`TestResult.NA` (dropped from the score, surfaced as a completeness caveat).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

from . import contract
from .calibration import CAL
from .contract import TestEvidence, TestResult
from .leakage import leakage_margin
from .probe import LinearProbe, cross_val_auc, point_head

# Retained-fraction bands (fraction of the naive effect that survives a control).
_SURVIVOR_RETAINED = CAL["survivor_retained"][0]   # 0.70 -> PASSED at/above
_KILL_RETAINED = CAL["kill_retained"][1]            # 0.40 -> FAILED below
# Biomarker-anchor gating on the 95% CI LOWER BOUND of the correlation (not raw
# r), so a lucky small-n correlation on noise cannot pass and a real anchor is
# not failed by seed. PASS => CI lower bound confidently positive.
_ANCHOR_CI_PASS = 0.12                              # CI lower bound >= this -> PASSED
_ANCHOR_CI_WEAK = 0.0                               # CI lower bound > this   -> WEAKENED


def _retained_fraction(auc_before: float, auc_after: float) -> float:
    eff = max(auc_before - 0.5, 1e-6)
    return float(np.clip((auc_after - 0.5) / eff, 0.0, 1.5))


def _result_from_retained(retained: float) -> TestResult:
    if retained >= _SURVIVOR_RETAINED:
        return TestResult.PASSED
    if retained >= _KILL_RETAINED:
        return TestResult.WEAKENED
    return TestResult.FAILED


def _residualize(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Regress covariates C out of every embedding column (return residuals)."""
    C = np.asarray(C, dtype=float)
    if C.ndim == 1:
        C = C.reshape(-1, 1)
    C = StandardScaler().fit_transform(C)
    C = np.column_stack([np.ones(len(C)), C])
    beta, *_ = np.linalg.lstsq(C, X, rcond=None)
    return X - C @ beta


def _outcome_keep_mask(df: pd.DataFrame, target: str) -> np.ndarray:
    if target == "conversion":
        return pd.to_numeric(df["conversion"], errors="coerce").notna().to_numpy()
    if target == "dx_binary":
        return df["dx"].astype("string").map({"AD": 1, "CN": 0}).notna().to_numpy()
    return np.ones(len(df), dtype=bool)


# ---------------------------------------------------------------------------
# 1. Age / sex adjustment
# ---------------------------------------------------------------------------
def test_age_sex(df: pd.DataFrame, target: str) -> TestEvidence:
    Xo, yo, go = point_head(df, target)
    if len(np.unique(yo)) < 2:
        return TestEvidence("age_sex", TestResult.NA, "target has <2 classes")

    keep = _outcome_keep_mask(df, target)
    sub = df.loc[keep]
    cov_cols = []
    age = sub["age"].to_numpy(float)
    if np.isfinite(age).sum() >= 3 and np.nanstd(age) > 0:
        cov_cols.append(np.nan_to_num(age, nan=np.nanmean(age)))
    if sub["sex"].nunique(dropna=True) > 1:
        cov_cols.append((sub["sex"].astype("string") == "F").to_numpy(float))
    if not cov_cols:
        return TestEvidence("age_sex", TestResult.NA, "no age/sex variation to adjust for")

    C = np.column_stack(cov_cols)
    auc_before = cross_val_auc(Xo, yo, groups=go)
    auc_after = cross_val_auc(_residualize(Xo, C), yo, groups=go)
    retained = _retained_fraction(auc_before, auc_after)
    res = _result_from_retained(retained)
    detail = (f"effect retained {retained:.0%} after age/sex adjustment "
              f"(AUC {auc_before:.2f} -> {auc_after:.2f})")
    return TestEvidence("age_sex", res, detail, {
        "auc_before": round(auc_before, 4), "auc_after": round(auc_after, 4),
        "retained": round(retained, 3), "n": int(len(yo)),
    })


# ---------------------------------------------------------------------------
# 2. Site / scanner leakage  (STAR)
# ---------------------------------------------------------------------------
def test_site_scanner(df: pd.DataFrame, target: str) -> TestEvidence:
    if df["scanner"].nunique(dropna=True) <= 1 and df["site"].nunique(dropna=True) <= 1:
        return TestEvidence("site_scanner", TestResult.NA,
                            "single scanner/site — no acquisition confound to test")
    m = leakage_margin(df, target=target)
    margin = m["margin"]
    if margin <= 0:
        res = TestResult.FAILED
        detail = (f"scanner predicted as well as outcome (margin {margin:+.2f}); "
                  f"likely acquisition artifact")
    elif margin < 0.10:
        res = TestResult.WEAKENED
        detail = f"outcome only narrowly exceeds scanner (margin {margin:+.2f})"
    else:
        res = TestResult.PASSED
        detail = f"outcome clearly exceeds scanner (margin {margin:+.2f})"
    return TestEvidence("site_scanner", res, detail, m)


# ---------------------------------------------------------------------------
# 3. Brain-age control  (STAR)
# ---------------------------------------------------------------------------
def test_brain_age(df: pd.DataFrame, target: str) -> TestEvidence:
    Xo, yo, go = point_head(df, target)
    if len(np.unique(yo)) < 2:
        return TestEvidence("brain_age", TestResult.NA, "target has <2 classes")

    # Train brain-age on cognitively-normal / healthy subjects only.
    healthy = df["dx"].astype("string").eq("CN").to_numpy()
    age_all = df["age"].to_numpy(float)
    train = healthy & np.isfinite(age_all)
    Xall = contract.embedding_matrix(df)
    if train.sum() < 8 or np.nanstd(age_all[train]) == 0:
        return TestEvidence("brain_age", TestResult.NA,
                            "too few healthy subjects with age to fit a brain-age model")

    reg = LinearRegression()
    n_splits = int(min(5, train.sum() // 3))
    yhat_cv = cross_val_predict(reg, Xall[train], age_all[train],
                                cv=max(n_splits, 2))
    ss_res = float(((age_all[train] - yhat_cv) ** 2).sum())
    ss_tot = float(((age_all[train] - age_all[train].mean()) ** 2).sum())
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    mae = float(np.mean(np.abs(age_all[train] - yhat_cv)))

    # Predicted brain age for every outcome subject. We control the outcome for
    # the brain's *apparent age* (the embedding-derived brain-age estimate), not
    # merely the residual gap: residualizing against the gap leaves the aging
    # component itself in the embedding, so a signal that is generic atrophy in
    # disguise would spuriously survive. Regressing out predicted brain age
    # removes the aging-aligned direction, so only age-independent disease signal
    # is left to be re-detected.
    reg.fit(Xall[train], age_all[train])
    keep = _outcome_keep_mask(df, target)
    # Predicted brain age for every subject: in-sample fit for subjects the
    # regressor never trained on, but the OUT-OF-FOLD prediction for training
    # (CN) subjects, so the control is never fit-and-applied on the same rows
    # (which would over-remove signal for a dx_binary outcome whose CN class is
    # the brain-age training set).
    brain_age_all = reg.predict(Xall)
    brain_age_all[train] = yhat_cv
    brain_age = brain_age_all[keep]
    age_kept = age_all[keep]
    gap = brain_age - age_kept                      # reported as brain-age gap
    gap = np.nan_to_num(gap, nan=0.0)
    control = np.nan_to_num(brain_age, nan=float(np.nanmean(brain_age)))

    auc_before = cross_val_auc(Xo, yo, groups=go)
    auc_after = cross_val_auc(_residualize(Xo, control), yo, groups=go)
    retained = _retained_fraction(auc_before, auc_after)
    res = _result_from_retained(retained)
    detail = (f"brain-age R2={r2:.2f}, MAE={mae:.1f}yr; effect retained "
              f"{retained:.0%} after brain-age-gap control")
    return TestEvidence("brain_age", res, detail, {
        "r2": round(r2, 3), "mae_yr": round(mae, 2),
        "auc_before": round(auc_before, 4), "auc_after": round(auc_after, 4),
        "retained": round(retained, 3), "n_healthy": int(train.sum()),
    })


# ---------------------------------------------------------------------------
# 4. Biomarker anchor  (HARD GATE)
# ---------------------------------------------------------------------------
#: Minimum complete-case n before a plasma anchor is even attempted. Below this
#: the correlation is too unstable to gate on (returns NA -> route to a cohort
#: with coverage rather than credit or condemn on noise).
_ANCHOR_MIN_N = 20


def _anchor_corr(score: np.ndarray, marker: np.ndarray) -> tuple[Optional[float], int, Optional[float]]:
    """Return (r, n, ci_lower_95). ci_lower is the lower bound of the two-sided
    95% Fisher-z confidence interval for the Pearson r. Gating on ci_lower rather
    than raw r is what stops a lucky ~2-sigma correlation on pure noise (small n)
    from spuriously PASSING the molecular anchor — and stops a real-but-noisy
    anchor from FAILING purely by seed."""
    ok = np.isfinite(marker)
    n = int(ok.sum())
    if n < _ANCHOR_MIN_N or np.nanstd(marker[ok]) == 0:
        return None, n, None
    r, _ = sstats.pearsonr(score[ok], marker[ok])
    if not np.isfinite(r):
        return None, n, None
    # Fisher z-transform -> symmetric CI in z -> back to r.
    r_c = float(np.clip(r, -0.999, 0.999))
    z = np.arctanh(r_c)
    se = 1.0 / np.sqrt(n - 3)
    ci_lo = float(np.tanh(z - 1.96 * se))
    return float(r), n, ci_lo


def _oof_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Out-of-fold P(positive) so the anchor cannot correlate with overfit
    in-sample residuals — a spurious anchor is exactly what we must not credit."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from .probe import _n_splits
    y_codes = np.searchsorted(np.unique(y), y)
    k = _n_splits(y_codes, None)
    pipe = Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    proba = cross_val_predict(pipe, X, y_codes, cv=k, method="predict_proba")
    return proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)


def test_biomarker_anchor(df: pd.DataFrame, target: str) -> TestEvidence:
    Xo, yo, _ = point_head(df, target)
    classes, counts = np.unique(yo, return_counts=True)
    if len(classes) < 2:
        return TestEvidence("biomarker_anchor", TestResult.NA, "target has <2 classes")
    # A minority class with <2 members cannot be cross-validated: StratifiedKFold
    # would place the lone member in a test fold, leaving a single-class training
    # fold that crashes LogisticRegression. Degrade to NA (route to a cohort with
    # enough labelled subjects) rather than crash — this is what makes per-cluster
    # refereeing on real, sparsely-labelled sub-cohorts complete gracefully.
    if int(counts.min()) < 2:
        return TestEvidence("biomarker_anchor", TestResult.NA,
                            "minority class too small to cross-validate safely "
                            "(singleton) — cannot anchor")

    # Out-of-fold probe score on the outcome-kept subset, correlated with markers.
    score = _oof_scores(Xo, yo)
    keep = _outcome_keep_mask(df, target)
    sub = df.loc[keep]

    ptau_r, ptau_n, ptau_lo = _anchor_corr(score, sub["p_tau217"].to_numpy(float))
    gfap_r, gfap_n, gfap_lo = _anchor_corr(score, sub["gfap"].to_numpy(float))

    stats = {"ptau217_r": None if ptau_r is None else round(ptau_r, 3), "ptau217_n": ptau_n,
             "ptau217_ci_lo": None if ptau_lo is None else round(ptau_lo, 3),
             "gfap_r": None if gfap_r is None else round(gfap_r, 3), "gfap_n": gfap_n,
             "gfap_ci_lo": None if gfap_lo is None else round(gfap_lo, 3)}

    # Pick the primary anchor (p-tau217 preferred, GFAP secondary).
    if ptau_r is not None:
        primary, ci_lo, n_used, marker = ptau_r, ptau_lo, ptau_n, "p-tau217"
    elif gfap_r is not None:
        primary, ci_lo, n_used, marker = gfap_r, gfap_lo, gfap_n, "GFAP"
    else:
        return TestEvidence("biomarker_anchor", TestResult.NA,
                            "no plasma p-tau217 / GFAP coverage at usable n — cannot "
                            "anchor (route to ADNI/EPAD)", stats)

    # Gate on the CI LOWER BOUND, not raw |r|: the anchor must be confidently
    # positive (its 95% CI excludes zero and clears a floor). This makes the
    # hard gate robust to seed/small-n noise in both directions.
    if ci_lo >= _ANCHOR_CI_PASS:           # confidently anchored
        res = TestResult.PASSED
    elif ci_lo > _ANCHOR_CI_WEAK:          # positive but thin
        res = TestResult.WEAKENED
    else:                                  # CI includes ~zero -> no molecular support
        res = TestResult.FAILED            # data present but unanchored -> gate fails
    detail = (f"{marker} correlation r={primary:+.2f} "
              f"(95% CI lower {ci_lo:+.2f}, n={n_used})")
    return TestEvidence("biomarker_anchor", res, detail, stats)


# ---------------------------------------------------------------------------
# 5. Replication split (held-out site / cohort)
# ---------------------------------------------------------------------------
def test_replication(df: pd.DataFrame, target: str) -> TestEvidence:
    Xo, yo, go = point_head(df, target)
    if len(np.unique(yo)) < 2 or go is None:
        return TestEvidence("replication", TestResult.NA, "no site/cohort grouping to split on")

    groups = np.asarray(go)
    uniq, counts = np.unique(groups, return_counts=True)
    if len(uniq) < 2:
        return TestEvidence("replication", TestResult.NA,
                            "single site/cohort — no held-out split available")

    # Hold out the smallest usable site; train on the rest.
    order = uniq[np.argsort(counts)]
    test_site = None
    for g in order:
        te = groups == g
        tr = ~te
        if (len(np.unique(yo[te])) >= 2 and len(np.unique(yo[tr])) >= 2
                and te.sum() >= 6 and tr.sum() >= 6):
            test_site = g
            break
    if test_site is None:
        return TestEvidence("replication", TestResult.NA,
                            "no site split yields two-class train and test folds")

    te = groups == test_site
    tr = ~te
    train_auc = cross_val_auc(Xo[tr], yo[tr], groups=None)
    probe = LinearProbe().fit(Xo[tr], yo[tr])
    from sklearn.metrics import roc_auc_score
    test_auc = float(roc_auc_score(yo[te], probe.decision_scores(Xo[te])))

    if test_auc >= 0.65:
        res = TestResult.PASSED
    elif test_auc >= 0.58:
        res = TestResult.WEAKENED
    else:
        res = TestResult.FAILED
    detail = (f"held-out cohort AUC={test_auc:.2f} (train {train_auc:.2f}); "
              f"n_test={int(te.sum())}")
    return TestEvidence("replication", res, detail, {
        "train_auc": round(train_auc, 4), "test_auc": round(test_auc, 4),
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
    })


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_gauntlet(df: pd.DataFrame, claim) -> list[TestEvidence]:
    """Run all five gauntlet tests for a claim; NA where data is insufficient.

    `claim` may be a `contract.Claim` or anything with a `.target` attribute;
    a bare string target is also accepted.
    """
    target = getattr(claim, "target", claim)
    if target not in contract.LABEL_TARGETS:
        target = "conversion"
    return [
        test_age_sex(df, target),
        test_site_scanner(df, target),
        test_brain_age(df, target),
        test_biomarker_anchor(df, target),
        test_replication(df, target),
    ]
