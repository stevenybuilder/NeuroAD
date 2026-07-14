"""
The five-test gauntlet — every way a structural-MRI finding can be an artifact.

Each test returns a `contract.TestEvidence(key, result, detail, stats)`:
  1. age_sex          — does it survive demographic covariates?
  2. site_scanner ⭐   — disease signal, or which machine acquired the scan?
  3. brain_age   ⭐   — more than generic aging / atrophy?
  4. biomarker_anchor — backed by molecular pathology (p-tau217 / GFAP)? HARD GATE.
  5. replication      — reproduces on a held-out site / cohort?

The confound flag thresholds (retained-fraction bands) come from the L3
`confound_priors` policy doc via `harness.policy.thresholds`, never from
free-floating constants; those policy values are transcriptions of
`calibration.CAL`, which stays the exact fallback if `policy/` is absent or
malformed. Where the data cannot support a test, its result is `TestResult.NA`
(dropped from the score, surfaced as a completeness caveat).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import contract
from .calibration import CAL
from .contract import TestEvidence, TestResult
from .harness import policy
from .leakage import leakage_margin
from .probe import (LinearProbe, N_BOOT, N_PERM, cross_val_auc, point_head,
                    residualized_cross_val_auc)

# Retained-fraction bands (fraction of the naive effect that survives a control) —
# the flag thresholds for the confound tests (age/sex, brain-age). Read from the
# L3 `confound_priors` policy doc, whose `retained_bands` transcribe
# CAL["survivor_retained"][0] / CAL["kill_retained"][1]. policy.thresholds already
# overlays policy onto that hardcoded CAL fallback key-by-key; the outer guard
# keeps the demo byte-identical even if the loader import itself ever fails.
try:
    _CONFOUND_BANDS = policy.thresholds("retained")     # from policy/confound_priors.yaml
    _SURVIVOR_RETAINED = _CONFOUND_BANDS["survivor_retained"]  # 0.70 -> PASSED at/above
    _KILL_RETAINED = _CONFOUND_BANDS["kill_retained"]          # 0.40 -> FAILED below
except Exception:                                       # pragma: no cover - defensive
    _SURVIVOR_RETAINED = CAL["survivor_retained"][0]    # 0.70 -> PASSED at/above
    _KILL_RETAINED = CAL["kill_retained"][1]            # 0.40 -> FAILED below
# Biomarker-anchor gating on the 95% CI LOWER BOUND of the correlation (not raw
# r), so a lucky small-n correlation on noise cannot pass and a real anchor is
# not failed by seed. PASS => CI lower bound confidently positive.
_ANCHOR_CI_PASS = 0.12                              # CI lower bound >= this -> PASSED
_ANCHOR_CI_WEAK = 0.0                               # CI lower bound > this   -> WEAKENED

# A brain-age control can only "control for aging" if the brain-age model is
# itself predictive. A non-predictive model (out-of-fold R2 at/below chance)
# scrubs essentially nothing, so ANY effect trivially "survives" and the control
# would earn full STAR credit while proving nothing — a negative R2 literally
# predicts age worse than the cohort mean. Require the model to clear a small
# positive out-of-fold R2 floor; below it the control is UNINFORMATIVE -> NA
# (dropped from numerator+denominator in contract.robustness_score), never a fake
# PASS. 0.10 = the model must explain at least ~10% of age variance out-of-fold
# to be treated as a working age predictor whose residualization means something.
_BRAIN_AGE_R2_FLOOR = 0.10
# RidgeCV penalty grid for the brain-age model. Unregularized OLS on a high-dim
# structural feeder overfits so hard out-of-fold that R2<<0, spuriously NAing a
# control that is in fact computable; a CV-penalized ridge is the standard
# brain-age estimator (Franke/Gaser).
_BRAIN_AGE_ALPHAS = np.logspace(1, 5, 20)


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
    # Fold-honest: the age/sex residualization is fit inside each CV fold on the
    # train rows only, never on the held-out fold (the whole-cohort residualize-
    # then-CV pattern let the control peek at the test rows).
    auc_after = residualized_cross_val_auc(Xo, C, yo, go, kind="covariate")
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
def test_site_scanner(df: pd.DataFrame, target: str,
                      n_boot: int = N_BOOT, n_perm: int = N_PERM) -> TestEvidence:
    if df["scanner"].nunique(dropna=True) <= 1 and df["site"].nunique(dropna=True) <= 1:
        return TestEvidence("site_scanner", TestResult.NA,
                            "single scanner/site — no acquisition confound to test")
    m = leakage_margin(df, target=target, n_boot=n_boot, n_perm=n_perm)
    margin = m["margin"]
    ci_lo = m.get("margin_ci_lo")
    ci_hi = m.get("margin_ci_hi")
    excludes_zero = m.get("margin_ci_excludes_zero", False)
    ci_txt = "" if ci_lo is None else f" [95% CI {ci_lo:+.2f}, {ci_hi:+.2f}]"

    # Reframe the star as a TEST, not a bare point cutoff: the outcome must beat
    # the scanner by more than noise (margin CI excludes zero), not merely have a
    # positive point estimate. This directly answers "is that margin significant?"
    if margin <= 0:
        res = TestResult.FAILED
        detail = (f"scanner predicted as well as outcome (margin {margin:+.2f}"
                  f"{ci_txt}); likely acquisition artifact")
    elif excludes_zero:
        res = TestResult.PASSED
        detail = (f"outcome exceeds scanner and the margin CI excludes zero "
                  f"(margin {margin:+.2f}{ci_txt})")
    else:
        # Point estimate positive but the CI still includes zero -> not
        # defensibly better than the machine.
        res = TestResult.WEAKENED
        detail = (f"outcome only narrowly exceeds scanner; margin CI includes "
                  f"zero (margin {margin:+.2f}{ci_txt})")
    return TestEvidence("site_scanner", res, detail, m)


# ---------------------------------------------------------------------------
# 3. Brain-age control  (STAR)
# ---------------------------------------------------------------------------
def test_brain_age(df: pd.DataFrame, target: str) -> TestEvidence:
    Xo, yo, go = point_head(df, target)
    if len(np.unique(yo)) < 2:
        return TestEvidence("brain_age", TestResult.NA, "target has <2 classes")

    # Train brain-age on cognitively-normal / healthy subjects only. Coerce the
    # nullable-boolean mask to a dense bool (NaN dx -> not-healthy) so a partial
    # dx column (real cohorts carry <NA>) can't poison the boolean `train` mask.
    healthy = df["dx"].astype("string").eq("CN").fillna(False).to_numpy(dtype=bool)
    age_all = df["age"].to_numpy(float)
    train = healthy & np.isfinite(age_all)
    Xall = contract.embedding_matrix(df)
    if train.sum() < 8 or np.nanstd(age_all[train]) == 0:
        return TestEvidence("brain_age", TestResult.NA,
                            "too few healthy subjects with age to fit a brain-age model")

    # Regularized brain-age estimator. Unregularized OLS on a 323-dim feeder
    # overfits out-of-fold (R2<<0, worse than the cohort mean), which spuriously
    # trips the R2 floor below and NAs a control that is in fact computable. A
    # CV-penalized ridge on standardized features is the standard brain-age model;
    # shuffled KFold so the out-of-fold R2 is not an artifact of source-row order.
    reg = make_pipeline(StandardScaler(), RidgeCV(alphas=_BRAIN_AGE_ALPHAS))
    n_splits = int(min(5, train.sum() // 3))
    yhat_cv = cross_val_predict(reg, Xall[train], age_all[train],
                                cv=KFold(max(n_splits, 2), shuffle=True,
                                         random_state=0))
    ss_res = float(((age_all[train] - yhat_cv) ** 2).sum())
    ss_tot = float(((age_all[train] - age_all[train].mean()) ** 2).sum())
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    mae = float(np.mean(np.abs(age_all[train] - yhat_cv)))

    # If the brain-age model itself does not predict age (out-of-fold R2 below the
    # floor), "controlling for brain age" removes nothing and the effect survives
    # by construction — that is not a passed control, it is an UNINFORMATIVE one.
    # Return NA (dropped from the score) rather than banking full STAR credit for a
    # control that does not work.
    if r2 < _BRAIN_AGE_R2_FLOOR:
        return TestEvidence("brain_age", TestResult.NA,
                            f"brain-age model not predictive (out-of-fold "
                            f"R2={r2:.2f} < {_BRAIN_AGE_R2_FLOOR:.2f}, "
                            f"MAE={mae:.1f}yr) — controlling for it removes nothing; "
                            f"uninformative control",
                            {"r2": round(r2, 3), "mae_yr": round(mae, 2),
                             "n_healthy": int(train.sum())})

    # Predicted brain age for every outcome subject. We compute BOTH the
    # embedding-derived apparent brain age (the strict control) and the
    # Franke/Gaser brain-age GAP (predicted - chronological), and report the
    # effect retained under each — see the two-way rationale below where the
    # controls are applied.
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
    gap = brain_age - age_kept                      # Franke/Gaser brain-age GAP
    gap = np.nan_to_num(gap, nan=0.0)
    control_pred = np.nan_to_num(brain_age, nan=float(np.nanmean(brain_age)))

    # We report the brain-age control BOTH ways so one over-aggressive
    # residualization cannot silently kill a real signal:
    #   * PREDICTED brain age (control_pred): removes the whole aging-aligned
    #     direction. This is the STRICT control — but for a dx_binary outcome the
    #     disease atrophy loads on the same axis, so it can over-remove the signal
    #     by construction (removes generic aging AND AD-specific atrophy together).
    #   * brain-age GAP (gap = predicted - chronological): the Franke/Gaser
    #     BrainAGE standard — residualize against the age-INDEPENDENT deviation,
    #     leaving age itself out of the scrub. This is the cited-standard control
    #     and the one the verdict is based on; the strict predicted-brain-age
    #     retained fraction is reported alongside as a secondary, conservative view.
    auc_before = cross_val_auc(Xo, yo, groups=go)
    # Fold-honest brain-age controls: residualize against the gap / predicted
    # brain age with the nuisance regression fit inside each fold on train rows
    # only. NOTE the brain-age model itself (yhat_cv above) is already out-of-fold
    # on the CN training set; here the per-fold residualization of the OUTCOME
    # probe is what becomes leakage-honest.
    auc_after_gap = residualized_cross_val_auc(Xo, gap, yo, go, kind="covariate")
    auc_after_pred = residualized_cross_val_auc(Xo, control_pred, yo, go,
                                                kind="covariate")
    retained_gap = _retained_fraction(auc_before, auc_after_gap)
    retained_pred = _retained_fraction(auc_before, auc_after_pred)

    # Verdict on the standard (gap) control; keep the primary "retained"/
    # "auc_after" keys pointing at it for the ledger's backward compatibility.
    retained = retained_gap
    res = _result_from_retained(retained)
    detail = (f"brain-age R2={r2:.2f}, MAE={mae:.1f}yr; effect retained "
              f"{retained_gap:.0%} after brain-age-GAP control "
              f"(Franke/Gaser standard), {retained_pred:.0%} under the stricter "
              f"predicted-brain-age control")
    return TestEvidence("brain_age", res, detail, {
        "r2": round(r2, 3), "mae_yr": round(mae, 2),
        "auc_before": round(auc_before, 4),
        "auc_after": round(auc_after_gap, 4),
        "retained": round(retained_gap, 3),
        # both adjustments, explicitly labelled:
        "auc_after_gap": round(auc_after_gap, 4),
        "retained_gap": round(retained_gap, 3),
        "auc_after_pred": round(auc_after_pred, 4),
        "retained_pred": round(retained_pred, 3),
        "n_healthy": int(train.sum()),
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


def _anchor_enrichment(
    score: np.ndarray, marker: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[Optional[float], Optional[float], int, Optional[int], Optional[int], Optional[float]]:
    """Amyloid-positivity ENRICHMENT anchor for a BINARY marker (0/1 positivity).

    Pearson correlation is the wrong statistic against a binary label — a two-point
    x-axis makes its magnitude an artifact of class balance, and the survivor
    narrative claims the probe is "enriched in amyloid-POSITIVE subjects", a
    group-contrast claim, not a linear-trend one. So we report a group-difference:
    the mean probe score in amyloid-positive minus amyloid-negative subjects
    (``delta``), the equivalent point-biserial correlation ``r`` (Pearson against
    the 0/1 label, reported for a scale-free companion number), and a bootstrap
    95% CI LOWER BOUND on the delta. Gating elsewhere stays on p-tau217; this is an
    ADDITIONAL reported anchor, so the CI lower bound is surfaced for evidence, not
    used to pass/fail the hard gate.

    Returns (delta, r, n, n_pos, n_neg, delta_ci_lo). Any of delta/r/ci_lo is None
    when the marker has too few complete cases or lacks both classes.
    """
    ok = np.isfinite(marker) & np.isfinite(score)
    n = int(ok.sum())
    s = score[ok]
    m = marker[ok]
    pos = m > 0.5
    neg = ~pos
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n < _ANCHOR_MIN_N or n_pos < 2 or n_neg < 2:
        return None, None, n, n_pos or None, n_neg or None, None
    delta = float(s[pos].mean() - s[neg].mean())
    # Point-biserial r == Pearson(score, 0/1 label); finite because both vary here.
    r, _ = sstats.pearsonr(s, m.astype(float))
    r = float(r) if np.isfinite(r) else None
    # Bootstrap the group difference: resample subjects with replacement, keep only
    # draws that retain both classes, take the 2.5th percentile as the CI lower
    # bound. This is the same "don't trust a single small-n point estimate" posture
    # the replication and leakage stars already use.
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    boot: list[float] = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=n, replace=True)
        pb = m[b] > 0.5
        if pb.sum() < 2 or (~pb).sum() < 2:
            continue
        boot.append(float(s[b][pb].mean() - s[b][~pb].mean()))
    ci_lo = float(np.percentile(boot, 2.5)) if boot else None
    return delta, r, n, n_pos, n_neg, ci_lo


def _oof_scores(X: np.ndarray, y: np.ndarray,
                groups: Optional[np.ndarray] = None) -> np.ndarray:
    """Out-of-fold P(positive) so the anchor cannot correlate with overfit
    in-sample residuals — a spurious anchor is exactly what we must not credit.

    Uses the SAME automatic PCA front-end as the rest of the probe path so the
    anchor correlation on a p >> n cohort is computed on the defensible reduced
    representation, not the inflated raw 768-d one. When site ``groups`` are
    supplied and there are enough distinct sites, the OOF folds are made
    site-disjoint (StratifiedGroupKFold) exactly like the headline probe, so the
    anchor score never rides on within-site leakage; otherwise it falls back to
    the plain StratifiedKFold path (unchanged for site/scanner-less cohorts).
    """
    from .probe import _n_splits, build_probe_pipeline
    from sklearn.model_selection import StratifiedGroupKFold
    X = np.asarray(X, dtype=float)
    y_codes = np.searchsorted(np.unique(y), y)
    k = _n_splits(y_codes, groups)
    pipe, _ = build_probe_pipeline(X.shape[0], X.shape[1])
    splits = None
    if groups is not None and len(np.unique(groups)) >= k:
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=0)
        candidate = list(cv.split(X, y_codes, np.asarray(groups)))
        # cross_val_predict cannot skip folds, so a site-disjoint split that
        # leaves ANY single-class training fold would crash the fit. Only use the
        # grouped splits when every train fold carries both classes; otherwise
        # fall back to the historical (non-grouped) path so sparsely-labelled
        # sub-cohorts still complete.
        if all(len(np.unique(y_codes[tr])) >= 2 for tr, _ in candidate):
            splits = candidate
    if splits is not None:
        proba = cross_val_predict(pipe, X, y_codes, cv=splits,
                                  method="predict_proba")
    else:
        # Preserve the historical fallback exactly: integer cv -> default
        # StratifiedKFold (no shuffle) for site/scanner-less cohorts.
        proba = cross_val_predict(pipe, X, y_codes, cv=k, method="predict_proba")
    return proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)


def test_biomarker_anchor(df: pd.DataFrame, target: str) -> TestEvidence:
    Xo, yo, go = point_head(df, target)
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
    # Site-disjoint OOF (like the headline) when site groups are available.
    score = _oof_scores(Xo, yo, go)
    keep = _outcome_keep_mask(df, target)
    sub = df.loc[keep]

    ptau_r, ptau_n, ptau_lo = _anchor_corr(score, sub["p_tau217"].to_numpy(float))
    gfap_r, gfap_n, gfap_lo = _anchor_corr(score, sub["gfap"].to_numpy(float))
    # C2N %p-tau217 (contract.EXTENDED_BIOMARKER_COLUMNS) — one of the best-validated
    # plasma tau markers. Read only when a richer feeder (the ADNI plasma ensemble)
    # triangulated it in; otherwise it stays None and the anchor is unchanged.
    if "pct_ptau217" in sub.columns:
        pct_r, pct_n, pct_lo = _anchor_corr(score, sub["pct_ptau217"].to_numpy(float))
    else:
        pct_r, pct_n, pct_lo = None, 0, None

    # Amyloid-positivity ENRICHMENT anchor. amyloid is BINARY (0/1 positivity), so
    # a Pearson r against it is the wrong statistic — we report the group
    # difference (probe score in amyloid-positive vs amyloid-negative subjects)
    # with a bootstrap CI lower bound, plus the equivalent point-biserial r. This
    # is an ADDITIONAL reported anchor that flows through the stats dict alongside
    # p-tau217/GFAP; the hard gate below still keys on p-tau217 (primary). It
    # directly answers the survivor narrative's claim that the probe is "enriched
    # in amyloid-positive subjects".
    if "amyloid" in sub.columns:
        amy_marker = pd.to_numeric(sub["amyloid"], errors="coerce").to_numpy(float)
        (amy_delta, amy_r, amy_n, amy_n_pos,
         amy_n_neg, amy_lo) = _anchor_enrichment(score, amy_marker)
    else:
        amy_delta = amy_r = amy_lo = None
        amy_n = 0
        amy_n_pos = amy_n_neg = None

    # Provenance: on a synthetic cohort the plasma markers are CALIBRATION TARGETS
    # drawn to sit inside a literature range (calibration.CAL), NOT measured plasma.
    # Badge them so no downstream artifact can present a fabricated r as a
    # measurement. No open MRI+plasma cohort exists, so this is always synthetic today.
    synthetic_anchor = bool(df.attrs.get("synthetic_biomarkers"))
    stats = {"ptau217_r": None if ptau_r is None else round(ptau_r, 3), "ptau217_n": ptau_n,
             "ptau217_ci_lo": None if ptau_lo is None else round(ptau_lo, 3),
             "gfap_r": None if gfap_r is None else round(gfap_r, 3), "gfap_n": gfap_n,
             "gfap_ci_lo": None if gfap_lo is None else round(gfap_lo, 3),
             "pct_ptau217_r": None if pct_r is None else round(pct_r, 3),
             "pct_ptau217_n": pct_n,
             "pct_ptau217_ci_lo": None if pct_lo is None else round(pct_lo, 3),
             # Amyloid-positivity enrichment (binary marker -> group difference +
             # point-biserial r + bootstrap CI lower bound), reported alongside the
             # ptau/gfap anchors. amyloid_r is the point-biserial companion number.
             "amyloid_delta": None if amy_delta is None else round(amy_delta, 3),
             "amyloid_r": None if amy_r is None else round(amy_r, 3),
             "amyloid_n": amy_n,
             "amyloid_n_pos": amy_n_pos, "amyloid_n_neg": amy_n_neg,
             "amyloid_ci_lo": None if amy_lo is None else round(amy_lo, 3),
             "synthetic": synthetic_anchor,
             "provenance": ("SYNTHETIC HARNESS — p-tau217/GFAP are calibration "
                            "targets (calibration.CAL), not measured plasma"
                            if synthetic_anchor else "measured")}

    # Pick the primary anchor (p-tau217 preferred, GFAP secondary, C2N %p-tau217
    # tertiary — only reached when neither of the two native markers is usable).
    if ptau_r is not None:
        primary, ci_lo, n_used, marker = ptau_r, ptau_lo, ptau_n, "p-tau217"
    elif gfap_r is not None:
        primary, ci_lo, n_used, marker = gfap_r, gfap_lo, gfap_n, "GFAP"
    elif pct_r is not None:
        primary, ci_lo, n_used, marker = pct_r, pct_lo, pct_n, "%p-tau217"
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
    if amy_delta is not None:
        detail += (f"; amyloid-enrichment Δ={amy_delta:+.3f} "
                   f"(point-biserial r={amy_r:+.2f}, 95% CI lower {amy_lo:+.3f}, "
                   f"n={amy_n}: {amy_n_pos}+/{amy_n_neg}−)")
    if synthetic_anchor:
        detail += " [SYNTHETIC HARNESS: calibration target, not measured plasma]"
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

    # Build a held-out fold by AGGREGATING whole sites (group-disjoint) until the
    # held-out set reaches a meaningful size with both classes, training on the
    # remaining sites. On a many-small-site cohort (ADNI: every individual site is
    # tiny) no single site is large enough to replicate on, so holding out one site
    # is always NA and the star never runs. Pooling the smallest sites into one
    # held-out cohort — while leaving the larger sites to train on — makes the
    # replication star actually informative, and it stays a genuine held-out-SITE
    # test: no site appears in both folds (group-disjoint), not row-level CV.
    MIN_TEST = 40                       # target held-out n for a meaningful fold
    order = uniq[np.argsort(counts)]    # smallest sites first -> largest sites train
    test_sites: list = []
    te = np.zeros(len(groups), dtype=bool)
    tr = ~te
    ok = False
    for g in order:
        test_sites.append(g)
        te = np.isin(groups, test_sites)
        tr = ~te
        if (te.sum() >= MIN_TEST and tr.sum() >= 6
                and len(np.unique(yo[te])) >= 2 and len(np.unique(yo[tr])) >= 2):
            ok = True
            break
    if not ok:
        return TestEvidence("replication", TestResult.NA,
                            "no aggregated held-out site fold reaches a two-class "
                            "cohort of sufficient size with a two-class train set")
    n_test_sites = len(test_sites)
    train_auc = cross_val_auc(Xo[tr], yo[tr], groups=None)
    probe = LinearProbe().fit(Xo[tr], yo[tr])
    from sklearn.metrics import roc_auc_score
    from .probe import _fast_binary_auc

    y_te = np.asarray(yo[te])
    scores_te = np.asarray(probe.decision_scores(Xo[te]))
    binary = len(np.unique(y_te)) == 2
    test_auc = float(roc_auc_score(y_te, scores_te))

    # Bootstrap a 95% CI on the held-out-site AUC instead of trusting a single
    # small-n point estimate (the bare-cutoff antipattern the tool rejects for the
    # other two stars). Gate PASS on the CI lower bound clearing the PASS
    # threshold; WEAKENED when the CI excludes chance (lower bound > 0.5) but does
    # not clear the threshold; FAILED when the CI includes chance. A lucky small-n
    # site whose CI straddles the threshold can therefore no longer be PASSED.
    REPL_PASS = 0.65        # AUC the CI lower bound must clear to PASS
    n_boot = N_BOOT
    rng = np.random.default_rng(0)
    n_te = int(te.sum())
    idx = np.arange(n_te)
    boot = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=n_te, replace=True)
        yb = y_te[b]
        if len(np.unique(yb)) < 2:
            continue
        if binary:
            a = _fast_binary_auc(yb, scores_te[b])
        else:
            try:
                a = float(roc_auc_score(yb, scores_te[b]))
            except Exception:
                a = None
        if a is not None:
            boot.append(a)
    boot = np.asarray(boot, dtype=float)
    if boot.size:
        ci_lo, ci_hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    else:
        ci_lo = ci_hi = None

    # A tiny, perfectly-separable held-out site yields a DEGENERATE bootstrap:
    # every valid resample returns the same boundary AUC, so the CI collapses to
    # a near-zero-width point (e.g. [1.00, 1.00]) that would spuriously clear the
    # PASS threshold. A handful of trivially-separable points is not evidence of
    # generalization — report it as uninformative (NA), dropped from the score.
    # A genuinely small-but-noisy site keeps its wide CI and is still judged
    # normally (typically FAILED when the CI includes chance).
    MIN_INFORMATIVE_TEST = 12
    degenerate = (ci_lo is not None and (ci_hi - ci_lo) < 0.02
                  and n_te < MIN_INFORMATIVE_TEST)

    if ci_lo is None or degenerate:
        res = TestResult.NA
    elif ci_lo >= REPL_PASS:               # CI lower bound clears the threshold
        res = TestResult.PASSED
    elif ci_lo > 0.5:                      # excludes chance but not the threshold
        res = TestResult.WEAKENED
    else:                                  # CI includes chance
        res = TestResult.FAILED
    ci_txt = "" if ci_lo is None else f" [95% CI {ci_lo:.2f}, {ci_hi:.2f}]"
    if degenerate:
        detail = (f"held-out site perfectly separable at n_test={n_te} "
                  f"(AUC={test_auc:.2f}, CI width ~0) — too small to be "
                  f"informative; reported NA")
    else:
        detail = (f"held-out cohort AUC={test_auc:.2f}{ci_txt} "
                  f"(train {train_auc:.2f}); n_test={n_te} "
                  f"over {n_test_sites} aggregated held-out site(s)")
    return TestEvidence("replication", res, detail, {
        "train_auc": round(train_auc, 4), "test_auc": round(test_auc, 4),
        "test_auc_ci_lo": None if ci_lo is None else round(ci_lo, 4),
        "test_auc_ci_hi": None if ci_hi is None else round(ci_hi, 4),
        "n_train": int(tr.sum()), "n_test": n_te,
        "n_test_sites": n_test_sites,
    })


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_gauntlet(df: pd.DataFrame, claim,
                 n_boot: int = N_BOOT, n_perm: int = N_PERM) -> list[TestEvidence]:
    """Run all five gauntlet tests for a claim; NA where data is insufficient.

    `claim` may be a `contract.Claim` or anything with a `.target` attribute;
    a bare string target is also accepted. ``n_boot``/``n_perm`` control the
    leakage-star bootstrap/permutation budget — callers that run many times
    (e.g. the seed-stability sweep) can lower them for speed.
    """
    target = getattr(claim, "target", claim)
    if target not in contract.LABEL_TARGETS:
        target = "conversion"
    return [
        test_age_sex(df, target),
        test_site_scanner(df, target, n_boot=n_boot, n_perm=n_perm),
        test_brain_age(df, target),
        test_biomarker_anchor(df, target),
        test_replication(df, target),
    ]
