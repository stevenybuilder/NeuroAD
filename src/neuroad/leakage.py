"""
Leakage instrumentation — the STAR mechanic, in the frontier's currency.

The headline number is a subject-disjoint **leakage margin**:

    margin = outcome_AUC - scanner_AUC

If the same frozen embedding predicts *which machine acquired the scan* as well
as it predicts the disease outcome, the "finding" may be acquisition physics,
not biology. We cite the published prior art for this mechanism
(calibration.PRIOR_ART) and ship the measurement.

Also here:
  * double_dissociation() — scrub the scanner-predicting direction out of the
    embedding and re-measure the outcome. Survivor retains; kill collapses.
  * confound_leaderboard() — rank how much variance each of {scanner/site, age,
    sex} explains in the probe score, so the scientist sees which artifact to fix.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from . import contract
from .probe import (LinearProbe, N_BOOT, N_PERM, auc_ci_perm, cross_val_auc,
                    point_head)

_log = logging.getLogger("neuroad.leakage")


# ---------------------------------------------------------------------------
def _scanner_target(df: pd.DataFrame) -> Optional[str]:
    """Prefer scanner; fall back to site; None if neither varies."""
    if df["scanner"].nunique(dropna=True) > 1:
        return "scanner"
    if df["site"].nunique(dropna=True) > 1:
        return "site"
    return None


def leakage_margin(df: pd.DataFrame, target: str = "conversion",
                   n_boot: int = N_BOOT, n_perm: int = N_PERM) -> dict:
    """outcome_AUC - scanner_AUC (site-disjoint outcome AUC), with uncertainty.

    Returns a JSON-safe dict:
        {outcome_auc, scanner_auc, margin, confound,
         outcome_ci, scanner_ci, outcome_p_perm, scanner_p_perm,
         scanner_ci_excludes_chance,
         margin_ci_lo, margin_ci_hi, margin_p, margin_ci_excludes_zero}

    When there is only one scanner/site, scanner_auc is reported as chance (0.5)
    and the confound is flagged as unavailable.

    Uncertainty. Both AUCs carry a bootstrap 95% CI and a stratified
    label-permutation p (site groups honored on the outcome side). The MARGIN CI
    is a percentile CI of ``outcome_boot - scanner_boot`` (the two AUCs are
    bootstrapped on their own row sets, so this is a slightly conservative,
    independent-resample combination); ``margin_p`` is the permutation null
    P(outcome_auc_perm - scanner_auc >= observed margin), i.e. the chance of the
    outcome beating the scanner by this much when the outcome labels carry no
    real (within-site) signal. This lets the star verdict be stated as
    "margin CI excludes zero" rather than a bare point cutoff.

    Note on the CV asymmetry: outcome_AUC is measured site-disjoint (the harder,
    honest estimate), while scanner_AUC cannot be — holding out the very group
    you are predicting is degenerate — so it uses standard stratified CV. The
    margin is therefore a CONSERVATIVE lower bound on the outcome's edge over the
    confound (it can only understate, never overstate, how much the outcome beats
    the machine), which is the right direction of bias for a skeptic's tool.
    """
    Xo, yo, go = point_head(df, target)
    o = auc_ci_perm(Xo, yo, groups=go, n_boot=n_boot, n_perm=n_perm,
                    return_arrays=True)
    outcome_auc = float(o.get("_auc_full", o["auc"]))
    o_boot = o.get("boot", np.empty(0))
    o_perm = o.get("perm", np.empty(0))

    # ICV (head-size) adjustment for ROI-VOLUME cohorts. On a FreeSurfer named-ROI
    # cohort (df.attrs['region_columns'] set) every feature is a raw regional
    # volume, so a bigger head inflates the ROI and the outcome AUROC partly rides
    # on head size, not atrophy. We report an ICV-residualized companion AUROC
    # (nuisance fit INSIDE each fold, site groups honored — the same fold-honest
    # machinery as the age/sex adjustment) so the displayed headline can be the
    # head-size-adjusted number, with the raw kept for audit. This is additive:
    # the score/margin path below still uses the raw `outcome_auc`, so promotion
    # decisions are unchanged — only the DISPLAYED region AUROC becomes honest.
    outcome_auc_icv_adj = None
    icv_adjusted = False
    if target in ("conversion", "dx_binary") and "icv" in df.columns \
            and df.attrs.get("region_columns"):
        try:
            if target == "conversion":
                m_icv = pd.to_numeric(df["conversion"], errors="coerce").notna().to_numpy()
            else:
                m_icv = df["dx"].astype("string").map({"AD": 1, "CN": 0}).notna().to_numpy()
            icv_v = pd.to_numeric(df["icv"], errors="coerce").to_numpy(float)[m_icv]
            if np.isfinite(icv_v).sum() >= 5 and np.nanstd(icv_v) > 0:
                icv_v = np.nan_to_num(icv_v, nan=np.nanmean(icv_v))
                from .probe import residualized_cross_val_auc
                adj = residualized_cross_val_auc(
                    Xo, icv_v.reshape(-1, 1), yo, go, kind="covariate")
                # residualized_cross_val_auc returns a 0.5 sentinel (WITHOUT
                # raising) when the CV is infeasible — too thin, or a single-class
                # evaluated fold. Treat an exact 0.5 as "not computed" rather than
                # a genuine head-size null, so the FE never labels an uncomputed
                # value "head-size corrected"; the header then shows raw only.
                if abs(float(adj) - 0.5) > 1e-9:
                    outcome_auc_icv_adj = round(float(adj), 4)
                    icv_adjusted = True
        except Exception as exc:  # noqa: BLE001
            _log.debug("ICV-adjusted outcome AUC failed, reporting raw only: %r", exc)

    conf = _scanner_target(df)
    if conf is None:
        scanner_auc = 0.5
        s = {"ci_lo": None, "ci_hi": None, "p_perm": None,
             "ci_excludes_chance": False}
        s_boot = np.empty(0)
    else:
        Xs, ys, _ = point_head(df, conf)
        # No group-aware CV here: we WANT to see the machine signal.
        s = auc_ci_perm(Xs, ys, groups=None, n_boot=n_boot, n_perm=n_perm,
                        return_arrays=True)
        scanner_auc = float(s.get("_auc_full", s["auc"]))
        s_boot = s.get("boot", np.empty(0))

    margin = outcome_auc - scanner_auc

    # Within-acquisition-stratum invariance — the DECISIVE de-confound. A confound
    # (e.g. field strength) cannot explain a contrast that is measured inside a
    # single level of it. We re-measure the outcome AUROC restricted to each
    # scanner/site level; if the effect holds at ~the pooled value within every
    # stratum, the pooled number is not the confound leaking. This is the honest
    # evidence to DISPLAY — the whole-cohort scanner_auc above can look optimistic
    # when the harmonization saw the held-out rows, so it must not be shown as
    # "scanner ruled out". Additive: the score/margin path is untouched.
    fs_invariance = _within_stratum_invariance(df, target, outcome_auc, conf)

    # Margin CI: percentile CI of outcome_boot - scanner_boot (paired by index;
    # truncated to the shorter run). Independent resampling of the two row sets
    # makes this a conservative (slightly wide) CI — the honest direction.
    margin_ci_lo = margin_ci_hi = None
    if o_boot.size and s_boot.size:
        m = min(o_boot.size, s_boot.size)
        diff = o_boot[:m] - s_boot[:m]
        margin_ci_lo, margin_ci_hi = (float(v) for v in np.percentile(diff, [2.5, 97.5]))
    elif o_boot.size and conf is None:
        margin_ci_lo, margin_ci_hi = (float(v) for v in
                                      np.percentile(o_boot - 0.5, [2.5, 97.5]))

    # Margin permutation p: does the outcome beat the (fixed) scanner AUC by more
    # than the null of shuffled-within-site outcome labels?
    margin_p = None
    if o_perm.size:
        null_margin = o_perm - scanner_auc
        margin_p = float((1 + int(np.sum(null_margin >= margin))) / (1 + o_perm.size))

    # Benjamini-Hochberg FDR correction across the three genuine permutation
    # p-values the star emits (outcome, scanner, margin). These are the only
    # true label-permutation p's in the gauntlet; the other tests gate on
    # retained-fraction / CI, not p_perm, so are not part of this family.
    from .harness.validation import benjamini_hochberg
    outcome_q, scanner_q, margin_q = benjamini_hochberg(
        [o["p_perm"], s["p_perm"], margin_p])

    return {
        "outcome_auc": round(float(outcome_auc), 4),
        "outcome_auc_icv_adj": outcome_auc_icv_adj,
        "icv_adjusted": icv_adjusted,
        "scanner_auc": round(float(scanner_auc), 4),
        "margin": round(float(margin), 4),
        "confound": conf if conf is not None else "none (single scanner/site)",
        "outcome_ci": None if o["ci_lo"] is None else [o["ci_lo"], o["ci_hi"]],
        "scanner_ci": None if s["ci_lo"] is None else [s["ci_lo"], s["ci_hi"]],
        "outcome_p_perm": o["p_perm"],
        "scanner_p_perm": s["p_perm"],
        "outcome_q": None if outcome_q is None else round(outcome_q, 4),
        "scanner_q": None if scanner_q is None else round(scanner_q, 4),
        "margin_q": None if margin_q is None else round(margin_q, 4),
        "scanner_ci_excludes_chance": bool(s["ci_excludes_chance"]),
        "margin_ci_lo": None if margin_ci_lo is None else round(margin_ci_lo, 4),
        "margin_ci_hi": None if margin_ci_hi is None else round(margin_ci_hi, 4),
        "margin_p": None if margin_p is None else round(margin_p, 4),
        "margin_ci_excludes_zero": bool(margin_ci_lo is not None and margin_ci_lo > 0),
        "field_strength_invariance": fs_invariance,
    }


# ---------------------------------------------------------------------------
def _within_stratum_invariance(df: pd.DataFrame, target: str,
                               pooled_auc: float, conf: Optional[str],
                               band: float = 0.05, min_n: int = 40) -> Optional[dict]:
    """Re-measure the outcome AUROC inside each single acquisition stratum.

    A confound (scanner / field strength / site) cannot inflate a contrast that
    is computed within one level of it. If the outcome holds at ~``pooled_auc``
    within every stratum, the pooled effect is not that confound leaking — the
    honest, decisive de-confound (mirrors the reference paper's ±0.05 AUROC
    field-strength equivalence band, Nature Medicine doi:10.1038/s41591-026-04497-1
    Extended Data Fig. 6d).

    Returns ``None`` when it does not apply (non-outcome target, single stratum,
    or every stratum too thin / single-class) so callers can fall back to the
    scanner-margin display. Uses the same site-disjoint ``cross_val_auc`` as the
    headline, so each stratum AUROC is measured on equal footing.
    """
    if target not in ("dx_binary", "conversion") or conf not in ("scanner", "site"):
        return None
    try:
        levels = [lv for lv in df[conf].dropna().unique()]
    except Exception:  # noqa: BLE001
        return None
    if len(levels) < 2:
        return None

    strata: list[dict] = []
    for lv in levels:
        sub = df[df[conf].astype("string") == str(lv)]
        try:
            Xs, ys, gs = point_head(sub, target)
        except Exception:  # noqa: BLE001
            continue
        if len(ys) < min_n or len(np.unique(ys)) < 2:
            continue
        try:
            auc = float(cross_val_auc(Xs, ys, groups=gs))
        except Exception:  # noqa: BLE001
            continue
        # A degenerate CV returns the 0.5 sentinel — skip rather than show a
        # spurious "chance within stratum".
        if abs(auc - 0.5) < 1e-9:
            continue
        strata.append({"level": str(lv), "auc": round(auc, 4), "n": int(len(ys))})

    if len(strata) < 2:
        return None
    max_abs_delta = max(abs(s["auc"] - pooled_auc) for s in strata)
    strata.sort(key=lambda s: s["n"], reverse=True)
    return {
        "confound": conf,
        "pooled_auc": round(float(pooled_auc), 4),
        "strata": strata,
        "max_abs_delta": round(float(max_abs_delta), 4),
        "band": band,
        "within_band": bool(max_abs_delta <= band),
    }


# ---------------------------------------------------------------------------
def label_shuffle_auc(df: pd.DataFrame, target: str = "conversion",
                      seed: int = 0) -> float:
    """Negative control: cross-validated AUC with the outcome labels PERMUTED.

    Under a leakage-free pipeline a shuffled label carries no signal, so this
    must sit near chance (~0.5). A value materially above chance flags residual
    leakage — e.g. a whole-cohort harmonization that let the batch correction
    peek at the held-out fold — because the probe is then decoding structure that
    survives label destruction. Uses the SAME site-disjoint CV machinery as the
    headline so the control is measured on equal footing.
    """
    X, y, go = point_head(df, target)
    if len(np.unique(y)) < 2:
        return 0.5
    y_shuffled = np.random.default_rng(seed).permutation(y)
    return cross_val_auc(X, y_shuffled, groups=go)


# ---------------------------------------------------------------------------
def _scanner_directions(X: np.ndarray, scanner_codes: np.ndarray,
                        max_dirs: int = 2) -> np.ndarray:
    """Orthonormal basis of the top scanner-discriminating direction(s).

    Uses LDA between scanner classes; falls back to the class-mean difference
    for the degenerate binary case. Columns are the directions to project out.
    """
    classes = np.unique(scanner_codes)
    if len(classes) < 2:
        return np.zeros((X.shape[1], 0))

    Xs = StandardScaler().fit_transform(X)
    try:
        lda = LinearDiscriminantAnalysis()
        lda.fit(Xs, scanner_codes)
        dirs = lda.scalings_[:, :min(max_dirs, len(classes) - 1)]
    except Exception:
        m1 = Xs[scanner_codes == classes[0]].mean(axis=0)
        m0 = Xs[scanner_codes != classes[0]].mean(axis=0)
        dirs = (m1 - m0).reshape(-1, 1)

    # Orthonormalize (QR); drop null columns.
    dirs = np.asarray(dirs, dtype=float)
    if dirs.shape[1] == 0 or not np.isfinite(dirs).all():
        return np.zeros((X.shape[1], 0))
    q, _ = np.linalg.qr(dirs)
    return q


def double_dissociation(df: pd.DataFrame, target: str) -> dict:
    """Residualize the embedding against the scanner-predicting direction, then
    re-measure the outcome AUC.

    Returns {'auc_before','auc_after','retained','confound','n'}. Survivors keep
    most of their effect (retained -> 1); kills collapse toward chance.
    """
    Xo, yo, go = point_head(df, target)
    auc_before = cross_val_auc(Xo, yo, groups=go)

    conf = _scanner_target(df)
    if conf is None or len(Xo) == 0:
        return {
            "auc_before": round(float(auc_before), 4),
            "auc_after": round(float(auc_before), 4),
            "retained": 1.0,
            "confound": "none (single scanner/site)",
            "n": int(len(Xo)),
        }

    # Scanner codes aligned to the SAME rows used for the outcome probe.
    Xs, ys, _ = point_head(df, conf)
    # point_head(outcome) and point_head(scanner) may drop different rows; align
    # by recomputing scanner codes on the outcome-kept rows.
    conf_col = df[conf].astype("string")
    if target == "conversion":
        keep = pd.to_numeric(df["conversion"], errors="coerce").notna().to_numpy()
    elif target == "dx_binary":
        keep = df["dx"].astype("string").map({"AD": 1, "CN": 0}).notna().to_numpy()
    else:
        keep = np.ones(len(df), dtype=bool)
    conf_codes = conf_col.fillna("__na__").to_numpy()[keep]
    _, conf_codes = np.unique(conf_codes, return_inverse=True)

    # Fold-honest: fit the scanner-direction (StandardScaler+LDA) on TRAIN rows of
    # each fold only and project the scanner subspace out of both train and test,
    # instead of the whole-cohort _scanner_directions fit + full-X projection
    # (which let the scrub see the held-out rows).
    from .probe import residualized_cross_val_auc
    auc_after = residualized_cross_val_auc(Xo, conf_codes, yo, go,
                                           kind="scanner_lda")

    eff_before = max(auc_before - 0.5, 1e-6)
    retained = float(np.clip((auc_after - 0.5) / eff_before, 0.0, 1.5))

    return {
        "auc_before": round(float(auc_before), 4),
        "auc_after": round(float(auc_after), 4),
        "retained": round(retained, 3),
        "confound": conf,
        "n": int(len(yo)),
    }


# ---------------------------------------------------------------------------
def _eta_squared(score: np.ndarray, groups: np.ndarray) -> float:
    """One-way ANOVA eta^2: fraction of score variance explained by a category."""
    grand = score.mean()
    ss_total = float(((score - grand) ** 2).sum())
    if ss_total <= 0:
        return 0.0
    ss_between = 0.0
    for g in np.unique(groups):
        s = score[groups == g]
        if len(s):
            ss_between += len(s) * (s.mean() - grand) ** 2
    return float(np.clip(ss_between / ss_total, 0.0, 1.0))


def _r2_continuous(score: np.ndarray, x: np.ndarray) -> float:
    """R^2 of a simple linear fit score ~ x (fraction of variance explained)."""
    ok = np.isfinite(x)
    if ok.sum() < 3:
        return 0.0
    s, xx = score[ok], x[ok]
    if np.std(xx) == 0 or np.std(s) == 0:
        return 0.0
    r = np.corrcoef(s, xx)[0, 1]
    return float(0.0 if not np.isfinite(r) else r ** 2)


def confound_leaderboard(df: pd.DataFrame, target: str = "conversion") -> list[dict]:
    """Rank the variance each confound explains in the in-sample probe score.

    Returns [{'confound','variance_explained'}, ...] sorted descending. The
    confounds considered are the acquisition batch (scanner or site), age, and
    sex — the three most likely to masquerade as biology.
    """
    Xo, yo, go = point_head(df, target)
    if len(np.unique(yo)) < 2:
        return []
    # Out-of-fold, site-disjoint probe score (mirrors gauntlet._oof_scores but
    # honors the site groups): an in-sample score would let overfit residuals
    # inflate every confound's apparent variance-explained. A full-length OOF
    # vector keeps the metadata alignment below unchanged.
    from sklearn.model_selection import cross_val_predict
    from .probe import _n_splits, build_probe_pipeline
    y_codes = np.searchsorted(np.unique(yo), yo)
    n_splits = _n_splits(y_codes, go)
    pipe, _ = build_probe_pipeline(Xo.shape[0], Xo.shape[1])
    splits = None
    if go is not None and len(np.unique(go)) >= n_splits:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=0)
        candidate = list(splitter.split(Xo, y_codes, np.asarray(go)))
        # cross_val_predict cannot skip folds, so only use the site-disjoint
        # splits when every train fold carries both classes; otherwise fall back
        # to plain stratified CV so sparsely-labelled sub-cohorts still complete.
        if all(len(np.unique(y_codes[tr])) >= 2 for tr, _ in candidate):
            splits = candidate
    if splits is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=0)
        splits = list(splitter.split(Xo, y_codes))
    proba = cross_val_predict(pipe, Xo, y_codes, cv=splits,
                              method="predict_proba")
    score = proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)

    # Recover the outcome-kept rows to align metadata with the score.
    if target == "conversion":
        keep = pd.to_numeric(df["conversion"], errors="coerce").notna().to_numpy()
    elif target == "dx_binary":
        keep = df["dx"].astype("string").map({"AD": 1, "CN": 0}).notna().to_numpy()
    else:
        keep = np.ones(len(df), dtype=bool)
    sub = df.loc[keep]

    rows: list[dict] = []
    batch = "scanner" if sub["scanner"].nunique(dropna=True) > 1 else "site"
    if sub[batch].nunique(dropna=True) > 1:
        g = sub[batch].astype("string").fillna("__na__").to_numpy()
        rows.append({"confound": batch, "variance_explained": round(_eta_squared(score, g), 4)})
    if sub["age"].notna().any():
        rows.append({"confound": "age",
                     "variance_explained": round(_r2_continuous(score, sub["age"].to_numpy(float)), 4)})
    if sub["sex"].nunique(dropna=True) > 1:
        g = sub["sex"].astype("string").fillna("__na__").to_numpy()
        rows.append({"confound": "sex", "variance_explained": round(_eta_squared(score, g), 4)})

    rows.sort(key=lambda r: r["variance_explained"], reverse=True)
    return rows
