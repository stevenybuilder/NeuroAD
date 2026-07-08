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

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from . import contract
from .probe import LinearProbe, cross_val_auc, point_head


# ---------------------------------------------------------------------------
def _scanner_target(df: pd.DataFrame) -> Optional[str]:
    """Prefer scanner; fall back to site; None if neither varies."""
    if df["scanner"].nunique(dropna=True) > 1:
        return "scanner"
    if df["site"].nunique(dropna=True) > 1:
        return "site"
    return None


def leakage_margin(df: pd.DataFrame, target: str = "conversion") -> dict:
    """outcome_AUC - scanner_AUC (site-disjoint outcome AUC).

    Returns {'outcome_auc', 'scanner_auc', 'margin', 'confound'}. When there is
    only one scanner/site, scanner_auc is reported as chance (0.5) and the
    confound is flagged as unavailable.

    Note on the CV asymmetry: outcome_AUC is measured site-disjoint (the harder,
    honest estimate), while scanner_AUC cannot be — holding out the very group
    you are predicting is degenerate — so it uses standard stratified CV. The
    margin is therefore a CONSERVATIVE lower bound on the outcome's edge over the
    confound (it can only understate, never overstate, how much the outcome beats
    the machine), which is the right direction of bias for a skeptic's tool.
    """
    Xo, yo, go = point_head(df, target)
    outcome_auc = cross_val_auc(Xo, yo, groups=go)

    conf = _scanner_target(df)
    if conf is None:
        scanner_auc = 0.5
    else:
        Xs, ys, _ = point_head(df, conf)
        # No group-aware CV here: we WANT to see the machine signal.
        scanner_auc = cross_val_auc(Xs, ys, groups=None)

    return {
        "outcome_auc": round(float(outcome_auc), 4),
        "scanner_auc": round(float(scanner_auc), 4),
        "margin": round(float(outcome_auc - scanner_auc), 4),
        "confound": conf if conf is not None else "none (single scanner/site)",
    }


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

    dirs = _scanner_directions(Xo, conf_codes)
    if dirs.shape[1] == 0:
        X_res = Xo
    else:
        # project each row out of the scanner subspace
        proj = Xo @ dirs @ dirs.T
        X_res = Xo - proj

    auc_after = cross_val_auc(X_res, yo, groups=go)

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
    Xo, yo, _ = point_head(df, target)
    if len(np.unique(yo)) < 2:
        return []
    score = LinearProbe().fit(Xo, yo).decision_scores(Xo)

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
