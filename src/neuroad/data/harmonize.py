"""
ComBat harmonization — a REAL de-confound, stronger than dropping a field strength.

The 3T-only survivor removes the 3T/1.5T acquisition confound by throwing away
every 1.5T scan (a third of the cohort). ComBat (Johnson, Li & Rabinovic 2007,
*Biostatistics*) instead *removes the batch location/scale effect* from the
features while preserving biological variation, so the whole cohort stays in
play. Empirical-Bayes shrinkage pools batch estimates, which matters when some
batches (here: sites) are small.

Two deliberate stances for a referee tool:

  * **Label-blind.** We protect `age` and `sex` as biological covariates but NOT
    `dx`. Harmonizing with the diagnosis label in the design would let the batch
    correction *see the outcome* on the full dataset before the probe is
    cross-validated — a subtle leakage the referee exists to catch. Protecting
    only age/sex keeps the correction honest: if AD-vs-CN still separates after
    a label-blind harmonization, the signal is not the harmonizer's doing.
  * **Auditable, dependency-free.** A compact parametric ComBat implemented here
    rather than a pinned third-party package, and validated empirically at call
    sites (batch should become ~unpredictable while age/dx signal survives).

This is a pure-pandas/numpy transform over a contract table: it rewrites the
`emb_*` columns and returns a new contract-valid frame, so the whole referee
runs on the harmonized cohort unchanged (it is just another feeder).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .. import contract


# ---------------------------------------------------------------------------
# Empirical-Bayes helpers (parametric ComBat, Johnson et al. 2007).
# ---------------------------------------------------------------------------
def _aprior(delta_hat: np.ndarray) -> np.ndarray:
    m = delta_hat.mean(axis=1)
    s2 = delta_hat.var(axis=1, ddof=1)
    s2 = np.where(s2 > 0, s2, np.finfo(float).tiny)   # degenerate batch -> no EB shrinkage
    return (2 * s2 + m**2) / s2


def _bprior(delta_hat: np.ndarray) -> np.ndarray:
    m = delta_hat.mean(axis=1)
    s2 = delta_hat.var(axis=1, ddof=1)
    s2 = np.where(s2 > 0, s2, np.finfo(float).tiny)
    return (m * s2 + m**3) / s2


def _postmean(g_hat, g_bar, n, d_star, t2):
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def _postvar(sum2, n, a, b):
    denom = n / 2.0 + a - 1.0
    denom = np.where(np.abs(denom) > 1e-12, denom, 1e-12)
    return (0.5 * sum2 + b) / denom


def _it_sol(s_data, g_hat, d_hat, g_bar, t2, a, b, tol=1e-4, max_iter=200):
    """Iterative empirical-Bayes solution for one batch (features x samples)."""
    n = (~np.isnan(s_data)).sum(axis=1)
    g_old, d_old = g_hat.copy(), d_hat.copy()
    for _ in range(max_iter):
        g_new = _postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = ((s_data - g_new[:, None]) ** 2).sum(axis=1)
        d_new = _postvar(sum2, n, a, b)
        dchange = max(np.max(np.abs(g_new - g_old) / (np.abs(g_old) + 1e-8)),
                      np.max(np.abs(d_new - d_old) / (np.abs(d_old) + 1e-8)))
        g_old, d_old = g_new, d_new
        if dchange < tol:
            break
    return g_old, d_old


# ---------------------------------------------------------------------------
# Core ComBat on a feature x sample matrix — as a fit/transform pair.
# ---------------------------------------------------------------------------
class ComBatModel:
    """Parametric ComBat as a leakage-honest fit/transform estimator.

    ``fit`` learns the batch location/scale corrections on the TRAIN rows only
    (grand mean, covariate coefficients, pooled variance, per-batch EB
    gamma*/delta*, and the batch vocabulary). ``transform`` applies them to any
    rows: a row whose batch was UNSEEN in fit passes through with the neutral
    location-only correction (gamma=0, delta=1), the same degenerate-batch
    handling the whole-cohort path uses. ``fit`` then ``transform`` on the SAME
    rows reproduces :func:`combat_matrix` exactly (it is now the thin wrapper).
    """

    def __init__(self, eb: bool = True):
        self.eb = eb
        self.grand: Optional[np.ndarray] = None       # (features,)
        self.beta_mod: Optional[np.ndarray] = None    # (n_cov, features) or None
        self.var_pooled: Optional[np.ndarray] = None  # (features,)
        self.batch_values: Optional[np.ndarray] = None
        self.gamma_star: Optional[np.ndarray] = None  # (K, features)
        self.delta_star: Optional[np.ndarray] = None  # (K, features)
        self._index: dict = {}

    def fit(self, X: np.ndarray, batch: np.ndarray,
            mod: Optional[np.ndarray] = None) -> "ComBatModel":
        X = np.asarray(X, dtype=float)
        n_feat, n_samp = X.shape
        uniq = np.unique(batch)
        batches = [np.where(batch == b)[0] for b in uniq]
        n_batch = len(batches)
        n_per = np.array([len(idx) for idx in batches], dtype=float)

        # Design: full batch dummies (K cols) + covariates. lstsq tolerates the
        # rank deficiency (batch dummies span the intercept).
        batch_design = np.zeros((n_samp, n_batch))
        for j, idx in enumerate(batches):
            batch_design[idx, j] = 1.0
        design = batch_design if mod is None else np.hstack([batch_design, mod])

        B_hat, *_ = np.linalg.lstsq(design, X.T, rcond=None)   # (params, features)
        grand = (n_per / n_per.sum()) @ B_hat[:n_batch]         # (features,)
        self.grand = grand
        self.beta_mod = None if mod is None else B_hat[n_batch:]
        stand_mean = np.tile(grand[:, None], (1, n_samp))
        if mod is not None:
            stand_mean += (mod @ B_hat[n_batch:]).T

        resid = X - (design @ B_hat).T
        var_pooled = (resid ** 2) @ np.ones((n_samp, 1)) / n_samp   # (features,1)
        var_pooled = var_pooled.ravel()
        var_pooled[var_pooled == 0] = 1e-8
        self.var_pooled = var_pooled
        s_data = (X - stand_mean) / np.sqrt(var_pooled)[:, None]

        # Batch effect estimates.
        gamma_hat = np.linalg.lstsq(batch_design, s_data.T, rcond=None)[0]  # (K, feat)
        # A size-1 batch has no ddof=1 variance; give it the neutral standardized
        # scale of 1.0 (location-only correction) rather than triggering a warning.
        delta_hat = np.array([
            s_data[:, idx].var(axis=1, ddof=1) if len(idx) >= 2 else np.ones(n_feat)
            for idx in batches])
        delta_hat = np.where(np.isfinite(delta_hat) & (delta_hat > 0), delta_hat, 1.0)

        if self.eb:
            gamma_bar = gamma_hat.mean(axis=1)
            t2 = gamma_hat.var(axis=1, ddof=1)
            a_prior, b_prior = _aprior(delta_hat), _bprior(delta_hat)
            gamma_star, delta_star = [], []
            for i, idx in enumerate(batches):
                g, d = _it_sol(s_data[:, idx], gamma_hat[i], delta_hat[i],
                               gamma_bar[i], t2[i], a_prior[i], b_prior[i])
                gamma_star.append(g)
                delta_star.append(d)
            gamma_star = np.array(gamma_star)
            delta_star = np.array(delta_star)
            # EB on a tiny batch can produce a non-finite / non-positive scale; fall
            # back to the raw batch estimate there so no feature becomes NaN.
            bad_d = ~(np.isfinite(delta_star) & (delta_star > 0))
            delta_star = np.where(bad_d, delta_hat, delta_star)
            gamma_star = np.where(np.isfinite(gamma_star), gamma_star, gamma_hat)
        else:
            gamma_star, delta_star = gamma_hat, delta_hat

        self.batch_values = uniq
        self.gamma_star = gamma_star
        self.delta_star = delta_star
        self._index = {b: i for i, b in enumerate(uniq)}
        return self

    def transform(self, X: np.ndarray, batch: np.ndarray,
                  mod: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n_feat, n_samp = X.shape
        stand_mean = np.tile(self.grand[:, None], (1, n_samp))
        if self.beta_mod is not None and mod is not None:
            stand_mean += (mod @ self.beta_mod).T
        var_pooled = self.var_pooled
        s_data = (X - stand_mean) / np.sqrt(var_pooled)[:, None]

        # Per-sample gamma/delta lookup; a batch unseen in fit gets the neutral
        # (gamma=0, delta=1) location-only correction.
        gamma_col = np.zeros((n_feat, n_samp))
        delta_col = np.ones((n_feat, n_samp))
        for j in range(n_samp):
            i = self._index.get(batch[j])
            if i is not None:
                gamma_col[:, j] = self.gamma_star[i]
                delta_col[:, j] = self.delta_star[i]
        bayes = (s_data - gamma_col) / np.sqrt(delta_col)
        return bayes * np.sqrt(var_pooled)[:, None] + stand_mean


def combat_matrix(X: np.ndarray, batch: np.ndarray,
                  mod: Optional[np.ndarray] = None,
                  eb: bool = True) -> np.ndarray:
    """Parametric ComBat (whole-cohort). Thin wrapper over :class:`ComBatModel`
    fit-then-transform on the same rows — output is identical to the historical
    inline implementation.

    X   : (features, samples) data.
    batch : (samples,) integer batch codes.
    mod : (samples, n_cov) biological covariates to PRESERVE (no intercept), or None.
    Returns harmonized (features, samples).
    """
    model = ComBatModel(eb=eb).fit(X, batch, mod=mod)
    return model.transform(X, batch, mod=mod)


# ---------------------------------------------------------------------------
# Contract-table wrapper.
# ---------------------------------------------------------------------------
def harmonize(df: pd.DataFrame, *, batch: str = "site",
              covariates: Sequence[str] = ("age", "sex"),
              eb: bool = True) -> pd.DataFrame:
    """Return a copy of contract table ``df`` with ComBat-harmonized ``emb_*``.

    ``batch`` is the acquisition variable to remove (``site`` or ``scanner``).
    ``covariates`` are the biological signals to PRESERVE (label-blind: ``dx`` is
    intentionally excluded). Batches with a single member cannot be scale-
    estimated and are passed through unchanged (merged into a residual group so
    the correction still runs on the rest).
    """
    cols = contract.embedding_columns(df)
    if not cols:
        raise contract.ContractError("harmonize: no emb_* columns to harmonize")
    out = df.copy().reset_index(drop=True)

    b_raw = out[batch].astype("string").fillna("NA")
    # Merge singleton batches into one 'other' group (ComBat needs >=2 to estimate scale).
    counts = b_raw.value_counts()
    singletons = set(counts[counts < 2].index)
    b_raw = b_raw.where(~b_raw.isin(singletons), other="__other__")
    batch_codes = pd.Categorical(b_raw).codes.astype(int)

    # Biological covariates to preserve: numeric, mean/mode-imputed, no intercept.
    mod_cols = []
    for c in covariates:
        if c not in out.columns:
            continue
        if c == "sex":
            v = (out["sex"].astype("string").str.upper() == "F").astype(float)
        else:
            v = pd.to_numeric(out[c], errors="coerce")
            v = v.fillna(v.mean())
        mod_cols.append(np.asarray(v, dtype=float))
    mod = np.column_stack(mod_cols) if mod_cols else None

    X = out[cols].to_numpy(dtype=float).T          # (features, samples)
    Xh = combat_matrix(X, batch_codes, mod=mod, eb=eb)
    out[cols] = Xh.T
    out.attrs.update(df.attrs)
    out.attrs["harmonized"] = f"combat(batch={batch}, covariates={list(covariates)}, eb={eb})"
    # LOUD leakage caveat: this whole-cohort ComBat is fit on ALL rows (including
    # what will later become the CV test folds), so the batch correction has seen
    # the held-out rows. It is a convenience feeder, NOT a fold-honest number.
    out.attrs["harmonized_leakage_caveat"] = (
        "whole-cohort ComBat (fit on all rows incl. test) — NOT fold-honest; "
        "use harmonize.combat_cv_auc for the leakage-honest AUC")
    return out


# ---------------------------------------------------------------------------
# Fold-honest ComBat: fit the batch correction INSIDE each CV fold on train rows.
# ---------------------------------------------------------------------------
def _covariate_matrix(sub: pd.DataFrame,
                      covariates: Sequence[str]) -> Optional[np.ndarray]:
    """Biological covariate design (mean/mode-imputed, no intercept) — the same
    construction :func:`harmonize` uses, over the given row subset."""
    cols = []
    for c in covariates:
        if c not in sub.columns:
            continue
        if c == "sex":
            v = (sub["sex"].astype("string").str.upper() == "F").astype(float)
        else:
            v = pd.to_numeric(sub[c], errors="coerce")
            v = v.fillna(v.mean())
        cols.append(np.asarray(v, dtype=float))
    return np.column_stack(cols) if cols else None


def combat_cv_auc(df: pd.DataFrame, target: str, batch: str = "scanner",
                  covariates: Sequence[str] = ("age", "sex")) -> float:
    """Leakage-honest AUC under ComBat harmonization.

    Runs the SAME site-disjoint CV as the headline probe, but fits ComBat on the
    TRAIN rows of each fold only (:class:`ComBatModel`), transforms both the train
    and test embeddings with that fitted model, then fits the linear probe on the
    transformed train and scores the transformed test. Because the batch
    correction never sees the held-out fold, this is the number the whole-cohort
    ``adni:combat`` feeder inflates. Returns 0.5 when the data is too thin.
    """
    from ..probe import (LinearProbe, RANDOM_STATE, _auc_from_oof, _n_splits,
                         point_head)
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    X, y, groups = point_head(df, target)               # X: (samples, features)
    classes = np.unique(y)
    if len(classes) < 2 or len(y) < 4:
        return 0.5
    y_codes = np.searchsorted(classes, y)

    # Align batch codes + covariates to the outcome-kept rows point_head returned.
    if target == "conversion":
        keep = pd.to_numeric(df["conversion"], errors="coerce").notna().to_numpy()
    elif target == "dx_binary":
        keep = df["dx"].astype("string").map({"AD": 1, "CN": 0}).notna().to_numpy()
    else:
        keep = np.ones(len(df), dtype=bool)
    sub = df.loc[keep]
    if batch not in sub.columns:
        return 0.5
    batch_codes = sub[batch].astype("string").fillna("__na__").to_numpy()
    mod_all = _covariate_matrix(sub, covariates)

    n_splits = _n_splits(y_codes, groups)
    if n_splits < 2:
        return 0.5
    use_groups = groups is not None and len(np.unique(groups)) >= n_splits
    if use_groups:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=RANDOM_STATE)
        split_iter = splitter.split(X, y_codes, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=RANDOM_STATE)
        split_iter = splitter.split(X, y_codes)

    oof = np.full((len(y_codes), len(classes)), np.nan)
    for tr, te in split_iter:
        if len(np.unique(y_codes[tr])) < 2:
            continue
        mod_tr = None if mod_all is None else mod_all[tr]
        mod_te = None if mod_all is None else mod_all[te]
        model = ComBatModel(eb=True).fit(X[tr].T, batch_codes[tr], mod=mod_tr)
        Xtr_h = model.transform(X[tr].T, batch_codes[tr], mod=mod_tr).T
        Xte_h = model.transform(X[te].T, batch_codes[te], mod=mod_te).T
        probe = LinearProbe().fit(Xtr_h, y_codes[tr])
        proba = probe.predict_proba(Xte_h)
        for j, cls in enumerate(probe.classes_):
            oof[te, cls] = proba[:, j]

    evaluated = ~np.isnan(oof).any(axis=1)
    if evaluated.sum() < 2 or len(np.unique(y_codes[evaluated])) < 2:
        return 0.5
    auc = _auc_from_oof(y_codes[evaluated], oof[evaluated], classes)
    return 0.5 if auc is None else float(auc)
