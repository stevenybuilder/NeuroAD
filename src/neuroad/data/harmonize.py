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
# Core ComBat on a feature x sample matrix.
# ---------------------------------------------------------------------------
def combat_matrix(X: np.ndarray, batch: np.ndarray,
                  mod: Optional[np.ndarray] = None,
                  eb: bool = True) -> np.ndarray:
    """Parametric ComBat.

    X   : (features, samples) data.
    batch : (samples,) integer batch codes.
    mod : (samples, n_cov) biological covariates to PRESERVE (no intercept), or None.
    Returns harmonized (features, samples).
    """
    X = np.asarray(X, dtype=float)
    n_feat, n_samp = X.shape
    batches = [np.where(batch == b)[0] for b in np.unique(batch)]
    n_batch = len(batches)
    n_per = np.array([len(idx) for idx in batches], dtype=float)

    # Design: full batch dummies (K cols) + covariates. lstsq tolerates the
    # rank deficiency (batch dummies span the intercept).
    batch_design = np.zeros((n_samp, n_batch))
    for j, idx in enumerate(batches):
        batch_design[idx, j] = 1.0
    design = batch_design if mod is None else np.hstack([batch_design, mod])

    B_hat, *_ = np.linalg.lstsq(design, X.T, rcond=None)   # (params, features)
    # Grand (batch-size-weighted) mean over the batch coefficients.
    grand = (n_per / n_per.sum()) @ B_hat[:n_batch]         # (features,)
    stand_mean = np.tile(grand[:, None], (1, n_samp))
    if mod is not None:
        stand_mean += (mod @ B_hat[n_batch:]).T

    resid = X - (design @ B_hat).T
    var_pooled = (resid ** 2) @ np.ones((n_samp, 1)) / n_samp   # (features,1)
    var_pooled = var_pooled.ravel()
    var_pooled[var_pooled == 0] = 1e-8
    s_data = (X - stand_mean) / np.sqrt(var_pooled)[:, None]

    # Batch effect estimates.
    gamma_hat = np.linalg.lstsq(batch_design, s_data.T, rcond=None)[0]  # (K, feat)
    # Per-batch, per-feature scale. A size-1 batch (or a zero-variance feature
    # within a batch) gives NaN/0 under ddof=1 — use the neutral standardized
    # scale of 1.0 there (s_data is globally unit-variance), a location-only
    # correction rather than a NaN.
    # A size-1 batch has no ddof=1 variance; give it the neutral standardized
    # scale of 1.0 (location-only correction) rather than triggering a warning.
    delta_hat = np.array([
        s_data[:, idx].var(axis=1, ddof=1) if len(idx) >= 2 else np.ones(n_feat)
        for idx in batches])
    delta_hat = np.where(np.isfinite(delta_hat) & (delta_hat > 0), delta_hat, 1.0)

    if eb:
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

    bayes = s_data.copy()
    for i, idx in enumerate(batches):
        bayes[:, idx] = ((s_data[:, idx] - gamma_star[i][:, None])
                         / np.sqrt(delta_star[i])[:, None])
    return bayes * np.sqrt(var_pooled)[:, None] + stand_mean


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
    return out
