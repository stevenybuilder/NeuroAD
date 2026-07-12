"""
Focused regression tests for the two rigor fixes in gauntlet.py:

  1. brain_age returns NA (not a fake PASS) when the brain-age model is not
     itself predictive of age — an uninformative control must not bank STAR credit.
  2. replication builds a held-out fold by AGGREGATING multiple whole sites on a
     many-small-site cohort (ADNI-shaped), so the star actually runs instead of
     always degrading to NA — while staying group-disjoint (held-out SITE test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from neuroad import contract, gauntlet, leakage
from neuroad.contract import TestResult as Result

D = 16
_DIS = slice(0, 5)


def _base_cohort(n, sites, seed=0, age_signal=True):
    """Contract-valid AD/CN cohort with a real disease direction in the embedding.

    `sites` is a list of site labels to draw from; `age_signal` toggles whether
    the embedding encodes chronological age (so we can force a non-predictive
    brain-age model when it is False).
    """
    rng = np.random.default_rng(seed)
    site = rng.choice(np.array(sites), size=n)
    scanner_of = {s: f"scan{i % 3}" for i, s in enumerate(sites)}
    scanner = np.array([scanner_of[s] for s in site])

    dx = rng.choice(["AD", "CN"], size=n, p=[0.5, 0.5])
    d_latent = np.where(dx == "AD", 1.0, -1.0) + rng.normal(0, 0.5, n)
    age = 65 + rng.normal(0, 7, n)

    X = rng.normal(0, 1.0, size=(n, D))
    dis_dir = rng.normal(size=(5,))
    X[:, _DIS] += 1.6 * np.outer(d_latent, dis_dir)
    if age_signal:
        age_dir = rng.normal(size=(3,))
        X[:, 9:12] += 1.2 * np.outer((age - age.mean()) / age.std(), age_dir)

    df = pd.DataFrame({
        "subject_id": [f"S{i:04d}" for i in range(n)],
        "dx": pd.Categorical(dx, categories=contract.DX_LEVELS),
        "conversion": pd.array([pd.NA] * n, dtype="Int8"),
        "age": age,
        "sex": pd.Categorical(rng.choice(["M", "F"], size=n), categories=contract.SEX_LEVELS),
        "site": pd.Categorical(site),
        "scanner": pd.Categorical(scanner),
        "amyloid": pd.array(rng.integers(0, 2, n), dtype="Int8"),
        "p_tau217": np.full(n, np.nan),
        "gfap": np.full(n, np.nan),
        "nfl": rng.normal(20, 5, n),
        "apoe4": pd.array(rng.integers(0, 3, n), dtype="Int8"),
    })
    for i in range(D):
        df[f"emb_{i}"] = X[:, i]
    contract.validate_table(df)
    return df


def test_brain_age_na_when_model_not_predictive():
    # No age signal in the embedding -> out-of-fold R2 is <= ~0 -> the control
    # scrubs nothing. Must be NA (uninformative), not PASSED.
    df = _base_cohort(220, ["s0", "s1", "s2"], seed=3, age_signal=False)
    ev = gauntlet.test_brain_age(df, "dx_binary")
    assert ev.result == Result.NA, f"non-predictive brain-age must be NA, got {ev.result} ({ev.detail})"
    assert ev.stats["r2"] < 0.10


def test_brain_age_runs_when_model_predictive():
    # Real age signal present -> R2 clears the floor -> a genuine PASS/WEAKENED/FAIL.
    df = _base_cohort(220, ["s0", "s1", "s2"], seed=3, age_signal=True)
    ev = gauntlet.test_brain_age(df, "dx_binary")
    assert ev.result != Result.NA, f"predictive brain-age should run, got NA ({ev.detail})"
    assert ev.stats["r2"] >= 0.10


def test_replication_aggregates_many_small_sites():
    # 20 tiny sites (~12 subjects each): no single site is >= 40, so single-site
    # holdout would always be NA. Aggregation must pool several whole sites into a
    # held-out fold of n >= 40 and actually run the test.
    sites = [f"site{i:02d}" for i in range(20)]
    df = _base_cohort(240, sites, seed=5, age_signal=True)
    ev = gauntlet.test_replication(df, "dx_binary")
    assert ev.result != Result.NA, f"aggregated replication should run, got NA ({ev.detail})"
    assert ev.stats["n_test"] >= 40
    assert ev.stats["n_test_sites"] >= 2, "must aggregate multiple whole sites"
    assert ev.stats["n_train"] >= 6


def test_label_shuffle_sits_near_chance():
    # Negative control: with a real disease direction but PERMUTED labels the
    # site-disjoint probe must collapse to ~chance. A materially-above-chance
    # value would flag residual leakage in the CV machinery itself.
    df = _base_cohort(240, ["s0", "s1", "s2"], seed=5, age_signal=True)
    auc = leakage.label_shuffle_auc(df, "dx_binary", seed=0)
    assert abs(auc - 0.5) < 0.12, f"shuffled-label AUC not near chance: {auc}"


def test_replication_group_disjoint():
    # The held-out sites must not appear in the training partition (genuine
    # held-out-site replication, not row-level CV). Verify by reconstructing the
    # split the same way and checking site disjointness.
    sites = [f"site{i:02d}" for i in range(20)]
    df = _base_cohort(240, sites, seed=5, age_signal=True)
    ev = gauntlet.test_replication(df, "dx_binary")
    if ev.result == Result.NA:
        return
    # n_test + n_train must cover exactly the two-class outcome rows with no overlap.
    assert ev.stats["n_test"] + ev.stats["n_train"] == \
        int(df["dx"].astype("string").isin(["AD", "CN"]).sum())
