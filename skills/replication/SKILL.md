---
name: replication-split
description: >-
  Gauntlet stage 5 of NeuroAD Discovery Engine. Tests whether a structural-MRI signal
  reproduces on a held-out site or cohort by training on the rest and evaluating
  on the held-out split. Use when a candidate MRI finding needs to be shown to
  generalize beyond the cohort/site it was discovered in. Weight 15/100.
  Implemented by neuroad.gauntlet.test_replication.
when-to-use: >-
  Run when the cohort spans at least two sites or two cohorts (e.g. OASIS-1 vs
  OASIS-2). NA on a single site/cohort. A finding that only exists where it was
  found is not a finding.
---

# Replication Split

**Adversarial question:** *Does it reproduce on a held-out site / cohort, or was
it cohort-bound?*

Weight **15**. A finding that only exists in the cohort it was found in is not a
finding. This stage does the most basic thing a sceptic would: hold out a site
(or a whole second cohort) and see if the effect is still there.

Implemented by `test_replication(df, target)` in `src/neuroad/gauntlet.py`.

## Exact statistic

1. `point_head(df, target)` yields `X`, `y`, and site `groups`.
2. **Pick the held-out site.** Iterate sites from smallest to largest and hold
   out the first that leaves a usable split: both train and test folds must have
   ≥2 outcome classes and ≥6 subjects each.
3. **Train on the rest, evaluate on the held-out site.** `train_auc` =
   `cross_val_auc` within the training split; a `LinearProbe` is fit on the
   training split and scored on the held-out site with `roc_auc_score` →
   `test_auc`.

Note: the verdict is the **absolute held-out AUC** (`test_auc`), not a
retention ratio relative to the training AUC.

## Verdict thresholds (`test_replication`)

- **PASSED** — `test_auc ≥ 0.65`: the signal generalizes to an unseen site.
- **WEAKENED** — `0.58 ≤ test_auc < 0.65`: reproduces but materially shrinks;
  partly cohort-specific.
- **FAILED** — `test_auc < 0.58`: collapses on the held-out split; the effect was
  cohort-bound (often another face of scanner/site leakage).
- **NA** — no site/cohort grouping, a single site, or no split yields two-class
  train and test folds.

## Stats emitted

`{"train_auc", "test_auc", "n_train", "n_test"}` — e.g. on `synthetic:SURVIVOR`
/ `conversion`: `train_auc ≈ 0.78`, `test_auc ≈ 0.76` → PASSED.

## Run it

```bash
PYTHONPATH=../../src ../../.venv/bin/python run.py
PYTHONPATH=src ./.venv/bin/python skills/replication/run.py
```

## Calibration & honesty notes

- On the vendored real data, replication is genuine: OASIS-2 gives a
  longitudinal held-out arm and OASIS-1 is a second cohort. Both are
  single-scanner, so a passing replication here does **not** clear the scanner
  ⭐ stage — the two tests are not interchangeable.
- Weak replication is a common way stage-2 leakage resurfaces; if this fails
  while age/sex passed, suspect a site confound.
