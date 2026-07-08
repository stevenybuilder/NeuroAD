---
name: brain-age-control
description: >-
  Gauntlet stage 3 (⭐ high weight) of NeuroAD Discovery Engine. Tests whether a
  structural-MRI signal is more than generic aging/atrophy by fitting an
  embedding-derived brain-age model on healthy subjects, regressing predicted
  brain age out of the embedding, and measuring how much of the effect remains.
  Use when a candidate MRI finding could be explained by accelerated aging
  rather than disease-specific change. Weight 25/100, star test. Implemented by
  neuroad.gauntlet.test_brain_age.
when-to-use: >-
  Run when the cohort has enough cognitively-normal subjects with age to fit a
  brain-age model (≥8). Alzheimer's brains look older, so a probe can score high
  by reading off accelerated aging — this stage removes that axis and checks
  what survives.
---

# Brain-Age Control (⭐)

**Adversarial question:** *Is the signal more than generic aging and atrophy?*

Weight **25**, ⭐. Alzheimer's brains look older; a probe can score high simply by
reading off accelerated aging. This stage removes the aging axis and checks what
survives.

Implemented by `test_brain_age(df, target)` in `src/neuroad/gauntlet.py`.

## Exact statistic

1. **Fit brain-age on healthy subjects only.** Train a `LinearRegression` from
   the embedding to chronological `age`, restricted to `dx == "CN"` subjects with
   finite age (needs ≥8, non-zero age variance). Cross-validated
   (`cross_val_predict`) to report honest fit:
   - `r2` = out-of-fold R² of predicted vs chronological age,
   - `mae_yr` = out-of-fold mean absolute error in years.
2. **Predict brain age for every outcome subject** with the refit model.
3. **Control on predicted brain age — not the gap.** The implementation
   regresses out **predicted brain age** itself (`control = brain_age`), because
   residualizing against the brain-age *gap* (`predicted − chronological`) would
   leave the aging component still inside the embedding, letting generic atrophy
   masquerade as disease. Regressing out predicted brain age removes the
   aging-aligned direction, so only age-independent disease signal remains to be
   re-detected. (The gap is still computed and reported for interpretation.)
4. `auc_before` = naive outcome AUC; `auc_after` = outcome AUC on the
   brain-age-residualized embedding; `retained` = `_retained_fraction(...)`.

## Verdict thresholds (`_result_from_retained`, same bands as stage 1)

- **PASSED** — `retained ≥ 0.70`: disease-specific, more than old brains.
- **WEAKENED** — `0.40 ≤ retained < 0.70`: a material share co-varies with
  brain age; report the shrunken effect.
- **FAILED** — `retained < 0.40`: collapses toward chance; the finding was
  accelerated aging.
- **NA** — target has < 2 classes, or too few healthy subjects with age to fit a
  credible brain-age model.

## Stats emitted

`{"r2", "mae_yr", "auc_before", "auc_after", "retained", "n_healthy"}` — e.g. on
`synthetic:SURVIVOR` / `conversion`: `r2 ≈ 0.78`, `mae_yr ≈ 2.9`, `retained ≈ 0.91`
→ PASSED.

## Run it

```bash
PYTHONPATH=../../src ../../.venv/bin/python run.py
PYTHONPATH=../../src ../../.venv/bin/python run.py synthetic:KILL
PYTHONPATH=src ./.venv/bin/python skills/brain_age_control/run.py
```

## Calibration & honesty notes — this control is a PROXY

- Brain-age here is **embedding-derived**: calibrated R²~0.85, MAE~3yr
  (`CAL["brain_age_r2"]`, `CAL["brain_age_mae_yr"]`), deliberately softened from
  optimistic ~0.89 claims that only hold on very wide-age healthy cohorts.
- Because the control is a proxy, residual generic-aging signal can survive the
  adjustment and masquerade as disease-specific — the reviewer stage must flag
  this. Brain-age gap is a *recognized* control for aging-vs-disease
  (Franke/Gaser 2013, `FACTS["brain_age_gap"]`), not a gold standard.
