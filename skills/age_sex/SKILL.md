---
name: age-sex-adjustment
description: >-
  Gauntlet stage 1 of NeuroAD Discovery Engine. Tests whether an imaging-derived signal
  (a linear probe on frozen structural-MRI embeddings) survives adjustment for
  age and sex. Use when a scientist has a candidate MRI finding and needs to
  rule out that the "signal" is just demographics. Weight 15/100. Implemented by
  neuroad.gauntlet.test_age_sex.
when-to-use: >-
  Run first on any claim whose target is `conversion` or `dx_binary`. It needs
  only the `age` and `sex` columns (always in the contract table), so it is
  almost never NA. It is the cheapest artifact to rule out before the heavier
  star tests.
---

# Age / Sex Adjustment

**Adversarial question:** *Does the signal survive demographic covariates, or is
it just age and sex in disguise?*

This is the lightest of the five NeuroAD Discovery Engine gauntlet stages (weight **15**),
but the cheapest artifact to rule out first — age and sex are the most
universally available confounds and the easiest to accidentally learn.

Implemented by `test_age_sex(df, target)` in `src/neuroad/gauntlet.py`.

## Exact statistic

1. `point_head(df, target)` extracts the embedding matrix `X`, outcome codes `y`,
   and site groups. `cross_val_auc` gives the naive subject/site-disjoint AUC
   (`auc_before`).
2. Build a covariate matrix `C` from `age` (mean-imputed, used if it has ≥3
   finite values and non-zero variance) and a female indicator (used if `sex`
   varies).
3. `_residualize(X, C)` z-scores `C`, prepends an intercept, and regresses `C`
   out of every embedding column via least squares — leaving only the
   age/sex-independent part of the embedding.
4. Re-measure the outcome AUC on the residualized embedding (`auc_after`).
5. **Retained fraction** (`_retained_fraction`): how much of the above-chance
   effect survives, `clip((auc_after − 0.5) / max(auc_before − 0.5, 1e-6), 0, 1.5)`.

## Verdict thresholds (`_result_from_retained`)

Bands come from `calibration.CAL`, never free-floating constants:

- **PASSED** — `retained ≥ 0.70` (`CAL["survivor_retained"][0]`). The signal is
  not merely demographic.
- **WEAKENED** — `0.40 ≤ retained < 0.70` (`CAL["kill_retained"][1]`). Age/sex
  explain part of it; report the shrunken effect, not the naive one.
- **FAILED** — `retained < 0.40`. Collapses toward chance after adjustment; the
  finding was demographics.
- **NA** — target has < 2 classes, or there is no age/sex variation to adjust for.

## Stats emitted

`{"auc_before", "auc_after", "retained", "n"}` — e.g. on `synthetic:SURVIVOR` /
`conversion`: `auc_before ≈ 0.71`, `auc_after ≈ 0.68`, `retained ≈ 0.86` →
PASSED. On `synthetic:KILL`: `retained = 0.0` → FAILED.

## Run it

```bash
# from this directory
PYTHONPATH=../../src ../../.venv/bin/python run.py                      # SURVIVOR / conversion
PYTHONPATH=../../src ../../.venv/bin/python run.py synthetic:KILL       # watch it collapse
# from the repo root
PYTHONPATH=src ./.venv/bin/python skills/age_sex/run.py
```

## Calibration & honesty notes

- Calibrated survivor retention is ~0.80 (loses 10–30%, `CAL["survivor_retained"]`);
  a KILL cohort retains ~0.25 (`CAL["kill_retained"]`).
- Passing here proves only that the effect is *more than* age/sex — it says
  nothing about scanner leakage or brain-age, which are the heavier ⭐ stages.
