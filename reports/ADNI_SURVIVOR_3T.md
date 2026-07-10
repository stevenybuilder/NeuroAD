# Real-ADNI SURVIVOR card — AD vs CN (3T only)

**Cohort:** 2109 subjects at 3T (269 AD / 924 CN), 68 sites, D=323 FreeSurfer features. Real ADNI (non-stub).

**Naive effect:** AUC = 0.924 (n=1193).

**Verdict:** STRONG CANDIDATE — score 85/100 — PROMOTED.

## Gauntlet

| Test | Result | Headline |
|---|---|---|
| age_sex | passed | auc_before=0.924, auc_after=0.885, retained=0.907 |
| site_scanner | weakened | outcome_auc=0.924, scanner_auc=0.5, margin=0.424 |
| brain_age | passed | r2=-0.379, mae_yr=5.29, auc_before=0.924 |
| biomarker_anchor | passed | ptau217_r=0.491, ptau217_n=755, ptau217_ci_lo=0.435 |
| replication | not_available | train_auc=0.924, test_auc=1.0, test_auc_ci_lo=1.0 |

## Why this survives when the full cohort is killed

On the full cohort the STAR site/scanner test FAILS — the FreeSurfer feeder predicts 3T-vs-1.5T field strength at AUC ~0.99, better than it predicts disease, so the finding is a batch artifact and scoring's honesty cap floors it to `fragile`. Restricting to a single field strength removes that dominant confound; the same AD-vs-CN signal then only weakens under the (now site-level) leakage test and holds its p-tau217 molecular anchor on real plasma, so it is promoted.

## Caveats

- Scanner label is field-strength-only (no manufacturer/model); the 3T restriction is the de-confound, but finer scanner/site structure remains and is what the residual site-leakage weakening reflects.
- Biomarker anchor holds on the p-tau217-complete subset only (~46% plasma coverage cohort-wide); n is reported per test.
- Replication returns NA rather than a pass — the held-out ADNI sites are too small to be individually informative (fixed: a perfectly-separable tiny split no longer counts as a pass).
