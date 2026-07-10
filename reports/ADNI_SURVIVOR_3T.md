# Real-ADNI SURVIVOR card — AD vs CN (3t de-confound)

**Cohort:** 2109 subjects — 3T scans only (field-strength slice) (269 AD / 924 CN), 68 sites, D=323 FreeSurfer features. Real ADNI (non-stub).

**Naive effect:** AUC = 0.924 (n=1193).

**Verdict:** ROBUST ENOUGH FOR FOLLOW-UP — score 83/100 — PROMOTED.

## Gauntlet

| Test | Result | Headline |
|---|---|---|
| age_sex | passed | auc_before=0.924, auc_after=0.885, retained=0.907 |
| site_scanner | weakened | outcome_auc=0.924, scanner_auc=0.5, margin=0.424 |
| brain_age | not_available | r2=-0.379, mae_yr=5.29, n_healthy=924 |
| biomarker_anchor | passed | ptau217_r=0.491, ptau217_n=755, ptau217_ci_lo=0.435 |
| replication | passed | train_auc=0.929, test_auc=0.836, test_auc_ci_lo=0.653 |

## Why this survives when the full cohort is killed

On the raw full cohort the STAR site/scanner test FAILS — the FreeSurfer feeder predicts 3T-vs-1.5T field strength at AUC ~0.99, better than it predicts disease, so the finding is a batch artifact and scoring's honesty cap floors it to `fragile`. Restricting to a single field strength removes that dominant confound; the same AD-vs-CN signal then only weakens under the (now site-level) leakage test and holds its p-tau217 anchor, so it is promoted. (ComBat mode is the stronger de-confound — it keeps the full cohort and makes the star pass, not just weaken.)

## Caveats

- The scanner label is field-strength-only (no manufacturer/model); the 3T restriction throws away every 1.5T scan and leaves finer site structure (the residual site-leakage weakening).
- Biomarker anchor holds on the p-tau217-complete subset only (~46% plasma coverage cohort-wide); n is reported per test. The anchor correlation is robust to the scan<->plasma date gap (r stays ~0.5 when restricted to pairs <=365d apart — see p_tau217_gap_days QC).
- Replication returns NA rather than a pass — the held-out ADNI sites are too small to be individually informative (a perfectly-separable tiny split no longer counts as a pass).
