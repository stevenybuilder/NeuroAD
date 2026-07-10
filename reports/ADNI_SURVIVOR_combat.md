# Real-ADNI SURVIVOR card — AD vs CN (combat de-confound)

**Cohort:** 2951 subjects — ComBat-harmonized full cohort (scanner batch removed, label-blind) (462 AD / 1153 CN), 72 sites, D=323 FreeSurfer features. Real ADNI (non-stub).

**Naive effect:** AUC = 0.923 (n=1615).

**Verdict:** STRONG CANDIDATE — score 100/100 — PROMOTED.

## Gauntlet

| Test | Result | Headline |
|---|---|---|
| age_sex | passed | auc_before=0.922, auc_after=0.897, retained=0.939 |
| site_scanner | passed | outcome_auc=0.922, scanner_auc=0.374, margin=0.549 |
| brain_age | passed | r2=-3.779, mae_yr=9.05, auc_before=0.922 |
| biomarker_anchor | passed | ptau217_r=0.449, ptau217_n=873, ptau217_ci_lo=0.394 |
| replication | not_available | train_auc=0.923, test_auc=1.0, test_auc_ci_lo=1.0 |

## Why this survives when the full cohort is killed

On the raw full cohort the STAR site/scanner test FAILS — the FreeSurfer feeder predicts 3T-vs-1.5T field strength at AUC ~0.99, better than it predicts disease, so the finding is a batch artifact and scoring's honesty cap floors it to `fragile`. ComBat harmonization removes that scanner batch effect from the features **label-blind** (it protects age/sex, NOT diagnosis, so it cannot manufacture the AD signal). The whole cohort stays in play, the scanner test now PASSES (scanner AUC ~0.37), and the AD signal plus its p-tau217 anchor survive — so it is promoted.

## Caveats

- The scanner label is field-strength-only (no manufacturer/model); ComBat by scanner removes the field-strength batch, but finer site/model structure it cannot see may remain.
- Biomarker anchor holds on the p-tau217-complete subset only (~46% plasma coverage cohort-wide); n is reported per test. The anchor correlation is robust to the scan<->plasma date gap (r stays ~0.5 when restricted to pairs <=365d apart — see p_tau217_gap_days QC).
- Replication returns NA rather than a pass — the held-out ADNI sites are too small to be individually informative (a perfectly-separable tiny split no longer counts as a pass).
