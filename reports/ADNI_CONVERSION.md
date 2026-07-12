# Real-ADNI CONVERSION card — MCI->AD prognosis

**Cohort:** 1199 conversion-labeled subjects (412 converters / 787 stable) — ComBat-harmonized full cohort (scanner batch removed, label-blind), 72 sites, D=323 FreeSurfer features. Real ADNI.

**Naive effect (OOF probe):** AUC = 0.644 (n=1199).

**Verdict:** ROBUST ENOUGH FOR FOLLOW-UP — score 77/100 — PROMOTED.

## Gauntlet

| Test | Result | Headline |
|---|---|---|
| age_sex | passed | auc_before=0.644, auc_after=0.628, retained=0.887 |
| site_scanner | passed | outcome_auc=0.644, scanner_auc=0.374, margin=0.27 |
| brain_age | not_available | r2=-3.779, mae_yr=9.05, n_healthy=1153 |
| biomarker_anchor | weakened | ptau217_r=0.133, ptau217_n=498, ptau217_ci_lo=0.046 |
| replication | weakened | train_auc=0.627, test_auc=0.82, test_auc_ci_lo=0.593 |

## Interpretation

Prognostic conversion prediction is a harder task than cross-sectional diagnosis: a cross-validated, permutation-significant AUC above chance is a REAL structural prognostic signal; an at-chance result means structure alone does not forecast conversion in this cohort (an honest negative).