# Methods

NeuroAD Discovery Engine evaluates a candidate structural-MRI finding by pointing a
single linear head at different columns of a cached embedding table and running
a five-test adversarial gauntlet. Every headline number below is pinned to a
literature-defensible range in `src/neuroad/calibration.py` (the `CAL` table and
`target()`), or computed live from data — nothing free-floating. Verdict language
stays deliberately hedged.

## Substrate and the reused head

The unit of analysis is one subject, represented by a frozen embedding vector
(`emb_0 … emb_{D-1}`) plus contract metadata (`contract.METADATA_COLUMNS`). The
embedding is *weight-free-swappable*: real frozen Neuro-JEPA structural
embeddings, a substitute open encoder, or structural-derived features (eTIV,
nWBV, ASF, hippocampal volume, cortical thickness) all satisfy the same
contract. On OASIS the "embedding" is the standardized structural-derived
feature block; MMSE/CDR are **never** used as probe features because they define
the labels.

The head is an L2-regularized logistic regression (the *linear probe*), scored by
**subject-disjoint cross-validated AUC** so no subject appears in both train and
test folds. Pointing the same head at `conversion`/`dx_binary`, at `site`/
`scanner`, or at a biomarker regression is the entire architecture.

**Naive effect.** Before any challenge, the head predicts the outcome. The values
in `calibration.CAL` are *synthetic-harness* calibration targets (what the
generated cohorts are tuned to reproduce), not claims about any real cohort:
AD-vs-CN diagnosis AUC ≈ **0.89** (`diagnosis_auc`, range 0.85–0.92); MCI→AD
conversion AUC ≈ **0.74** (`conversion_auc`, range 0.68–0.80). Conversion is a
genuinely harder prognostic task, scored separately from diagnosis. On the **real
single-cohort OASIS** data the AD-vs-CN probe lands a touch lower (~**0.82**),
which is expected for a small weight-free-feature cohort and is reported as-is —
the calibration range governs the synthetic harness, not the real run.

## The gauntlet (five tests)

Each test returns a `contract.TestEvidence` with a `TestResult`
(passed / weakened / mixed / failed / not_available) and the statistics that
justify it. Dimension weights (sum = 100) live in `contract.GAUNTLET`; the two
starred tests carry the most weight because a scanner or generic aging can most
easily fake them.

### 1. Age / sex adjustment (weight 15)

Re-fit the probe with age and sex added as covariates (or residualize the
embedding against them) and measure the **effect retained**: the ratio of the
adjusted effect size to the naive one. A survivor retains ≈ **80%**
(`survivor_retained`, 0.70–0.90); a kill retains ≈ **25%** (`kill_retained`,
0.10–0.40) and collapses. Result bands: retained ≥ 0.70 → *passed*,
0.40–0.70 → *weakened*, < 0.40 → *failed*.

### 2. ⭐ Site / scanner leakage (weight 25)

Point the **same head** at the `scanner` (or `site`) label and compute its
subject-disjoint AUC. The headline metric is the

> **leakage margin = outcome_AUC − scanner_AUC**

expressed in the frontier's currency (cf. arXiv:2606.09189).
A KILL shows scanner AUC ≈ **0.92** (`site_auc_kill`, 0.88–0.96) — leakage meets
or exceeds the outcome, so the margin is near zero or negative and the test
*fails*. A SURVIVOR shows scanner AUC ≈ **0.64** (`site_auc_survivor`, 0.58–0.70),
so the outcome clearly exceeds the confound and the margin is materially
positive.

**Double dissociation (control).** We residualize the embedding against a
scanner-predicting direction and re-probe the outcome. The survivor still
predicts the outcome after scrubbing; the kill collapses. This dissociates
disease signal from acquisition signal rather than merely comparing two AUCs.

**Confound leaderboard.** We rank the fraction of variance in the probe direction
explained by each confound (scanner, age, sex), so the scientist sees *which*
artifact to fix first.

On single-scanner real data (OASIS-1, OASIS-2) the star test is honestly
reframed as **cohort/batch leakage** — the head predicts OASIS-1 vs OASIS-2
membership as a pseudo-site. The ground-truth scanner-confound KILL lives in the
synthetic harness.

### 3. ⭐ Brain-age control (weight 25)

Generic aging and atrophy are the most common confounds masquerading as disease.
We fit a brain-age regressor from the embedding on a wide-age cohort and use the
**brain-age gap** (predicted − chronological age; BrainAGE / brain-PAD, Franke &
Gaser 2013) as a control covariate, then measure the effect drop.

The brain-age model is reported at **R² ≈ 0.85 with MAE ≈ 3 yr**
(`brain_age_r2` 0.80–0.90; `brain_age_mae_yr` 2.5–4.0) — deliberately
**softened** from the optimistic R²≈0.89 that only holds for large, wide-age
healthy cohorts (wide age span inflates R²; Peng 2021 SFCN MAE≈2.14yr; Bashyam
2020). This is a **proxy** control, and the reviewer agent is required to flag it
as such. Post-adjustment outcome AUC: survivor ≈ **0.81** (`survivor_post_auc`),
kill ≈ **0.58** (`kill_post_auc`, near chance).

### 4. Biomarker anchor (weight 20) — strongest corroboration

When plasma markers are available, a promoted claim should show a biomarker
correlation on the *complete* subset. We correlate the probe score with p-tau217
and GFAP and report the correlation *and the n* of the complete subset. A failed
biomarker anchor is a hard refutation and blocks promotion.

Calibrated targets: p-tau217 r ≈ **0.43** (`ptau217_r`, 0.30–0.55) and GFAP
r ≈ **0.35** (`gfap_r`, 0.25–0.45) — a **modest** structural↔molecular link, not
redundancy (plasma p-tau217 alone reaches AD-vs-CU AUC ~0.93–0.98). Realistic
missingness is high (`PTAU217_MISSINGNESS` ≈ 0.45), so the completeness caveat is
surfaced on every card. The **gated ADNI cohort ships real *measured* plasma
p-tau217** (~1,377 non-null of 2,951 subjects), so the served ADNI substrate
anchors on measurement, not calibration — the demo card reports r=+0.49, n=876,
`"synthetic": false, "provenance": "measured"`. Open-only cohorts that lack
plasma treat the anchor as `not_available` and rely on leakage-clean held-out
replication instead. The synthetic harness demonstrates the molecular-anchor
mechanic offline and is labeled as a harness rather than treated as evidence.

**Biomarker routing** (used by the biology bridge, survivors only):
amyloid + p-tau → amyloid-cascade; GFAP / weak-amyloid → neuroinflammatory /
glial; NfL + WMH → vascular / axonal.

### 5. Replication split (weight 15)

Refit and evaluate on a held-out site/cohort (OASIS-1 vs OASIS-2, or a held-out
synthetic site). A signal that reproduces out-of-cohort *passes*; one that only
holds in-sample is *weakened* or *failed*.

Replication is also the accepted open-data corroboration fallback when a cohort
does not ship plasma biomarkers. It can promote a claim only if the biomarker
anchor is `not_available`, the held-out replication passes, and the site/scanner
leakage test passes. Replication alone is not enough, because a scanner artifact
can replicate across similarly biased cohorts.

## Scoring and verdict

`contract.robustness_score` computes a weighted sum of per-test credit
(passed = 1.0, weakened / mixed = 0.5, failed = 0.0), **renormalized over only
the tests that actually ran** — a not-available test is dropped from both
numerator and denominator, and the resulting completeness gap is surfaced as an
explicit caveat rather than silently penalizing the score. The integer 0–100
score maps through `VERDICT_BANDS`:

| Score | Verdict |
|---|---|
| ≥ 85 | strong candidate |
| ≥ 70 | robust enough for follow-up |
| ≥ 40 | partially robust |
| < 40 | fragile |

Only claims at or above `PROMOTION_FLOOR` (partially robust) are eligible for
promotion (`contract.is_promoted`). Eligibility is then gated by independent
corroboration:

- **Molecular path:** biomarker anchor is passed or weakened. This is the
  strongest path and is required when the biomarker test is available.
- **Open-data replication path:** biomarker anchor is unavailable, held-out
  replication passes, and the site/scanner leakage test passes.

Biology speaks only about promoted survivors, and every claim is paired with its
evidence ledger. If promotion uses replication rather than a molecular anchor,
the card says so and routes the next experiment to ADNI/EPAD for plasma
confirmation.

## Adjudication and self-review (Claude)

For promoted claims a **courtroom** stage runs a Prosecution argument (this is an
artifact), a Defense argument (this is real biology), and a Judge that renders
reasoning — each a consequential step, not decoration. A **reviewer** agent then
argues *against* the verdict: it flags the proxy brain-age control, p-tau217
missingness, and that "partially robust ≠ robust." Every Claude call uses the
live Anthropic API when `ANTHROPIC_API_KEY` is set and a deterministic template
fallback otherwise, so the pipeline is reproducible offline.
