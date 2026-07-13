# P1 Flagship Audit — is ADNI AD-vs-CN AUROC 0.922 (adni:combat) de-confounded?

**Date:** 2026-07-13  **Auditor:** NeuroAD Data-QA (read-only audit)
**Question:** Is the flagship headline AUROC **0.922** for **ADNI dx_binary (AD vs CN)** on
substrate **adni:combat** genuinely de-confounded of scanner / field-strength leakage — safe to
ship as the demo headline?

---

## VERDICT

**SAFE WITH CAVEAT — ship 0.922, but the displayed honesty caption MUST be corrected.**

The number is **NOT field-strength-inflated** and is safe to ship. The decisive proof is not the
ComBat scanner test the UI currently shows — it is that the AD-vs-CN effect **holds at ~0.92
inside each single field strength** (3T-only 0.9245, 1.5T-only 0.9365), where a field-strength
confound is impossible by construction. If 0.922 were the 1.5T/3T split leaking, restricting to
one strength would collapse it. It does not.

BUT the currently-displayed evidence for the de-confound — **"scanner AUC 0.374 (below chance)"**
— is an **optimistic artifact of whole-cohort ComBat** and overstates how thoroughly field
strength was scrubbed. The honest, fold-honest residual scanner leakage is **AUC ≈ 0.65**, not
below chance. The ship decision does not change (outcome still beats scanner by +0.28 honestly,
and survives within-strata), but the caption must stop claiming scanner was driven below chance.

---

## 1. Substrate + cohort (what actually loads)

- `adni:combat` → `loaders.py:82-87` loads `data/real/_gated/adni.csv` (2951 rows, 323 `emb_*`
  columns) and applies **parametric ComBat** via `harmonize.harmonize(batch="scanner",
  covariates=("age","sex"))` (`harmonize.py:213-260`).
- The `emb_*` columns are **FreeSurfer named-ROI morphometry**, NOT NeuroJEPA embeddings. This is
  a **different substrate** from the `adni:neurojepa` cohort the handoff worried about. The
  worry ("NeuroJEPA reads field strength at AUC 0.990") is about a substrate the flagship does
  **not** use — though, notably, raw FreeSurfer morphometry leaks field strength almost as badly
  (0.989, see §3).
- ComBat batch = **scanner** = field strength (values: `3T`, `1.5T`). So ComBat's batch variable
  IS field strength. It is **label-blind**: age/sex are protected covariates; `dx` is
  deliberately excluded from the design (`harmonize.py:13-18`), so the correction never sees the
  outcome.
- **dx_binary cohort:** n = **1615** subjects (**462 AD / 1153 CN**), **one row per subject**
  (1615 unique subject_ids = 1615 rows → cross-sectional, **no subject leakage possible**), 70
  sites. CV is **site-disjoint** (StratifiedGroupKFold on site).

## 2. Field-strength composition — the confound is REAL in the raw data

| dx | n | 1.5T | 3T | %1.5T |
|----|----|------|-----|-------|
| AD | 462 | 193 | 269 | **41.8%** |
| CN | 1153 | 229 | 924 | **19.9%** |

χ²(1) = 80.9, **p = 2.3×10⁻¹⁹**, Cramér's V = 0.224, **OR(AD↔1.5T) = 2.89**. AD subjects are
~2.9× more likely to be imaged at 1.5T. This is exactly the imbalance the handoff flagged, and it
is genuinely present. Age (AD 74.9±7.8 vs CN 71.1±7.0) and sex (AD 44.8%F vs CN 60.7%F) are also
confounded. **So a de-confound is required — the question is whether adni:combat delivers one.**

## 3. Scanner-leakage evidence (measured on the SAME substrate + cohort)

All reproduced with `.venv/bin/python`, `PYTHONPATH=src:.`, `leakage.leakage_margin(df,
target="dx_binary")` and `harmonize.combat_cv_auc`:

| Pipeline | Outcome AUC (AD vs CN) | Scanner/field-strength AUC | Margin |
|---|---|---|---|
| **Raw morphometry (no ComBat)** | 0.9349 | **0.9894** | **−0.055 (KILL: scanner wins)** |
| **Whole-cohort ComBat = shipped `adni:combat`** | **0.9225** | **0.3736** | **+0.549 [0.522, 0.575]** |
| **Fold-honest ComBat (fit on train only)** | **0.9252** | **0.648** | **+0.277** |

Reads:
- **Raw** morphometry is field-strength-confounded: features predict 1.5T-vs-3T at **0.989**,
  *above* the outcome. Un-harmonized, this is a KILL. ComBat is doing real work.
- **Whole-cohort ComBat** (the shipped feeder) reproduces the cache cell **exactly**:
  outcome 0.9225 (→ displayed 0.922), scanner 0.3736, margin +0.549 [0.522, 0.575], excludes
  zero, outcome p_perm 0.001, scanner p_perm 1.0. **scanner_auc = 0.374 is measured on the same
  substrate+cohort as the 0.922** — it correctly bounds *this* cell, not a stale one.
- **BUT** whole-cohort ComBat fits the batch correction on ALL rows incl. the test folds
  (`harmonize.py:256-259` carries its own LOUD caveat). That makes scanner *look* unpredictable
  even on held-out rows. The **fold-honest** scanner leakage is **0.648**, not 0.374 — residual
  field-strength signal survives ComBat. The 0.374 "below chance" is optimistic and must not be
  presented as evidence that scanner was scrubbed.
- **Crucially, the outcome is NOT inflated by ComBat leakage:** fold-honest ComBat gives
  **0.9252 ≈** whole-cohort 0.9225. The 0.922 is not a harmonization-leakage artifact.

## 4. The decisive de-confound — within-field-strength invariance

A field strength cannot confound a comparison that contains only one field strength. Restricting
to a single strength (site-disjoint CV, no ComBat needed):

| Stratum | n (AD / CN) | Outcome AUC | Δ vs pooled 0.9225 |
|---|---|---|---|
| **3T only** | 1193 (269 / 924) | **0.9245** | +0.002 |
| **1.5T only** | 422 (193 / 229) | **0.9365** | +0.014 |

Both strata land at 0.92–0.94, essentially identical to the pooled ComBat number, and both Δ are
**within the ±0.05 AUROC equivalence band** the reference paper uses for subgroup robustness
(Nature Medicine, doi:10.1038/s41591-026-04497-1, *Extended Data Fig. 6d*, per-diagnosis
robustness by MRI field strength). **By the field-leading model's own field-strength test, the
effect shows no material field-strength deviation.** This is the number that proves de-confound,
and it is stronger than the ComBat scanner test.

## 5. Other confounds

- **Age/sex:** cache `value_adjusted = 0.905` (age/sex-residualized, fold-honest) — effect
  retains 96% (0.922 → 0.905). Passes.
- **Brain-age:** in the ComBat cohort brain-age IS predictive (R² = 0.48, MAE 4.0 yr); effect
  retains 94% under brain-age-GAP control and **68% (→ 0.786) under the stricter
  predicted-brain-age control** — still well above chance.
- **Effect-size sanity:** the reference paper's ADNI CN-vs-AD classifier generalizes to external
  cohorts at AUROC **93.49 (AIBL)** and **88.09 (OASIS-1)** (Nature Medicine, main text, ADNI
  external-generalization paragraph). 0.922 is squarely in the expected AD-vs-CN band — not
  implausibly high.
- **Replication (cache cell):** held-out AUROC **0.92 [95% CI 0.79, 1.00]**, n_test = 41 over 11
  aggregated held-out sites. Positive but **thin (n=41, CI touches 1.00)** — treat as supportive,
  not confirmatory.

## 6. Report-vs-cache reconciliation

They describe **two different experiments** and the demo serves only one:

| | `reports/adni_dx_3T_survivor.json` (static) | `app/investigate_cache.json` `adni:combat\|dx_binary` (LIVE) |
|---|---|---|
| Substrate | z-standardized FreeSurfer, **3T-only slice** | **ComBat full cohort** |
| n | 1193 | 1615 |
| Naive AUC | 0.924 | 0.9225 (headline 0.922) |
| scanner_auc | 0.5 (placeholder, CI null) | 0.3736 |
| site_scanner | **weakened**, margin CI incl. zero | **passed**, margin excludes zero |
| brain_age | **not_available** | passed (R²=0.48) |
| score / verdict | **83** / "robust enough for follow-up" | **100** / "strong candidate" |
| held-out | 0.84 [0.65, 0.97] | 0.92 [0.79, 1.00] |

`app/server.py:462` serves the **cache** (`investigate_cache.get(...)` → `personalize`). The
static 3T report is an **older artifact, NOT served**. They are not contradictory — the 3T report
is corroborating within-stratum evidence (§4) — but its "weakened / score 83 / brain-age N/A"
framing is stale relative to the ComBat cell that actually ships. **Authoritative for the demo:
the cache cell.**

## 7. The score = 100 question

For the **shipped adni:combat cell**, all five gauntlet tests genuinely PASS (age_sex,
site_scanner, brain_age R²=0.48, biomarker_anchor, replication) → **score 100 is legitimate and
NOT renormalized over 4/5.** The renorm-overclaim from the prior memory note applies to the OLD
3T survivor (score 83, brain_age "not_available", 1/5 could not run) — which is **not what
ships**. So **no renormalization caption is required** for the flagship.

However: the `site_scanner` PASS that feeds the score rests on the optimistic scanner 0.374.
Under the honest fold-honest scanner AUC (0.648), the margin is still +0.277 (outcome ≫ scanner),
so the test still passes and the score stays 100 — but the displayed *detail text* for that test
("scanner 0.374, below chance") is misleading and should be corrected.

## 8. Rigor delta vs the reference paper

- **Meets/exceeds:** subject-disjoint (trivially — 1 row/subject) and **site-disjoint** CV; the
  reference paper likewise stresses site-level evaluation (Extended Data Fig. 6c,e). Within-strata
  field-strength robustness (§4) mirrors its Extended Data Fig. 6d equivalence-band method.
- **Diverges (defensibly):** the reference paper **does not ComBat away** field strength — it
  reports composition and stratified robustness. This project ComBats AND stratifies; the
  stratified result is the trustworthy one.
- **Falls short:** (a) the shipped feeder uses **whole-cohort ComBat**, which the codebase itself
  flags as not fold-honest, and the leakage panel reports the optimistic scanner number rather
  than the fold-honest 0.65; (b) external replication is **n=41** — far below the reference
  paper's multi-cohort external validation.

---

## Required caption fix (exact recommendation)

Replace the leakage line so it does **not** claim scanner was scrubbed below chance. Ship copy:

> **AD vs CN — ADNI FreeSurfer morphometry, ComBat-harmonized by field strength.**
> AUROC **0.922 (95% CI 0.906–0.938)**, n = 1615 (462 AD / 1153 CN), site-disjoint CV, 70 sites.
> **De-confound:** effect holds within each field strength (3T-only 0.92, 1.5T-only 0.94; both
> within ±0.05 of pooled), so it is not the 1.5T/3T split. Residual field-strength predictability
> after fold-honest harmonization is AUC ≈ 0.65 (not fully removed), but the outcome exceeds it by
> +0.28. Age/sex-adjusted 0.90; strictest brain-age-adjusted 0.79. External held-out 0.92
> [0.79–1.00], n = 41 — supportive, not confirmatory.

Do **not** display "scanner AUC 0.374 / below chance" as the de-confound evidence.
