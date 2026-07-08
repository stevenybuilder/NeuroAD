---
name: biomarker-anchor
description: >-
  Gauntlet stage 4 of NeuroAD Discovery Engine, and the HARD GATE for promotion. Tests
  whether a structural-MRI signal is backed by molecular pathology by
  correlating the out-of-fold probe score with plasma p-tau217 / GFAP on the
  complete-case subset, requiring a correlation statistically distinguishable
  from zero. A promoted claim MUST clear this anchor. Use when deciding whether
  an imaging finding is worth advancing to a biology hypothesis. Weight 20/100.
  Implemented by neuroad.gauntlet.test_biomarker_anchor.
when-to-use: >-
  Run once a claim has survived the imaging-only stages, to decide promotion.
  Imaging finds it, proteins confirm it — a claim cannot reach "robust enough
  for follow-up" on imaging alone. NA when the cohort has no plasma marker
  coverage (route to ADNI/EPAD).
---

# Biomarker Anchor (HARD GATE)

**Adversarial question:** *Is the signal backed by molecular pathology (plasma
p-tau217 / GFAP), or is it an unanchored imaging pattern?*

Weight **20**, and the **promotion gate**: imaging finds it, proteins confirm it.
A claim cannot reach "robust enough for follow-up" on imaging alone — it must
show a plasma-biomarker correlation, on the subjects that actually have the
marker, that is statistically distinguishable from zero.

Implemented by `test_biomarker_anchor(df, target)` in `src/neuroad/gauntlet.py`.

## Exact statistic

1. **Out-of-fold probe score.** `_oof_scores(X, y)` produces cross-validated
   `P(positive)` from a standardized logistic probe, so the anchor cannot
   correlate with in-sample overfit residuals — a spurious anchor is exactly
   what we must not credit.
2. **Complete-case correlation.** On the outcome-kept subset, correlate that
   score against plasma `p_tau217` and `gfap` (Pearson r), reporting the
   complete-case `n` for each. A marker with <5 finite values or zero variance
   yields `None` (not usable). Missingness is real and is surfaced, not hidden.
3. **Primary anchor.** p-tau217 is preferred; GFAP is the secondary fallback.
4. **Significance guard (hard gate).** The verdict requires the correlation to be
   statistically distinguishable from zero — the magnitude of the anchor must
   clear a molecular-support bar *and* its confidence interval must exclude (or
   be bounded away from) zero, so a large-but-noisy `r` on a handful of subjects
   does not pass the gate. (The exact CI/threshold logic lives in
   `test_biomarker_anchor`; treat the concept — "real, non-zero molecular
   correlation" — as the contract, and read the current stats for the live
   numbers.)

## Verdict thresholds (conceptual)

- **PASSED** — a molecular correlation that is both adequately sized and
  statistically distinguishable from zero on an adequate subset. Molecular
  backing present; the claim may be promoted.
- **WEAKENED** — a suggestive but weak or thin-subset correlation; the anchor is
  soft, not solid.
- **FAILED** — the marker is present but shows no meaningful, significant
  correlation. The imaging signal is not molecularly anchored → **cannot be
  promoted**.
- **NA** — no plasma p-tau217 / GFAP coverage in the cohort. Cannot anchor;
  route to ADNI/EPAD.

## Stats emitted

`{"ptau217_r", "ptau217_n", "gfap_r", "gfap_n", ...}` (plus the CI-lower-bound
fields the significance guard reports). On `synthetic:SURVIVOR` / `conversion`
the primary anchor is p-tau217 (`r ≈ 0.41`, n ≈ 58, CI lower bound > 0) → PASSED.

## Run it

```bash
PYTHONPATH=../../src ../../.venv/bin/python run.py
PYTHONPATH=src ./.venv/bin/python skills/biomarker_anchor/run.py
```

## Routing signal (feeds the Bridge / biology step)

Which marker dominates decides the mechanism the biology step proposes:
p-tau217 / amyloid → amyloid-cascade; GFAP with weak amyloid →
neuroinflammatory / glial; NfL (+ WMH) → vascular / axonal.

## Calibration & honesty notes

- Correlation with a structural probe is expected to be **modest** (p-tau217
  r ~0.30–0.55, `CAL["ptau217_r"]`; GFAP r ~0.25–0.45, `CAL["gfap_r"]`), not
  redundant — p-tau217 is among the strongest blood AD markers (AD-vs-CU AUC
  ~0.93–0.98) but measures a different thing than an atrophy probe
  (`FACTS["ptau217"]`, `FACTS["gfap"]`).
- Realistic p-tau217 missingness (~0.56, `PTAU217_MISSINGNESS`) means the anchor
  often rests on far fewer subjects than the headline cohort — which is exactly
  why the significance guard, not the point estimate alone, decides the gate.
