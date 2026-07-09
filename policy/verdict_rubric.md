---
policy: verdict_rubric
layer: L3
kind: brief
consumers: [code, claude]
version: 1.0.0
schema_version: 1
last_verified: "2026-07-08"
mirrors:
  - src/neuroad/contract.py         # VERDICT_BANDS, verdict_for, PROMOTION_FLOOR, robustness_score
# Machine-readable band table — MUST equal contract.VERDICT_BANDS (inclusive lower bounds).
verdict_bands:
  - { min_score: 85, verdict: "strong candidate",              key: STRONG }
  - { min_score: 70, verdict: "robust enough for follow-up",   key: ROBUST_FOLLOWUP }
  - { min_score: 40, verdict: "partially robust",              key: PARTIALLY_ROBUST }
  - { min_score: 0,  verdict: "fragile",                       key: FRAGILE }
promotion_floor: PARTIALLY_ROBUST   # source: contract.PROMOTION_FLOOR (score >= 40 promotes)
promotion_floor_min_score: 40
score:
  range: [0, 100]
  method: >-
    weighted over the gauntlet dimensions that actually ran, renormalized so NA
    tests are dropped from BOTH numerator and denominator (contract.robustness_score).
  dimension_weights:                # source: contract.GAUNTLET (sum == 100)
    age_sex: 15
    site_scanner: 25                # STAR
    brain_age: 25                   # STAR
    biomarker_anchor: 20
    replication: 15
  result_credit:                    # source: contract.RESULT_CREDIT
    passed: 1.0
    weakened: 0.5
    mixed: 0.5
    failed: 0.0
    not_available: 0.0
---

# Verdict Rubric — bands, promotion floor, hedged language

The referee turns a 0–100 robustness score into one of four verdicts. This table
**mirrors `contract.VERDICT_BANDS` / `verdict_for`** exactly — the bands are
inclusive lower bounds, evaluated top-down.

## Verdict bands

| Score (inclusive) | Verdict | Meaning |
|-------------------|---------|---------|
| **85–100** | **strong candidate** | Survives the full gauntlet with little erosion; the strongest thing the engine will say. |
| **70–84** | **robust enough for follow-up** | Survives the STAR tests and the molecular anchor; worth a confirmatory experiment. |
| **40–69** | **partially robust** | Clears the promotion floor but with material weakening; promote with explicit caveats. |
| **0–39** | **fragile** | Collapses under one or more challenges (often the scanner/site star). Not promoted. |

## Promotion floor

`PROMOTION_FLOOR = partially robust` → **a claim is promoted iff its score is
`>= 40`**. Only promoted claims reach the biology / Bridge step
(`propose_biology` fires only for `card.promoted`). A fragile finding never gets
a mechanism.

## How the score is computed (no free-floating numbers)

- Each gauntlet dimension earns `weight × result_credit[result]`
  (`passed`=1.0, `weakened`/`mixed`=0.5, `failed`/`NA`=0.0).
- **NA dimensions are dropped from both numerator and denominator** — the score
  reflects only evidence actually gathered — and the dropped tests are surfaced
  as an explicit **completeness caveat**, never hidden.
- Weights sum to 100; the two STAR tests (site/scanner leakage, brain-age
  control) carry 25 each because they are the artifacts a machine or generic
  aging can most easily fake.

## Hedged-language rule (honesty contract)

The verdict word is load-bearing, so the prose that accompanies it must match its
altitude. Claude and any report generator MUST obey:

1. **Never upgrade the noun.** A score of 72 is "robust enough for follow-up,"
   not "a discovery," "a breakthrough," or "proof." Below 85 the word
   "candidate" is the ceiling.
2. **Hedge below 70.** A "partially robust" (40–69) claim must be stated with an
   explicit qualifier ("partially robust, pending …") and its weakest test named.
3. **State the missing tests.** If any dimension is NA, the sentence must carry
   the completeness caveat ("… on the tests that could be run; the X test was
   not available because …").
4. **A failed STAR test caps the language.** If site/scanner leakage FAILED, the
   finding is described as *likely an acquisition artifact* regardless of the
   numeric score, and it is not promoted.
5. **No unhedged mechanism talk for non-survivors.** Biology is proposed only for
   promoted survivors; for everything else the mechanism section says the finding
   did not clear the floor.
