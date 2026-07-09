---
policy: atn_framework
layer: L3
kind: brief                       # Claude-consumed Markdown brief (front-matter is machine-readable)
consumers: [code, claude]
version: 1.0.0
schema_version: 1
last_verified: "2026-07-08"
mirrors:
  - src/neuroad/gauntlet.py         # test_biomarker_anchor (the anchor gate)
  - src/neuroad/contract.py         # BIOMARKER_COLUMNS
  - src/neuroad/calibration.py      # FACTS
# Machine-readable staging table (the loader reads `anchors`; prose below is for Claude).
anchors:
  A:
    axis: amyloid
    columns: [amyloid]
    role: "amyloid deposition (A+/A-)"
    anchor_eligible: false          # supporting/enriching, not the correlation gate
    routes_to: amyloid_cascade
  T:
    axis: tau
    columns: [p_tau217]
    role: "phosphorylated-tau / tangle pathology (primary molecular anchor)"
    anchor_eligible: true           # PRIMARY biomarker-anchor marker (preferred)
    routes_to: amyloid_cascade
    fact_key: ptau217
  N:
    axis: neurodegeneration
    columns: [nfl]                  # plasma neurodegeneration marker; structural MRI is the (N) imaging axis
    role: "neurodegeneration / neuronal-axonal injury (the imaging signal itself is the (N) atrophy axis)"
    anchor_eligible: false          # NfL routes vascular/axonal; structural (N) is what is being tested, not the anchor
    routes_to: vascular
  I:
    axis: inflammation
    columns: [gfap]
    role: "astrogliosis / neuroinflammation (secondary molecular anchor)"
    anchor_eligible: true           # SECONDARY biomarker-anchor marker (fallback after p-tau217)
    routes_to: glial
    fact_key: gfap
anchor_gate:
  primary_marker: p_tau217          # gauntlet prefers p-tau217
  secondary_marker: gfap            # falls back to GFAP
  min_complete_case_n: 20           # source: gauntlet._ANCHOR_MIN_N
  statistic: "95% CI lower bound of Pearson r between out-of-fold probe score and marker"
  ci_lower_pass: 0.12               # source: gauntlet._ANCHOR_CI_PASS  -> PASSED
  ci_lower_weak: 0.0                # source: gauntlet._ANCHOR_CI_WEAK  -> WEAKENED (if > 0)
---

# AT(N)(+I) Framework — the molecular-anchor eligibility brief

This brief tells the referee **which molecular axis a structural finding must be
anchored to** before it can be promoted, and **which plasma markers are eligible
to serve as that anchor**. It is the domain-knowledge backing for the
biomarker-anchor **HARD GATE** (`gauntlet.test_biomarker_anchor`).

The engine speaks in the field-standard **A / T / (N)** research framework
(Jack et al. NIA-AA), extended with **(+I)** for the inflammatory axis that
plasma GFAP indexes:

| Axis | Meaning | Plasma marker in this contract | Anchor-eligible? | Routes to |
|------|---------|-------------------------------|------------------|-----------|
| **A** | Amyloid deposition | `amyloid` (positivity) | No — *enriches* the tau/amyloid pole | amyloid-cascade |
| **T** | Tau / tangle pathology | `p_tau217` | **Yes — PRIMARY anchor** | amyloid-cascade |
| **(N)** | Neurodegeneration | `nfl` (plasma) / the structural MRI itself | No — the (N) imaging axis is *what is being tested* | vascular / axonal |
| **(+I)** | Neuroinflammation / astrogliosis | `gfap` | **Yes — SECONDARY anchor** | glial |

## Why (N) is the finding, not the anchor

The structural-MRI probe **is** the imaging read-out of the **(N)** axis
(atrophy / neurodegeneration). You cannot anchor an (N) imaging pattern to
itself — that would be circular. So the anchor gate demands a **molecular**
axis: **T** (p-tau217) first, then **(+I)** (GFAP). **A** (amyloid positivity)
*enriches* the tau/amyloid pole for routing but is not the correlation gate.
Plasma **NfL** is another neurodegeneration/axonal marker, so it routes the
mechanism (vascular/axonal) rather than serving as the anchor.

## The biomarker-anchor gate (exact, mirrors gauntlet.py)

"Imaging finds it, proteins confirm it." A claim cannot reach the promotion
floor on imaging alone — it must show a plasma-biomarker correlation, on the
subjects that actually have the marker, that is statistically distinguishable
from zero.

1. **Out-of-fold probe score** — cross-validated `P(positive)` (no in-sample
   overfit residuals to correlate against).
2. **Complete-case correlation** — Pearson `r` of that score against
   `p_tau217` (primary) then `gfap` (secondary), requiring at least
   **`min_complete_case_n = 20`** finite values with non-zero variance
   (`_ANCHOR_MIN_N`); below that the marker yields `NA`, route to a cohort with
   coverage (ADNI / EPAD).
3. **Gate on the 95% CI LOWER BOUND** (Fisher-z), not raw `r`, so a lucky
   small-n correlation on noise cannot pass and a real anchor is not failed by
   seed:
   - **PASSED** if `ci_lower >= 0.12` (`_ANCHOR_CI_PASS`) — confidently anchored.
   - **WEAKENED** if `0.0 < ci_lower < 0.12` — positive but thin.
   - **FAILED** if `ci_lower <= 0.0` — marker present but no molecular support
     (data present, unanchored → the gate fails).
   - **NA** if no usable p-tau217 / GFAP coverage — route, don't credit or condemn.

## Calibration expectation (so a "modest" anchor is not read as failure)

The correlation of a molecular marker with a *structural* atrophy probe is
expected to be **modest, not redundant**: p-tau217 `r ~ 0.30–0.55`
(`CAL["ptau217_r"]`), GFAP `r ~ 0.25–0.45` (`CAL["gfap_r"]`). p-tau217 is among
the strongest blood AD markers (AD-vs-CU AUC ~0.93–0.98) but measures a
different biological thing than an atrophy probe, so a modest correlation is the
*correct* expectation — the significance guard (CI lower bound), not the point
estimate, decides the gate. Realistic p-tau217 missingness (~45%,
`PTAU217_MISSINGNESS`) means the anchor often rests on far fewer subjects than
the headline cohort; that is exactly why the n-floor and CI rule exist.
