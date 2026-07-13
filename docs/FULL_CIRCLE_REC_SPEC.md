# Full-circle recommendation — approved build spec (pending, do AFTER the protein-tab session)

**Status:** approved by user 2026-07-12. **Do NOT edit `app/neuroad.html` for this until the
protein-structure-tab session has released the file** (avoid collision). Additive only — the
existing right-rail card and its tabs (Brain data, Artifacts, Summary, …) are UNCHANGED.

## What it is
A new intermediate step in the flow. Today, choosing/investigating the top candidate goes
straight to the right-rail detail card. Instead, insert a **ranked-candidate list** in the same
right-rail spot, ONE STEP BEFORE the detail card. Clicking the top (or any) candidate opens the
existing right-rail card exactly as before.

## Layout (minimalist — match the existing ZUI aesthetic)
- A header block showing the **measured statistical significance of the imaging finding** (this is
  the real AUROC+CI; it is NOT per-protein):
  - `AUROC 0.85 [0.81–0.89], p<0.001` (the finding's discrimination + CI)
  - `anchored to plasma p-tau217 (r=0.49, n=876)`
  - `ranking validated vs GWAS: AUC 0.728, p=0.003` (non-circular proof the ranking surfaces
    independently-known AD genes; source: reports/target_prioritization_validation.json →
    validation.honest_tests, gold set Bellenguez 2022 Nat Genet PMID:35379992)
- A list of **rounded-edge boxes**, one per candidate, TOP ONE HIGHLIGHTED, each showing REAL data:
  `rank · gene · priority <priority_score> · PI4AD #<rank>` with a → affordance.
- Clicking a box → the existing right-rail card for that target.

## Data source (all real — no fabrication)
`case.translation.ranked_targets[]` → each item has `{gene, priority_score, rank, source, evidence_note}`.
Current values (ADNI SURVIVOR): APP 8.60 / PI4AD #18 (top), ESR1 7.99 #61, MAPT 7.30 #151,
APOE 7.15 #185, PSEN1 6.21 #492, BACE1 (candidate-only, no score).
Finding AUROC+CI + p-tau217 anchor come from the case's gauntlet/anchor fields (same numbers the
detail card already shows). The 0.728 ranking-validation stat is in the reports file above.

## HONESTY RULES (the reason this spec deviates from "AUROC per candidate")
- **Do NOT show a per-protein AUROC.** Per-candidate AUROC does not exist — candidates are ranked
  by evidence-aggregation `priority_score`, not a classifier AUROC. Fabricating one is the exact
  overclaim the project's thesis forbids.
- The ONE measured AUROC+CI belongs to the imaging finding and lives in the header, shared by the
  whole ranking. Per-candidate boxes show the real `priority_score` + PI4AD rank only.
- Positioning for the "unknown protein/compound significance" question: statistical significance
  lives at the IMAGING layer; the protein layer is prioritized cited evidence + structure,
  validated as a ranking (AUC 0.728) — not a per-protein significance test.

## Flow
investigate → **[NEW] ranked-candidate list (this spec)** → click top → existing right-rail card.
This closes the loop and sets up the visceral end-on-protein-structure + measured-significance close.
