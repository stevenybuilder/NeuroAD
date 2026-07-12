# Biomarker & Candidate-Protein Ranking — Methodology + Narrowed Recommendations

_Status: v1 (2026-07-11). Live-data results in `reports/candidate_ranking.md` and
`reports/target_prioritization_validation.md`._

This documents (a) how the pipeline ranks **biomarkers**, (b) how it ranks
**candidate proteins/targets** today, (c) the **fleshed-out composite** that
narrows the target list using multiple independent live signals, and (d) the
**honest caveats** that decide how much to trust each recommendation.

---

## 1. Biomarker ranking — how it's done

`harness/translation._dominant_biomarker` ranks the plasma markers by **effect
size** for disease-vs-control separation:

- For each of `p_tau217`, `gfap`, `nfl`, compute the standardized mean difference
  (`bridge._effect_size`) between the AD and CN groups.
- The marker with the largest effect size is named the **dominant biomarker**.

**Result (validated):** **p-tau217 dominates.** It alone predicts MCI→AD
conversion at **AUC 0.814** — above structure (0.726) and above naive multimodal
concat (0.741). This is the single strongest modality in the pipeline and is
consistent with p-tau217 capturing the tau/amyloid molecular cascade. GFAP
(astrocytic) and NfL (neurodegeneration) are secondary.

**Narrowing rule:** lead with p-tau217; use GFAP/NfL as mechanism-disambiguating
context (GFAP → glial, NfL → neurodegeneration severity), not as primary
predictors.

---

## 2. Target ranking today — how it's done

`harness/translation._rank_targets` is currently **single-signal**:

1. The referee classifies a survivor's **mechanism** (`amyloid_cascade`, `glial`,
   `vascular`).
2. Each mechanism maps to a **narrow candidate gene set** (`MECHANISM_GENES`).
3. Each candidate gets its **PI4AD priority** (0–10, from the live 14,676-gene
   portal table).
4. Sort by PI4AD priority → **top target**.
5. `_network_hubs` additionally propagates the seeds over STRING (RWR) and returns
   the non-seed hubs; the top target also gets an AlphaFold structure + STRING
   interaction partners.

This works but rests on **one** score (PI4AD), which the validation shows is
**residually circular** vs the GWAS gold set (PI4AD's Priority Index uses genetic
evidence as an input).

---

## 3. The fleshed-out composite (the narrowing) — how it's done now

`scripts/rank_candidates.py` fuses **four independent LIVE signals**, each min-max
normalized to [0,1] and weighted by how much the live validation trusts it:

| Signal | Source (LIVE) | Weight | Rationale |
|---|---|---|---|
| PI4AD priority | PI4AD portal 0–10 (14,676 genes) | 0.30 | prioritisation prior; **capped** because it's residually circular vs GWAS |
| OT non-genetic assoc (held-out) | Open Targets GraphQL | **0.35** | the **one clean, non-circular** signal (validated **AUC 0.728, p=0.003**) |
| STRING-RWR centrality | STRING v12 live network | 0.20 | network support from known-AD seeds |
| AlphaFold mean pLDDT | EBI AlphaFold DB (keyless) | 0.15 | druggability proxy (structural foldedness) |

`composite = Σ wᵢ·normᵢ` over **present** signals, weights renormalized so a
missing signal never silently zeros a gene. Every raw signal is kept, so the score
is fully auditable.

### Live result (pooled shortlist)

| Rank | Gene | Composite | PI4AD (rank) | OT-heldout | STRING deg | pLDDT |
|---|---|---|---|---|---|---|
| 1 | MAPK1 | 0.892 | 7.97 (r64) | **None** | 9 | 90.4 |
| 2 | HRAS | 0.789 | 8.19 (r45) | **None** | 5 | 91.9 |
| 3 | APP | 0.726 | 8.60 (r18) | 0.76 | 66 | 67.4 |
| 4 | ESR1 | 0.704 | 7.99 (r61) | **None** | 10 | 66.4 |
| 5 | TREM2 | 0.524 | 8.14 (r49) | 0.64 | 32 | 76.8 |
| 6 | BIN1 | 0.521 | 6.75 (r287) | **0.98** | 43 | 66.7 |

---

## 4. The honest caveat that changes the recommendation

**The composite's top two (MAPK1, HRAS — the "Ras hub") have NO Open Targets AD
association signal** (`OT-heldout = None`: they aren't in OT's top-200
AD-associated targets). Their high composite rests on PI4AD (circular) + high
pLDDT (well-folded, but that's druggability not disease-relevance) + weak network
degree. **They are network/prioritisation hypotheses, not association-supported
targets.**

Conversely, **BIN1 carries the single cleanest signal** (OT held-out non-genetic
**0.98**) yet ranks only #6 because its PI4AD priority is lower.

So there are **two defensible narrowings**, and we should present both:

- **Composite-first (exploratory):** MAPK1, HRAS, APP — surfaces the Ras-hub
  hypothesis. Flag loudly that MAPK1/HRAS lack independent AD-association evidence.
- **Clean-signal-first (defensible):** rank by the validated OT held-out signal —
  **BIN1 (0.98), APP (0.76), TREM2 (0.64), PSEN1 (0.59), CLU (0.51)**. These are
  the genes independent non-genetic evidence actually elevates. This is the list I
  would put in front of a wet-lab partner.

**Recommended narrowed shortlist for wet-lab prioritisation:**
**APP, TREM2, BIN1** — each ranks well on BOTH the composite and the clean OT
signal, spans the amyloid (APP), glial (TREM2), and endocytic/vascular (BIN1) axes,
and each has a concrete organoid readout already specified in
`translation._ORGANOID_READOUT`.

---

## 5. Honesty framing (unchanged, now earned)

This is **decision-support / hypothesis-generation**, not an efficacy claim. The
validation harness shows the ranking is not yet outcome-predictive for drug
efficacy (drug-target held-out AUC ≈ 0.52, at chance). The honest deliverable is a
**rigorously-filtered, wet-lab-testable shortlist**, validated prospectively in
iPSC/organoid models — not "an AI that predicts AD drug targets."

## 6. Next steps

1. Wire the composite (`rank_candidates.py`) into `translation._rank_targets` as an
   optional `method="composite"` so the referee can emit the multi-signal ranking
   with per-signal provenance.
2. Page Open Targets deeper (>200) so more candidates carry the clean signal.
3. Add the Boltz-2 complex-confidence signal (once the GPU fold lands) as a 5th
   column for target-pair prioritisation.
