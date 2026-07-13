# Framing — positioning, Q&A, and honest guardrails

The master positioning doc for NeuroAD: how to frame it, what to claim, what NOT to
claim, and how to answer the hard questions. Consolidates the strategic decisions from
build/review sessions. Companion to `docs/DEMO_SCRIPT.md` (the video) and
`docs/DEMO_PLASMA_FRAMING.md` (the plasma talking track). Every number cites a saved
report — cite, don't improvise.

---

## 1. The positioning in one line

> **NeuroAD closes the translation gap between neuroimaging and the wet lab — and it
> only lets *validated* signal across. Plasma is the translator; the referee is the
> gatekeeper.**

Diagnosis is where the imaging foundation-model papers stop. It's where we start.

---

## 2. What it actually is (and isn't)

- **It is:** a scientific referee + discovery engine. Imaging phenotype in →
  scrutiny → biomarker grounding → ranked, falsifiable protein targets out.
- **It is not:** a diagnostic model competing on AUROC, and not a foundation model we
  pretrained. We stand on a *frozen* encoder and build the unsolved part downstream.
- **The through-line:** the imaging→biomarker→target chain runs end-to-end on real
  ADNI data. The targets it emits are **hypotheses to test**, not validated drug
  targets (Layers 5–6 are real-evidence adapters, not outcome-validated). It *feeds*
  the wet lab; it doesn't replace it.

---

## 3. Who is it for? — both, as a hand-off pipeline

The two worlds are separated by a **translation gap**: imaging produces a *phenotype*
("this atrophy predicts conversion"), the bench needs a *molecular target* ("which
protein to knock down"). Neither output is directly usable by the other.

- **Neuroimaging researchers — the front door.** They bring scans + the
  frozen-encoder/probe/referee machinery (Layers 1–3). Value: an honest referee that
  stops a plausible imaging finding from becoming a wasted quarter.
- **Wet-lab / translational researchers — the back door.** They receive ranked,
  falsifiable protein targets with suggested experiments (Layers 5–6). Value: a
  biology-grounded shortlist, not a black-box classifier.
- **Plasma is the bridge.** The biomarker anchor converts an imaging phenotype
  (neuroimaging's language) into a molecular hypothesis (the bench's language).

**Answer to give:** *"It's built for the hand-off — neuroimaging in, wet-lab
experiments out, with the biomarker anchor as the translator. Neither audience alone
is the customer; the pipeline between them is the product. That's also why we have six
layers instead of stopping at diagnosis."*

---

## 4. Plasma as the spine (the biomarker-discovery story)

Plasma p-tau217 isn't a feature we fuse — it's the anchor that makes the imaging
signal biologically legible. The four-beat arc:

1. **Signal** — imaging predicts who converts. `conversion_imaging_only.json`
2. **Grounding** — plasma decodes *what it means*: p-tau217 is the top attributed
   driver, so the signal is **tau**. `attentive_probe_ad.json`
3. **Discovery** — where imaging & plasma **diverge**, imaging flags converters the
   blood test calls low-risk → a candidate **new imaging biomarker**.
   `conversion_biomarker_negative.json`
4. **Targets** — the anchor routes tau → ranked druggable targets (APP, MAPT…).
   `candidate_ranking.json`

**The biomarker blind-spot (the differentiated result):** split the plasma-tested
ADNI cohort (n=498, 142 converters) by the blood marker —

| Subgroup | Imaging AUC | Plasma AUC | Wins |
|---|---|---|---|
| **p-tau217 LOW** ("test says low-risk") | **0.77** [0.68–0.85] | 0.64 [0.51–0.76] | imaging |
| p-tau217 HIGH (plasma's wheelhouse) | 0.64 | **0.69** | plasma |
| Amyloid-negative | 0.66 | 0.63 | imaging (edge) |

*"Each modality carries the signal exactly where the other goes quiet."* Mockup:
`mri_visualizations/biomarker_blindspot_card.png`.

---

## 5. How we compare to the Nature Medicine paper (NeuroVFM, s41591-026-04497-1)

The paper does *almost exactly our Layer 2* as one experiment: frozen encoder +
attentive MLP probe, AD classifier trained on ADNI, validated on OASIS-1 / AIBL.

| Dimension | Verdict |
|---|---|
| Probe methodology | ✅ Matched (same frozen-encoder + attentive-MLP design; they have per-patch tokens for spatial maps, we have a pooled vector → LOGO attribution) |
| Sample size / power (diagnosis) | ✅ Comparable — bounded by the same public cohorts |
| Rigor toolkit | ✅ Matched, arguably exceeded (we add permutation nulls + negative controls + ComBat) |
| Sound reasoning | ✅ Yes (honest "report don't assume" discipline) |
| Pretraining scale | ❌ Not close (their 5.24M volumes vs our frozen encoder) |
| Breadth / clinical validation | ❌ Not close (156 tasks, prospective n=1,155, blinded experts) |

**The key number:** their AD AUROC is ~0.89–0.93; ours ~0.81–0.86 **on the same
cohorts**. That gap is **the encoder, not the science** — a 5.24M-volume purpose-built
encoder simply carries a stronger representation. Our statistical discipline matches or
exceeds theirs. And 4 of our 6 layers have no analog in the paper at all — "matching
the paper" is the wrong yardstick for most of what we built.

---

## 6. Q&A bank (anticipated hard questions)

- **"Is this for neuroimaging or wet-lab researchers?"** → Both, as a hand-off
  pipeline. See §3.

- **"Doesn't fusion (imaging + plasma) beat plasma alone?"** → No, and we don't claim
  it. +0.012, DeLong p=0.12, not separable. Plasma is the workhorse; imaging's value
  is **coverage, localization, and discovery**, not incremental accuracy. (Owning this
  is the credibility move.)

- **"Why didn't you pretrain your own encoder?"** → Pretraining a neuroimaging
  foundation model is a solved, published problem that needs a health system's data
  (NeuroVFM: 5.24M volumes, ~1,000 GPU-hours). We stand on a frozen encoder — exactly
  as the SOTA papers do downstream — and spend effort on the unsolved part: ranked,
  falsifiable targets. Also: our weights are CC-BY-NC-ND (frozen, no derivative), so
  fine-tuning would violate the license.

- **"Should you flesh out the MLP more?"** → No. At n in the hundreds the frozen
  embedding is already linearly separable — the nonlinear MLP *matches or loses to* the
  linear probe on every cohort (ADNI Δ−0.0015; OASIS Δ−0.068). Adding capacity would
  overfit; the honest verdict logic reports this. It's in a good state.

- **"Is the spatial attention heatmap worth building?"** → It's demo polish, not
  scientific differentiation. A hippocampus heatmap is *confirmatory* (shows what's
  known) and the paper already does spatial maps. Our novel interpretability is the
  **molecular attribution** (p-tau217 drives the signal) — lead with that. The heatmap
  needs a per-patch re-embed (inference, license-clean) with no accuracy gain; a mockup
  risks reading as fabricated. Deprioritized for the deadline.

- **"How do you know the imaging signal is real, not batch effect?"** → Site-disjoint
  CV, in-fold PCA, ComBat (cohort leakage 0.9996→0.563), permutation nulls, and a
  referee that *refuses* claims failing the scanner-leakage test (the KILL beat: 0.87 →
  0.41 after adjustment, scored 39/100).

- **"Do you have enough data / statistical power?"** → For **diagnosis** (AD vs CN),
  yes — AUC ~0.85, tight CIs, p<0.001, replicated across ADNI/OASIS. For
  **discovery-grade** claims (fusion > plasma), no — underpowered, and we say so. The
  binding lever is converters-with-plasma; ADNI is at its ceiling, more need external
  cohorts (AIBL is wired and ready, `data/gated.py`).

- **"What does 'serving the plasma-negative population' mean?"** → Two groups: (a)
  **plasma-unavailable** — no blood test at all (701 ADNI converter-labeled subjects;
  imaging alone predicts conversion at AUC 0.68 for them — most of the world, where the
  assay isn't deployed); (b) **biomarker-negative** — tested, came back low (the 0.77
  result). *"A tool that only works when you have a p-tau217 result is useless for the
  majority who don't."*

- **"Is it good enough for the hackathon?"** → Yes, comfortably. The risk is
  over-claiming, not under-delivering. Nature-league methodology on the diagnostic
  slice + 4 novel downstream layers + real 3-assay plasma + rigor that kills its own
  weak claims. Rubric: Demo 30% / Impact 25% / Claude 25% / Depth 20%.

- **"How can 1,600 subjects generalize to new/unseen data without a priori (external)
  validation?"** → It can't, and we don't claim it does. Bigger n tightens the
  *in-distribution* estimate (ADNI-like subjects); it does **nothing** for
  *distribution shift*. Internal site-disjoint CV ≈ transfer across ADNI *sites*, NOT
  to a new cohort/scanner/population. The honest claim is ADNI-internal; true
  generalization is an **external-cohort test** (OASIS today; AIBL/NACC next), and the
  referee **measures and reports** transfer rather than assuming it. "Not proven
  out-of-distribution" is an *output*, not a bug. (See §9.)

- **"If it doesn't generalize, what's the point when a researcher enters a
  hypothesis?"** → Because the tool is a **falsification engine, not a predictor**. It
  never promises a model that works everywhere; it tells a researcher whether the
  signal in *their* data survives adversarial scrutiny (leakage, permutation,
  replication, molecular anchor) — *before* they burn a quarter chasing an artifact.
  The **method is cohort-agnostic** and generalizes even when a specific *finding* is
  cohort-limited. In a field where most brain-ML doesn't replicate, an honest referee
  that says "this is scanner leakage, don't chase it" IS the product.

- **"Does NeuroJEPA (and the layers) fix the sample-size problem?"** → For
  *efficiency*, yes; for *power/generalization*, no. Transfer learning off a frozen
  foundation model means *hundreds* of labels suffice where a from-scratch CNN needs
  *tens of thousands*; PCA + ComBat + fusion further stretch small cohorts. But nothing
  manufactures statistical power for a thin claim (58 converters stays a wide CI) or
  proves out-of-distribution transfer. Sample-**efficient**, honest about the ceiling.

- **"Does the 5.7× AD expansion help the conversion result?"** → No — it strengthens
  *diagnosis* (AD-vs-CN imaging arm: 87 → 494 embedded AD, tighter CIs), not
  *conversion*. Cross-sectional AD can't join a baseline-MCI cohort. Conversion's
  binding lever is converters-with-plasma → external cohorts (AIBL/NACC). Do **not**
  point the AD number at the conversion claim.

- **"What's your backend pipeline — how does a number actually get produced?"** → A
  hypothesis + dataset flows through a **five-stage referee**: (0) standardize into the
  data contract → (1) probe for a leakage-honest OOF signal → (2) attack it with the
  **5-test gauntlet** → (3) score with a **hard honesty gate** → (4) Claude narrates/
  argues (read-only, never scores) → (5) if it survives, a **composite multi-signal
  ranker** emits the ranked 1–5 targets. Full detail + module names in §12.

- **"What if I enter a hypothesis the data can't measure — say a random protein?"** →
  It's parsed onto the nearest measurable target (`dx_binary`/`conversion`), but the
  **substrate check refuses to fabricate a result** for a predictor that isn't a
  measured analyte or a wired target. Known analyte (p-tau217/GFAP/NfL) → supervised
  probe; known AD target/gene → the evidence-ranker (STRING/LINCS/OpenTargets);
  neither → honest "no substrate to test this here." The tool says "I can't test that"
  rather than invent a correlation — same ethos as everything else. See §12.

---

## 7. Honesty guardrails (the non-negotiables)

Owning these *is* the credibility, and credibility is the product.

- **Never claim fusion beats plasma.** It doesn't (not separable). Say complementarity.
- **The biomarker blind-spot (0.77) is directional** — 23 converters, wide CI. Say
  "directional," not a locked estimate.
- **Targets are hypotheses**, not validated drug targets. Layers 5–6 are real-evidence
  adapters + scaffold, not outcome-validated.
- **Plasma is real but gated** (ADNI, 3 assays) — no open cohort pairs MRI with plasma.
  That gap is also the whitespace that makes this novel.
- **The AUROC gap vs the Nature paper is the encoder, not the rigor** — state it
  plainly; it turns a disadvantage into a scope difference.
- **Diagnosis is our input, not our product.** Lead with the discovery engine.

---

## 8. Backing numbers (all from saved reports)

| Claim | Number | Source |
|---|---|---|
| AD vs CN (frozen embedding, ADNI) | AUC ~0.85 [0.81–0.89] | `adni_neurojepa_crosscohort.json` |
| AD vs CN cross-cohort (site-disjoint) | 0.83 | `adni_neurojepa_crosscohort.json` |
| AD vs CN (OASIS clinical) | 0.81 | `oasis_neurojepa_ad.json` |
| MLP vs linear head | Δ −0.0015 (ADNI), matches | `attentive_probe_ad.json` |
| Conversion — plasma alone | 0.814 | `adni_conversion_multimodal.json` |
| Conversion — fused vs plasma | +0.012, p=0.12 (not separable) | `power_analysis_conversion.md` |
| Conversion — imaging, plasma-absent (n=701, 270 conv) | 0.68 | `conversion_imaging_only.json` |
| Biomarker blind-spot (p-tau low) | imaging 0.77 vs plasma 0.64 | `conversion_biomarker_negative.json` |
| ComBat cohort-leakage drop | 0.9996 → 0.563 | `adni_neurojepa_crosscohort.json` |
| Plasma p-tau217 (real, 3 assays) | 1,377 subjects | `plasma_ensemble.py` |

---

## 9. Sample hypotheses a researcher can enter (input → meaningful output)

The unit of input is a **Claim** (`contract.Claim(claim_text, target=…, group_a/b=…)`);
the output is a **verdict card** (OOF AUC + bootstrap CI + permutation p + leakage /
replication / anchor tests → promoted / fragile / killed, scored /100). These all run
on real wired data *today* — they are demo inputs, not aspirations:

| Hypothesis you type | What the tool returns | Why it's meaningful |
|---|---|---|
| "AD is decodable from structural MRI" (`dx_binary`, AD/CN) | Naive AUC **0.935**, then scanner-leakage test fires (scanner AUC **0.989**) → **fragile / not promoted, 39/100** | The flagship: the tool **catches its own feeder cheating** — flags a flashy result as scanner artifact before anyone ships it |
| "MCI→AD conversion is predictable from baseline MRI" (`conversion`) | OOF AUC ~**0.72–0.82**, p_perm=0.001, site-disjoint, honest CI | A **real but modest** prognostic signal, reported with the uncertainty the sample supports |
| "Plasma p-tau217 anchors the imaging AD axis" (anchor test) | Correlation **r≈0.46** (AD/CN), r≈0.26 (conversion) on measured plasma | The imaging signal aligns with **tau biology** — not purely artifact; this is what survives the KILL |
| "Fusion (imaging+plasma) beats plasma alone" (fusion + leave-one-out) | Fused **0.82** vs plasma **0.80**, Δ≈+0.001, CIs overlap → **no CI-supported gain** | An **honest negative** — the tool refuses the overclaim; complementarity, not superiority |
| "The imaging signal generalizes across cohorts" (train ADNI → test OASIS) | Cross-cohort AUC + cohort-leakage **0.9996→0.563** after ComBat | Turns "does it generalize?" into a **measured out-of-cohort number**, not an assumption |
| "MAPT / tau is a prioritized target for AD" (`rank_candidates`) | Ranked druggable targets (APP, MAPT, APOE, BACE1…) + evidence adapters (STRING/LINCS/Boltz) + suggested experiment | A **falsifiable target shortlist** for the bench — the discovery-engine output |

The strongest three for the demo: the **self-catching AD artifact** (rigor story), the
**honest fusion negative** (credibility story), and the **target ranking** (the "so
what — feeds the wet lab" story).

---

## 10. Generalization & external validity — the referee reframe

The sharpest line of attack, and the answer that turns it into a strength:

- **Internal CV ≠ external validation.** Our numbers are site-disjoint OOF *within*
  ADNI — they estimate transfer to new **ADNI-like** subjects (and, because whole sites
  are held out, give partial evidence against scanner overfitting). They do **not**
  prove out-of-distribution transfer. **n does not buy generalization** — a model
  trained on a million ADNI scans can still fail on a new scanner. Only an external
  cohort answers it.
- **The tool measures generalization; it doesn't promise it.** Cross-cohort transfer
  is one of the referee's *tests* (OASIS-1/2 + OpenBHB embeddings exist;
  `run_adni_crosscohort.py`), with AIBL/NACC as the roadmap (`data/gated.py` drop-in).
- **Sample size: efficiency vs power vs generalization.** The frozen NeuroJEPA encoder
  (transfer learning) + PCA + ComBat + fusion make the tool **sample-efficient** —
  hundreds of labels suffice where a from-scratch CNN needs tens of thousands. They do
  **not** manufacture statistical power for a thin claim (58 converters stays a wide
  CI) or prove distribution-shift transfer. Sample-efficient, honest about the ceiling.
- **One-liner:** *"We don't assume generalization — we adversarially test it, and
  report the honest out-of-cohort number rather than the flattering in-sample one."*

---

## 11. Recent findings (session 2026-07-12)

- **AD imaging expansion (diagnosis arm) — precision, NOT discrimination or
  generalization.** Pulled **407 new AD MPRAGEs** from IDA (verified, 0 missing),
  embedded via frozen NeuroJEPA on Colab; embedded AD **87 → 494 (5.7×)**. Adversarial
  verification (3 skeptics, `reports/AD_EXPANSION_ANALYSIS.md`) **caught our own
  headline as overclaimed**: the raw pooled AUC "gain" 0.857 → 0.891 does NOT survive
  scanner matching. 40% of the added AD are **1.5T** while **100% of CN are 3T**, so
  field strength (readable at AUC **0.990**) became a partial AD proxy. Honest numbers:
  down-sampled to original n → ~0.853 (≈ baseline); **scanner-matched 3T-only → 0.861
  [0.835, 0.885]** vs 0.857 baseline (**+0.004**, not +0.034). The **one real win** is
  precision: the 95% CI **halved (0.081 → 0.039)**, an audited √n gain on balanced,
  site-disjoint folds (not a Card-A degenerate split). The expansion bought a *more
  precise estimate of a partly-confounded quantity* — it did **not** de-confound (needs
  ComBat) or prove OOD transfer (needs an external cohort). **Report 0.861 (3T-matched),
  not 0.891.** This is the tool working as designed: our own referee refused our own
  inflated number. Strengthens *precision* of the imaging arm — **not** conversion
  (cross-sectional AD can't join a baseline-MCI cohort).
- **Conversion NeuroJEPA-fusion** (334-subject slice, 58 converters / 276 stable):
  attention fusion of imaging(FreeSurfer) + plasma + NeuroJEPA. Fused **AUC 0.82
  [0.76, 0.88], p_perm=0.001**; plasma dominant (gate 0.50, leave-one-out −0.069);
  **NeuroJEPA adds +0.0008** (LOO −0.0008) → no incremental signal over FreeSurfer +
  plasma. Honest negative. `reports/ADNI_CONVERSION_FUSION.md`.
- **CIs + permutation nulls confirmed present** everywhere (`probe.auc_ci_perm`), with
  a self-honest caveat that `p_perm` is a lower bound.
- **Demo-hardening TODO** (targets the exact judge questions above): (a) surface CI
  bars + p-values next to every verdict in the UI; (b) a visible "connected to real
  ADNI/OASIS (n=…)" indicator + the bring-your-own-cohort drop-in; (c) an
  external-validation slide (train-ADNI / test-OASIS transfer number).

---

## 12. The pipeline — how a hypothesis becomes a verdict (backend data-science layer)

The whole engine in one line: **standardize → probe for a signal → attack it with 5
adversarial tests → score behind a hard honesty gate → (if it survives) rank druggable
targets by a transparent multi-signal composite.** The referee decides *whether to
believe it*; the ranker decides *what to do about it*.

### The five stages (`pipeline.run_referee`)

| Stage | Module | What it does |
|---|---|---|
| **0. Data contract** | `data/loaders.py`, `gated.py`, `contract.py` | Any cohort (ADNI/OASIS or a user CSV) → one standard table: `subject_id, dx, conversion, site, scanner, plasma (p-tau217/GFAP/NfL), age/sex`, and `emb_*` feature columns (FreeSurfer or frozen NeuroJEPA). Cohort-agnostic. |
| **1. Probe (naive effect)** | `probe.py` | Deliberately simple, leakage-honest classifier (StandardScaler + in-fold PCA + logistic) through **site-disjoint, repeated OOF CV** → headline AUC + bootstrap 95% CI + permutation p. |
| **2. Gauntlet (5 tests)** | `gauntlet.py` | The adversarial battery (below). |
| **3. Scoring + honesty caps** | `scoring.py` | Weighted **score /100** + **verdict** (promoted/fragile/killed). **HARD GATE:** a claim that fails leakage or lacks a biomarker anchor **cannot be promoted** no matter how high the score. |
| **4. Claude layer** | `claude/` | `claim_parser` (NL→Claim), `courtroom`/`reviewer` (argue *against* the verdict), `narrator` (explain). **Read-only — never changes the numbers.** |
| **5. Translation → targets** | `harness/translation.py`, `harness/ranking.py`, `integrations/*` | Validated phenotype → the ranked **1–5 druggable targets** (composite below). |

### The 5 gauntlet tests (`gauntlet.py`)
1. **Naive effect** — the raw OOF signal from Stage 1.
2. **Site/scanner leakage (STAR)** — can the same features predict acquisition hardware? (Caught the AD-expansion field-strength confound at AUC 0.990.)
3. **Permutation null** — above chance?
4. **Biomarker anchor (HARD GATE)** — correlates with measured plasma p-tau217/GFAP, judged on the CI *lower bound* so lucky small-n noise can't pass?
5. **Replication** — holds on a held-out site/cohort?

### The ranked 1–5 (`harness/ranking.py`) — a transparent composite, not a black box
`Composite = Σ wᵢ · normalized_signalᵢ` over the signals present (weights renormalized if
some are missing): **PI4AD priority**, **STRING-RWR network centrality (0.20)**, **LINCS
L1000 drug-signature reversal**, **Boltz-2/AlphaFold structural confidence**, **OpenTargets
evidence**. Each min-max normalized to [0,1]; sorted → targets 1–5, each with component
values, a source stamp, and a suggested wet-lab experiment. Gated behind the gauntlet, so
only validated phenotypes reach it.

### How a hypothesis is categorized & routed (entry point)
On entry, two classifications happen before any number is computed:
1. **Parse → structured Claim** (`claude/claim_parser.py`): free text → `target` (must be
   an allowed label column: `dx_binary`, `conversion`, …), `group_a`/`group_b`,
   `covariates`. The hypothesis is *coerced onto a measurable outcome*.
2. **Novelty + mode**: `novelty_class` = **known** (re-measuring prior art —
   scanner/site/leakage) / **novel** (undiscovered structure — subtype/latent/cluster) /
   **adjacent**; and `discovery_router.route` picks the engine — **novel-pattern** →
   unsupervised **Detective** (`discovery.discover_and_referee`, clusters embeddings and
   referees each recovered phenotype); **named-contrast** → supervised
   **`pipeline.run_referee`**.

**Edge case — "X random protein correlated to Alzheimer's":** the phenotype parses to
`dx_binary`; the protein is the predictor, and it branches on what X is — a **measured
analyte** (p-tau217/GFAP/NfL) → supervised probe/anchor; a **known AD target/gene** → the
evidence-ranker (STRING/LINCS/OpenTargets); **neither** → **no substrate**, and the tool
says so instead of fabricating a correlation. Because `target` is constrained to real
label columns and predictors to measured features, the engine **cannot silently invent a
result for an ungrounded input** — by design. (Caveat: the parser maps vague inputs to the
*nearest* target by default; the substrate check is what stops that from producing a
meaningless number.)

---

## 13. Team Q&A — imaging→omics connection, PI4AD & data provenance (2026-07-13)

Answers to the questions that come up when a teammate traces the pipeline end to end.
State these as design decisions, not confessions — the boundaries here are the credibility.

**The connection in one line:** *imaging routes, PI4AD ranks — the plasma biomarker is the
hinge.* The MRI finds and validates a signal and picks the **mechanism**; the genetics/omics
side supplies the **ranking** for that mechanism's genes. That split is deliberate: the
ranking is an external, citable authority, not a number we tuned.

- **"How does the ADNI data connect to the PI4AD ranking table?"** → Through the
  **plasma-biomarker column**. The ADNI CSV carries, per subject, the 768-d NeuroJEPA
  embedding (`emb_*`) + measured plasma (`p_tau217`/`gfap`/`nfl`/`amyloid`) +
  `dx/age/sex/site/scanner`. Flow: probe `emb_*` for a signal → referee kills confounds →
  **correlate the survivor with the biomarker columns; whichever it tracks selects the
  mechanism** (`p_tau217→amyloid_cascade`, `gfap→glial`, `nfl→vascular`, in
  `_MECH_BIOMARKER`) → the mechanism indexes a gene set → looked up in the PI4AD priority
  table → ranked targets → AlphaFold on the top one. The biomarker is a real molecular
  readout (p-tau217 = measured phospho-tau), so it's a legitimate hinge, not a leap.

- **"Is the ranking computed from the MRI?"** → No, and it can't be — for anyone. A
  structural scan carries brain shape/atrophy; gene *priority* is a population-genetics
  statement (GWAS/eQTL/networks across thousands of people). The MRI's job is discovery +
  validation + **mechanism routing**; the genetics ranking is PI4AD's. Say **"imaging
  routes, PI4AD ranks,"** not "the MRI computes the ranking." (Optional upgrade: condition
  the ranking on regional atrophy via Allen-atlas expression — scoped at ~2 days for a
  demo-tier version; documented, not built.)

- **"Is this based on PI4AD? Do we run it?"** → The target ranking **is** PI4AD (real
  published priorities: APP 8.60/#18, MAPT 7.30/#151, …). We don't run its R package (no R
  runtime, by design); we ship a **provenance-stamped snapshot** of its portal output (74
  top genes, cross-checked vs the paper) with an **optional live-fetch** path, and run our
  **own STRING-RWR network propagation** on top. Using an external authority is the point —
  the calibration check (recovers APP/ESR1) only carries weight *because* the numbers aren't
  ours. (The fuller 1–5 ranker in §12 blends PI4AD with STRING/LINCS/AlphaFold/OpenTargets;
  the demo card the team saw is the PI4AD-priority view.)

- **"What omics data do we use?"** → GWAS / eQTL / PCHi-C are **upstream, baked into
  PI4AD's precomputed ranking** — we consume the output, we don't ingest those files. The
  one omics-type dataset we compute on directly is **STRING** (protein-interaction network,
  our RWR propagation). Everything we ingest is **imaging (MRI→embeddings) + plasma
  biomarkers + clinical** — not raw omics. Say "we stand on PI4AD's multi-omics ranking and
  run our own network propagation," not "we do multi-omics integration."

- **"Are the ranked proteins the biomarkers?"** → No — different jobs. **Biomarkers
  (p-tau217/GFAP/NfL) are what we measure** to confirm the signal is real biology and route
  the mechanism. **Ranked proteins (APP/MAPT/APOE) are drug targets** — what to intervene
  on. They overlap in biology (MAPT = the tau gene; p-tau217 = measured tau protein) but one
  is the evidence, the other is the target.

- **"What genome do we derive from?"** → None per-subject — we don't sequence anyone. Genes
  resolve to the **human reference via Ensembl IDs** (Open Targets on GRCh38); proteins/
  structures via **UniProt** (APP = P05067). The genomic evidence lives in **PI4AD's
  population genetics** (GWAS/eQTL). State the limitation plainly: the underlying GWAS is
  **European-ancestry-biased**.

- **"Raw ADNI scans, or preprocessed tables?"** → Raw, and processed by us. We pulled raw
  DICOM MPRAGE (590 + 407 + 334 across three collections) and ran our own preprocessing —
  skull-strip (deepbet / SynthStrip), resample, **NeuroJEPA-encode into our own
  embeddings**, FastSurfer for volumes. The embeddings are ours, not someone else's
  spreadsheet. Note: these are **structural T1, not fMRI** — classical fMRI steps (temporal
  motion correction, slice-timing) don't apply; the structural analogs (skull-strip,
  orientation/affine, QC annotation) are already built and run in the SFG module
  (`mri_visualizations/backend/sfg/`).

**Language — use vs avoid (this cluster):**
- **Use:** "imaging routes, PI4AD ranks"; "the biomarker is the hinge to the omics side";
  "PI4AD is an external, citable authority"; "we run our own STRING propagation"; "our own
  NeuroJEPA embeddings from raw scans."
- **Avoid:** "the MRI computes the ranking"; "we do multi-omics integration"; "these
  proteins are the biomarkers"; "we derive from genome X"; "classical fMRI on our data."

---

## 14. Live vs frozen — how the demo actually runs (deployment architecture, 2026-07-13)

**The one line:** the backend is **live cloud compute**, not a replay. On Cloud Run,
`/api/investigate` recomputes the full referee per request and `/api/ask` is live Claude
(Opus) — and the **ADNI / Neuro-JEPA headline recomputes live**, not from a cache.

**Three tiers — what runs where:**

| Layer | Where | Live or precomputed |
|---|---|---|
| Neuro-JEPA embedding of raw MRI + FastSurfer | Colab GPU, offline | **Precomputed** — Cloud Run has no GPU; the output is a small embedding table |
| Referee (probe → 5-gauntlet → score → verdict) | Cloud Run CPU, per request | **Live recompute** — numpy/sklearn, no GPU in the request path |
| Claude narration / courtroom (`/api/ask`) | Cloud Run → Anthropic API | **Live** — Opus, `claude_live:true` |
| `demo_data.json` prefilled happy-path | static asset | **Frozen** — the offline "insurance" replay for the static / GitHub-Pages demo; the live backend recomputes instead |

**Verified live receipts (deployed URL, 2026-07-13):**
- `adni:neurojepa` → **live** AUC 0.857 (adj 0.764), n=590, real Neuro-JEPA embeddings + real
  gated p-tau217 anchor; robustness computed live (age/sex passed, scanner weakened,
  brain-age passed).
- `oasis` (weight-free feeder) → live AUC 0.468/0.482, n=86.
- `synthetic:KILL` → live AUC 0.935 (the KILL beat).

- **"How is the API live without a GPU — is it all cached?"** → No. The *only* precomputed
  step is the **GPU Neuro-JEPA embedding of raw scans** (Cloud Run has no GPU; the embedding
  is a one-time transform). Everything downstream — the whole referee + Claude — runs **live
  per request** on CPU against the cohort tables baked into the container. A production
  version doesn't "add live compute"; it already has it. It only adds an **on-demand GPU
  worker** so a user could upload a *brand-new* MRI and embed it live.

- **"The Neuro-JEPA embeddings are license-restricted — how are they in the live ADNI demo?
  Isn't that a leak?"** → No, and the boundary is deliberate. The ADNI embedding tables + the
  gated ADNI clinical table are **baked into the PRIVATE Cloud Run image** (explicit
  `.gcloudignore` negations) but stay **OUT of public GitHub** (gitignored). The distinction
  that keeps it clean: a **private service** that exposes only **aggregate results** (an AUC,
  a verdict) over the API is **not public distribution** of the tables — users never receive
  the subject-level data. Non-commercial (hackathon), de-identified, inside the private GCP
  project. **Public GitHub = no embeddings; private Cloud Run image = embeddings baked in.**
  *(Design rationale, not legal advice — license read in `docs/HF_ACCESS.md` + ignore-file
  headers.)*

- **⚠️ Demo-safety:** for a **live** "watch it compute" moment, drive typed hypotheses to
  `adni:neurojepa` (live, the headline) or the open cohorts (`oasis`/`openbhb`/`synthetic`).
  Do **not** type an `oasis:neurojepa` hypothesis live — that specific embedding table isn't
  shipped, so it errors honestly. The prefilled happy-path is the frozen fallback (real
  numbers) if the network/API is down.

**Language — use vs avoid (this cluster):**
- **Use:** "the backend is live cloud compute"; "ADNI recomputes live on the private Cloud
  Run image"; "only the GPU embedding is precomputed"; "embeddings ship to the private image,
  never to public GitHub."
- **Avoid:** "we cached the results and replay them"; "it's not really live"; "the embeddings
  are on GitHub."

## 15. Deep dive — permutations, precompute, and the LLM router (2026-07-13)

*Folded in from a code-grounded review of the real implementation + external best-practice.
Every claim below traces to a file:line in the repo; the external talking points carry citations.*

### 15a. Permutation testing — how we establish significance (not a bare cutoff)

**Plain version.** The referee turns a frozen-embedding linear probe into a *hypothesis test*.
It computes an out-of-fold, site-disjoint cross-validated AUROC once, then reuses those frozen
scores to build (a) a percentile **bootstrap 95% CI** (resample subjects) and (b) a **label-
permutation null** (shuffle the outcome labels *within each site group*, recompute the AUROC).
Pointing the same probe at the disease outcome vs. at the scanner/site gives two AUROCs whose
difference is the **leakage margin**; the same machinery yields `outcome_p_perm`, `scanner_p_perm`,
and `margin_p`, which are then **Benjamini-Hochberg FDR-corrected** together. The star's verdict is
stated as **"margin CI excludes zero,"** not a hand-picked threshold. Code: `src/neuroad/probe.py`
(`auc_ci_perm`, `_shuffle_within_groups`, vectorized rank-AUC), `src/neuroad/leakage.py`
(`leakage_margin`, margin CI/p, BH-FDR), `src/neuroad/gauntlet.py` (only `test_site_scanner` emits
the star p's; the other four gate on retained-fraction bands / a Fisher-z CI / a bootstrap CI).

**Q&A.**
- *What is the permutation null actually testing?* For grouped (outcome) targets it's stronger than
  plain chance: labels are shuffled **within each site**, so site class-balance is held fixed and the
  p-value answers "is the label↔score link better than what site membership alone explains?" — not
  merely "better than random."
- *How is `margin_p` different from comparing two p-values?* It's its own permutation null on the
  *difference*: P(outcome_perm − fixed scanner_auc ≥ observed margin), add-one smoothed.
- *Why isn't the scanner AUROC also site-disjoint — doesn't that inflate it?* Holding out the group
  you're predicting is degenerate, so scanner uses ordinary stratified CV. That can make scanner
  *optimistic*, which biases the margin **downward** — the safe direction for a skeptic's tool (the
  margin can only understate the outcome's edge).
- *Are the p-values honest given the speed tricks?* The code discloses it: the null fixes the fitted
  OOF scores (the probe isn't refit per permutation), so `p_perm` is an admitted **lower bound**
  (anticonservative) — a documented tradeoff, and add-one smoothing means it's never reported as 0.
- *Which of the 5 gauntlet tests produce permutation p's?* Only the site/scanner star (via
  `leakage_margin`). Age/sex and brain-age gate on retained-fraction bands (0.70 / 0.40); the
  biomarker anchor on a Fisher-z CI lower bound; replication on a held-out-site bootstrap CI ≥ 0.65.

### 15b. Precomputing everything — a finite grid of *real* results

**Plain version.** The real cost is the referee (~25-36 s/hypothesis). But any free-text hypothesis
collapses to a tiny **finite coordinate** — `(dataset, target ∈ {conversion, dx_binary, site,
scanner}, region, anchor ∈ {amyloid, p_tau217, gfap, nfl, none})` — so the whole answerable space is
a small enumerable grid. We sweep it **offline** (`scripts/warm_investigate_cache.py`) running the
*real engine* per cell, and a live `/api/investigate` becomes an **O(1) dict lookup** of that cell
(`app/investigate_cache.py`). Two honesty properties: every cached cell is a genuine full-rigor
engine output (the `demo_data.json` **"frozen-seam"** pattern generalized from one cell to the whole
grid — never a fabricated number), and a **miss still computes live and back-fills**. A live miss is
itself ~53× cheaper because `orchestrator._base_memo` caches the anchor-*invariant* referee base, so
switching anchor re-applies only the ~0.5 s translation, not another ~30 s referee.

**Q&A.**
- *How does a 25-36 s recompute become O(1)?* The text is reduced to the coordinate key
  `dataset|target|region|anchor|want_api`; the handler does a plain in-memory dict `.get` on the
  preloaded grid and returns the personalized real cell before ever entering the referee.
- *Isn't a cached cell a fake?* No — the warmer calls the **identical** `compute_investigate →
  orchestrator.investigate` code path the live server uses on a miss, so cache and live are
  byte-identical by construction.
- *If it just serves a cell, is the displayed hypothesis wrong?* `personalize()` deep-copies the cell
  and overwrites **only display text** (`_meta.hypothesis`, `claim_text`); the score/verdict/tests/
  effect stay the coordinate's genuine values. Two wordings that route to the same coordinate legitimately
  share one real result, each shown with its own text.
- *What's the ~53× speedup?* It's the *miss* path: the anchor only routes read-only translation
  artifacts — never a probe input, gauntlet test, score, verdict, or promotion — so one base is the
  genuine result for every anchor (verified byte-identical), turning ~36.8 s into ~0.7-2.2 s.
- *Can it desync from the shipped grid?* Only via staleness: if the engine math changes, the grid must
  be **re-warmed** (as it was this session after the translation + invariance changes). A routing
  mismatch is at worst a miss, never a wrong number. Out-of-scope hypotheses bypass both cache get and
  put so they can't collide with a promoted cell — they hit the honest refusal.

### 15c. LLM-as-router, *not* judge

**Plain version.** We use the model as a **router**: it maps free text to one of the finite enum
`{conversion, dx_binary, site, scanner}` — it never produces or changes a number. A judge *evaluates
an output after the fact*; a router *classifies the input before work happens*, and intent-routing
over a bounded enum is a router problem. `route_target()` goes normalized-text route-cache hit →
one Sonnet-5 **enum-constrained** structured call on a miss (temp 0, reason-before-label) → keyword
regex **backstop** on no-key/low-confidence/error, and never raises. One canonical function feeds
**both** the engine target (`claim_parser._fallback`) and the cache key
(`investigate_cache._infer_target`), so they can't drift. It fixes the exact bug: "p-tau217 predicts
hippocampal atrophy in preclinical AD" is a cross-sectional `dx_binary` claim (0.92) that the keyword
regex misrouted to conversion (0.64). Backed by a 58-item golden set + a CI eval (offline
no-regression + a gated live eval that hard-asserts LLM ≥ keyword per class and materially better on
the adversarial collision bucket).

**Q&A.**
- *Why "router" not "judge"?* Its output space is a four-value enum enforced at the API grammar level
  (strict tool call). It classifies intent into a coordinate; it doesn't adjudicate anything — the
  verdict/scoring is deterministic Python.
- *If the LLM picks the wrong target, does it corrupt the result?* No. It only selects which precomputed
  cell is looked up; a wrong/novel route is at worst a cache miss that recomputes the genuine result live.
- *How do you guarantee the cache-key target matches the engine target?* By construction — both call the
  identical `route_target`. One router, so no drift.
- *What stops it being worse than the old regex?* The regex **is** the backstop; the always-on test
  asserts `route_target == keyword` exactly when offline, so the router is a strict superset.
- *Is determinism guaranteed?* We don't overclaim it. Temperature 0 is *near*-deterministic, not
  bit-guaranteed (floating-point / GPU nondeterminism). Determinism actually comes from the **route
  cache** + the keyword backstop, not the model.

### 15d. External best-practice — defensible talking points (cited)

- **Significance:** reporting the margin above a **label-permutation null** with p = (C+1)/(N+1) is the
  scikit-learn `permutation_test_score` / Ojala-Garriga (JMLR 2010) method and the Nichols-Holmes
  (2002) neuroimaging standard — strictly stronger than an "AUROC > 0.7" cutoff.
- **Leakage:** the permutation null doubles as a leakage detector (a null centered *above* chance flags
  a site/scanner shortcut). Fitting ComBat/feature-selection outside the CV loop can inflate performance
  by Δr up to ~0.47 (Rosenblatt et al., *Nat Commun* 2024; ComBat class-imbalance leakage, arXiv
  2410.19643) — we fit fold-wise and audit site-predictability, which is exactly why our margin is
  credible.
- **Precompute:** a precomputed finite grid behind one semantic-layer definition is the dbt/AtScale
  "metrics layer + intelligent cache" pattern — every displayed number is a real computed result served
  from cache (memoized truth, one source, many read-only consumers), not a mock.
- **Routing:** classify-before-act over a bounded enum with **constrained decoding** at temp 0, plus a
  **routing cache** and a deterministic keyword backstop, is the standard production ensemble
  (heuristics → lightweight classifier → LLM escalation on low confidence). A small classify model adds
  ~5-100 ms vs 500-2000 ms for a full LLM call, and the cache means the model is hit at most once per
  novel hypothesis.

**Language — use vs avoid (this cluster):**
- **Use:** "the referee reports how far above a permutation null the effect sits"; "every cached cell is
  a real full-rigor engine output"; "the LLM is a router over a finite enum — it never touches a number";
  "determinism comes from the route cache + keyword backstop, not from temp 0."
- **Avoid:** "the AUROC clears our threshold" (we don't use a bare cutoff); "the LLM judges the
  hypothesis"; "temperature 0 makes it deterministic"; "the grid is mocked/hardcoded."
