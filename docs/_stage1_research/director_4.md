# NeuroAD Discovery Engine — Scientific North Star

*Director of Science synthesis. This is the direction the biostatistics and engineering directors build against. Decisive, not a survey.*

---

## 0. The one-sentence north star

**NeuroAD is not a detector and not a treatment tool. It is a discovery-and-referee accelerator: it turns a researcher's one-sentence hypothesis into a stability-vetted, confound-screened, biologically-anchored *candidate* imaging phenotype — plus the single experiment that would confirm or kill it.** Everything below defends and operationalizes that sentence.

---

## 1. End-outcome framing: DISCOVERY+VALIDATION accelerator (definitive)

**Position: upstream discovery + validation accelerator that feeds both detection and treatment — and is neither.** This is not a hedge; it is the only framing that survives contact with both the 2025 science and the Gladstone bar.

**Why not DETECTION.** As of May 16, 2025 the FDA cleared plasma p-tau217/Aβ42 (Lumipulse); three assays now hit AUC 0.94–0.97 for amyloid-PET status. *A blood draw already answers "is there amyloid?" as well as PET.* Any tool whose end-outcome is "detect AD" is racing a cleared, cheap, deployed diagnostic on the one axis (amyloid) where imaging no longer wins. On 61 single-site subjects you cannot beat p-tau217, and it is the wrong claim: the 2024 NIA-AA criteria explicitly retain imaging for the **N-axis, topography, and staging** — *where*, *how much*, *how fast*, not *whether*. Detection is a solved-enough problem we would lose at.

**Why not TREATMENT.** No path from 61 embeddings to a therapeutic in 5 days. Overclaim red line.

**Why DISCOVERY+VALIDATION is the winning frame — tied directly to the Gladstone "advance the field" bar.** The frontier-science report is unambiguous: in 2025 a contribution *advances* AD imaging when it (i) generates a **testable biological hypothesis, not a prediction**; (ii) **survives external replication + confound control**; (iii) **anchors to an orthogonal, independently-measured axis** (plasma p-tau217/GFAP/NfL, genetics, conversion); and (iv) **fills a gap plasma cannot** (topography, N-axis, heterogeneity/subtype, rate-of-change). "Higher benchmark accuracy" is explicitly *not* advancing. The field is **saturated with classifiers and starved of trustworthy, reproducible, mechanistic phenotypes.** A discovery+validation accelerator is a machine that manufactures exactly (i)–(iv). That is the definition of advancing, mechanized.

**Why this also wins the white space.** The competitive scan proves the empty cell precisely: imaging foundation models (Neuro-JEPA, Brain-JEPA, BrainDINO) own the embedding substrate and *stop*; the AD co-scientists (Biomni-AD, Prima Mente) own hypothesis→experiment but on omics/epigenome and never touch pixels; POPPER is a real falsification referee with a free-text entry point but on tabular CSV, no imaging, no discovery front-end; SuStaIn discovers imaging subtypes but on engineered volumes with no referee, no hypothesis entry, no next-experiment. **We are "POPPER for SSL neuroimaging phenotypes, with the discovery front-end POPPER lacks and the referee SuStaIn lacks."** Discovery+validation is the only framing that lands us in that intersection.

**It feeds both detection and treatment without being either:** a refereed candidate phenotype is an input to a future diagnostic panel (detection) and a stratification hypothesis for a trial (treatment). We are the upstream engine, honestly scoped.

---

## 2. The single most impressive HONEST demo

**Concept: "The referee catches its own false discovery, then finds a real one."** A two-act live run on real frozen Neuro-JEPA embeddings.

**What the researcher types (entry point):**
> *"I think there's a non-hippocampal atrophy pattern in this cohort that tracks disease severity beyond normal aging. Show me if it's real or if it's scanner noise."*

**What the researcher sees, in order:**

1. **Hypothesis-in → plan-out card.** Claude parses the sentence into a structured `HypothesisSpec`: target = unsupervised phenotype in frozen embedding space; contrast = CDR/diagnosis; confounds to prioritize = age, sex, scanner/field-strength; **pre-registered falsification criterion** = "reject if scanner-AUC > threshold OR bootstrap-Jaccard < 0.6 OR anchor CI lower-bound crosses zero." The researcher sees the tool commit to what would kill the finding *before* it runs. (This puts Claude on the critical path, not narrating.)

2. **Act I — the trap, sprung on real data.** The Detective clusters the real embeddings; the gauntlet runs live, ticking. A cluster that looked promising lights up **red on the scanner-leakage test** (double-dissociation via LDA scanner directions on the 6-site OpenBHB split). Verdict: **KILLED — scanner artifact.** The courtroom (Prosecution/Defense over the same evidence) narrates *why*. This is the field's central methodological anxiety (arXiv:2604.14441 — embeddings leak site) demonstrated live, and it is more convincing than any accuracy number: *we caught our own false discovery.*

3. **Act II — the survivor.** The same machinery, pointed at the real disease-bearing OASIS-1 cohort, surfaces a phenotype that **passes** bootstrap-Jaccard stability (≥0.6), **passes** age/sex confound control, **passes** the scanner-leakage gauntlet, and **anchors** to an orthogonal axis we actually hold (CDR / diagnosis severity, with the anchor's specificity stated: severity-anchor = "AD-like," not "amyloid-specific"). Verdict meter climbs to **SURVIVOR — candidate**.

4. **The Gladstone close — one card.** The surviving candidate + its **mechanism hypothesis** (stated as hypothesis, never fact) + **the single falsifiable next experiment**: "replicate in ADNI/EPAD; anchor cluster membership to plasma p-tau217 (AD-specificity) vs NfL (generic neurodegeneration); confirmatory n and cohort named." That card *is* the "advance the field" moment.

**The honest claim we make — verbatim, no overclaim:**
> "This is **not a new biomarker.** It is a **stability-vetted, scanner-invariant candidate imaging phenotype**, discovered unsupervised from frozen foundation-model embeddings, that survived a five-test falsification gauntlet and anchors to an independent severity axis. External replication and a plasma anchor are the stated next experiment — here is the exact cohort to run it."

**The wow line:**
> **"Watch it kill our best-looking result because it was scanner noise — then trust the one it lets through."**

That single sentence carries all four judging axes: Claude on the critical path (Claude Use), a real produced lead not just a rejection (Impact), grad-level statistics shown live (Depth), and a demo whose hero is *real* with synthetic demoted to a labeled positive control (Demo). It directly answers the judge audit's credibility cliff.

---

## 3. Ranked demo concepts (honest-novelty first)

### Concept A — "Confound-robust replicated subtype anchored to severity" *(PRIMARY — build this)*
- **Scientific claim:** An unsupervised phenotype in frozen Neuro-JEPA space that is stability-vetted (bootstrap-Jaccard ≥0.6), scanner-invariant (passes leakage gauntlet on 6-site OpenBHB), age/sex-adjusted, and anchored to CDR/diagnosis. A *candidate*, not a biomarker.
- **Data it needs:** The 61 real OASIS-1 embeddings you already have (36 CN / 17 MCI / 8 AD) — runnable this afternoon. Stretch: embed the ~174 additional CDR-labeled OASIS-1 subjects (open data, Colab GPU) to n≈235. OpenBHB (96 healthy, 6-site, 2 field strengths) as the scanner-leakage exercise set.
- **Referee beats:** bootstrap-Jaccard stability → age/sex confound → scanner-leakage double-dissociation → brain-age residual → biomarker/severity anchor (Fisher-z CI lower bound). Prosecution/Defense + scored rubric verdict.
- **Overclaim risk & mitigation:** Risk = "we found a new AD subtype." Mitigation = the Prevot/Oxtoby reproducibility literature (5,444 subjects; Subcortical subtype survived only 3/8 models) is our own citation that single-cohort subtypes routinely fail — so we say "**internally stability-vetted; external replication is the stated next experiment**," never "validated" or "reproducible."

### Concept B — "Structure beyond accelerated aging" (brain-age residual)
- **Scientific claim:** The SSL embedding predicts a severity/conversion signal *after regressing out chronological age, sex, AND brain-predicted age* — i.e. disease structure beyond a Brain Age Gap. A well-scoped, respected claim (BAG correlates amyloid r=0.43, tau r=0.58; the novelty is the residual).
- **Data:** Same 61 embeddings; brain-age model on OpenBHB healthy controls.
- **Referee beats:** brain-age control (regress out predicted brain age, not the gap) → age/sex → replication.
- **Overclaim risk:** low; the claim is inherently modest and self-limiting. Weaker *wow* than A because it lacks the live "caught the artifact" theatre. Use as a **supporting panel inside A**, not a standalone demo.

### Concept C — "Confound-robustness as the result itself" (the positive control)
- **Scientific claim:** A naive probe learns scanner (high scanner-AUC); the referee catches and rejects it. The *methodology* is the finding.
- **Data:** 6-site OpenBHB; the synthetic `tau_hot` planted cluster (ARI=1.0) demoted to an **explicitly labeled calibration phantom**.
- **Referee beats:** scanner-leakage gauntlet as hero.
- **Overclaim risk:** near-zero, but this is *defensive* — it only says NO. **This is Act I of Concept A, not a standalone.** The judge audit is explicit: a tool that only rejects caps its own impact ceiling. Never ship C alone.

**Directive: ship A, with C as its Act-I control and B as a supporting panel. One demo, three moves.**

---

## 4. The researcher-hypothesis entry point

**The mechanism: a couple sentences → structured `HypothesisSpec` → parameterizes the actual run → pre-registers its own falsification.** This is the Databricks-Genie pattern (plain-language in, shown-reasoning + provenance out) fused with POPPER's pre-registered falsification.

**Flow:**
1. **Parse (Claude, on critical path).** Free-text hypothesis → structured `HypothesisSpec`:
   - `target`: unsupervised phenotype vs. specific region/pattern
   - `contrast/label`: CDR, diagnosis, conversion (via the neuro semantic layer — synonym map: "severity"→CDR, "decline"→conversion)
   - `confounds_to_prioritize`: age, sex, scanner, ICV — from a known-confound registry, with directionality priors (hippocampal↓ in AD)
   - `falsification_criterion`: **pre-registered, machine-checkable** — the Jaccard/scanner-AUC/anchor-CI thresholds that would kill it, committed *before* the run.
2. **Plan to memory.** Orchestrator writes the investigation plan to a notes file (provenance survives context limits), scales effort (quick probe vs. full Detective sweep).
3. **Parameterize the real run.** The `HypothesisSpec` *drives* which label column the anchor test uses, which confounds the gauntlet prioritizes, and the accept/reject thresholds — it is load-bearing, not decorative. This is the current build's single biggest gap: today `claim_parser.py` is a target-column mapper that doesn't parameterize the run. Close it.
4. **Show reasoning + provenance (trust UI).** Every verdict sentence links to its numeric evidence (Jaccard, scanner-AUC, Fisher-z CI); deterministic gauntlet tests carry a "Trusted" badge (Genie pattern); the pre-registered criterion is shown next to the outcome so the researcher sees the tool held itself to its own falsification.

**Harness principle (from the Anthropic/Databricks synthesis):** Neuro-JEPA is a **frozen tool behind the harness** — the substrate, never the reasoner (exactly as Databricks wires FMs behind Model Serving and Claude Science wires BioNeMo behind skills). Domain knowledge lives in **editable, versioned config** (semantic.yaml synonyms, confound registry, referee_rubric.md, the 5 gauntlet tests as deterministic "Trusted" UDFs), not weights. **Deterministic where the neuroscience formula must not vary; ML for the SSL discovery; Claude only for orchestration, adversarial refereeing, and the pre-registered plan.** The moat is the harness + referee, not the embeddings — everyone can cluster embeddings; no one ships the guardrail the field's own batch-effects literature proves is missing.

---

## 5. Hard red lines (flag these in the demo ourselves)

- ❌ "New AD biomarker" → **candidate phenotype** (biomarker needs Context-of-Use + analytical/clinical validation; impossible in 5 days).
- ❌ "Detects preclinical/asymptomatic AD" → **hypothesis-generating in the symptomatic/MCI range** (preclinical is where subtypes are least reproducible; 61 single-site subjects can't support it).
- ❌ "Better than plasma p-tau217 / replaces PET" → wrong axis entirely; imaging's value is topography/N-axis, not amyloid detection.
- ❌ "Generalizes across scanners" → only "confound-robust to the extent the 6-site OpenBHB test exercises it"; the labeled cohort is single-site — say so.
- ❌ "Validated / reproducible" → "internally stability-vetted; external replication is the stated next experiment."
- ❌ Any causal/mechanism claim as fact → always "mechanism *hypothesis*."
- ⚠️ Small-n (n=8 AD): report effect sizes + bootstrap CIs, no p-value theater; let the referee visibly *reject* underpowered findings.

---

## 6. Marching orders for the other directors

**Biostatistics director:** Point the existing gauntlet at the **real 61-subject OASIS-1 disease-bearing embeddings today** (currently the referee only runs on synthetic + healthy OpenBHB — this is the credibility cliff). Own: bootstrap-Jaccard on real clusters, the scanner-leakage double-dissociation as the live Act-I kill, the severity/CDR anchor with Fisher-z CI, brain-age residual. If the anchor is weak on n=61, that is *honest* and the referee should show it — a visible rejection is a feature. Stretch: embed the ~174 extra CDR-labeled OASIS-1 subjects on Colab to n≈235; do not let this block the honest run you can do this afternoon.

**Engineering director:** (1) Upgrade `claim_parser.py` into the `HypothesisSpec` generator that **parameterizes the run end-to-end and pre-registers the falsification criterion** — Claude on the critical path. (2) Re-choreograph the demo so **real is the hero**, synthetic `tau_hot` is a labeled positive-control phantom, and the closing card is "best real surviving candidate + one falsifiable next experiment + named confirmatory cohort." (3) Keep Neuro-JEPA a frozen tool behind the harness; keep the gauntlet tests deterministic and badge them "Trusted."

**The whole ballgame:** convert the weakest axis (real discovery / Gladstone fit) into the demo's hero, using rigor we already have, on real embeddings we already hold.