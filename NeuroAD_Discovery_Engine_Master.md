# NeuroAD Discovery Engine — Master Brief, Action Plan & Data Plan

**Date:** July 8, 2026
**Event:** Built with Claude — Life Sciences (Anthropic × Gladstone Institutes)
**Status:** Canonical doc. Led by the product thesis, then the action plan; folds in the full data-acquisition plan, competitive landscape, and sourced sponsor signals. Confirmed NeuroJEPA facts baked in (structural encoder; CC BY-NC-ND weights; native attentive-probing + feature-extraction support; brain-age as a native strength). **Substrate stance: real data first, synthetic as harness + insurance.**

> **Find the signal. Kill the artifact. Run the experiment.**

---

## Identity

**NeuroAD Discovery Engine** — a self-supervised AD neuroimaging discovery engine with a built-in scientific referee.

**Subtitle options:**

1. Self-supervised signal discovery for Alzheimer's neuroimaging research.
2. From MRI embeddings to validated Alzheimer's hypotheses.
3. A discovery engine that turns neuroimaging data into biomarker-backed experiments.
4. Finding latent Alzheimer's signals, filtering artifacts, and accelerating the next experiment.

**One-line description:**

> NeuroAD Discovery Engine helps Alzheimer's researchers find latent disease signals in neuroimaging data, filter out scanner noise and aging artifacts, anchor surviving patterns to biomarkers, and generate the next experiment to run.

**Imaging finds it. Proteins confirm it. The system tells you what to do next.**

---

## Core thesis

Alzheimer's researchers do not just need another dashboard or generic AI copilot. They need a faster way to move from messy neuroimaging data to trustworthy, testable biological hypotheses.

**NeuroAD Discovery Engine accelerates AD research by using self-supervised MRI embeddings to surface hidden disease signals, then stress-testing those signals against confounds before promoting only the survivors into biomarker-backed experiment cards.**

The core loop is:

```text
neuroimaging data → self-supervised embeddings → latent signal discovery → confound/referee gauntlet → biomarker anchoring → next experiment
```

In plain English:

> The system finds candidate Alzheimer's signals researchers may not have pre-labeled, asks whether those signals are real biology or artifacts, and turns the strongest survivors into concrete experiments.

### Why this accelerates Alzheimer's R&D

Most AD imaging work is slowed down by three hard problems:

1. **Signal discovery is difficult.** Important disease patterns may live in high-dimensional MRI structure, not in a single hand-crafted feature like hippocampal volume.
2. **False signals are common.** A model can accidentally learn scanner/site differences, generic aging, atrophy, or cohort artifacts instead of AD biology.
3. **Translation is slow.** Even when a signal looks promising, researchers still need to connect it to biomarkers, mechanism, and the next validation experiment.

NeuroAD Discovery Engine compresses that workflow. It helps researchers go from raw or embedded MRI datasets to a prioritized set of candidate AD phenotypes, each with a robustness verdict, biomarker evidence, and a proposed experiment.

### Positioning

**Not just:** an Alzheimer's imaging QC tool.
**Not just:** a generic research assistant.
**Not just:** a model that predicts diagnosis or conversion.

**Better framing:**

> A self-supervised AD neuroimaging discovery engine with a built-in scientific referee.

The discovery engine finds candidate signals. The referee decides which signals are robust enough to trust. The biology bridge explains what to test next.

### The Detective vs. the Referee (the pitch metaphor)

The product is the fusion of two ideas that are really two halves of one product:

- **The Detective** (phenotype discovery) — goes looking for hidden patient subgroups in imaging data and proposes what disease biology they might reflect. Ambitious, impressive, but easy to overclaim.
- **The Referee** (claim-level falsification) — takes a finding and tries as hard as it can to break it through a gauntlet of artifact tests. Rigorous, trustworthy, but on its own never tells you anything new about the disease.

Run the Detective's bold guesses *through* the Referee's reality check first, and you only ever speak biology about signals that already survived. **The fact-checker earns the detective the right to speak.**

```text
Detective finds a candidate signal
        ↓
Referee tries to kill it (site/scanner, demographics, brain-age, replication)
        ↓
Only survivors are promoted
        ↓
Bridge explains the likely biology + names the next experiment
```

---

## How a researcher uses it (the happy path)

The user is a computational/translational AD researcher with imaging + partial metadata already in hand and one question she can't answer alone: *"Is this signal worth a quarter of my time, or is it scanner noise, aging, or atrophy in disguise?"* Her flow:

1. **Point it at the data** — subject table + embeddings (or run the frozen encoder to produce them). Out comes a **cohort card**: n, sites, label coverage, biomarker missingness. She sees the shape and the gaps up front.
2. **State a hunch** — plain language ("MRI embeddings predict conversion"), turned into a structured, testable claim she confirms. *(Decision 1 of 2.)*
3. **Run** — the probe trains, then the referee gauntlet fires. She watches a checklist tick (age/sex → site-leakage → brain-age → biomarker anchor → replication), not code.
4. **Read the verdict** — a card, not a dashboard: *"partially robust — survives age/sex and replication, but the same embeddings predict scanner nearly as well as the outcome, weakens ~30% after brain-age control, and does correlate with p-tau217 (n=168)."* This is the emotional core: it names the exact assumption under which her finding stops being trustworthy. *(Decision 2 of 2: does she trust it enough to continue?)*
5. **Biology + next experiment** (survivors only) — hedged, biomarker-routed mechanism + one falsifiable experiment + the condition that would kill it.
6. **Export** — card + methods paragraph + code + evidence ledger. Reproducible, hand-off-able, grant-ready.

**The design tell:** she makes exactly two decisions — *which claim*, and *whether to trust the verdict*. The machine does the adversarial grind between. Her scarce judgment goes to the science, not to remembering last Tuesday's preprocessing.

### Two usage modes

**Mode A — Claim-driven.** The researcher states a hypothesis in plain language ("MRI embeddings predict which MCI patients convert to Alzheimer's disease"). The system turns that into a structured claim:

```text
target = conversion
population = MCI subjects
features = NeuroJEPA structural embeddings
controls = age, sex, site, scanner, brain-age
biomarkers = p-tau217, GFAP, NfL if available
```

**Mode B — Discovery / unsupervised.** The researcher does not need a predefined hypothesis. The system searches for latent AD-relevant signals in the embeddings:

- clusters enriched for diagnosis/conversion
- embedding directions associated with biomarkers
- subgroups with unusual brain-age residuals
- phenotypes that replicate across cohorts or sites

This is where the self-supervised angle matters most: the model can surface structure learned from MRI data that was not manually labeled in advance.

### The four-card output taxonomy

The product outputs a small number of high-value cards, not a giant dashboard.

1. **Cohort card** — is the dataset usable? n, diagnosis/conversion label coverage, site/scanner distribution, age/sex distribution, biomarker completeness, missingness warnings.
2. **Discovery card** — candidate signals found from embeddings: an embedding direction associated with AD vs control, a cluster enriched for converters, a latent phenotype associated with p-tau217 or GFAP, or a subgroup that looks high-risk but may be scanner-driven.
3. **Referee verdict card** — does the signal survive the gauntlet? age/sex adjustment, site/scanner leakage, brain-age/atrophy control, held-out replication split, biomarker anchor (if available). Verdict bands: `fragile → partially robust → robust enough for follow-up → strong candidate`.
4. **Experiment card** — generated only for signals that survive enough of the gauntlet: biological interpretation, biomarker evidence, next experiment, falsification condition, caveats and missingness.

---

# THE ACTION PLAN

Two ideas that sound opposed but aren't: **contract-first is the build method; real-data-first is the substrate goal.** You build against a fixed embedding-table contract so nothing blocks, and you feed that contract with real data wherever access + preprocessing time allow, keeping synthetic as the harness and the insurance policy. Steps 1–6 require no external access.

## Step 1 — Contract + requests (the keystone; do this first)

- **Define the cached-embedding table schema — the contract every downstream piece reads:**
  `subject_id, embedding[d], dx, conversion, age, sex, site, scanner, amyloid, p_tau217, gfap, nfl, apoe`
- **Fire all external requests in parallel** (latency you don't control): NeuroJEPA weights (granted), ADNI, OASIS-3, NACC. See the Data Plan for the full tiering.
- **Split ownership** — one owns the Referee, one owns the Bridge. Both code against the contract.
- Scope is locked: structural encoder → **structural conversion anchor** (on open-only data, pivot the claim to AD-vs-CN diagnosis; see Data Plan).

Why first: the moment the schema exists, downstream is data-source-independent. Swapping real embeddings in later changes one thing — the file.

## Step 2 — Stand up the substrate (real-first, synthetic as harness)

- **Synthetic harness (immediate):** generate a schema-matched table with an injected conversion signal, a deliberately injected site/scanner confound (so the star leakage test has something real to catch), and realistic biomarker missingness. This is the validation harness and the guaranteed fallback.
- **Real, in parallel (preferred where feasible):** pull the open datasets that need no application — **IXI** (encoder smoke-test), **MIRIAD** / **OASIS-1/2** (real labeled structural MRI), **EPAD** (real multi-site, for the leakage beat). Budget for the preprocessing tax (see Data Plan).

## Step 3 — Reusable head + cohort card

Build the one small probe (linear probe; small MLP only if linear leaves signal on the table). The cohort card falls out of the table for free.

## Step 4 — Referee gauntlet

Point the same head at three targets by swapping the label column, then wrap with the rest:
- conversion (signal) · **site/scanner (star leakage test)** · biomarker (molecular anchor)
- plus age/sex adjustment, brain-age control, replication split → verdict + robustness score.

## Step 5 — Clustering layer (the Detective)

Unsupervised phenotype discovery (k-means / HDBSCAN on embeddings or a UMAP). The discovery entry point; everything above can run per-cluster.

## Step 6 — Biology bridge (thin) + export

Survivors only: one biomarker-routed mechanism hypothesis → one falsifiable experiment → falsification criteria → exported claim/phenotype card.

## Step 7 — Swap real for synthetic wherever it clears

- Weights → run the encoder (IXI first to lock embedding dim/scale), then real cohorts, into the same schema; rerun.
- Open data (MIRIAD/OASIS/EPAD) → real diagnosis + site-leakage + brain-age beats.
- Gated data (ADNI/OASIS-3) if it clears → upgrades to real conversion + real plasma-biomarker anchor.

## Step 8 — Demo polish

Lock the two-example script (one kill, one survivor), the wow line, the exported card. Rehearse.

---

# DATA PLAN

## The one hard constraint before any real data helps: the preprocessing tax

NeuroJEPA is a frozen encoder trained on T1w/T2w/FLAIR that went through a specific pipeline (MNI registration, skull-strip, bias-field correction). Real embeddings require running raw scans through a matching pipeline first. **This preprocessing — not access — is the real cost of "real data" in a one-week build.** Soften it two ways: prefer datasets shipping FreeSurfer features (volumes, cortical thickness — the weight-free feeder that skips most of the tax), and keep synthetic as the fallback. (Aside: preprocessing/skull-strip choices are themselves a documented confound source in AD MRI — which is on-theme for the referee.)

## Tier 1 — No application, download today (rely on this week)

| Dataset | Imaging | Labels | Biomarkers | Notes |
|---|---|---|---|---|
| **IXI** | T1, T2, PD (~600) | none (healthy) | none | Encoder smoke-test: proves NeuroJEPA loads + emits real vectors; locks embedding dim/scale. No analysis value. |
| **MIRIAD** | T1, longitudinal (AD + CN) | AD vs CN, MMSE | none | Best open labeled structural set. **Single scanner → no site-leakage test.** Great for diagnosis + brain-age beats. |
| **OASIS-1** | T1 cross-sectional (416; 100 AD) | AD vs CN (CDR) | none | Open tier (lighter than OASIS-3). Diagnosis probe. |
| **OASIS-2** | T1 longitudinal (150) | dementia / CDR over time | none | Open tier. Closest open thing to a progression label. |
| **EPAD** | 3D T1w, FLAIR (~1356) | preclinical/prodromal | imaging-derived | Open-access, **multi-site → enables the leakage test.** Few frank converters. |
| **OpenBHB** | T1 + anatomical measures | healthy (brain-age) | none | Aggregated healthy cohort; good for the brain-age control model. |

**Skip:** OpenNeuro's dementia sets are mostly EEG/MEG (ds004504, BioFIND), not structural MRI.

## Tier 2 — Applied for, timing uncertain (best data, may miss the deadline)

| Dataset | Imaging | Labels | Biomarkers | Access |
|---|---|---|---|---|
| **ADNI** | T1/T2/FLAIR, multi-site | MCI→AD conversion | **plasma p-tau217, GFAP, NfL**, amyloid, APOE | committee, ~1–2 wk |
| **OASIS-3** | multimodal, multi-site | dementia/CDR, converters | amyloid PET, APOE, some CSF | restricted DUA |
| **NACC** | SCAN MRI summaries | UDS diagnosis/CDR | CSF (no plasma panel) | data request |

## What analysis runs on what

- **Conversion probe** (anchor claim) → conversion labels → **ADNI / OASIS-3** (gated). On open data, **pivot to AD-vs-CN diagnosis** (MIRIAD / OASIS-1) — still demonstrates the method.
- **Site/scanner leakage** (star beat) → **multi-site** → **EPAD** (open) or ADNI/OASIS-3. NOT MIRIAD (single scanner).
- **Brain-age control** → any aging cohort → **MIRIAD / OASIS / OpenBHB / IXI**. Easy, open.
- **Biomarker anchor** (p-tau217 / GFAP) → **ADNI only.** No open set has the plasma panel; runs on **synthetic** otherwise (or CSF amyloid/tau on OASIS-3 if it clears).

**Honest implication:** open data alone gives a genuinely real demo of *diagnosis + site-leakage + brain-age* (EPAD for leakage, MIRIAD for diagnosis/brain-age). The *conversion* framing and the *plasma-biomarker anchor* essentially require gated cohorts — those stay synthetic unless ADNI/OASIS-3 clear.

## Order of operations for the week

1. **Encoder online** — weights → run on IXI; lock real embedding dim + scale so synthetic matches.
2. **Real labeled imaging** — MIRIAD (and/or OASIS-1/2); preprocess or use shipped FreeSurfer features.
3. **Multi-site for the star beat** — EPAD, so site-leakage has genuine cross-scanner structure.
4. **Synthetic as harness + insurance** — keep the full arc (incl. biomarker anchor) runnable end-to-end.
5. **Gated swap-in** — if ADNI/OASIS-3 clear, upgrade to real conversion + real plasma anchor.

## Synthetic's role (demoted, not gone)

- The **only** guaranteed substrate for the p-tau217/GFAP biomarker-anchor beat.
- The **validation harness**: planted ground-truth confounds prove the referee catches what it should before trusting messy real data.
- The **insurance policy**: a complete, runnable demo needing zero external access.

## Links

**Open (today):** IXI — https://brain-development.org/ixi-dataset/ · OASIS (1–4) — https://sites.wustl.edu/oasisbrains/ · MIRIAD — UCL DRC release (paper DOI 10.1016/j.neuroimage.2012.12.044) · EPAD — open MRI release (paper DOI 10.1016/j.nicl.2022.103106) · OpenBHB — search "OpenBHB"
**Gated (applied for):** ADNI — https://ida.loni.usc.edu/collaboration/access/appApply.jsp?project=ADNI · OASIS-3 — https://sites.wustl.edu/oasisbrains/ · NACC — https://naccdata.org
**Encoder:** NeuroJEPA — https://github.com/NYUMedML/Neuro-JEPA (weights: HuggingFace gated; granted)

---

# SCIENTIFIC MOTIVATION — the AD research challenges we map against

Alzheimer's research is hard because the disease is biologically and clinically heterogeneous. A recent review describes AD etiology as complex and diverse, involving aging, genetics, environment, amyloid, tau, inflammation, oxidative stress, glutamate excitotoxicity, microbiota-gut-brain factors, autophagy, and other mechanisms, and notes that unraveling the interplay among these pathological aspects and validating primary disease initiators remains difficult. [[Zhang et al., 2024, Nature / Signal Transduction and Targeted Therapy](https://www.nature.com/articles/s41392-024-01911-3)]

| Research challenge | Why it matters | Tool opportunity |
|---|---|---|
| Disease heterogeneity | AD varies by stage, phenotype, genotype, sex, comorbidities, pathology burden, and progression rate. | Subtype-aware evidence scoring rather than global target ranking. |
| Preclinical / early-stage detection | Pathology can begin years or decades before clinical symptoms; intervention timing matters. | Longitudinal imaging + biomarker + cognition modeling. |
| Correlation vs causation | Omics and imaging often surface associations, not causal mechanisms. | Hypothesis validation and experiment-selection logic. |
| Multimodal fragmentation | Imaging, omics, cognition, genetics, clinical records, and wet-lab data live in different systems. | Cross-modal evidence graphs with reproducible manifests. |
| Scanner/site/protocol confounding | Neuroimaging models can learn acquisition artifacts rather than disease signals. | Confound audits and disease-preserving harmonization checks. |
| Translation failure | Mouse, iPSC, organoid, human tissue, imaging, and clinical signals often conflict. | Contradiction mining across species, cell type, assay, and disease stage. |
| Trust and reproducibility | Researchers need provenance, code, assumptions, and failure modes. | Claude-generated auditable artifacts, notebooks, model cards, and reviewer reports. |

---

# REFERENCE

## Architecture reality (constrains everything)

**NeuroJEPA is a frozen structural encoder.** A V-JEPA-2-style 3D model using a JEPA-style latent predictive objective and a Mixture-of-Experts architecture, pretrained on roughly **1.55M** curated T1w/T2w/FLAIR scans — all structural. "Multimodal" = multiple structural sequences fused, not structural+functional. The paper reports evaluations across diagnosis, prognosis, time-to-event, and age prediction, including cross-cohort AD transfer (e.g. NACC-to-ADNI). We do **not** fine-tune. [[Neuro-JEPA arXiv](https://arxiv.org/html/2606.14957v1)]

**License:** code is MIT; **weights are CC BY-NC-ND 4.0** (non-commercial, **NoDerivatives**). The ND clause is why fine-tuning-and-sharing is off the table (a fine-tuned checkpoint is a derivative). Our **frozen-encoder + own-head** approach is clean: weights used as-is (no derivative), non-commercial research. *(Not legal advice — read the license before any public repo.)*

**The repo ships our approach natively:** attentive probing (frozen backbone + trainable head) + a feature-extraction path + `load_backbone_from_hf`. "Run once, cache the vectors, probe the table" is first-class.

**Build against the embedding interface, not NeuroJEPA specifically.** Downstream expects one fixed-dim vector per subject. Three interchangeable feeders: real NeuroJEPA embeddings · a substitute open encoder · weight-free structural features (hippocampal volume, cortical thickness, WMH burden).

## How NeuroJEPA is involved (rationale)

NeuroJEPA is the **representation engine**. It acts as a frozen self-supervised structural MRI encoder that converts brain scans into dense embedding vectors. Those embeddings become the substrate for discovery, probing, clustering, and robustness testing. The MVP does **not** need to fine-tune NeuroJEPA; it uses it as a frozen backbone and builds lightweight analysis layers on top.

**Why frozen embeddings beat hand-crafted features.** A frozen self-supervised encoder may capture structural brain patterns that are not obvious from hand-engineered features alone. Instead of starting with only predefined measurements like hippocampal volume or cortical thickness, the system searches across richer latent representations learned from large-scale MRI data. That lets researchers ask: Are there latent MRI phenotypes associated with AD diagnosis? Are there embedding clusters enriched for MCI converters? Do any embedding-derived phenotypes correlate with p-tau217, GFAP, NfL, or amyloid? Does the signal survive scanner/site and brain-age controls?

```text
T1w/T2w/FLAIR MRI scans → Frozen NeuroJEPA encoder → Subject-level embedding vector
    → Discovery + probes + clustering → Referee gauntlet → Biomarker/mechanism bridge → Experiment card
```

**Why this is not just a NeuroJEPA demo.** NeuroJEPA provides the self-supervised representation, but the product value comes from the workflow around it. The novelty is not simply using a foundation model on MRI; it is using self-supervised MRI embeddings as a *discovery substrate*, then wrapping them in a *falsification and biomarker-translation loop*.

```text
NeuroJEPA finds rich structure.
NeuroAD Discovery Engine decides which structure matters for Alzheimer's research.
```

*(Functional counterpart: **Brain-JEPA** is a JEPA-style foundation model for fMRI/brain dynamics using Brain Gradient Positioning and Spatiotemporal Masking [[Brain-JEPA arXiv](https://arxiv.org/abs/2409.19407)]. It is the more direct fit if a learned functional backbone is used — see the functional-track roadmap below. NeuroJEPA is structural; Brain-JEPA is functional.)*

## Components — four researcher-facing tools from one small head

The layer you build is a small head that maps a frozen NeuroJEPA embedding vector to a number — a linear probe (logistic regression) to start, optionally a tiny 2-layer MLP for nonlinear capacity. That's the entire architecture. The power isn't the layer; it's that **pointing the same head at different label columns gives you three of the four tools** — you write one small thing and get most of the product. Each tool answers a concrete decision the researcher would otherwise resolve by hand over weeks:

1. **Conversion probe** (target = conversion, or AD-vs-CN on open data) — *"Is this patient/subgroup on a trajectory toward AD?"* Prioritization and trial-enrichment: who to watch, recruit, study. Also the proof the frozen embeddings carry real signal at all.
2. **Site/scanner probe** (same head, label = scanner) — *"Is my disease signal real, or acquisition artifact?"* If this predicts scanner as well as #1 predicts the outcome, the "disease signal" is partly the machine. The referee's **star test** — and it costs almost nothing: same code, different label column.
3. **Biomarker regression** (same head, target = p-tau217 / GFAP) — *"Is this imaging pattern tied to molecular pathology, or just brain shape?"* If the embedding predicts p-tau217, that's protein-level evidence the signal is real biology — the anchor a scanner can't fake — and *which* biomarker it tracks routes the mechanism (tau-driven vs. glial/inflammatory).
4. **Clustering layer** (unsupervised — k-means / HDBSCAN on the vectors or a UMAP) — the genuinely different one. No label, no training target. *"What patient subgroups exist that nobody labeled?"* The Detective / discovery step. Everything above then runs **per cluster** (does this subgroup convert faster? is it a scanner artifact? does it carry a tau signature?).

Optional: a **survival/Cox head** — *"how fast does this subgroup convert?"* (timing, not just yes/no). Native NeuroJEPA task (time-to-event). Nice-to-have, not core.

**The stage line:** *"We built one probe and asked it three different questions — what disease, which scanner, which protein — and the disagreement between those answers is the whole product."* The researcher's judgment goes to interpreting those disagreements; that's the actual science.

Start linear everywhere. A linear probe that predicts the outcome is a *stronger* claim than an MLP that does — it proves the signal is cleanly present in the embedding rather than something the head had to manufacture, and it keeps "is the signal the encoder's or ours?" unambiguous. Reach for the MLP only if linear leaves obvious signal on the table. Don't touch adapters/LoRA/unfreezing — they re-enter the license and you don't need them.

**Build order (priority):** linear probe on conversion → same probe on site → clustering → biomarker regression → structural/brain-age-controlled re-fit.

## Referee gauntlet — MVP (structural track)

| Test | Question | How |
|---|---|---|
| Age / sex adjustment | Survives demographics? | re-fit with covariates |
| **Site / scanner leakage** ⭐ | Disease signal, or which machine? | same head, label = scanner |
| Brain-age control | More than generic aging/atrophy? | control for embedding-derived brain age (native NeuroJEPA strength, R²≈0.89) |
| **Biomarker anchor** | Backed by molecular pathology? | correlate phenotype with p-tau217 / GFAP |
| Replication split | Reproduces held-out? | held-out site / cohort |

Verdict bands: fragile → partially robust → robust enough for follow-up → strong candidate. Only *partially robust* and up reach biology.

## Referee gauntlet — functional track (Phase 1+ roadmap)

When a learned functional backbone (Brain-JEPA) or standard connectome features enter, NeuroJEPA flips role to being the **structural control**, and the gauntlet gains functional-specific robustness tests:

| Test | Question it answers |
|---|---|
| Motion matching | Does the effect survive when groups are matched on head motion (framewise displacement)? |
| Denoising / GSR sensitivity | Does it survive alternative denoising and global-signal-regression choices? |
| Parcellation sensitivity | Does it survive using a different brain atlas? |
| Structural counterfactual | Does the functional signal add anything beyond the brain physically shrinking? |

**The structural counterfactual (where NeuroJEPA fits on the functional track).** Many apparent functional Alzheimer's signals are partly **anatomy wearing an fMRI costume** — atrophy, white-matter disease, or vascular burden masquerading as circuit dysfunction. Using frozen NeuroJEPA structural embeddings (or weight-free proxies) as a control lets the Referee ask: *"Does this functional signal still matter after we control for structural neurodegeneration?"*

**Functional anchor claim:** *"Posterior cingulate–hippocampal connectivity is reduced in MCI vs. controls."* (On-brand for medial-temporal / memory-circuit work.) A concrete functional vertical slice: one processed rs-fMRI or ADNI-like derivative table → naive significant effect → five stress tests (age/sex, motion matching, site split, denoising sensitivity, structural control) → verdict → survivor's mechanism + next experiment + falsification → exported card. [[ADNI rs-fMRI pipeline arXiv, 2026](https://arxiv.org/abs/2602.03278)]

## Biomarkers — three roles

1. **Stratifier** — "survives within amyloid-positive only?" Tightens the gauntlet.
2. **Ground-truth anchor** — correlation with plasma p-tau217 / GFAP = molecular evidence of real pathology.
3. **Bridge to mechanism** — signature routes to biology + experiment:
   - amyloid + p-tau → amyloid-cascade → neuron/organoid validation
   - GFAP / sTREM2, weak amyloid → neuroinflammatory/glial → iPSC microglia/astrocyte
   - NfL + WMH, amyloid absent → vascular/axonal → BBB/endothelial readout

Menu (AT(N)+X): A = Aβ42/40, amyloid PET; T = p-tau181/217/231, tau PET; N = t-tau, plasma NfL, atrophy; X = GFAP, sTREM2, YKL-40; + APOE as stratifier.

**Coverage caveat (in-product):** biomarker data is partial. Every biomarker-backed claim carries a completeness label ("holds on complete subset, n=X").

## Biology bridge (payoff — narrow)

Survivors only: one biomarker-routed mechanism hypothesis → one experiment → falsification. Not a dashboard; a single earned claim, each statement paired with its artifact + protein evidence. Because the biology only speaks about gated signals, every mechanistic statement can be paired with the artifact evidence that backs it — the difference between narrating a plot and standing behind a conclusion.

**Discipline: Referee deep, Bridge narrow.** Build the Referee for real. For the Bridge, ship one survivor phenotype → one mechanism hypothesis → one falsifiable experiment. Do **not** build a full six-artifact suite — that is how the biology step becomes shallow.

---

# COMPETITIVE LANDSCAPE

The broad product category already exists: Alzheimer's-specific agentic co-scientists, AD data workbenches, target-evidence portals, neuroimaging pipelines, and general Claude Science workflows all cover parts of the problem. The differentiated opportunity is narrower — **the missing interpretability and validation layer between neuroimaging foundation models and Alzheimer's biological discovery.**

| Category | Existing examples | What they already cover | Why not build another one |
|---|---|---|---|
| AD co-scientist | **Biomni-AD** | General-purpose AI co-scientist with unified AD data integration; an Alzheimer's Insights AI Prize winner. [[ADDI: Biomni-AD](https://www.alzheimersdata.org/adpd2026-knowledge-hub/teambiomni-ad)] | A generic "AD co-scientist" will look derivative. |
| Virtual wet lab | **Prima Mente / PARTHENON** | Multi-agent therapeutic hypothesis evaluation using virtual neurons, microglia, astrocytes. [[ADDI AI Explorations](https://www.alzheimersdata.org/accelerating-research/ai-explorations)] | Strong biology-first wedge; hard to beat directly in a short hackathon. |
| AD data infrastructure | **AD Workbench** | Secure cloud data sharing/analytics, multimodal tools, compute, workspaces, audit trails. [[AD Workbench](https://www.alzheimersdata.org/ad-workbench)] | Infrastructure exists; a new workspace is not novel. |
| AD target evidence | **Agora** | Human transcriptomic/proteomic/metabolomic evidence for AD genes; 20k+ genes, 900+ nominated targets. [[Agora](https://agora.adknowledgeportal.org/)] | Target evidence browsing is already covered. |
| General scientific workbench | **Claude Science** | Literature, code, compute, artifacts, reviewer agents, 60+ skills/connectors. [[Claude Science](https://www.anthropic.com/news/claude-science-ai-workbench)] | A generic Claude workflow does not differentiate. |
| Neuroimaging preprocessing | **fMRIPrep, FSL, AFNI, SPM, FreeSurfer, Nilearn** | Robust fMRI preprocessing, QC, connectivity, statistical modeling. [[fMRIPrep](https://fmriprep.org/)] | "Upload fMRI, get preprocessing/connectivity output" is not enough. |
| Neuroimaging foundation models | **Neuro-JEPA, Brain-JEPA, Brain-DiT, SLIM-Brain, BrainSymphony, BrainDINO** | Representation learning + downstream prediction/classification. | Predictive embeddings alone do not deliver biological interpretation or experimental action. |

**The gap:** the outputs of modern multimodal AI and neuroimaging models are still hard to translate into (1) disease-stage-specific AD biology, (2) human-relevant mechanisms, (3) subtype-aware hypotheses, (4) artifact-resistant conclusions, and (5) falsifiable next experiments.

## Competitive wedge matrix

| Competitor / adjacent tool | Their strength | Our wedge |
|---|---|---|
| Biomni-AD | Broad AD co-scientist and data integration | Specialized neuroimaging-to-mechanism interpretability module. |
| Prima Mente / PARTHENON | Virtual wet lab with neurological virtual cells | Feed it better imaging-derived subtype hypotheses. |
| AD Workbench | Data access, compute, shared analytics | A plugin/skill that could run inside the workbench. |
| Agora | AD gene/target evidence | Link imaging phenotypes to target/cell/pathway evidence. |
| Claude Science | General scientific workbench | A domain-specific, AD-neuroimaging workflow skill. |
| fMRIPrep / Nilearn | Preprocessing and analysis | Use their outputs; do not compete with them. |
| Neuro-JEPA / Brain-JEPA | Powerful representations | Interpret, audit, and biologically contextualize their embeddings. |

---

# SPONSOR SIGNALS (why this aligns with the judges)

**Claude / Anthropic side.** Anthropic launched **Claude Science** as an AI workbench for scientists on June 30, 2026 — explicitly designed to integrate fragmented scientific tools, produce auditable artifacts, connect to compute environments, support domains such as genomics/single-cell/proteomics/structural biology/cheminformatics, and include reviewer-agent behavior for citations and calculations. [[Claude Science](https://www.anthropic.com/news/claude-science-ai-workbench)] The event, "Built with Claude: Life Sciences," positions **Claude Science + Claude Code + Gladstone Institutes** and emphasizes working software that outlasts the event. [[Built with Claude: Life Sciences](https://cerebralvalley.ai/e/built-with-claude-life-sciences)] Anthropic's AI-for-science evaluation language emphasizes **scientific merit, potential impact, technical feasibility, and team credibility**. [[Anthropic AI for Science Program FAQ](https://support.claude.com/en/articles/11199177-anthropic-s-ai-for-science-program)]
**Implication:** the winning build needs to look like real scientific infrastructure — reproducible, auditable, useful to a named researcher, integrated into an actual workflow.

**Gladstone side.** Gladstone has been publicly emphasizing AI coupled to experimental biology — notably the "thinking microscope," an AI-powered system that can design and conduct experiments on diseased cells, learn from results, and potentially accelerate work in neurodegenerative diseases including Parkinson's, Alzheimer's, and ALS. [[Gladstone: AI Designing Its Own Experiments](https://gladstone.org/news/ai-designing-its-own-experiments)]
**Implication:** a project that only summarizes literature or produces predictions is weak. A project that moves a researcher toward a **better next experiment** is much stronger.

## Judge alignment

| Criterion | How it maps |
|---|---|
| Scientific merit | Attacks confounding, heterogeneity, artifact-vs-biology, trustworthiness. |
| Impact | Saves months chasing artifacts; prioritizes signals worth validating. |
| Gladstone fit | Terminates at a concrete experiment; biomarker→cell-type routing matches experimental-biology emphasis. |
| Anthropic / Claude Science | Auditable artifacts, reviewer-style critique, reproducible code, evidence ledger. |
| Originality | Foundation-model interpretability + falsification layer, not a generic co-scientist. |
| Feasibility | Narrow skill on cached embeddings; one reused head; repo-native probing. |

---

# EXAMPLE OUTPUT CARD

```yaml
claim_id: ad_master_001
claim_text: A structural MRI embedding phenotype predicts MCI-to-AD conversion.
substrate: frozen NeuroJEPA structural embeddings
head: linear probe
population: {group_a: MCI converters, group_b: MCI non-converters}
robustness:
  age_sex_adjustment: passed
  site_scanner_leakage: WEAKENED (embeddings predict scanner ~AUC 0.75)
  brain_age_control: weakened (effect size drops ~30%)
  biomarker_anchor: passed (correlates with plasma p-tau217, n=168)
  replication_split: passed
verdict: partially robust
promoted: true
biology_hypothesis: >
  Surviving component consistent with early medial-temporal disruption not
  fully explained by atrophy; p-tau217 enrichment points to tau-driven
  rather than primarily inflammatory biology.
next_experiment:
  - restrict to amyloid-positive, p-tau217-high converters matched on scanner
  - re-fit; test residual against a memory-encoding readout
  - replicate on an independent held-out cohort/site
falsification:
  - deprioritize if the effect vanishes once site and brain-age are jointly controlled
caveats:
  - p-tau217 missing in ~56% of cohort; anchor holds only on complete subset
  - brain-age control is a proxy in MVP
```

---

# DEMO CHOREOGRAPHY (what the audience sees)

1. Open on a claim/cluster that looks significant in the naive plot.
2. Run the referee live — survives some tests, weakens on others.
3. **The turn:** *"the same embeddings predict scanner almost as well as the outcome"* — the tool catches its own encoder cheating.
4. **The recovery:** *"but it correlates with p-tau217"* — molecular evidence pulls part of the signal back into 'real.'
5. Promote the survivor → biology + next experiment → export the card.

**Wow line:** *"Here is the precise assumption under which this Alzheimer's finding stops being true — and here's the protein evidence for the part that's real, plus the experiment that would confirm it."*

**Demo line:** *"NeuroAD Discovery Engine does not just predict Alzheimer's. It discovers candidate disease signals, tries to falsify them, and only promotes the survivors into experiments."*

**Show one KILL + one SURVIVOR.** The survivor is the satisfying arc (ends on biology + experiment); the kill is the more honest, more surprising moment — a five-minute *"don't chase this"* that saves months is arguably the highest-value output the tool produces. To generate the kill from the synthetic harness: turn `SITE_COUPLE` up and `DISEASE_LOAD` down (a finding that's almost all acquisition artifact and collapses under control).

**Build assets already exist:** `generate_synthetic_cohort.py` + `validate_cohort.py` emit the validated survivor arc — naive AUC ≈0.74, site-leakage ≈0.75 (predicts scanner as well as outcome), p-tau217 r ≈0.51, dropping to ≈0.68 after site removal (partially robust). Swap real embeddings into the same schema to upgrade.

---

# RISKS AND MITIGATIONS

| Risk | Mitigation |
|---|---|
| "Just automated QC / multiverse." | Lead with the falsification→biomarker→biology loop and the survivor gate. |
| Biology overclaims. | Gate + protein anchor; hedge wording; speak only about gated survivors. |
| Gated data slips (ADNI/OASIS-3). | Open data (MIRIAD/EPAD) covers diagnosis+leakage+brain-age; synthetic covers conversion+biomarker anchor. |
| Preprocessing tax eats the week. | Use shipped FreeSurfer features; synthetic as guaranteed fallback. |
| Structural encoder can't do fMRI claim. | Two-track framework; MVP is structural; fMRI is functional track. |
| Biomarker coverage partial. | Completeness labels + report n. |
| Too broad for timebox. | One reused head + narrow bridge — one survivor, one hypothesis, one experiment. |
| Model finds site/scanner artifact. | That's the point — catching false signals earns trust. |

---

## Final line

> NeuroAD Discovery Engine: a self-supervised discovery platform for Alzheimer's neuroimaging that finds latent embedding-derived signals, falsifies them through the referee gauntlet against scanner leakage, demographics, brain-age, and replication, anchors the survivors to a fluid biomarker, and — for what's left standing — names the likely biology and the next experiment to run.

**Find the signal. Kill the artifact. Run the experiment.**
