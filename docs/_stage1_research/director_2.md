Research complete. Synthesizing the competitive-intelligence scan below.

---

# Competitive Scan: SSL/Foundation-Model-Driven Neuroimaging Biomarker Discovery

**Bottom line up front:** No existing tool does end-to-end SSL-driven *imaging*-biomarker discovery with a falsification/referee layer AND a researcher-hypothesis entry point. The category splits into four silos that never touch: (1) foundation models that emit embeddings but stop there; (2) AD co-scientists that reason over *omics/literature/molecular* data, not imaging embeddings; (3) generic discovery harnesses (one of which — POPPER — is a genuine falsification engine, but on tabular data, not SSL imaging); (4) classical imaging-subtype tools (SuStaIn) that discover phenotypes but have no LLM referee and no hypothesis entry point. **NeuroAD Discovery Engine sits in the empty intersection of all four.**

---

## (a) Neuroimaging foundation models + their tooling

| Tool | What it does | Closes hyp→discovery→validation→next-exp loop? | Gap |
|---|---|---|---|
| **Neuro-JEPA** (NYU Langone, [HF: NYUMedML/Neuro-JEPA](https://huggingface.co/NYUMedML/Neuro-JEPA)) | V-JEPA2 extended to 3D brain MRI; pretrained on ~1.55M T1w/T2w/FLAIR; emits embeddings; benchmarked on 12 public + 3 clinical cohorts. This is your substrate. | No. It is a frozen feature extractor. Ships weights + benchmark numbers, not a discovery workflow. | No probe orchestration, no unsupervised discovery, no referee, no hypothesis entry. It's an *engine*, not a product. |
| **Brain-JEPA** ([arXiv 2409.19407](https://arxiv.org/abs/2409.19407)) | fMRI JEPA on UK Biobank (40k subjects); Brain Gradient Positioning + spatiotemporal masking; SOTA on demographic/trait/disease tasks. | No. Supervised fine-tuning benchmark model. | Same as above — representation only. Functional (fMRI), not structural. |
| **BrainLM** ([bioRxiv 2023.09.12.557460](https://www.biorxiv.org/content/10.1101/2023.09.12.557460v1.full), [HF: vandijklab/brainlm](https://huggingface.co/vandijklab/brainlm)) | fMRI foundation model, 77k recordings; can discover intrinsic functional networks unsupervised, predict future brain states. | Partial — mentions unsupervised network discovery, but no closed validation loop, no referee. | No falsification, no confound audit, no next-experiment. Discovery is a demo, not a governed workflow. |
| **BrainDINO** ([arXiv 2604.27277](https://arxiv.org/abs/2604.27277)) | Self-distilled structural-MRI FM, 6.6M slices; frozen reps generalize across tumor seg, neurodegeneration classification, brain-age. | No. Transfer-learning benchmark. | Shows "pathology-sensitive feature structure" but never converts it to an auditable biomarker candidate. |
| **Batch-effects-in-FM-embeddings** ([arXiv 2604.14441](https://arxiv.org/pdf/2604.14441)) | Critical finding, not a tool: BrainLM/SwiFT embeddings **encode substantial batch/site variability that often dominates diagnosis signal**. | N/A | This paper is your *justification for a scanner-leakage referee test* — it proves the field ships embeddings that leak site. No one ships the guardrail. |

**Verdict on (a):** These are pure representation engines. The entire value-add above "frozen embedding" is unclaimed. The batch-effects literature proves the field *knows* embeddings leak scanner/site but ships no automated audit — exactly your scanner-leakage gauntlet test.

---

## (b) AD-specific discovery / co-scientist tools

| Tool | What it does | Closes loop? | Gap |
|---|---|---|---|
| **Biomni-AD** ([Biomni: bioRxiv 2025.05.30.656746](https://www.biorxiv.org/content/10.1101/2025.05.30.656746v1); winner of [AD Data Initiative agentic prize](https://www.alzheimersdata.org/news/alzheimer-s-disease-data-initiative-doubles-1m-prize-competition-for-agentic-ai-solutions-to-accelerate-alzheimer-s-research)) | AD "co-scientist" grounded in expert-curated **omics, single-cell, biomarker, clinical** data; agents do literature search + data integration; separate agents check hallucinations and biological-rationale soundness. | Partially — has a self-critique/hallucination-check agent (its "referee"), and moves data→insight. | **No imaging modality. No SSL embeddings. No unsupervised phenotype discovery from images.** Its referee checks LLM factuality/rationale, NOT statistical confounds (age/sex/scanner leakage) on a learned representation. Different substrate entirely. |
| **Prima Mente — PARTHENON / Athena / Pleiades** ([GEN](https://www.genengnews.com/topics/artificial-intelligence/prima-mente-unlocks-early-stage-alzheimers-diagnostics-with-epigenome-model/)) | Virtual-cell "wet lab" + AI co-scientist Athena; Pleiades = 7B epigenetic FM (methylation/cfDNA) predicting AD from neuron/microglia cell-of-origin. | Closest to a loop in *epigenomics* — model→hypothesis→virtual experiment. | **Epigenome, not imaging.** No MRI, no SSL image embeddings, no imaging-confound referee. Adjacent domain, non-overlapping. |
| **AD Workbench** (AD Data Initiative, [alzheimersdata.org](https://www.alzheimersdata.org/)) | Data-sharing + analytics platform; the *distribution channel* where the above winners are hosted free. | No — it's infrastructure/hosting, not a discovery engine. | A data platform, not a method. Actually a *deployment target* for you, not a competitor. |
| **Agora** (Sage Bionetworks, [agora.adknowledgeportal.org](https://agora.adknowledgeportal.org/)) | Curated **genomic** target-nomination browser from AMP-AD; explore gene-level AD evidence. | No. Read-only evidence browser. | Genomics targets, no imaging, no discovery, no referee, no next-experiment. |

**Verdict on (b):** The AD co-scientist space is crowded but **entirely molecular/omics/literature**. Biomni-AD even has a rationale-check "referee," but it validates *LLM claims*, not *ML representations*. None ingests imaging embeddings or does unsupervised imaging-phenotype discovery. Your imaging-native + statistical-confound referee is orthogonal to all of them.

---

## (c) General scientific-discovery harnesses

| Tool | What it does | Closes loop? | Gap |
|---|---|---|---|
| **POPPER** (Stanford/Harvard, [arXiv 2502.09858](https://arxiv.org/abs/2502.09858), [github snap-stanford/POPPER](https://github.com/snap-stanford/POPPER)) | **The single closest analog to your referee.** Agentic *sequential falsification*: takes a free-text hypothesis, designs falsification experiments, converts p-values→e-values with strict Type-I error control; matched PhD scientists 10× faster. | Yes for validation — falsification + statistical rigor + accept/continue decision. Free-text hypothesis IS the entry point. | **Operates on structured CSV/tabular data (gene perturbation, DiscoveryBench economics/sociology). No neuroimaging, no image embeddings, no SSL substrate, no unsupervised discovery front-end.** It validates a *given* hypothesis; it does not *surface* candidate phenotypes from a frozen imaging model. It's the referee without the SSL discovery engine or the imaging domain. |
| **Google/DeepMind AI Co-Scientist** ([Nature s41586-026-10644-y](https://www.nature.com/articles/s41586-026-10644-y), [arXiv 2502.18864](https://arxiv.org/abs/2502.18864)) | Gemini multi-agent: generate/critique/rank hypotheses via Elo tournaments; wet-lab-confirmed on AML repurposing + liver-fibrosis targets. | Generates + ranks + tournament-critiques hypotheses; validation is *external* wet-lab, not in-loop statistical falsification. | Literature/reasoning-driven, general biomedical. No imaging, no embeddings, no SSL, no confound audit. Debate/ranking ≠ statistical referee. |
| **Claude for Life Sciences / Claude Science** ([anthropic.com/news/claude-for-life-sciences](https://www.anthropic.com/news/claude-for-life-sciences)) | Scientific workbench: literature review, hypothesis generation, data analysis; connectors to Benchling, PubMed, Synapse, 10x Genomics. | No closed loop; no falsification/referee as a distinct feature (confirmed absent in search). | General LLM harness. No imaging FM integration, no unsupervised discovery, no statistical referee. This is the *platform you build on*, not a competitor. |
| **Databricks Mosaic AI / AiChemy** ([InfoWorld](https://www.infoworld.com/article/4154467/databricks-launches-aichemy-multi-agent-ai-for-drug-discovery.html)) | Multi-agent framework: domain "skills" + tool orchestration + Unity Catalog governance; AiChemy targets **drug/molecule** discovery via OpenTargets/PubChem/PubMed MCP. | No scientific-validation loop; it's an orchestration/governance layer. | This is the **architectural template your "Databricks-style harness" vision emulates**, but it has zero neuroimaging content and no falsification referee. Confirms the pattern is proven and unclaimed for imaging. |
| **Nilearn / BrainIAK** ([nilearn.github.io](https://github.com/nilearn/nilearn), [BrainIAK](https://www.humanbrainmapping.org/)) | Mature Python neuroimaging ML: MVPA, decoding, connectivity, parcellation on scikit-learn. | No. Analysis libraries, human-driven. | Toolboxes, not agents. No FM embeddings, no hypothesis entry, no automated referee, no discovery orchestration. Building blocks, not a product. |

**Verdict on (c):** POPPER proves the falsification-referee concept works and takes free-text hypotheses — but on tabular data, no imaging, no SSL discovery front-end. The generic co-scientists reason over literature/omics. None couples a frozen imaging FM to a statistical referee. The Databricks pattern (deterministic skills + ML + governance) is validated for *drug* discovery and wide open for imaging.

---

## (d) Biomarker-discovery / imaging-subtype platforms

| Tool | What it does | Closes loop? | Gap |
|---|---|---|---|
| **SuStaIn** (Subtype & Stage Inference, [Nature Comms s41467-018-05892-0](https://www.nature.com/articles/s41467-018-05892-0), [github ucl-pond/pySuStaIn]) | The gold-standard unsupervised imaging-subtype discovery: disentangles phenotypic + temporal heterogeneity from cross-sectional MRI; recovered FTD genotypes from imaging alone; stratifies AD/ADNI. | Discovery: yes. Validation/referee: no. Next-experiment: no. Hypothesis entry: no. | **No LLM layer, no confound/leakage referee, no free-text hypothesis entry, no mechanism→next-experiment.** Operates on hand-engineered regional volumes, NOT SSL embeddings. Requires expert setup. This is your closest *discovery* analog — and it stops exactly where your referee begins. |
| **ComBat / harmonization tooling** ([Nature s41598-025-25400-x](https://www.nature.com/articles/s41598-025-25400-x)) | Removes site/scanner batch effects from multi-site MRI features. | No — a preprocessing step. | A *component* of a referee (your scanner-leakage test), not a discovery system. Requires manual batch-variable specification. |
| **icometrix / QMENTA** (commercial) | Clinical volumetric quantification + regulatory-grade MRI biomarker reporting. | No discovery — they *quantify known* biomarkers for clinical reporting. | Deliver established measures (hippocampal volume etc.), not *novel* biomarker discovery. No SSL, no referee, no hypothesis entry. Opposite end of the maturity spectrum. |
| **Nature Methods unified multimodal embedding** ([s41592-026-03070-5](https://www.nature.com/articles/s41592-026-03070-5)) | Agentic multi-agent system generating *explanatory* embeddings for single-object phenotypes; traces which visual features drove interpretation. | Partial — interpretable embedding→biological state. | **Pathology/microscopy morphology, not brain MRI.** Interpretability, but no falsification referee, no AD/neuro focus. Nearest philosophical cousin in a different imaging domain. |

**Verdict on (d):** SuStaIn is the incumbent for imaging-phenotype discovery but runs on engineered features, has no LLM referee, no hypothesis entry, and no next-experiment output. Commercial platforms quantify *known* biomarkers. The discovery→validation→referee→next-experiment chain is unbuilt for imaging.

---

## THE WHITE SPACE

Map every tool onto five required capabilities. **No single tool has more than three; NeuroAD targets all five.**

| Capability | Neuro-JEPA/Brain-JEPA/BrainDINO | Biomni-AD / Prima Mente | POPPER | Google/Claude co-scientists | SuStaIn | **NeuroAD** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **1. SSL/FM imaging substrate** (frozen brain embeddings) | ✅ | ❌ | ❌ | ❌ | ❌ (engineered feats) | ✅ |
| **2. Unsupervised phenotype DISCOVERY** | ~ (demo) | ❌ | ❌ | ❌ | ✅ | ✅ |
| **3. Statistical FALSIFICATION / referee** (age/sex, scanner-leakage, brain-age, biomarker-anchor, replication) | ❌ | ~ (LLM rationale only) | ✅ (tabular) | ~ (debate/Elo) | ❌ | ✅ |
| **4. Researcher free-text HYPOTHESIS entry** | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **5. Mechanism hypothesis + falsifiable NEXT-EXPERIMENT** | ❌ | ~ | ~ | ✅ | ❌ | ✅ |

**The precise unclaimed intersection:** *SSL imaging embeddings (1) → unsupervised discovery (2) → statistical confound-aware referee (3), entered via a researcher's free-text hypothesis (4), ending in a falsifiable next experiment (5).*

- The **imaging foundation models** own row 1 and stop.
- The **AD co-scientists** own rows 4–5 but on omics/literature, never touching imaging embeddings; their "referee" checks LLM factuality, not representation confounds.
- **POPPER** is the only true statistical-falsification referee with a free-text entry point (rows 3–4) — but on CSV tabular data, with no SSL discovery engine and no imaging domain. *This is your single most important comparable: you are "POPPER for SSL neuroimaging phenotypes," with the discovery front-end POPPER lacks.*
- **SuStaIn** owns row 2 for imaging — but on engineered features, with no referee, no hypothesis entry, no next-experiment.

**Why the gap exists (and is defensible):** The batch-effects literature ([arXiv 2604.14441](https://arxiv.org/pdf/2604.14441)) proves the field ships imaging embeddings that leak scanner/site, yet no tool automates the guardrail. Combining a *frozen* imaging FM (row 1) with a *statistical* referee that explicitly tests scanner-leakage and biomarker-anchoring (row 3) is precisely the missing bridge — and it's the honest-discovery framing the Gladstone "advance the field" prize rewards.

---

## Positioning takeaways for differentiation

1. **Name your closest comparables explicitly and beat them on the axis they lack:** "SuStaIn discovers imaging phenotypes but can't tell you if they're scanner artifacts; POPPER falsifies hypotheses but can't discover them from a brain and doesn't do imaging; Biomni-AD reasons over omics, never pixels. NeuroAD is the only one that discovers a phenotype from frozen SSL brain embeddings *and* runs it through a statistical gauntlet before letting you believe it."
2. **The referee IS the moat.** Everyone can emit embeddings and cluster them. The 5-test gauntlet (age/sex, scanner-leakage, brain-age, biomarker-anchor, replication) with a Claude prosecution/defense adversary is the unclaimed layer — and it directly answers the field's documented batch-effect crisis.
3. **Frame vs. the AD Data Initiative winners** (Biomni-AD, Prima Mente): they won a *sibling* prize months ago and are omics/epigenome. You are the imaging-native, honesty-first complement — not a competitor, a missing modality. That is a strong "advance the field" narrative.
4. **What NOT to claim:** do not position as a general AD co-scientist (Biomni/Google own that), a foundation model (Neuro-JEPA owns that), or a clinical biomarker product (icometrix owns that). Position narrowly as *the falsification-refereed SSL imaging-phenotype discovery bridge* — the empty cell in the table.

**Key sources:** [Neuro-JEPA](https://huggingface.co/NYUMedML/Neuro-JEPA) · [Brain-JEPA](https://arxiv.org/abs/2409.19407) · [BrainLM](https://www.biorxiv.org/content/10.1101/2023.09.12.557460v1.full) · [BrainDINO](https://arxiv.org/abs/2604.27277) · [Batch effects in FM embeddings](https://arxiv.org/pdf/2604.14441) · [Biomni](https://www.biorxiv.org/content/10.1101/2025.05.30.656746v1) · [Prima Mente](https://www.genengnews.com/topics/artificial-intelligence/prima-mente-unlocks-early-stage-alzheimers-diagnostics-with-epigenome-model/) · [AD Data Initiative prize](https://www.alzheimersdata.org/news/alzheimer-s-disease-data-initiative-doubles-1m-prize-competition-for-agentic-ai-solutions-to-accelerate-alzheimer-s-research) · [Agora](https://agora.adknowledgeportal.org/) · [POPPER](https://arxiv.org/abs/2502.09858) · [Google Co-Scientist](https://www.nature.com/articles/s41586-026-10644-y) · [Claude for Life Sciences](https://www.anthropic.com/news/claude-for-life-sciences) · [Databricks AiChemy](https://www.infoworld.com/article/4154467/databricks-launches-aichemy-multi-agent-ai-for-drug-discovery.html) · [SuStaIn](https://www.nature.com/articles/s41467-018-05892-0) · [Nilearn](https://github.com/nilearn/nilearn)