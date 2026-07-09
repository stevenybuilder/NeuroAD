# LATEST SCIENCE (2025–2026): What TRUE Innovation on the AD Imaging+Biomarker Frontier Requires

## 1. The field has re-centered: plasma biomarkers now do the amyloid-detection job; imaging's remaining value is TOPOGRAPHY, STAGING, and MECHANISM

The single biggest shift: on **May 16, 2025 the FDA cleared the first blood test for AD** (Fujirebio Lumipulse G pTau217/β-Amyloid 1-42 plasma ratio), for symptomatic adults 55+. Plasma p-tau217 now detects amyloid-PET-positive pathology at **AUC 0.94–0.97 / 89–95% accuracy** across Simoa, Ella, and Lumipulse assays — comparable to a PET scan for ruling amyloid in/out. This is the pivotal fact for a discovery tool: **a cheap blood draw already solves "is there amyloid?"** So a new *structural-MRI* signature that merely re-predicts amyloid status adds little. Imaging earns its keep only where plasma is weak.

Where plasma is weak / imaging still wins (the honest value gap you can target):
- **Positive predictive value ceiling.** p-tau217 as a stand-alone amyloid marker in *cognitively unimpaired* people needs PET/CSF confirmation to exceed 90% PPV — i.e. it is a good rule-in/rule-out screen but not a locater. It says *whether*, not *where* or *how fast*.
- **Attenuation in the very elderly** — p-tau217's discriminative accuracy drops in the oldest-old vs p-tau181/GFAP, a known confound.
- **Topography, staging, neurodegeneration burden.** The **2024 NIA-AA revised criteria (Jack et al.)** made the disease *biologically* defined and explicitly kept **amyloid-PET and tau-PET for biological staging** because "imaging captures both topographic and quantitative information." Blood covers the **A** and **T** axes; the **N (neurodegeneration)** axis and *spatial pattern* remain imaging's domain. That is the crack an imaging-SSL tool legitimately lives in.

The plasma panel context you must anchor to: **p-tau217** = amyloid/early-tau proxy; **GFAP** = astrocytic/amyloid-associated, rises early, useful in oldest-old; **NfL** = non-specific neurodegeneration/axonal damage (elevated across many diseases, so a *good "N-anchor"* but a *bad "AD-specificity" anchor*). Anchoring an imaging phenotype to **p-tau217 = "is this AD-like?"**; to **NfL/GFAP = "is this neurodegeneration, of any cause?"** — that distinction is scientifically load-bearing and reviewers respect it.

- [FDA-cleared Lumipulse p-tau217/Aβ42 & assay accuracy (AAIC 2025 highlights)](https://aaic.alz.org/releases-2025/highlights-aaic-2025.asp)
- [Three p-tau217 assays vs PET (AUC 0.94–0.97)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12959247/)
- [Ruling in / ruling out: clinical utility & PPV ceiling of plasma biomarkers](https://pmc.ncbi.nlm.nih.gov/articles/PMC12483549/)
- [p-tau217 attenuated in very elderly vs p-tau181/GFAP](https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2025.1668512/full)
- [2024 NIA-AA revised criteria — imaging retained for biological staging (Jack et al.)](https://alz-journals.onlinelibrary.wiley.com/doi/10.1002/alz.13859)

## 2. What "novel biomarker discovery" CREDIBLY looks like now (not "we beat ADNI accuracy")

The credible-discovery playbook in 2025 is **unsupervised imaging phenotype → external replication → anchoring to an independent biological axis (plasma/genetic/cognitive) → falsifiable next step**. Three concrete, respected templates:

**(a) Unsupervised deep imaging-derived phenotypes that carry independent biological signal.** The gold-standard proof that an SSL/unsupervised imaging representation is "real" is that it associates with *genetics or fluid biology you never trained on*. Precedents: **UDIPs** — a 3D convolutional autoencoder on UK Biobank T1/FLAIR yielding 128-d representations that surfaced **97 GWAS loci** for brain structure; and **iGWAS**, self-supervised deep phenotyping whose learned features drive genome-wide associations. The transferable move for your tool: *the phenotype earns credibility not by classification accuracy but by anchoring to an orthogonal biological variable.* This is exactly the "biomarker anchor" referee test you already built.

**(b) Data-driven progression subtypes (SuStaIn family), now with a hard reproducibility literature.** The 2026 Prevot/Oxtoby study (**5,444 subjects, ANMerge+OASIS+ADNI**) is the current bar: the **Typical (hippocampal/amygdala) subtype replicated in 8/8 models (Kendall's τ 0.53–1.0)**; the **Cortical subtype in 6/8**; the **Subcortical subtype in only 3/8** — it *collapsed when controls were excluded, at 1.5T vs 3T, and in memory-enriched cohorts*. This is the single most important paper for calibrating your ambition: it tells you **which subtypes are robust and which are artifacts of site/field-strength/cohort composition**, and it validates *your* bootstrap-Jaccard stability + replication gauntlet as the correct methodology.

**(c) Normative modeling / individual deviation maps.** The field has moved from group averages to **per-subject deviation z-maps** benchmarked against population "brain charts" (PCN Toolkit, CentileBrain, BrainChart, Brain MoNoCle). Crucially, **site/scanner + age + sex are modeled as covariates so deviations are confound-adjusted by construction**, and calibrating to a new scanner needs only a small local healthy-control reference set. This is the current best answer to "how do I get a confound-robust subtype."

- [UDIPs: unsupervised deep imaging phenotypes → 97 GWAS loci (Comms Biol 2024)](https://www.nature.com/articles/s42003-024-06096-7)
- [iGWAS: self-supervised deep phenotyping for genetic association](https://www.medrxiv.org/content/10.1101/2022.05.26.22275626.full.pdf)
- [How reproducible are data-driven AD atrophy subtypes? (Prevot & Oxtoby, 5,444 subj)](https://arxiv.org/html/2412.00160v1)
- [Normative modeling in neuroimaging — practical guide (2025)](https://arxiv.org/html/2509.07237v1)
- [Personalizing AD brain-structure change via normative modeling](https://pmc.ncbi.nlm.nih.gov/articles/PMC11633367/)

## 3. Self-supervised / foundation models on brain MRI: the 2025–2026 state

Foundation models pretrained on tens of thousands of unlabeled scans are now the substrate, exactly matching your frozen-NeuroJEPA approach:
- **BrainIAC** — pretrained ~49,000 MRIs (SSL + contrastive), one frozen core adapted to many disorders including AD.
- **BrainFound** / **UMBIF** (~51,000 exams, masked-image + contrastive) and **npj/Nature Neuroscience "generalizable foundation model for human brain MRI"** — all report the same headline: SSL wins **most in label-scarce and cross-dataset settings**, i.e. exactly the regime you're in (61–96 labeled subjects).

Implication for your tool: **the frozen-embedding + lightweight-probe + unsupervised-discovery architecture is the current mainstream, not a gimmick.** Your novelty is NOT "another foundation model" (that race is over and you can't win it in 5 days) — it's the **referee/discovery harness on top**. The field has plenty of embeddings and almost no rigorous, automated *adjudication* of what a discovered pattern means. That is a genuine white space.

- [Towards generalisable foundation models for brain MRI (npj Imaging 2026)](https://www.nature.com/articles/s44303-026-00176-5)
- [A generalizable foundation model for human brain MRI (Nature Neuroscience 2026)](https://www.nature.com/articles/s41593-026-02202-6)
- [Masked & Predictive SSL foundation models for 3D brain MRI (arXiv)](https://arxiv.org/html/2606.13315)

## 4. The field's open problems (your tool should visibly *address* these, not ignore them)

1. **Heterogeneity.** AD is biologically and spatially heterogeneous; ~41% of AD subjects show *atypical* (non-hippocampal) atrophy. Group-average models miss this — the field explicitly wants per-subject / subtype resolution.
2. **Preclinical detection.** Prevention requires intervening in the asymptomatic phase; neuropathology precedes symptoms by years. But this is also where subtypes are least stable and controls confound normal aging — the hardest, highest-value target.
3. **Scanner/site confounds — THE credibility killer.** Site effect is entangled with clinical status; **ComBat and GAN-based harmonization help but are imperfect**, and a "discovery" that is really a scanner signature is the classic embarrassing failure. Your explicit **scanner-leakage referee test is squarely on the field's central methodological anxiety** — lean into it. Note your own OpenBHB data spans **2 field strengths / 6 sites**, which is both a risk and a chance to *demonstrate* confound-robustness.
4. **Translation / reproducibility.** Regulators require a **Context of Use** and **analytical + clinical validation** before a biomarker "qualifies" — validation alone is insufficient. A 5-day project cannot qualify a biomarker; it can produce a *qualification-ready candidate with a stated COU and a pre-registered validation plan.*

- [MRI harmonization survey — acquisition/image/feature levels (2025)](https://arxiv.org/pdf/2507.16962)
- [Mapping heterogeneous AD structural subtypes via normative models (Transl Psych 2026)](https://www.nature.com/articles/s41398-026-03902-0)
- [FDA Biomarker Qualification: Evidentiary Framework (Context of Use)](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/biomarker-qualification-evidentiary-framework)

## 5. The "ADVANCE THE FIELD" bar (Gladstone Award)

Synthesizing what 2025 review/consensus pieces call *advancing*: a contribution advances AD imaging when it (i) **generates a testable biological hypothesis, not just a prediction**; (ii) **survives external replication + confound control**; (iii) **anchors to an orthogonal, independently-measured axis** (plasma p-tau217/GFAP/NfL, genetics, or longitudinal conversion); and (iv) **fills a gap plasma cannot** (topography, staging, N-axis, heterogeneity/subtype, rate-of-change). "Higher accuracy on a benchmark" is explicitly *not* advancing — the field is saturated with classifiers and starved of *trustworthy, mechanistic, reproducible* phenotypes. Your differentiator = **an automated adversarial referee that enforces (i)–(iv)**. That reframes the tool from "yet another classifier" to "a reproducibility/skepticism engine for imaging discovery" — which is what translational reviewers actually reward.

## 6. DISTILLED — credible, respected novel-discovery contributions achievable for an imaging-SSL tool (in your data/compute reality)

Ranked by credibility-per-effort given 61 OASIS-1 (CN/MCI/AD, single-site) + 96 OpenBHB (healthy, 6-site), frozen NeuroJEPA, no ADNI:

**A. A confound-robust, replicated imaging subtype anchored to a plasma axis (HIGHEST value, matches your Detective+gauntlet).**
Discover a phenotype in the frozen-embedding space via KMeans/GMM/HDBSCAN → require **bootstrap-Jaccard ≥ ~0.6 stability** (SuStaIn-literature standard for "real") → **pass scanner-leakage test on the 6-site OpenBHB split** → anchor to a biological axis. Frame honestly: "a *stability-vetted, scanner-invariant* candidate atrophy phenotype." The Prevot/Oxtoby reproducibility framework is your citation that this *is* the correct bar.

**B. A brain-age-residual signature that carries information beyond the Brain Age Gap.**
BAG correlates with amyloid (r=0.43) and tau (r=0.58) PET. The credible novelty is **the residual**: show your SSL embedding predicts an outcome (MCI→AD conversion, or a plasma anchor) *after regressing out chronological age, sex, AND brain-predicted age* — i.e. "structure beyond accelerated aging." This directly maps to your brain-age referee test and is a respected, well-scoped claim.

**C. An imaging-phenotype ↔ plasma-biomarker anchor as the *validation currency*.**
Because plasma p-tau217/GFAP/NfL now exist cheaply, the respected move is: an imaging cluster whose members differ on an *independently measured* fluid marker. Even without plasma in OASIS-1, you can **anchor to the orthogonal axes you DO have** (CDR, conversion, or — if you embed the ~174 additional CDR-labeled OASIS-1 subjects — a larger labeled anchor). State the anchor explicitly and its specificity (p-tau217→AD-like; NfL→generic neurodegeneration).

**D. Confound-robustness *as itself a demonstrable result*.**
Using the 2-field-strength / 6-site OpenBHB data, *show* that a naive probe learns scanner (high scanner-AUC) and that your referee catches and rejects it. A live "we caught our own false discovery" is more convincing to translational reviewers than any accuracy number — it demonstrates the exact skepticism the field says is missing.

## 7. What would be OVERCLAIMING (hard red lines — flag these in the demo yourself)

- ❌ "We discovered a **new AD biomarker.**" → You have a *candidate imaging phenotype*. A biomarker requires a stated **Context of Use + analytical + clinical validation** (FDA); none is achievable in 5 days.
- ❌ "**Detects preclinical/asymptomatic AD.**" → Preclinical detection is the field's hardest problem *and* where subtypes are least reproducible; with 61 single-site subjects this is unsupportable. Downgrade to "hypothesis-generating in symptomatic/MCI range."
- ❌ "**Better than plasma p-tau217 / replaces PET.**" → Plasma already matches PET for amyloid at AUC~0.95; you cannot beat that on 61 subjects, and it's the wrong claim anyway (imaging's value is topography/N-axis, not amyloid detection).
- ❌ "**Generalizes across scanners**" from single-site OASIS-1 → you can only claim confound-robustness *to the extent your 6-site OpenBHB test exercises it*; be explicit about the labeled-cohort being single-site.
- ❌ "**Validated / reproducible**" from one dataset → the reproducibility literature (8/8 vs 3/8 subtype survival) shows single-cohort subtypes routinely fail to replicate. Say "internally stability-vetted; external replication is the stated next experiment."
- ❌ Any **causal / mechanism** claim stated as fact → your mechanism hypothesis + falsifiable experiment framing is correct; keep "hypothesis," never "mechanism shown."
- ⚠️ Small-n statistics (n=8 AD): report effect sizes + bootstrap CIs, avoid p-value theater, and let the referee *reject* underpowered findings visibly.

The honest, high-impact framing that threads all of this: **"Not a new biomarker — a discovery-and-referee engine that turns a one-sentence hypothesis into a stability-vetted, confound-screened, biologically-anchored imaging *candidate* plus the one experiment that would confirm or kill it."** That is precisely what "advances the field" means in 2025 (reproducibility + anchoring + falsifiability over benchmark accuracy), and it is defensible on the data you actually have.

**Key sources:** [Biomarkers in AD clinical trials 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12914139/) · [Brain Age as AD biomarker — BAG–amyloid/tau correlations](https://www.mdpi.com/2076-3425/16/1/33) · [Innovations in AD diagnostics: novel biomarkers, multimodal, non-invasive (Frontiers 2025)](https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2025.1651708/full) · [Genetic analysis of imaging-derived phenotypes (Nat Rev Genet 2026)](https://www.nature.com/articles/s41576-026-00989-5) · [2025 NIH ADRD Research Progress Report](https://www.nia.nih.gov/about/2025-nih-dementia-research-progress-report) · [Reproducibility of data-driven AD atrophy subtypes (Sage 2026)](https://journals.sagepub.com/doi/full/10.1177/13872877251415019)