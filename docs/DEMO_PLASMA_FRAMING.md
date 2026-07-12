# Demo framing — plasma as the spine, and the biomarker blind-spot

Companion to `docs/DEMO_SCRIPT.md` (the 3-min autopilot video). This is the
**biomarker / plasma** talking track: the story to tell in the live walkthrough and,
especially, in Q&A. Every number here is backed by a saved report — cite, don't
improvise.

Mockup of the in-UX card: `mri_visualizations/biomarker_blindspot_card.png`
(styled to the ZUI; the panel it depicts is the "biomarker blind-spot" finding).

---

## The one thing to land

> **Plasma p-tau217 isn't a feature we fuse — it's the anchor that makes our imaging
> signal biologically legible.** It tells us *what* the signal is (tau), benchmarks us
> against the best blood test, and shows us exactly where imaging sees what the blood
> test misses. That divergence is where biomarker discovery starts — and it feeds
> straight into target prioritization.

Diagnosis is where the Nature Medicine imaging models stop. It's where we start.

---

## The four-beat arc (say it in this order)

| # | Beat | What it shows | Backing number |
|---|---|---|---|
| 1 | **Signal** | Imaging predicts who converts (MCI→AD). | imaging-only AUC 0.65–0.73 · `conversion_imaging_only.json` |
| 2 | **Grounding** | Plasma decodes *what it means* — p-tau217 is the top attributed driver, so the signal is **tau**, not noise. | LOGO attribution · `attentive_probe_ad.json` |
| 3 | **Discovery** | Where imaging & plasma **diverge**, imaging flags converters the blood test calls low-risk → a candidate **new imaging biomarker**. | imaging 0.77 vs plasma 0.64 (p-tau-low) · `conversion_biomarker_negative.json` |
| 4 | **Targets** | The plasma anchor routes tau → ranked druggable targets (APP, MAPT…). | PI4AD ranking · `candidate_ranking.json` |

Beat 3 is the demo's differentiated moment — it's the panel in the mockup, and it's
the thing the imaging-only foundation-model papers structurally cannot produce
(they have no biomarker to attribute to or diverge from).

---

## The biomarker blind-spot (Beat 3, in detail)

Split the plasma-tested ADNI cohort (n=498, 142 converters) by the blood marker:

| Subgroup | Imaging AUC | Plasma AUC | Who wins |
|---|---|---|---|
| **p-tau217 LOW** ("test says low-risk") | **0.77** [0.68–0.85] | 0.64 [0.51–0.76] | **imaging** |
| p-tau217 HIGH (plasma's wheelhouse) | 0.64 | **0.69** | plasma |
| Amyloid-negative | 0.66 | 0.63 | imaging (edge) |

**The line:** *"Each modality carries the signal exactly where the other goes quiet.
Among patients the blood test would reassure, imaging still flags the converters —
at 0.77, where plasma is near-silent."*

**Honesty guardrails (state them; they build trust):**
- This is **complementarity, not "imaging beats plasma."** On p-tau-high people, plasma wins. Never claim fusion beats plasma outright (it doesn't: +0.012, p=0.12).
- The low-p-tau subgroup has **23 converters** — a strong *directional* result with a wide CI, not a locked point estimate. Say "directional."
- Plasma p-tau217 is **real** ADNI data, triangulated across 3 assays — but **gated** (no open cohort pairs MRI with plasma). That gap is also the whitespace that makes this novel.

---

## Coverage story (the "serving the plasma-negative population" line)

Two distinct groups — keep them separate:
- **Plasma-unavailable** (no blood test at all): 701 ADNI converters-labeled subjects
  the fusion arm discards. Imaging alone predicts conversion at **AUC 0.68** for them
  — a population plasma literally can't serve (most of the world: assay is new,
  costly, undeployed). `conversion_imaging_only.json`.
- **Biomarker-negative** (tested, came back low): the 0.77 result above.

Framing: *"A tool that only works when you have a p-tau217 result is useless for the
majority of patients who don't. Imaging extends risk stratification to them."*

---

## Who is this for? (anticipated judge question)

**Both — sequentially, not simultaneously. It's a hand-off pipeline.**

- **Neuroimaging researchers** are the *front door*: they bring the scans and the
  frozen-encoder + probe machinery (Layers 1–3). For them the value is the honest
  referee — leakage tests, ComBat, kill/survive — that stops a plausible imaging
  finding from becoming a wasted quarter.
- **Wet-lab / translational researchers** are the *back door*: they receive the output
  — ranked, falsifiable protein targets with suggested experiments (Layers 5–6). For
  them the value is a prioritized, biology-grounded shortlist instead of a black-box
  classifier.
- **Plasma is the bridge between them.** The biomarker anchor is what converts an
  imaging phenotype (neuroimaging's language) into a molecular hypothesis (the wet
  lab's language). Without it the two audiences don't connect; with it, the imaging
  signal becomes a target the bench can test.

So the honest answer: *"It's built for the hand-off — neuroimaging in, wet-lab
experiments out, with the biomarker anchor as the translator. Neither audience alone
is the customer; the pipeline between them is the product."*

---

## Q&A quick-fire

- **"Doesn't fusion beat plasma?"** — No, and we don't claim it. +0.012, not
  separable. Plasma is the workhorse; imaging's value is coverage, localization, and
  discovery. (Owning this is the credibility move.)
- **"Why not pretrain your own encoder?"** — That's a solved, published problem needing
  a health system's data. We stand on a frozen encoder — as the SOTA papers do
  downstream — and spend effort on the unsolved part: ranked, falsifiable targets.
- **"How do you know the imaging signal is real, not batch effect?"** — Site-disjoint
  CV, ComBat (cohort leakage 0.9996→0.563), permutation nulls, and a referee that
  refuses claims failing the scanner-leakage test (see the KILL beat).
