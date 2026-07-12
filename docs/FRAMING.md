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
