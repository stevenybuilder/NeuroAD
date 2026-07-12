# Scientific-Rigor + Frontend/Backend E2E Audit

_Read-only audit. No code changed. Sources: deep research on Haoxu Huang / NeuroJEPA
(verified against arXiv:2606.14957 + HuggingFace + GitHub), a file:line confound-control
code audit, a frontend-integration audit, and a live end-to-end test of the running demo._

## TL;DR verdicts

1. **NeuroJEPA is real and we use it as its authors sanction.** arXiv:2606.14957 (Haoxu Huang
   first author, Narges Razavian senior; NYU Langone). 3D ViT-Base-MoE, 768-d, V-JEPA-2 latent
   prediction, pretrained on ~1.55M **internal NYU clinical** T1w/T2w/FLAIR (no ADNI/OASIS). The
   model card explicitly supports **frozen extraction + attentive probing** — exactly our regime.
2. **Confound control is partial:** **site IS controlled in the headline estimate** (site-disjoint
   GroupKFold); **age and sex are NOT — they are only tested post-hoc at 15% weight.** A tool that
   markets "deconfounding" currently ships a demographically-unadjusted headline AUC.
3. **One HIGH-severity leak:** ComBat is fit on the whole cohort before CV (label-blind, so dx isn't
   leaked, but feature-distribution structure crosses train/test).
4. **The headline result is field-honest, not an embarrassment:** plasma p-tau217 dominance and
   conversion imaging 0.72 (site-disjoint) sit squarely in the literature's honest band.
5. **Frontend E2E:** the served demo (`neuroad.html`) is a **frozen replay of `demo_data.json`** that
   **never calls the live pipeline**, surfaces a **synthetic cohort with its honest badge stripped in
   rendering**, shows **hardcoded fabricated narrative numbers** as if they were engine output, and
   **hides the real backend depth (L2–L6) and the strongest real-data proofs**. "Ask Claude" is
   honestly wired but runs **canned offline templates** with no API key.

---

## Part A — NeuroJEPA (Haoxu Huang): what it is, methods, rigor

**Verified** (arXiv:2606.14957 HTML/abstract + NYUMedML/Neuro-JEPA HF card + GitHub):
- 3D ViT-Base **Mixture-of-Experts**, 768-d hidden (the dim we ingest), 576 tokens, patch 12³,
  96×108×96 input, 3D RoPE. Objective: **V-JEPA-2 latent prediction** (EMA momentum teacher,
  foreground-aware L1 loss, multiscale masking 0.75) — **not** contrastive, **not** reconstructive.
- Pretraining: ~1.55M T1w/T2w/FLAIR, 282,693 patients, **entirely internal NYU Langone clinical MRI**.
- Eval: 47 tasks / 12 public + 3 clinical cohorts; **headline numbers fine-tune attentive layers**
  (not frozen probe). Frozen + attentive probing is officially supported ("suggested" on the card).
- Their strongest control: **patient-level (subject-disjoint) splits** (biometric-leakage rationale).
- Their rigor **gaps** (absent in the main text we could read): no ComBat/site harmonization; splits
  are **random patient-level, not site/scanner-disjoint**; no age/sex covariate adjustment; brain age
  is a **target** (R²=0.894), never a nuisance control; no fluid-biomarker anchor.

**Do NOT overstate (unverified):**
- Appendix H.3 "Fairness Analysis" is referenced but was **not readable** — say "none in the verified
  main text," not "proven absent."
- Absolute AD-vs-CN AUROCs were not surfaced (only deltas: +4.4–6.4% AUROC, MCI→AD C-index +6.6%).
  **Do not quote an absolute NeuroJEPA AD AUROC, and do not claim to beat it** (they fine-tune; we freeze).
- **Name collision:** a second "Haoxu Huang" (robotics/vision, Tsinghua; CoPa/CcHarmony) is conflated in
  DBLP/Scholar. VICReg/spectral-MAE belongs to a **different** paper (arXiv:2606.13315, Ergun et al., no
  Huang) — do **not** attribute VICReg to NeuroJEPA.

---

## Part B — Did we control confounds properly? (direct verdict)

| Confound | Controlled *in the headline estimate*? | Mechanism |
|---|---|---|
| **Site** | **Yes** | Site-disjoint StratifiedGroupKFold on the primary AUC (`probe.py:248-256`); permutation nulls shuffle *within* site (`probe.py:390-407`). |
| **Scanner** | Partial (via site grouping) + tested | Post-hoc STAR margin = outcome_AUC − scanner_AUC (`gauntlet.py:134-163`, `leakage.py:43-129`). |
| **Age** | **No — only tested** | Headline is on **raw 768-d embeddings**, no age adjustment (`pipeline.py:147`). Residualized only inside the `age_sex` gauntlet at 15% weight (`gauntlet.py:102-128`). |
| **Sex** | **No — only tested** | Same as age. |

**Important nuance:** in ComBat, age/sex are **preserved, not removed** (`harmonize.py:186-196`) — the
*opposite* of controlling for them. So on both the raw and ComBat feeders, demographic signal stays in
the features that make the headline. Defensible as "we test whether the effect survives demographic
residualization," but **not** "we control for age/sex."

**Credit where due (genuinely strong for a hackathon):** per-fold PCA-10 + scaler inside the Pipeline
(no preprocessing leakage on the headline, `probe.py:92-98,262`); deliberately reporting the deflated
PCA AUC (~0.93) over the inflated raw-768d (~0.998); label-blind ComBat (correct anti-leakage choice);
bootstrap CI + permutation p with self-disclosed anticonservatism; NA guardrails for a non-predictive
brain-age model; anchor gates on CI lower bound; repeated-CV OOF ensembling.

---

## Part C — Scientific gaps

**Domain:**
1. **Single-modality under-use.** NeuroJEPA is multimodal (T1w/T2w/FLAIR, MoE); we feed only **T1w
   MPRAGE**. "Imaging adds little" may partly reflect degraded single-modality input, not the model ceiling.
2. **Domain shift is real.** NeuroJEPA saw zero research MPRAGE in pretraining; frozen ADNI use is OOD —
   name it as a co-explanation for weak imaging, not just a framing win.
3. **The headline is field-*predicted*.** p-tau217 ~0.88–0.91; conversion imaging honest ceiling ~0.70
   (Wen 2020: 0.665/0.702). Our 0.72 is in-band → evidence we resist confound-inflation, not underperformance.
4. **Brain-age-as-control must be age-bias-corrected + CN-trained** or it re-imports the confound.
5. **No true external held-out cohort** for the headline (internal 590/334, CV-only; plasma synthetic on
   synthetic cohorts). Huang's signature is cross-health-system OOD — CV-only is discounted.

**Data science (severity):**
1. **ComBat leaks (HIGH):** fit on the whole cohort before CV (`loaders.py:67-72`→`harmonize.py:162-203`).
   Re-fit inside each fold (fit train / transform test).
2. **Power / EPV (HIGH):** conversion n=58 events ÷ PCA-10 ≈ **5.8 EPV** (< 10 floor). Make the **linear
   probe primary** for conversion, report EPV, present wide CIs honestly.
3. **Nonlinear confound-leakage (HIGH-ish, cheap):** ComBat features → attentive MLP can paradoxically leak
   the confound (Hamdan 2023). Add a **label-shuffle sanity test**.
4. **"After" arms not fold-honest (MED):** residualizers (`gauntlet.py:80-88`) + scanner-LDA
   (`leakage.py:196`) fit on full X incl. test rows.
5. **biomarker_anchor uses non-grouped CV (MED):** plain StratifiedKFold, not site-disjoint (`gauntlet.py:305-311`).
6. **No multiple-comparison correction (MED):** `benjamini_hochberg` exists (`harness/validation.py:367`) but
   is never wired into the gauntlet.
7. Permutation p / bootstrap CI anticonservative (LOW, self-disclosed); confound_leaderboard is in-sample (LOW).

---

## Part D — Frontend / backend end-to-end (live test + audit)

**What the live backend does (verified via `/api/*`):** `POST /api/investigate` runs the real pipeline —
`adni:neurojepa` → AUC 0.857 (n=590); `adni:conversion` → AUC 0.712 (n=334), framed "MCI converters vs
non-converters." `POST /api/orchestrate` (Ask Claude) → HTTP 200, `path:"scripted_offline"`, real tool calls.

**What the served frontend actually does:**
- `neuroad.html` (served at `/`) has **zero calls to `/api/investigate`** — confirmed by grep and by the
  server log (a browser "Investigate" click produced **no** API POST). It **replays frozen `demo_data.json`**;
  any typed hypothesis yields the same canned "p-tau217 → hippocampal atrophy" storyline on a **synthetic
  360-subject / 2-site cohort**.
- **Synthetic data is surfaced with its honest badge stripped:** the app boots `substrate='synthetic'`; the
  JSON self-badges "SYNTHETIC HARNESS / NOT Neuro-JEPA / not measured plasma," but the cohort card
  (`neuroad.html:2135-2143`) renders none of that — only "n participants, k sites."
- **Hardcoded fabricated narrative numbers** are shown as engine output: the "Confound gauntlet" card
  (`neuroad.html:2154-2161`) uses fixed Δ 0.16/0.22/0.14/0.53/0.27 under **invented test names** ("Head
  motion", "Reverse causation") that don't match the real 5-dim gauntlet; kill cards hardcode "0.71 vs 0.86"
  and "Before 0.61 → After 0.79." No "illustrative" badge.
- **The real depth is invisible:** L2 grounding, L3 cross-attention, L4 Boltz, L5 pathways, L6 biomarker
  fusion, the real gauntlet `case.tests`/verdicts, discovery clustering, and the strongest real-data proofs
  (`real_evidence`: OpenBHB n=3984 scanner AUC 0.891; `neurojepa_evidence`: frozen NeuroJEPA scanner-leakage
  0.958, AD-signal 0.811) are **all in `demo_data.json` but rendered nowhere** in `neuroad.html`.
- **Ask Claude is honest but canned offline:** wired to `/api/orchestrate`, reads server `path`, labels
  "(offline template)" vs "(Claude · live)"; with no `ANTHROPIC_API_KEY` it runs client-side templates. One
  weakness: `claude_live` is fetched but never rendered → no upfront offline indicator.

**Net:** the demo is polished and renders end-to-end, but it is a **frozen, partly-synthetic replay decoupled
from the real backend**, and it foregrounds scripted/fabricated numbers while hiding the defensible real
evidence. This is the documented "deterministic demo" design — but it directly means: backend components are
baked-in (not live-loaded), there **is** synthetic data on the frontend, and typed hypotheses do **not** drive
real computation.

---

## Part E — Top prioritized fixes

**Science/rigor**
1. **Move ComBat (and the "after"-arm residualizers/LDA) inside the CV loop** — closes the one HIGH leak. (M)
2. **Report an age/sex-residualized *primary* AUC** next to the naive one; add an age/sex-matched AD-vs-CN
   run; stop calling it "deconfounding" until then. (S–M)
3. **Add the label-shuffle test + embedding-space batch metrics** (kBET, silhouette-by-site, pre/post-ComBat
   site-decodability delta). (S)
4. **Fix the conversion power story:** linear probe primary for MCI→AD, report EPV≈5.8, honest wide CIs,
   confirm no same-subject visit spans folds. (S–M)
5. **Wire `benjamini_hochberg` into the gauntlet; switch biomarker_anchor to site-disjoint CV; make
   confound_leaderboard OOF.** (M)

**Frontend (if the demo should show real work, not a canned replay)**
6. Render `co.badge` + `co.substrate_line` + `case.tests` so the SYNTHETIC-HARNESS / "not measured plasma"
   caveats reach the screen; replace hardcoded gauntlet/kill numbers with the real `case.tests`/verdicts.
7. Surface `real_evidence` / `neurojepa_evidence` and a persistent Claude offline/live badge.
8. (Optional) wire the "Investigate" button to the working `POST /api/investigate` for a live path, or badge
   the frozen storyline clearly as an illustrative walkthrough.

**Write-up guardrails:** don't claim to beat NeuroJEPA's AD numbers (frozen-probe vs fine-tuned); present
plasma-dominance as the field-honest result; name single-modality + domain shift as co-explanations; where
research was unverifiable (NeuroJEPA absolute AUROCs, Appendix H.3, exact "Cautionary Tale" collapse figures,
JEPA-vs-MAE direction), say so rather than quoting hard numbers.
