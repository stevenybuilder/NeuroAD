# Data Expansion Spec — increasing sample size where it actually helps

_Status: DRAFT v1 (2026-07-11). Targets the two rate-limiting classes identified by
the power analysis, not "more data" broadly._

## 1. What the numbers say we need

From `reports/power_analysis_conversion.md` and the class audit:

| Lever | Current | Target | Gap |
|---|---|---|---|
| **MCI→AD converters WITH plasma** (fusion power) | 142 (n=498) | **~650–710** (80% power) | **+~500–570 converters** |
| **AD imaging cases** (imaging-arm CIs) | 87 ADNI + ~41 OASIS ≈ 128 | ~300+ | +~175 AD scans |
| AD-vs-CN diagnosis (n=1,615, AUC 0.92) | 462 AD / 1,153 CN | — | already well-powered; **do not expand** |
| CN subjects generally | plentiful | — | **do not expand** (won't move any claim) |

**The binding lever is converters-with-plasma.** The local audit is definitive:
ADNI is at its ceiling — 270 more ADNI converters exist but **lack plasma p-tau217**,
so they can't join the fusion arm. New converters-with-plasma must come from
**external cohorts.**

## 2. Cohorts to acquire (ranked by converter-with-plasma yield)

| Cohort | Adds (converters w/ plasma) | Adds (AD imaging) | Access | Effort |
|---|---|---|---|---|
| **AIBL** (Australian Imaging, Biomarkers & Lifestyle) | ~150–250 MCI, plasma p-tau + longitudinal | ~200 AD T1w | **LONI/IDA** (same portal as ADNI — you already have access) | **Low** — mirrors the ADNI pipeline |
| **NACC** (National Alzheimer's Coordinating Center) | ~several hundred (plasma subset growing) | large | naccdata.org data request (free, ~2 wk) | Medium (harmonize UDS fields) |
| **OASIS-3** | modest converters | ~1,000+ sessions incl. AD | central.xnat.org (free registration) | Low–Medium |
| **A4 / LEARN** | plasma-rich, preclinical | some | LONI | Medium |

**Recommended first move: AIBL.** It's on the LONI/IDA portal you already use for
ADNI, carries plasma p-tau217 + longitudinal conversion, and drops straight into the
existing NeuroJEPA + triangulated-plasma + ComBat pipeline. AIBL + NACC together
plausibly reach the ~650–710 converter target.

## 3. How each folds into the existing pipeline (no new machinery)

The pipeline is already cohort-agnostic — expansion is data-plumbing, not new code:

1. **Imaging** → same `scripts/neurojepa_embed_colab.py` (frozen NeuroJEPA), same
   GCS-resumable driver. New cohort = new manifest of T1w NIfTI.
2. **Plasma** → the triangulated-plasma ensemble (`data/plasma_ensemble.py`) already
   harmonizes multiple assays; add the new cohort's assay as another source.
3. **Structure** → same FreeSurfer/FastSurfer `aseg` → contract volumes.
4. **Harmonization** → **ComBat** (already in the contract) removes the site/scanner
   batch effect between ADNI and the new cohorts. This is the critical step — the
   cross-cohort leakage check (currently AUC 1.0 on raw features) must drop toward
   ~0.5 after harmonization for a valid pooled analysis.
5. **Labels** → map each cohort's diagnosis/conversion coding to the contract schema
   (`data/gated.py` already has per-cohort column maps for CDR etc. — add the new
   cohort's columns there).

## 4. What I can vs cannot do

- **I can:** build the manifest-ingestion + harmonization + label-mapping code for
  each cohort, run the embeds (Colab + GCS-resume), and re-run the conversion /
  cross-cohort analyses once the raw data is downloaded.
- **I cannot:** download the cohorts — they are **DUA-gated** (LONI/NACC/XNAT
  require your authenticated account, exactly like ADNI did). You initiate the
  download; I wire everything else.

## 5. Sequencing (against the power target)

1. **Finish ADNI + decoder** (in progress) — establishes the baseline the expansion
   is measured against.
2. **AIBL** (you download via LONI) → embed + harmonize → re-run conversion. Expected
   to roughly double converters-with-plasma toward ~300–400.
3. **NACC** (data request in parallel) → push toward the ~650–710 target for 80%
   power on the fusion-vs-plasma question.
4. Re-run `scripts/power_analysis_conversion.py` after each addition to track power
   empirically rather than by assumption.

## 6. Honest expectation-setting

Even at ~700 converters, recall the power analysis caveat: this proves the
**observed +0.012 fusion lift** at 80% power *if that lift is the true effect*. If
the true lift is smaller (regression to the mean), you'd need still more. The
expansion is worth doing for the **prognostic robustness and cross-cohort
generalization** it buys regardless — but "fusion beats plasma" may remain a
marginal, hard-won result. Plasma p-tau217 stays the workhorse.
