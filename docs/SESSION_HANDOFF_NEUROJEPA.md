# Session Handoff — NeuroJEPA Structural-MRI AD Pipeline

_Last updated: 2026-07-11 (turn 2)_

---

## 0. Turn-2 update — real triangulated plasma executed + 2 confirmed defects fixed

**The gated LONI plasma CSVs were present locally after all** (`../download/`:
UPENN + C2N + LILLY plasma, DXSUM, UCSFFSX7, APOERES, ADSP PET, ADNIMERGE2) —
the prior turn assumed they were outside the environment. So the blocked
next-steps were **executed on real data**:

- **Ran the real ADNI contract build with the triangulated plasma ensemble**
  (`scripts/build_adni_contract.py`). The committed `adni.csv` had been built
  single-assay; the rebuild adds `p_tau217_native`, `p_tau217_n_assays`,
  `ab42_40`, `pct_ptau217`. Verified a **clean superset**: all 335 shared columns
  byte-identical, only `p_tau217` changes (native pg/mL → z-harmonized), native
  preserved exactly in `p_tau217_native`.
- **Plasma wiring confirmed solid end-to-end on real data:**
  `run_adni_survivor.py` promotes 100/100; biomarker anchor is
  `provenance=measured, synthetic=false`, reading triangulated `p_tau217`
  (r=0.434, n=876) **and** the new `pct_ptau217` fallback (r=0.239, n=464).
- **Fixed the `p_tau217` "pg/mL" docstring** in `contract.py` (next-step #3).

**Adversarial verification (multi-agent) found + I fixed 2 CONFIRMED defects:**

1. **[HIGH] Lilly assay silently dropped** (`plasma_ensemble.py`). Lilly MSD600 is
   LONG-format (`TESTCD=='PTAU217'`, value in `ORRES`); the config listed TESTCD
   *values* as column names, so Lilly never contributed and triangulation capped
   at 2 assays — defeating the module's whole purpose. Added `_normalize_assay`
   to pivot Lilly long→wide. **Now reaches depth 3** (real data: union 1593→1605,
   triangulated 917→957, imaging-cohort coverage 1366→1398). Regression test
   added (`test_lilly_long_format_triangulates_to_three`) + real-data smoke now
   asserts all-three-assays/max-depth-3.
2. **[HIGH] No embedding-dim guard** (`run_adni_crosscohort.py`). A misplaced 323-d
   FreeSurfer contract at the 768-d NeuroJEPA path would silently intersect on 323
   dims and fabricate a passing verdict. Added a `NEUROJEPA_DIM=768` guard that
   fails loud; verified it rejects 323-d and accepts 768-d.

Lower-severity audit notes (documented, not silent-changed): ensemble `p_tau217`
is a multi-visit subject mean distinct from the nearest-date `p_tau217_native`
(docstring clarified; prefer native for conversion targets); `combine_first`
scale-mix is guarded today (0 leakage). **Full suite: 255 passed / 2 skipped.**

Still the ONE hard blocker: **ADNI raw T1w MRI** (NeuroJEPA 768-d imaging feed) —
DUA-gated manual IDA/LONI download, absent. The imaging scaffold is verified and
guards correctly; it will run the instant the NIfTIs land.

---

## 1. TL;DR

This session built and validated **Layer 1** of the discovery engine: a frozen
**NeuroJEPA** foundation-model pipeline that turns raw **structural T1w MPRAGE** MRI
(no fMRI anywhere) into 768-d embeddings and probes them for an Alzheimer's signal
across **two independent cohorts** (OASIS-1 and OASIS-2, n=360 total). Under
leakage-free cross-validation with permutation testing, embeddings separate AD from
cognitively normal at **AUC 0.88 [0.82, 0.93] on OASIS-1** and hold up under the full
rigor battery: ComBat harmonization removes site/cohort batch effects (leakage
1.0 → chance) while the disease signal survives (pooled **AUC 0.87 [0.81, 0.92]**),
the signal is **not brain-age** (age-adjusted 0.844; age alone only 0.602), it
**beats the classical nWBV atrophy baseline** (0.868 vs 0.829), and all headline
contrasts **survive Benjamini-Hochberg FDR at q=0.05**. This is an honest,
reproducible two-cohort result, tempered only by a modest AD sample (41 AD in the
pooled AD-vs-CN analysis).

---

## 2. What was built this session

### Code / scripts
- **`.gitignore` hardened** — blocks weights (`*.safetensors/*.pt/*.pth/*.ckpt/*.bin`),
  volumes (`*.nii/*.nii.gz/*.mgz/*.img/*.hdr`, `OASIS1_RAW/`), embedding tables
  (`*neurojepa*.csv`, `*embeddings*.csv`), and token files. A `.githooks/pre-commit`
  hook (installed via `core.hooksPath`) **hard-blocks** weights/volumes/token literals
  even under `git add -f`.
- **`scripts/neurojepa_embed_colab.py`** — self-contained Colab T4 embedding job.
  Streams OASIS-1 disc tarballs / OASIS-2 `OAS2_RAW` parts on the runtime, extracts
  only the volumes it needs, fetches the gated `NYUMedML/Neuro-JEPA` weights
  **ephemerally** via `HF_TOKEN` (never committed), and runs frozen inference → 768-d.
  Flags: `--dataset oasis1|oasis2|adni`, `--skull-strip` (deepbet), `--fast-resample`
  (~10x), `--resume`, `--checkpoint-every` (durable gzip+base64 stdout blobs that
  survive runtime drops), `--skip-install`, `--skip-fetch`. Forces MONAI
  `NibabelReader` for Analyze `.img`. (ADNI branch added this turn — see §5.)
- **`scripts/run_oasis_neurojepa.py`** — reproducible single-cohort AD-signal report
  via a leakage-free probe.
- **`scripts/run_oasis_combined.py`** — per-cohort replication + cohort-leakage +
  pooled + site-disjoint cross-cohort analysis.
- **`scripts/run_oasis_harmonized.py`** — ComBat (batch=cohort, **label-blind**) with
  leakage / AD before-vs-after comparison.
- **`scripts/run_oasis_rigor.py`** — age-adjustment, nWBV classical baseline,
  Benjamini-Hochberg FDR.
- **`docs/COLAB_RUNBOOK.md`** updated. Nine fMRI→structural-MRI doc fixes applied
  (8 in `alzheimers_ai_tool_development_plan.md`, 1 in `product improvement 1.md`);
  functional-track / Brain-JEPA / fMRIPrep references were **deliberately kept**.

### Data produced (git-ignored, local only; CC-BY-NC-ND compliant)
- **`data/real/oasis1_neurojepa_embeddings.csv`** — 210 OASIS-1 subjects
  (119 CN / 63 MCI / 28 AD), T1w MPRAGE team-masked (T88), 768-d.
- **`data/real/oasis2_neurojepa_embeddings.csv`** — 150 OASIS-2 baseline subjects
  (85 CN / 52 MCI / 13 AD), deepbet skull-stripped raw `mpr-1`, 768-d.

> These CSVs are **not** in git (embedding tables are gitignored). They exist only in
> the local working tree. Regenerate via the Colab job (§8) if lost.

### Reports (tracked in git)
- `reports/oasis_neurojepa_ad.json`
- `reports/oasis_neurojepa_combined.json`
- `reports/oasis_neurojepa_harmonized.json`
- `reports/oasis_neurojepa_rigor.json`

---

## 3. Key findings — AUC table

All AUCs are from **leakage-free cross-validation** with **permutation testing**.
Modality is **structural T1w MPRAGE only — zero fMRI**.

| Analysis | Contrast / detail | AUC | 95% CI | n | p_perm |
|---|---|---|---|---|---|
| **OASIS-1** | AD vs CN | **0.88** | [0.82, 0.93] | 210 | 0.001 |
| OASIS-1 | Impaired vs CN | 0.81 | — | 210 | — |
| **OASIS-2** (skull-stripped) | AD vs CN | **0.75** | [0.57, 0.87] | 150 | — |
| Cohort leakage (RAW pooled) | site discriminability | ~1.0 | — | 360 | — |
| **ComBat leakage** (dx-blind) | site discriminability | **0.47** (chance) | — | 360 | — |
| **ComBat pooled** | AD vs CN, post-harmonization | **0.87** | [0.81, 0.92] | 245 (41 AD) | — |
| Cross-cohort transfer | train-one/predict-other, site-disjoint CV | **0.89** | — | — | — |
| Age-adjusted | AD vs CN, age-linear component removed | 0.844 | — | — | — |
| Age alone | AD vs CN | 0.602 | — | — | — |
| Classical baseline | nWBV atrophy | 0.829 | — | — | — |
| Pre-adjustment reference | AD vs CN (pooled, raw) | 0.868 | — | — | — |

**Interpretation highlights**
- **Leakage is real but removable.** RAW pooled cohort discriminability is ~1.0, and
  **skull-stripping did NOT remove it** — the residual batch effect is spatial
  normalization / intensity / mask method, not skull. ComBat (batch=cohort, preserve
  age+sex, **dx label-blind**) drives leakage 1.0 → 0.47 (chance) while the pooled
  AD-vs-CN signal **holds** 0.865 → 0.87. Because batch removal was label-blind, the
  surviving disease signal is **separable from site**, enabling **honest pooling** at
  n=360 (245 in the AD-vs-CN analysis, 41 AD).
- **Not brain-age.** Removing the age-linear component drops AD-vs-CN only 0.868 →
  0.844, and age alone predicts at just 0.602 → the signal is disease, not aging.
- **Beats the classical baseline.** NeuroJEPA 0.868 > nWBV atrophy 0.829 > age 0.602.
- **Survives multiple-comparison correction.** Under BH-FDR at q=0.05, AD-vs-CN,
  age-adjusted AD-vs-CN, and impaired-vs-CN all survive (p_perm 0.0005).

---

## 4. Pipeline status vs the architecture diagram (honest, per-layer)

| Layer | Component | Status |
|---|---|---|
| **Layer 1** | NeuroJEPA structural-MRI foundation | **BUILT + VALIDATED this session** (two cohorts, leakage-free, FDR-passing). |
| Layer 2 | U-Net (segmentation) | **Not built** — deliberately deprioritized. |
| Layer 3 | Fusion | **Offline contract ADAPTER, not a trained model.** |
| Layer 4 | Refinement Engine (referee) | **Mature deterministic core.** |
| Tabular feed | ADNI cognitive + plasma + demographics | Real ADNI **n=2,951** (cognitive + plasma p-tau217 / GFAP / NfL + demographics); plasma **triangulated ensemble wired this turn** (§5). |
| Layers 5/6/Output | PI4AD / AlphaFold / translation | **Real-evidence adapters + scaffold, NOT outcome-validated.** |

---

## 5. This turn — two integrations

### (a) Plasma-biomarker ensemble wired into the ADNI contract
The triangulated plasma ensemble (`data/plasma_ensemble.py`) fuses ADNI's three assays
— UPenn Fujirebio/Quanterix, C2N PrecivityAD2, Lilly MSD600 — into a z-harmonized
p-tau217 plus plasma Aβ42/40 and C2N %p-tau217. It is now **merged into the ADNI
contract anchor** so those markers actually gate promotion and route mechanism instead
of sitting standalone.

**What it does now**
- **Promotion gate:** the referee's `biomarker_anchor` (the HARD GATE in
  `scoring.build_claim_card`) correlates the OOF probe score against the now
  **triangulated, higher-coverage** `p_tau217` column, and additionally reads
  %p-tau217 as a fallback anchor (reported in stats / ledger / ATN).
- **Mechanism routing:** `bridge._route` (also used by
  `harness/orchestrator._mechanism_enrichment`) now folds Aβ42/40 (amyloid A-axis) and
  %p-tau217 (tau axis) into the p-tau217/amyloid-cascade dominance calculation.

**Files changed:** `src/neuroad/data/plasma_ensemble.py` (added `merge_into_contract` +
`MERGED_COLUMNS`), `src/neuroad/contract.py` (`EXTENDED_BIOMARKER_COLUMNS`, extended
coverage block), `scripts/build_adni_contract.py` (`--plasma-download`, degrades to
single-assay if absent), `src/neuroad/data/gated.py` (float64 coercion for extended
markers), `src/neuroad/claude/bridge.py` (routing), `src/neuroad/gauntlet.py`
(`pct_ptau217` correlation + tertiary fallback anchor).

**State:** all changes compile; **9/9** `test_plasma_ensemble` pass; regression sets
green (84 + 40 passing across `test_claude/gated/data/harness/translation_and_honesty`
and `test_engine/discovery/discovery_router/gauntlet_rigor/discovery_loop`).

**Caveats:**
- For ADNI, `p_tau217` is now **z-harmonized, not pg/mL** (intentional — anchor uses
  scale-free Pearson r, routing uses standardized effect size). Native pg/mL is
  retained in **`p_tau217_native`**. The contract docstring still says "pg/mL" —
  worth tidying. GFAP/NfL remain in native units.
- Ensemble coverage is a superset of the single-assay UPenn draw for the imaging
  cohort, so `combine_first` is effectively all-ensemble; the native branch only
  guards an impossible edge.
- The **real-data build path was not executed** here — it needs the gated LONI CSVs
  (outside this environment). Wiring is verified by compile/parse + synthetic tests;
  `test_real_data_smoke` stays skipped.

### (b) ADNI raw-MRI pipeline scaffold
Extends the embedding + analysis pipeline to ADNI. **No `src/neuroad/**`, `tests/**`,
`run_oasis_*.py`, or git was touched** beyond the additive changes below.

**Files created**
- **`scripts/build_adni_image_manifest.py`** — reads `data/real/_gated/adni.csv`,
  emits `data/real/_manifests/adni_image_manifest.csv` (**2,951 rows**:
  `subject_id, image_path, dx, age, sex, site, scanner, p_tau217, gfap, nfl, amyloid`).
  `image_path` is a documented **placeholder** (`ADNI_MRI/<subject>/T1.nii.gz`);
  `--image-root` / `--image-pattern` repoint it at the real download folder. Carrying
  the plasma+amyloid columns means the eventual embedding CSV lands **already joined**
  to biomarkers. Ran clean: plasma present for 1,366 subjects, amyloid for 2,028.
- **`scripts/run_adni_crosscohort.py`** — ready-to-run analysis (guards with a clear
  "ADNI embeddings not present — run the embed first" message; verified exit 1). Four
  blocks via `probe.auc_ci_perm`: (a) within-ADNI AD-vs-CN, (b) cross-scanner
  cohort-leakage OASIS-vs-ADNI (raw), (c) ComBat-harmonized pooled AD-vs-CN +
  site-disjoint cross-cohort CV, (d) biomarker anchoring — embedding → plasma p-tau217
  high/low (median split) and amyloid A+/A−. Writes
  `reports/adni_neurojepa_crosscohort.json`. Smoke-tested end-to-end on a synthetic
  768-d file against the real OASIS embeddings: harmonize drove raw leakage 1.0 → 0.40
  while pooled AD-vs-CN survived; both anchoring probes serialized.

**Files edited (additive, backward-compatible)**
- **`scripts/neurojepa_embed_colab.py`** — added `adni` to `--dataset`; ADNI branch
  does **no tarball fetch** (expects user-supplied local T1w NIfTIs) and warns if
  `--skull-strip` is omitted; `skull_strip_batch(..., dataset=)` derives the
  per-subject brain-file id from the immediate parent dir for ADNI's 2-level layout
  (parent-of-parent would collide every subject on the shared root). OASIS-1/OASIS-2
  paths unchanged. `py_compile` passes.

**State / assumptions:** ADNI embeddings assumed 768-d (same frozen space as OASIS,
confirmed 768-d); analysis intersects the shared `emb_*` set defensively. Embedding
CSV is assumed to carry manifest metadata through (embed loop copies non-image
columns). ADNI raw preprocessing mirrors OASIS-2 raw (deepbet + trilinear
fast-resample) for a shared representation space; residual scanner batch handled by
ComBat. ADNI is one cohort/site for leakage; OASIS files are optional (script degrades
to ADNI-only run).

**UPDATE — ADNI raw MRI is IN HAND (no longer blocked on the download).** The n=590
ADNI T1w collection is already on this machine (and mirrored in Google Drive folder id
`1Qd754tBNX-CfkjYG_fztdjVszIbdM8Jh`), as three zips in the PARENT dir
`/Users/stevenyang/Documents/claude-life-sciences-hack/`:
- `adni t1 mprage n=590.zip` (5.47 GB) — **406 unique subjects**, DICOM.
- `adni t1 mprage n=590_dataset.zip` (3.10 GB) — **184 unique subjects**, DICOM.
- `adni_t1_mprage_n=590_IDA_Metadata.zip` (678 KB) — per-scan `.xml` metadata.

The two image zips are **complementary halves (0 overlap → 590 unique subjects total)**.
Layout is standard IDA: `ADNI/<RID e.g. 000_S_0000>/<sequence>/<date>/<imageID>/*.dcm`.
Subject RIDs join directly to `data/real/_gated/adni.csv` (biomarkers). Sequences are
all **T1-weighted** (Accelerated Sagittal MPRAGE dominant, plus IR-FSPGR / Sagittal 3D
MPRAGE) — a Siemens/Philips/**GE multi-vendor 3T** mix, i.e. a genuine cross-scanner
test vs OASIS's 1.5T Siemens.

**Remaining work to embed (fresh-session task):**
1. **Format is DICOM, not NIfTI** — `dcm2niix` is NOT installed locally (`brew install
   dcm2niix`, or convert on the Colab runtime). Convert each subject's DICOM series →
   one T1w NIfTI. Recommend converting locally (CPU, fast) → ~590 small NIfTIs, then
   upload those to Colab (much lighter than the 8.5 GB of DICOM).
2. Point `build_adni_image_manifest.py --image-root <nifti_dir> --image-pattern ...` at
   the converted NIfTIs (join to biomarkers happens automatically).
3. Embed on Colab with `--dataset adni --skull-strip --fast-resample`. **n=590 is a big
   job** (~30+ min) for the fragile runtime → run in CHUNKS with `--resume` +
   `--checkpoint-every` (see the runtime-release learning in §7); harvest durable blobs.
4. Concat → `data/real/adni_neurojepa_embeddings.csv`, then
   `run_adni_crosscohort.py` for the cross-scanner + biomarker-anchoring report.

Until the NIfTIs are converted+embedded, `data/real/adni_neurojepa_embeddings.csv`
does not exist and `run_adni_crosscohort.py` exits with its guard message.

### Verify pass — VERDICT: GREEN
- pytest **254 passed / 2 skipped** (baseline 249/2 — five more passing, none failing).
- All `scripts/*.py` compile cleanly.
- `run_oasis_rigor.py` exit 0, wrote `reports/oasis_neurojepa_rigor.json` (all FDR
  checks PASS).
- `run_oasis_harmonized.py` exit 0, wrote `reports/oasis_neurojepa_harmonized.json`
  (verdict PASS).
- No regressions from either build. No fixes needed. Git was **not** run.

---

## 6. Next steps (prioritized)

1. **Embed the ADNI n=590 T1w cohort (images already in hand — see §5b UPDATE).**
   Convert DICOM→NIfTI (`dcm2niix`), point `build_adni_image_manifest.py` at the NIfTIs,
   then embed on Colab in chunks (`--dataset adni --skull-strip --fast-resample
   --resume --checkpoint-every 15`) → `data/real/adni_neurojepa_embeddings.csv`, then
   `run_adni_crosscohort.py`. This is THE highest-value next step: a large 3T multi-vendor
   third cohort → definitive cross-scanner generalization + plasma/amyloid biomarker
   anchoring (does the imaging embedding predict molecular pathology, not just clinical dx?).
2. **Execute the real ADNI contract build** (`scripts/build_adni_contract.py`) with the
   gated LONI plasma CSVs to exercise the newly wired triangulated ensemble on real
   data and confirm coverage numbers (currently only synthetic-tested).
3. **Fix the `p_tau217` "pg/mL" docstring** in `contract.py` to reflect that the ADNI
   column is now z-harmonized (native pg/mL lives in `p_tau217_native`).
4. **Grow the AD sample** — the pooled AD-vs-CN rests on 41 AD subjects; ADNI should
   materially strengthen the CIs.
5. Optionally revisit **Layer 2 (U-Net)** and turn **Layer 3 fusion** from an adapter
   into a trained model once the imaging feed is multi-cohort.

---

## 7. Blockers & operational learnings

- **ADNI T1w MRI is DUA-gated and manual** — must be pulled by the user from
  ida.loni.usc.edu (Image Collections → MRI/MPRAGE → DICOM), converted DICOM→NIfTI via
  dcm2niix. This is the one hard external blocker.
- **Modest AD sample** — 41 AD subjects in the pooled AD-vs-CN analysis. Results are
  honest but CIs are wide (OASIS-2 alone: [0.57, 0.87]). Treat headline numbers as
  encouraging-but-preliminary until ADNI lands.
- **Colab runtimes release on idle / websocket EOF.** The working pattern is
  **FOREGROUND exec** (keeps the kernel busy) plus **frequent durable checkpoints**
  (gzip+base64 stdout blobs) so a mid-run drop never loses embeddings. Use `--resume`.
- **`colab exec` does NOT forward argv** to `sys.argv` — a runpy wrapper is required,
  and script flags must be passed after `--`.
- **Secrets never as command literals.** `HF_TOKEN` rides as an **uploaded file**, is
  read ephemerally, and is never a command argument or committed literal. The
  pre-commit hook blocks token literals even under `git add -f`.
- **Skull-stripping does not fix cross-cohort leakage** — residual batch is spatial
  normalization / intensity / mask-method. ComBat (label-blind) is the tool that
  actually removes it while preserving disease signal.
- **Gated-weight license (CC-BY-NC-ND):** frozen inference producing embeddings is used
  as a non-derivative, non-commercial output; weights and embedding tables are
  gitignored and never published.

---

## 8. How to reproduce

### Re-run the rigor + harmonization reports on existing local embeddings
```bash
cd /Users/stevenyang/Documents/claude-life-sciences-hack/neuroad-discovery-engine
PYTHONPATH=src ./.venv/bin/python scripts/run_oasis_rigor.py
PYTHONPATH=src ./.venv/bin/python scripts/run_oasis_harmonized.py
# also available:
PYTHONPATH=src ./.venv/bin/python scripts/run_oasis_neurojepa.py
PYTHONPATH=src ./.venv/bin/python scripts/run_oasis_combined.py
```

### Regenerate OASIS embeddings from scratch (Colab T4)
See **`docs/COLAB_RUNBOOK.md`** for the full runbook (foreground exec + checkpoint
pattern, token-as-uploaded-file). Then:
```bash
# OASIS-1 (team-masked T88 MPRAGE):
python scripts/neurojepa_embed_colab.py -- --dataset oasis1 --fast-resample --resume --checkpoint-every 10
# OASIS-2 (raw mpr-1, deepbet skull-strip):
python scripts/neurojepa_embed_colab.py -- --dataset oasis2 --skull-strip --fast-resample --resume --checkpoint-every 10
# place results at data/real/oasis{1,2}_neurojepa_embeddings.csv
```

### ADNI flow (after the manual IDA/LONI download)
```bash
# 1) point the manifest at the real NIfTI folder (files at <root>/<subject_id>/T1.nii.gz)
./.venv/bin/python scripts/build_adni_image_manifest.py --image-root /data/ADNI_MRI

# 2) embed the T1w NIfTIs (Colab T4, per docs/COLAB_RUNBOOK.md):
#    upload manifest + script + hf_token, then:
python scripts/neurojepa_embed_colab.py -- \
    --dataset adni --skull-strip --fast-resample \
    --manifest data/real/_manifests/adni_image_manifest.csv \
    --id-col subject_id --out adni_neurojepa_embeddings.csv
#    -> place result at data/real/adni_neurojepa_embeddings.csv

# 3) cross-scanner + biomarker anchoring report:
PYTHONPATH=src ./.venv/bin/python scripts/run_adni_crosscohort.py
```

### Real ADNI contract build with the triangulated plasma ensemble
```bash
# needs the gated LONI plasma CSVs in the download dir:
PYTHONPATH=src ./.venv/bin/python scripts/build_adni_contract.py --plasma-download <gated_dir>
```

---

_Data files (`*embeddings*.csv`, volumes, weights, tokens) are git-ignored and local
only. Reports under `reports/` are tracked. Do not commit weights or embedding tables._
