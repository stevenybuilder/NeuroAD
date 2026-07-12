# Session Handoff #3 — ADNI Imaging Complete, Live Discovery Validation, Decoder Built

_Last updated: 2026-07-11. Tests: **404 passed / 2 skipped** (green; +16 this session's
discovery-half work). Continues `docs/SESSION_HANDOFF_2.md` (read it for Layer-1/plasma
context)._

---

## 0. ADDENDUM (this session) — Discovery-half data-science hardening (backend-only)

Strengthened the **discovery half** across 4 tracks, all **read-only side artifacts** —
the frozen seam (`translation.translate()` schema, default `method="pi4ad"`, `agent.py`,
`app/*`, `app/demo_data.json`) was **not touched** (verified: seam files predate every
edit; `app/demo_data.json` unchanged → demo surfaces byte-identical output). Live results:

| Track | What | Key honest result |
|---|---|---|
| **4. Conclusive temporal test** | Paginated OT past 200 + genome-scale universe; **rebuilt NOVEL_2022 = the complete, source-verified 41-gene Bellenguez-2022 Table-2 new-loci set** (the old 20-gene subset wrongly included TSPOAP1/CCDC6/TMEM163) | Underpowered null → **properly-powered null**: novel-in-universe 4→27 (OT) & 5→13 (network). Clean OT non-genetic AUC **0.506** (n=27); network 0.587 (p=0.14). Circular ceilings only 0.63–0.66 ⇒ most retrospective "signal" is genetic circularity. |
| **1. Statistical rigor** | Bootstrap AUC CIs, BH-FDR, **negative controls** (housekeeping decoys + degree-matched network null), shortlist rank-stability | Clean OT-nongenetic **AUC 0.845 [0.783,0.908], q=0.0015**, decoy at chance (0.54, ns) ⇒ **specific**. **PI4AD scores decoys at 0.872** ⇒ non-specific (hub bias) — controls caught it. Network novel-AUC only marginally above degree-matched null (0.489, emp_p=0.073). |
| **3. LINCS efficacy axis** | New keyless **SigCom LINCS** integration: does a gene's KO **reverse** an AD signature? Orthogonal perturbational/efficacy axis; committed snapshot (1567 genes) | Honest **null** (efficacy vs GWAS 0.583/p=0.27; vs drugs 0.372; decoy ns) — a cancer-line proxy, correctly weak. A genuinely orthogonal feature for the learned ranker. |
| **2. Learned ranker** | New calibrated logistic model: rank-norm features (incl. LINCS), LOO-CV, learned weights replace hand-set | **OOF AUC 0.740 [0.626,0.848], p=0.001, Brier 0.024.** Learned weights recover **OT-heldout as the workhorse (+0.877)**; correctly down-weights LINCS (−0.89) & imputed pLDDT. Multi-signal (0.74) does **not** beat OT alone (0.845) ⇒ one clean signal carries the discovery half. |

**New code (all off-seam):** `integrations/lincs.py`, `harness/ranking_model.py`;
scripts `run_discovery_rigor.py`, `build_lincs_signature.py`, `rank_candidates_learned.py`.
**Edited (off-seam):** `harness/validation.py` (bootstrap CI / BH-FDR / degree-matched
null / DECOY_GOLD / rebuilt NOVEL_2022 / `validate(with_ci=)`), `integrations/opentargets.py`
(`disease_targets` pagination for `top_n>200`; ≤200 path byte-identical), `run_temporal_validation.py`.
**New reports:** `discovery_rigor`, `lincs_efficacy`, `candidate_ranking_learned`; regenerated
`temporal_validation`. **New docs:** `LINCS_SPEC.md`, `LEARNED_RANKER_SPEC.md`. **New snapshot:**
`integrations/data/lincs_ad_reversal_snapshot.json`. **+16 tests** (`test_lincs.py`,
`test_ranking_model.py`, rigor tests in `test_validation.py`).

**Reproduce (live; no GPU, keyless):**
```bash
PYTHONPATH=src ./.venv/bin/python scripts/run_temporal_validation.py       # Track 4
PYTHONPATH=src ./.venv/bin/python scripts/run_discovery_rigor.py            # Track 1
PYTHONPATH=src ./.venv/bin/python scripts/build_lincs_signature.py --limit 1000   # Track 3
PYTHONPATH=src ./.venv/bin/python scripts/rank_candidates_learned.py --top-n 600  # Track 2
```

### Boltz-2 / 5th ranking signal (handoff §7 item 3) — DONE (GPU run by a parallel session)

`boltz_snapshot.json` was populated by a real Colab GPU run earlier (3 complexes:
APP–APOE 0.508 / APP–BACE1 0.587 / APP–MAPT 0.373; 2 affinities: Nilotinib–ABL1,
Bexarotene–RXRA — confidence+affinity scalars only). `translation.translate` already
surfaces it (`boltz_targeting`; wired by that session). Re-ran `scripts/rank_candidates.py`
so the **5th signal now lights up in `reports/candidate_ranking.md`** (APP leads at 5-signal
composite 0.782; APP/BACE1 boltz 0.587, APOE 0.508, MAPT 0.373; unfolded genes honestly
`None`). **Not done (demo owner's call, keeps frozen seam):** `app/demo_data.json` predates
the translation boltz-wiring and does NOT yet surface boltz — run `python app/build_demo_data.py`
to show it in the demo. Learned ranker left at 5 features (boltz coverage is ~4/600 genes —
too sparse to train a weight; it belongs in the shortlist-focused hand composite).

### Item 2 (imaging → conversion fusion) — IN PROGRESS, resumable with NO re-download

334 MCI-conversion T1 scans were pulled from IDA and everything is **durable on GCS**
(`gs://neuroad-adni-project-flash-490419/adni_conversion/`): `raw/part1.zip`+`part2.zip`
(the DICOM), `raw/*_IDA_Metadata.zip`, and `inputs/{crosswalk.csv, manifest_full.csv,
neurojepa_embed_colab.py (fixed), adni_colab_dicom_to_embed.py}`. The pipeline is PROVEN:
a run converted **all 334 → NIfTI** and built the manifest; it only failed on Colab infra
— (1) deepbet `fill_voids` dep (**fixed** in `neurojepa_embed_colab.py` + re-staged to GCS),
then (2) the runtime was **reclaimed mid-embed** (~15-min reclaim + websocket EOF during the
silent skull-strip). **Lesson:** a one-shot embed can't survive the reclaim — needs the
resumable auto-restart harness (checkpoint NIfTIs + embeddings to GCS per-unit, keepalive
output, `colab start`+resume loop on each death; see `docs/DATA_INGESTION_ETL.md` §5).

**To finish (no A100 needed to START planning; the embed step needs a GPU cycle):**
`scripts/run_conversion_embed_colab.py` is the GCS-native driver — make it resumable
(NIfTI+CSV → GCS per-unit) and wrap in the auto-restart loop, then the output lands at
`adni_conversion/adni_conversion_neurojepa_embeddings.csv`. Once that CSV exists, fusion is
CPU-only: `attention_fusion(conversion_df, imaging_embedding=<that CSV as emb_* frame>)` on
the 334-subject slice (58 converters — **underpowered**, expect inconclusive vs plasma).

### Still blocked (handoff §7)
- **Item 2** (fold imaging → L3 **conversion** fusion): infeasible as data-plumbing — the
  590 NeuroJEPA embeddings are the AD/CN cohort (0 subject_id overlap with the 498 MCI
  conversion+plasma cohort) and use a different ID space (RID vs re-indexed) with no join
  key. Real fix = **embed the MCI conversion subjects' MRIs via NeuroJEPA (a GPU job)**.
- **Item 5** (AIBL/NACC): needs the user to run the DUA-gated LONI/IDA download.
- **Item 6** (decoder labeling+training): GPU + FastSurfer-on-Colab labeling quagmire — defer.

This session: (a) **finished the ADNI imaging embed** (the #2 handoff's blocked item)
and proved its science; (b) ran the **discovery half fully LIVE** and got an honest,
nuanced verdict; (c) built the **NeuroJEPA-conditioned U-Net decoder** (L2 Option B)
and wired a **multi-signal composite ranker**; (d) stood up **GCS storage**; (e) did a
**power analysis** and **temporal validation**; and (f) hit — and documented — a hard
wall on decoder *labeling* under free-Colab constraints.

---

## 1. TL;DR — what to know first

- **ADNI imaging embed is DONE.** All 590/590 subjects embedded (NeuroJEPA 768-d).
  Final CSV on GCS **and** local (`data/real/adni_neurojepa_embeddings.csv`).
- **Cross-cohort now validly pools** (the whole point of the embed): ComBat cohort
  leakage dropped **1.0 → 0.56**, pooled AD-vs-CN **AUC 0.861** (n=835, 128 AD),
  imaging embedding anchored to p-tau217/amyloid. `reports/adni_neurojepa_crosscohort.json`.
- **Discovery half, live verdict:** the one CLEAN non-circular signal is Open Targets
  held-out-non-genetic **AUC 0.728, p=0.003**; PI4AD-vs-GWAS 0.869 is strong but
  **residually circular**; drug-target prediction is at chance. It is a
  **hypothesis engine**, now *earned* rather than asserted. `reports/target_prioritization_validation.md`.
- **Constraint that shaped everything:** user runs on **paid Colab Pro+ units only,
  NO dollar spend**. GCE GPU quota is 0; trial credits expired. Colab CLI runtimes get
  **reclaimed at ~10–22 min even while busy**, and only **one GPU runtime at a time**.
- **Blocked:** the NeuroJEPA-conditioned decoder **labeling** (FastSurfer-on-Colab dep
  quagmire). Recommend **defer** — model + code + spec are done and ready for a stable GPU.
- **All background jobs are STOPPED. No active Colab runtime.** Nothing is running.

---

## 2. Headline results (this session)

| Result | Value | Where |
|---|---|---|
| ADNI imaging embed | 590/590 (503 CN + 87 AD) | GCS + `data/real/adni_neurojepa_embeddings.csv` |
| Cross-cohort ComBat leakage | 1.0 → **0.56** (valid pooling) | `reports/adni_neurojepa_crosscohort.json` |
| Pooled imaging AD-vs-CN | **0.861** [0.829, 0.892], n=835, 128 AD | same |
| Within-ADNI imaging AD-vs-CN | 0.848 [0.805, 0.888], n=590, 87 AD | same |
| **Discovery: OT held-out non-genetic (CLEAN)** | **AUC 0.728, p=0.003** | `reports/target_prioritization_validation.md` |
| Discovery: PI4AD-vs-GWAS (residually circular) | 0.869, p=0.001 (CAVEAT flagged) | same |
| Discovery: drug-target held-out | 0.516 (at chance) | same |
| Composite narrowed shortlist | **APP, TREM2, BIN1** | `reports/candidate_ranking.md` |
| Power: converters for 80% power (fusion vs plasma) | **~650–710** (total n≈2,300–2,500) | `reports/power_analysis_conversion.md` |
| Temporal validation (widened, 41-gene gold) | **properly-powered null** — network 0.587 (p=0.14, 13/41), OT non-genetic 0.506 (p=0.46, 27/41); circular comparators 0.63–0.66 (p<0.01) | `reports/temporal_validation.md` |
| AlphaFold | proven LIVE/keyless (EBI AF DB); live UniProt resolution | `run_target_prioritization_validation.py` |

---

## 3. What was built (files)

**New src modules:**
- `integrations/gcs_store.py` — GCS helper (headless ADC auth; upload/download/exists/try_download/list).
- `integrations/neurojepa_decoder.py` — **L2 Option B model**: raw-MRI 3D U-Net (anatomical
  skips) + FiLM conditioning on the 768-d NeuroJEPA embedding. **GPU-verified**: 5.7M params,
  2.8–6.6 GB, correct output shapes + volume readout. (Torch at import; intentionally NOT in
  `integrations/__init__` so the offline test suite stays torch-free.)
- `harness/ranking.py` — shared 4-signal composite ranking helper.
- `harness/validation.py` — **edited**: added `add_nodes` STRING neighbor expansion (via
  `pi4ad.py`), and `KNOWN_2019` / `NOVEL_2022` temporal gold sets.
- `integrations/pi4ad.py` — **edited**: `add_nodes` threaded through `_fetch_string_live` /
  `fetch_string_subgraph` / `propagate_hits` (default 0 = unchanged).
- `harness/translation.py` — **edited**: `_rank_targets(method='composite')` opt-in (default unchanged).

**New scripts:**
- `scripts/run_target_prioritization_validation.py` — LIVE full-universe discovery validation.
- `scripts/rank_candidates.py` — 4–5-signal composite candidate ranker (5th = optional Boltz).
- `scripts/power_analysis_conversion.py` — DeLong + bootstrap power analysis.
- `scripts/run_temporal_validation.py` — prospective novel-target validation.
- `scripts/prep_decoder_data.py` — decoder data-prep (FastSurfer labels → .npz → GCS). **BLOCKED, see §6.**
- `scripts/train_neurojepa_decoder.py` — resumable decoder trainer (GCS checkpoint/resume).
- `scripts/run_decoder_training_loop.py` — local auto-restart driver for training.

**New docs:** `docs/L2_UNET_SPEC.md`, `docs/RANKING_METHODOLOGY.md`,
`docs/TEMPORAL_VALIDATION_SPEC.md`, `docs/DATA_EXPANSION_SPEC.md`, `docs/L3_FUSION_SEAM.md`,
`docs/EXECUTION_STATUS.md`.

**Reports (tracked):** `adni_neurojepa_crosscohort.json`, `target_prioritization_validation.{json,md}`,
`candidate_ranking.{json,md}`, `power_analysis_conversion.{json,md}`, `temporal_validation.{json,md}`.

---

## 4. Infrastructure & hard-won lessons (READ before touching Colab)

- **GCS bucket:** `gs://neuroad-adni-project-flash-490419` (us-west1). Auth on the runtime
  via the user's ADC file uploaded as `/content/adc.json` +
  `GOOGLE_APPLICATION_CREDENTIALS`. (Service-account keys are **org-policy-blocked**.)
  - `adni/nifti/<rid>/T1.nii.gz` — 590 converted NIfTI (durable).
  - `adni/adni_neurojepa_embeddings.csv` — final embeddings.
  - `decoder_data/<sid>.npz` — decoder training triples (currently EMPTY — labeling blocked).
- **Colab CLI reality:** runtimes reclaimed at **~10–22 min even while actively computing**;
  **only ONE GPU runtime at a time** (starting a second reclaims the first). ~1792 units left,
  ~1/hr burn — effectively unlimited, but **no dollar/GCE spend allowed**.
- **The winning pattern for long jobs** (used for the ADNI embed): *incremental
  GCS-resumable bootstrap* (checkpoint each unit — NIfTI/partial-CSV — to GCS) + a local
  **auto-restart loop** that re-launches on each death + a **45s keepalive** poll. Cumulative
  progress converges over cycles. See scratch scripts referenced in git history / this session.
- **Detached process gotcha:** launch with `start_new_session=True` (else the detached child
  gets SIGINT when the `colab exec` cell completes — this was the misdiagnosed "§7 numpy blocker").
- **Blocking `colab exec` does NOT stream a subprocess's stdout** and drops its websocket at the
  end — don't rely on it for long jobs.
- **deepbet `--no-deps` misses `fill_voids` / `connected-components-3d` / `fastremap`** — install
  them explicitly or `run_bet` fails mid-run.
- **Keep Colab's numpy 2.0.2** — do NOT pin `numpy<2` (it forces a re-resolve that trips pip's
  version parser on Colab's stack).
- **venv quirk:** the repo's `.venv/bin/pip` has a stale shebang (project was renamed). Use
  `./.venv/bin/python -m pip`. Run tests: `PYTHONPATH=src ./.venv/bin/python -m pytest -q`.

---

## 5. Pipeline layer status (updated)

| Layer | Status | Notes |
|---|---|---|
| Input: Raw MRI | done (T1w) | 590 ADNI on GCS. **T2w/FLAIR unused** (aspirational). |
| L1 NeuroJEPA | **validated + ADNI done** | cross-cohort pooling now valid |
| Tabular (plasma) | validated | p-tau217 dominant (0.814); cognitive *scores* unused (CDR used for labels only) |
| L2 U-Net decoder | **model built + verified; labeling BLOCKED** | FastSurfer volumes (Option A) serve L2 today; NeuroJEPA-conditioned decoder (Option B) built, awaiting a stable GPU |
| L3 Fusion | mature; seam ready | imaging-embedding + decoder-volume seam documented (`docs/L3_FUSION_SEAM.md`); imaging NOT yet folded into the conversion fusion |
| L4 Refinement | mature | unchanged |
| L5 PI4AD | **validated live** | full 14,676-gene table; live STRING neighbor-expanded RWR surfaces real AD hubs |
| L6 Molecular targeting | AF DB live; **Boltz fold deferred** | `boltz_snapshot.json` empty → 5th ranking signal honestly `None` |
| Output prioritization | **validated live** | composite ranker wired into referee |

---

## 6. BLOCKED: decoder labeling (why, and the recommendation)

The NeuroJEPA-conditioned decoder needs voxel-level segmentation labels from FastSurfer.
On free Colab this hit a **dependency quagmire**, fixed one layer at a time only to reveal
the next: (1) `numpy<2` pin crashed pip → fixed (per-package `--no-deps`); (2) `from
neuroad.integrations import gcs_store` failed (flat-staged) → fixed (flat fallback);
(3) **FastSurfer's install broke torch's CUDA** — `libcusparseLt.so.0: cannot open shared
object file` → torch fell back to CPU → deepbet/FastSurfer unusable. This is on top of the
~20-min runtime instability (per-cycle FastSurfer install + weights download + 200-NIfTI
pull can exceed the runtime lifetime).

**Recommendation: DEFER.** The model, `train_neurojepa_decoder.py`, `run_decoder_training_loop.py`,
`prep_decoder_data.py`, and `docs/L2_UNET_SPEC.md` are all done and ready to run on a **stable
GPU** (a ~$2–3 spot GCE L4 does the whole label+train cleanly in ~3 h — currently ruled out by
the no-$ constraint). FastSurfer volumes (Option A) already give L2 real volumes, so the
pipeline is not blocked on this — it's the research add-on.

---

## 7. Recommended next steps (prioritized)

**Unblocked, cheap, high-value (do these first — no GPU, no $):**
1. **Temporal validation — DONE and CONCLUSIVE.** Widened (`--add-nodes 1000 --ot-top-n 2000`
   + PI4AD full-table comparator) over the complete **41-gene** `NOVEL_2022` set → a
   **properly-powered null**: no clean non-circular signal reaches p<0.05 (network 0.587;
   OT non-genetic 0.506); circular comparators (0.63–0.66) confirm the retrospective signal is
   genetic circularity. **Interpretation:** the engine is a validated hypothesis engine for
   KNOWN biology, NOT a demonstrated novel-target anticipator. If you want to push efficacy,
   the remaining move is the **LINCS pivot** (item 4).
2. **Fold the ADNI imaging embedding into the L3 conversion fusion** (seam exists per
   `docs/L3_FUSION_SEAM.md`) — may improve the multimodal MCI→AD result; minimal compute.

**One Colab job each (Colab-resumable pattern; watch the dep pitfalls in §4):**
3. **Boltz-2 fold** (`scripts/boltz_fold_colab.py`) → populate `boltz_snapshot.json` → lights up
   L6 + the 5th ranking signal. (Boltz also had install friction — budget for it.)

**Pivot (bigger, more ambitious — if temporal stays null after widening):**
4. **LINCS L1000 connectivity mapping** — genome-scale efficacy proxy (does a gene's knockdown
   reverse the AD signature?). ~1–2 day integration. The honest path toward *efficacy*, not just
   association.

**Needs the user (external, DUA-gated):**
5. **Data expansion** — download **AIBL** (LONI, same portal as ADNI) then **NACC** for
   converters-with-plasma toward the ~650–710 target. `docs/DATA_EXPANSION_SPEC.md`. The user
   initiates downloads; wire ingestion/ComBat/embed on our side. **Note:** ADNI is maxed at 142
   converters-with-plasma (270 more exist but lack plasma).

**Deferred (needs stable GPU):**
6. Decoder labeling + training (§6).

---

## 8. Honest framing for the writeup / demo (unchanged, now earned)

- **Predictive half (L1–L4):** real, cross-cohort-validated structural-MRI + plasma AD
  classifier/prognostic model. Imaging now pools across 3 cohorts (leakage 0.56). MCI→AD
  conversion ~0.83 fused — but the fusion advantage over plasma is only **+0.012 AUC** and
  needs ~5× more converters to prove; **plasma p-tau217 (0.814) is the workhorse**.
- **Discovery half (L5/L6/Output):** a **provenance-honest, wet-lab-testable hypothesis engine**
  with ONE clean non-circular signal (OT held-out 0.728). NOT an efficacy predictor. Do not oversell.
- One-liner: *"A cross-cohort-validated structural-MRI + plasma AD classifier/prognostic model
  feeding a provenance-honest, wet-lab-testable hypothesis engine."*

---

## 9. How to reproduce the key results

```bash
cd neuroad-discovery-engine
# LIVE discovery validation (network fetches; no GPU):
PYTHONPATH=src ./.venv/bin/python scripts/run_target_prioritization_validation.py
# composite candidate ranking (LIVE):
PYTHONPATH=src ./.venv/bin/python scripts/rank_candidates.py
# power analysis (reproduces OOF scores; ~3 min):
PYTHONPATH=. ./.venv/bin/python scripts/power_analysis_conversion.py
# temporal validation (LIVE):
PYTHONPATH=src ./.venv/bin/python scripts/run_temporal_validation.py
# cross-cohort with the ADNI imaging embeddings (already in data/real/):
PYTHONPATH=src ./.venv/bin/python scripts/run_adni_crosscohort.py
# full suite:
PYTHONPATH=src ./.venv/bin/python -m pytest -q      # 383 passed / 2 skipped
```

_Data files (`*embeddings*.csv`, `_gated/adni.csv`, GCS objects, tokens) are git-ignored /
local / in GCS. Reports under `reports/` and docs under `docs/` are tracked. Nothing was
committed this session — changes are in the working tree._
