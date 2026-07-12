# Session Handoff — Layers build-out (L2 probe, L3/L5/L6, AlphaFold-3)

Branch `feat/molecule-translation-loop`. **Nothing committed.** **Suite: 462 passed, 2 skipped.**
All work is offline-deterministic, honesty-stamped, additive (default `translate()` path is
byte-compatible; new capabilities are opt-in). Sibling handoffs exist for other workstreams
(`SESSION_HANDOFF_3.md` = discovery-half/LINCS; `SESSION_STATUS.md`, `SESSION_HANDOFF_2.md`,
`SESSION_HANDOFF_NEUROJEPA.md`). This one covers the **pipeline layers** work.

> A **separate session** owns the conversion embed (`run_conversion_embed_colab.py`) + the LONI
> raw-MRI pulls (AD 5× expansion + MCI). Do NOT duplicate that here.

---

## What this session built

### AlphaFold-3 / molecular targeting (L6) — now REAL
- Ran Boltz-2 (open, MIT, AF3-class) on Colab A100 → committed to
  `src/neuroad/integrations/data/boltz_snapshot.json` (scalars only, never coordinates):
  - Complexes: APOE|APP (conf 0.508), APP|BACE1 (0.587), APP|MAPT (0.373).
  - Affinities: ABL1::Nilotinib (−1.524, P 0.67), RXRA::Bexarotene (−0.613, P 0.93).
- `translation.py`: surfaces the first STRING partner **with** a folded result + attaches per-compound
  `boltz_affinity`; unfolded targets stay honest-`deferred`.
- Colab gotchas ([[neuroad-boltz-colab-gotchas]] + `docs/DATA_INGESTION_ETL.md`): `--no_kernels`
  (missing `cuequivariance_torch`); bake defaults (`colab exec` drops `-- argv`); **stream, don't
  capture** (silent long fold → websocket EOF); re-fold via `scripts/boltz_fold_colab.py`.

### Attentive MLP probe (L2, replaces U-Net) — `src/neuroad/attentive_probe.py`
- `MLPProbe` via a new `probe.probe_factory` hook → identical site-disjoint CV / bootstrap /
  permutation / `n_repeats` machinery. `evaluate()` + `feature_grounding()` (LOO attribution).
  Wired as `signal_grounding` (opt-in `include_grounding`).
- **Chose MLP over U-Net** (matches NeuroVFM frozen-encoder+MLP; no new data; FastSurfer gives
  volumes). See `../pipeline.txt` L2 note. MLP ≈ linear at n=590 (near-linearly separable) — reported honestly.

### L3 cross-attention fusion — `src/neuroad/integrations/cross_attention.py`
- Real numpy multi-head scaled-dot-product cross-attention (fixed feature transformer) → reused head.
  NOT the vkola transformer, NOT trained end-to-end (honest). `include_cross_attention`. 18 tests.
- `multimodal_transformer.py` (vkola ADRD) now **wired** via `_biomarker_fusion(df)` → `biomarker_fusion`
  (surrogate default; auto-upgrades to real weights with `NCOMMS2025_CKPT`/clone — GPU path).

### L5 pathway enrichment — `src/neuroad/integrations/pathway_enrichment.py` + `data/ad_pathway_genesets.json`
- Hypergeometric ORA + BH-FDR over a curated AD pathway snapshot. Amyloid → "Alzheimer disease" top
  (q=0.008). `include_pathways`. 17 tests.

### L6 target druggability — `src/neuroad/integrations/targeting.py`
- Fuses AlphaFold pLDDT + Boltz complex conf + Boltz affinity + PI4AD → transparent per-target score,
  renormalized over present components (absent ones marked, never faked). BACE1 #1 (0.731).
  `include_targeting`. 17 tests.

### Data / probe / demo
- `probe.py`: **repeated-CV OOF ensembling** (`n_repeats`, `N_REPEATS_ENSEMBLE=8`) + `probe_factory`.
  `pipeline.py:_naive_effect` headline AUC uses the 8-ensemble.
- `data/adni_jepa.py` + `adni:neurojepa` loader: real **590 ADNI** (87 AD/503 CN, multi-site, real
  plasma) → AD-vs-CN **AUC 0.857**. `apoe4` joined from gated `APOERES` (560/590) so the plasma block
  is complete (lets L3 fusion run on this cohort).
- `scripts/build_cohort_crosswalk.py`: reusable IDA-metadata → `RID,PTID,IMAGEUID` (tested on MCI:
  334 subjects). AD worklist: `data/real/_manifests/ad_expansion_imageid_worklist.txt` (375 new AD IDs).
- `app/build_demo_data.py` → `app/demo_data.json` regenerated with all new layers surfaced.
  **Frontend `app/index.html` UNTOUCHED** (data loaded, no UI clutter). OASIS (no plasma) correctly
  omits cross-attention. Build ~4 min now (cohort-level CV per plasma cohort); one-off.

---

## Key data facts (verified)
- **590 ADNI** real, 87 AD / 503 CN, **no MCI**, one image/subject.
- Manifest: **2,951 (462 AD, 1,299 MCI, 1,153 CN)**. AD 5×: 375 new AD available (182 3T / 193 1.5T);
  clean 3T-only ≈ 3× (269), full 5× needs 1.5T + `adni:combat` ComBat.
- MCI-conversion download: 334 subjects, raw DICOMs local, crosswalk built — ready to embed.
- **Consistent across 3 methods (grounding, concat fusion, cross-attention): plasma p-tau217 dominates
  AD-vs-CN (~0.93); imaging adds little on top.** Imaging's real value is the **conversion** arm
  (sMCI→pMCI) — what the other session's MCI embed unlocks.

## Architecture Q answered this session
`demo_data.json` is the **frontend data seam, NOT a DB tier.** Recommendation: **frozen JSON for the
hackathon** (deterministic → demo can't fail live); real DB (Firestore for JSON docs) only when users
generate data worth persisting; GCS-blob-per-run is the middle path; `server.py` live-compute + JSON
fallback if "looks canned" is a concern.

---

## Next steps (backend; NOT the other session's embed/LONI work)
1. **Commit** — 462 green on `feat/molecule-translation-loop`, nothing committed (177 uncommitted files,
   many from sibling sessions). Coordinate a commit/PR with the other workstreams.
2. **Real vkola transformer on Colab** — currently surrogate; run the real ADRD model (adrd = conda/git-LFS
   pkg, GPU) → per-subject amyloid/tau snapshot, mirror the Boltz pattern; loader auto-upgrades.
3. **When MCI/AD embeddings land (other session):** wire `adni:mci` / expanded feeders (mirror
   `adni_jepa.py`), re-validate, run **cross-cohort / leave-one-cohort-out** — the honest single-cohort
   fix, and where imaging should finally beat plasma (conversion arm).
4. **Optional:** faster demo build (lower cross-attention `n_boot/n_perm`); `server.py` live-compute;
   GCS-blob run persistence.

## Pointers
- Opt-in `translate()` kwargs (all default False; demo build turns them on): `include_grounding`,
  `include_cross_attention`, `include_targeting`, `include_pathways`.
- ETL: `docs/DATA_INGESTION_ETL.md`. Boltz re-fold: `scripts/boltz_fold_colab.py`.
- New tests: `test_attentive_probe.py`, `test_cross_attention.py`, `test_pathway_enrichment.py`,
  `test_targeting.py`, `test_adni_jepa_and_fusion.py`, `test_probe_ensemble.py`, `integrations/test_boltz.py`.
