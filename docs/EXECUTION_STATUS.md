# Execution Status

_Last updated: 2026-07-11_

Summary of what the plan-execution workflow delivered, split by **code-complete
and runnable now** vs **needs a Colab GPU run**. Honesty invariants (no fabricated
data, missing signal -> `None` + provenance, offline-first adapters) hold across
every item below.

## Test suite

- **372 passed, 2 skipped** (`PYTHONPATH=src ./.venv/bin/python -m pytest -q`, ~72s).
- Baseline was 365 passed / 2 skipped; the workflow added **7 tests** (the new
  `tests/test_composite_ranking.py`).
- **No regressions.** The two skips are pre-existing (unrelated to this workflow).

## Code-complete and runnable now (no GPU required)

| Item | Deliverable | How to run / verify |
| --- | --- | --- |
| Composite referee | `src/neuroad/harness/ranking.py` (new shared 4-signal helper), `translation._rank_targets(..., method='composite')` (opt-in, default `'pi4ad'` unchanged), `tests/test_composite_ranking.py` | `PYTHONPATH=src pytest tests/test_composite_ranking.py` (7 passed). Offline live check: `_rank_targets('glial', method='composite')` yields scored, provenance-stamped rows (e.g. TREM2 comp=0.7258, n=4). |
| Boltz 5th signal | `scripts/rank_candidates.py` — optional `boltz_confidence` weight (0.15), gated on `BZ.has_precomputed_results()` | `scripts/rank_candidates.py --offline` runs; boltz=None for all genes (committed snapshot intentionally empty), 4-signal composites byte-identical to `reports/candidate_ranking.md`. Producing real Boltz values needs the Colab fold job (below). |
| L3 fusion seam | `docs/L3_FUSION_SEAM.md` (audit doc) | Seam already exists + is complete in `src/neuroad/integrations/fusion.py`; `PYTHONPATH=src pytest tests/test_fusion.py` -> 27 passed. Doc describes how to feed the 768-d NeuroJEPA embedding and decoder volumes as a gated modality. No code change. |
| Training-loop driver | `scripts/run_decoder_training_loop.py` — local unattended orchestration driver | Syntax + argparse verified (`--help`). It *drives* Colab but runs locally; end-to-end requires the GPU + prepped data below. |

## Needs a Colab GPU run

| Item | Deliverable | Blockers | Notes |
| --- | --- | --- | --- |
| ADNI NeuroJEPA embeddings | `scripts/neurojepa_embed_colab.py` (pre-existing) | Colab GPU, gated HF backbone, DUA-gated ADNI T1w | **Already running** (per workflow context). Feeds the embeddings CSV consumed by decoder data-prep. |
| Decoder data-prep | `scripts/prep_decoder_data.py` (new) | Colab GPU, deepbet + FastSurfer, ADNI T1w NIfTIs, GCS creds | Builds per-subject `.npz` triples (mri[1,D,H,W], jepa[768], label[D,H,W] to `neurojepa_decoder.LABELS`) and uploads to `gs://neuroad-adni-project-flash-490419/decoder_data/<sid>.npz`. Resumable (uploaded npz is the checkpoint); honest per-subject skip on any missing input. Locally verified: syntax, offline `--help`, pure `remap_aseg_to_labels`, `gcs_store` import. |
| Decoder training | `scripts/run_decoder_training_loop.py` (new) driving `scripts/train_neurojepa_decoder.py` (pre-existing) | Colab GPU + `decoder_data/*.npz` present in GCS | Trains across Colab runtime reclaims using only Colab units; resumes from GCS checkpoint. Aborts honestly (exit 2) if GCS holds no `*.npz`. Depends on decoder data-prep completing first. |
| Boltz-2 fold | `scripts/boltz_fold_colab.py` (pre-existing) producing `boltz_snapshot.json` | Colab GPU | Only this run populates the 5th ranking signal; until then `rank_candidates.py` honestly reports boltz=None. |

## Dependency order for the Colab runs

1. ADNI embeddings (running) -> embeddings CSV.
2. `prep_decoder_data.py` -> `decoder_data/*.npz` in GCS.
3. `run_decoder_training_loop.py` -> trained decoder checkpoint in GCS.
4. `boltz_fold_colab.py` (independent) -> populates the optional 5th signal.

## Known non-blocking notes (disclosed by build agents)

- `ranking.py` docstring says the 4-signal core was "factored out ... no
  duplication", but `scripts/rank_candidates.py` still keeps its own copy (now a
  5-signal Boltz variant). The 4 core weights match; this is a doc-accuracy nit,
  not a defect.
- Decoder data-prep MRI/label voxel alignment relies on deepbet preserving the T1
  affine and FastSurfer seg sharing world space — standard but unverified on real
  ADNI volumes; subjects are honestly skipped rather than fabricated if outputs are
  missing.
- `fusion.py` / `tests/test_fusion.py` are untracked in the working tree, so the
  "already existed" claim can't be git-confirmed against a committed baseline; 27
  passing tests corroborate the seam is real and complete.
