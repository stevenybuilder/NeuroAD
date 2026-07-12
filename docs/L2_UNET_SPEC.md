# L2 — Structural Segmentation ("U-Net Decoder") Integration Spec

_Status: DRAFT v1 (2026-07-11). Owner: pipeline. Consumes L1 (NeuroJEPA) + raw
MRI; feeds L3 (multimodal fusion)._

## 1. What L2 is for

In the pipeline diagram L2 is the **U-Net decoder (few-shot 3D biomarker
segmentation)** that takes L1's NeuroJEPA latent embedding plus **anatomical skip
connections** from the raw MRI and emits **exact hippocampal & cortical volumes**.
Those volumes are the interpretable, clinically-anchored structural biomarkers
(hippocampal atrophy, ventricular enlargement, cortical thinning) that join the
tabular contract and flow into L3 fusion alongside plasma p-tau217.

The point of L2 is **grounding**: L1's 768-d embedding is powerful but opaque;
L2 turns the same anatomy into named mm³ volumes a neurologist can read and that
map onto the established AD atrophy signature.

## 2. Current state (what actually ships today)

`integrations/structural_segmenter.py` provides two honest layers:

1. **`parse_aseg_stats(path)`** — pure, offline, deterministic parser: a
   FreeSurfer/FastSurfer `aseg.stats` → normalized volume dict on the contract
   schema (`hippocampal_volume`, `ventricle_volume`, `whole_brain_volume`,
   `cortex_volume`, `intracranial_volume`). Fully tested now.
2. **`segment_volume(nifti)`** — the GPU path: shells out to **FastSurfer**
   (`run_fastsurfer.sh --seg_only`, Apache-2.0, ~1 min/volume on a GPU), parses
   its `aseg.stats` via layer 1. Honestly degrades to `None` (never fabricates
   volumes) when torch/GPU/FastSurfer is absent.

**The gap vs the diagram:** today's L2 is **FastSurfer** — a *separate* pretrained
segmentation CNN — not a decoder conditioned on the frozen NeuroJEPA encoder with
skip connections. FastSurfer gives real, validated volumes; it just doesn't reuse
L1's representation. That is a legitimate substitute, and the honest framing is:
**FastSurfer is the shipping L2; the NeuroJEPA-conditioned decoder is the research
extension.** This spec covers both and the seam between them.

## 3. Two designs, and the recommendation

### Option A — FastSurfer-as-L2 (ship now, validated)
- **What:** run FastSurfer `--seg_only` per subject → `aseg.stats` → volumes.
- **Pros:** real Apache-2.0 model, FreeSurfer-compatible output, no training, no
  labeled data needed, already scaffolded. Volumes are directly comparable to the
  ADNI `WholeBrain`/`Hippocampus` fields the contract already uses.
- **Cons:** doesn't reuse L1; ~1 min/volume GPU; another model dependency.
- **Recommendation:** **this is L2 for the milestone.** It makes the "exact
  hippocampal & cortical volumes" box real and validated.

### Option B — NeuroJEPA-conditioned U-Net decoder (research extension)
- **What:** a 3D U-Net decoder whose **bottleneck is the frozen NeuroJEPA
  encoder's multi-scale features**, with **skip connections tapped from the raw
  T1w** (the diagram's "anatomical skip connections"), trained **few-shot** to
  predict the subcortical/cortical segmentation.
- **Labels (few-shot):** FastSurfer/FreeSurfer `aseg` on a small subject subset
  are the pseudo-ground-truth masks — i.e. Option A *bootstraps the labels* for
  Option B. This is the honest few-shot story: distill a fast, L1-aware decoder
  from a few hundred FastSurfer segmentations.
- **Pros:** reuses L1 (one forward pass yields both the embedding and the
  segmentation), potentially faster at inference, and the skip connections let the
  decoder recover fine anatomy the JEPA bottleneck discards.
- **Cons:** needs training + a labeled subset + GPU; must be validated to *beat*
  FastSurfer before it replaces it.
- **Recommendation:** build after Option A ships; gate its adoption on the
  validation in §6.

## 4. Interface contract (both options produce the same output)

```
segment(subject) -> {
  "hippocampal_volume":  float|None,   # mm^3, bilateral
  "ventricle_volume":    float|None,   # mm^3, lateral+inf-lat+3rd+4th
  "whole_brain_volume":  float|None,   # mm^3, BrainSegVolNotVent
  "cortex_volume":       float|None,   # mm^3, CortexVol
  "intracranial_volume": float|None,   # mm^3, eTIV (normalizer)
  "source": "fastsurfer_segment" | "neurojepa_unet" | "fastsurfer_aseg_stats",
  "subject_id": str,
}
```
Invariants: never fabricate a volume (missing → `None` + honest source stamp);
always emit `intracranial_volume` so downstream can head-size-normalize; keys are
exactly the contract's `VOLUME_KEYS`.

## 5. Data flow into L3

1. Per-subject volumes → join the contract table on `subject_id`.
2. **Normalize** each structural volume by `intracranial_volume` (eTIV) to remove
   head-size confound before fusion (standard for hippocampal/cortical measures).
3. **ComBat-harmonize** across sites/scanners (the contract already does this for
   the FreeSurfer block; L2 volumes enter the same harmonization).
4. Feed as a **modality block** to L3 `attention_fusion` — the imaging-structure
   block sits alongside the plasma block and the L1 embedding block. Because
   naive concat dilutes plasma (validated: 0.741 < 0.814), L2 volumes must enter
   through the **modality-balanced / late-fusion** path, not raw concatenation.

## 6. Validation plan (how we earn each claim)

**L2 volumes are correct:**
- Agreement vs FreeSurfer on the same scans: intraclass correlation (ICC) per
  volume key; target ICC ≥ 0.9 for hippocampus/cortex.
- (Option B only) segmentation Dice vs the FastSurfer masks on a held-out subset;
  target Dice ≥ 0.85 subcortical.

**L2 volumes help the pipeline (the claim that matters):**
- Ablation in L3: AD-vs-CN and MCI→AD conversion AUC **with vs without** the L2
  block. L2 earns its place only if it lifts (or at least doesn't dilute) the
  fused AUC over plasma+L1 alone — reported with CIs, OOF, permutation-tested,
  exactly like the existing conversion cards.
- (Option B) must **match or beat FastSurfer volumes' downstream lift** to
  replace it; otherwise FastSurfer stays as L2.

## 7. Execution notes / risks

- **GPU runtime instability (observed this session):** the Colab CLI runtimes were
  reclaimed unpredictably at ~10–22 min even while busy. FastSurfer at ~1 min/vol
  over 590 subjects is ~10 min of GPU — feasible per-batch, but the job MUST
  checkpoint each subject's volume row to durable storage (Drive/GCS) so a
  mid-batch death resumes instead of restarting (see the storage plan). Run in
  batches of ~150 subjects with per-subject durable append.
- **Compliance:** FastSurfer weights (and any trained decoder weights) live only
  on the GPU runtime, never committed; only the derived volume table leaves the
  runtime — same contract as the NeuroJEPA embed and Boltz snapshot.
- **License:** FastSurfer Apache-2.0 (permissive). Option B's decoder weights are
  ours.

## 8. Concrete next steps

1. **Ship Option A:** run `scripts/fastsurfer_volumes_colab.py` (exists) over ADNI
   + OASIS in ~150-subject batches with durable per-subject checkpointing; land a
   `structural_volumes.csv` and join to the contract.
2. **Ablation:** add the L2 block to `integrations/fusion.py` and run the
   with/without AUC ablation for AD-vs-CN and MCI→AD.
3. **Option B prototype:** freeze NeuroJEPA, attach a 3D U-Net decoder head with
   raw-MRI skip connections, train few-shot on ~300 FastSurfer-labeled subjects,
   evaluate per §6. Adopt only if it beats Option A downstream.
