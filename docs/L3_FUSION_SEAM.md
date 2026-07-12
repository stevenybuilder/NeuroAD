# L3 Fusion Seam — feeding the NeuroJEPA embedding and L2 volumes

_How to wire (a) the 768-d NeuroJEPA imaging embedding and (b) the
`neurojepa_decoder` structural volumes (hippocampal / ventricle / cortex) into
the attention-weighted late fusion in `src/neuroad/integrations/fusion.py`._

**Status: the seam already exists and is complete + tested.** This note is
documentation only — it describes how to feed the two imaging modalities through
the existing, back-compatible entrypoint. Nothing below asks you to modify
`fusion.py`; the audit confirmed the public API already carries everything needed
(`attention_fusion(df, imaging_embedding=...)`, the `row_mask` alignment seam, and
graceful `seam_open` degradation), all covered by `tests/test_fusion.py`.

---

## 1. The seam contract (what `fusion.py` already exposes)

`attention_fusion(df, *, imaging_embedding=None, ...)` runs an attention-weighted
late fusion over the frozen contract embedding (`imaging`) + the plasma/tabular
block (`plasma`). It accepts an **optional third modality** through the
`imaging_embedding` keyword:

- The frame must carry a **`subject_id`** column plus **`emb_*`** columns
  (`emb_0`, `emb_1`, … — the same `EMBED_PREFIX` convention as
  `contract.embedding_columns`). Any width is accepted; the 768-d NeuroJEPA
  vector is the intended payload.
- Alignment is by `subject_id` (`_align_external_embedding`): the frame is
  reindexed onto `df.subject_id`, duplicate ids drop to first, and only subjects
  with a **complete** embedding row survive. The shared row set is then tightened
  to the intersection of `AD/CN ∩ complete-plasma ∩ complete-embedding` — one
  identical, imputation-free row set across all three modalities.
- If too few subjects overlap (`< 4` or only one class), the third modality is
  **not** wired: `neurojepa_wired=False`, `seam_open=True`, and an honest `error`
  note is stamped. Left `None`, fusion runs on imaging + plasma only and the seam
  stays open. (Covered by `test_seam_open_by_default_without_third_modality`,
  `test_third_modality_wired_when_embedding_frame_supplied`,
  `test_seam_stays_open_when_subjects_do_not_overlap`.)

Once wired, the third modality is gated in exactly like the others: it gets its
own leakage-free, site-disjoint out-of-fold P(AD) score, a softmax attention
weight over its above-chance contribution, and a leave-one-out attribution row.

---

## 2. Feeding (a): the 768-d NeuroJEPA imaging embedding

Build a per-subject frame with the embedding columns and hand it to the seam:

```python
import pandas as pd
from neuroad.integrations.fusion import attention_fusion

# jepa_vecs: dict[subject_id -> np.ndarray shape (768,)]  (the frozen L1 embedding)
nj = pd.DataFrame(
    {"subject_id": list(jepa_vecs)}
    | {f"emb_{i}": [jepa_vecs[s][i] for s in jepa_vecs] for i in range(768)}
)

res = attention_fusion(df, imaging_embedding=nj)   # third modality == "neurojepa"
res.gates            # {"imaging": .., "plasma": .., "neurojepa": ..}
res.neurojepa_wired  # True when the overlap was sufficient
```

The vector is treated as a frozen score source — `attention_fusion` fits a
`LinearProbe` (with its automatic PCA front-end) per fold, so a 768-d block over a
modest slice is handled without leakage. No standardization is required on your
side; the gate standardizes each modality's OOF score internally.

---

## 3. Feeding (b): the L2 structural volumes as a modality block

The `neurojepa_decoder` read-out (`DecoderVolumes`) emits three real, never-
fabricated `mm^3` volumes — `hippocampal_volume`, `ventricle_volume`,
`cortex_volume` — plus a `source` stamp. These are a legitimate imaging modality,
but they must enter fusion the **modality-balanced / late-fusion** way, not by raw
concat: per `docs/L2_UNET_SPEC.md` a naive concat dilutes plasma (validated
0.741 < 0.814), so route the volumes through the same `imaging_embedding=` seam as
their own gated modality.

### 3.1 The `docs/L2_UNET_SPEC.md` pre-fusion path (do this before the seam)

Per subject, in order:

1. **Collect volumes.** `hippocampal_volume`, `ventricle_volume`,
   `cortex_volume` (mm^3). Missing signal → `None` + the decoder's `source`
   stamp; never fabricate.
2. **eTIV-normalize.** Divide each structural volume by the subject's
   `intracranial_volume` (eTIV) to remove the head-size confound. eTIV is
   **not** produced by `neurojepa_decoder` (its `DecoderVolumes` has no eTIV
   field); it comes from the L2 FastSurfer contract block
   (`intracranial_volume`, per `docs/L2_UNET_SPEC.md`'s volume dict). If eTIV is
   absent for a subject, that subject's normalized volumes are `None` and it drops
   from the complete-case slice — do not substitute a cohort mean.
3. **ComBat-harmonize** the eTIV-normalized volumes across sites/scanners, using
   the same harmonization the contract already applies to the FreeSurfer block, so
   the L2 volumes live on the same harmonized scale.
4. **Late-fuse** by handing the harmonized block to `attention_fusion` — never
   raw-concat it onto the plasma or embedding blocks.

### 3.2 Packaging the block for the seam

The seam keys the modality on `emb_*` columns, so name the three harmonized
volumes `emb_0`, `emb_1`, `emb_2` (a 3-wide "embedding"):

```python
# vol_rows: list of dicts, one per subject, AFTER eTIV-normalize + ComBat:
#   {"subject_id": sid,
#    "emb_0": hippo_norm, "emb_1": vent_norm, "emb_2": cortex_norm}
l2_block = pd.DataFrame(vol_rows)           # subject_id + emb_0..emb_2

res = attention_fusion(df, imaging_embedding=l2_block)
```

That runs a **3-modality** late fusion — plasma + contract-imaging + the L2
volume block — with per-modality gates, a leave-one-out attribution table (which
modality drives the fused AUC), and Brier/ECE calibration of the OOF fused P(AD).
Because the volume block is complete-case aligned by `subject_id`, subjects
lacking any of the three harmonized volumes simply fall out of the shared slice —
honest, imputation-free, no raise.

> **One block at a time.** The current seam takes a single `imaging_embedding`
> frame. To fuse both the 768-d NeuroJEPA embedding **and** the L2 volume block as
> two separate gated modalities in one call would need an additive multi-frame
> parameter (not yet added — out of scope for this doc, which prefers
> documentation over edits to a 1000+ line file). Today, either (i) run them in
> separate `attention_fusion` calls and compare, or (ii) horizontally concatenate
> the 768 `emb_*` and the 3 volume `emb_*` into one frame (re-indexed `emb_0..`)
> to gate the combined imaging block against plasma.

---

## 4. What the seam guarantees (honesty)

- **Leakage-free:** every modality (including a wired third block) is scored
  out-of-fold with site-disjoint `StratifiedGroupKFold`, identical machinery to
  `probe.auc_ci_perm` / the gauntlet.
- **Never fabricates:** absent embeddings/volumes → the subject drops from the
  complete-case slice; absent third modality → `seam_open=True`,
  `neurojepa_wired=False`, `error` note stamped. No raise on thin slices.
- **Provenance:** results stamp `source="fitted_fusion"`,
  `model="adni_attention_late_fusion"`, and `gate_learner` records the learner
  that actually ran (`"numpy"` unless the optional torch extra is installed).
- **ADNI-only decision support**, dependent on the gated ADNI export; **not**
  outcome-validated against known AD drugs.
