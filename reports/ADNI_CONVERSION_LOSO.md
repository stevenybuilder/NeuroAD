# MCI->AD Conversion — Leave-One-Site-Out (adni:conversion feeder)

**Cohort:** 334 baseline-MCI subjects (58 pMCI / 276 sMCI), 58 sites. **Target:** conversion (pMCI vs sMCI). **Split:** leave-one-site-out (site-disjoint) — no acquisition site in both train and test.
**Substrate:** frozen Neuro-JEPA structural embeddings.

| Block | Features | AUC (site-disjoint) |
|---|---|---|
| Neuro-JEPA imaging | 768-d frozen MRI embedding | AUC 0.7181 [0.6405, 0.7797], p_perm=0.001 |
| Plasma | p-tau217, GFAP, NfL | AUC 0.8098 [0.748, 0.87], p_perm=0.001 |
| Fused | Neuro-JEPA + plasma | AUC 0.723 [0.6448, 0.7828], p_perm=0.001 |

**Fused − plasma:** -0.0868 AUC (CIs overlap).

**Verdict:** Plasma alone (0.8098) is not beaten by adding imaging (naive-concat fused 0.723, -0.0868): imaging does not add conversion signal over plasma on this single cohort. (The concat block PCA-10s 768 imaging + 3 plasma dims together, so it under-weights plasma and understates fusion — the attention-weighted fusion in scripts/run_conversion_fusion.py reaches ~0.82 but still shows no CI-supported gain over plasma. Both fusion methods agree: plasma dominates conversion at this sample size.)

## Honesty

Single-cohort, underpowered (58 converters). A TRUE cross-dataset leave-one-cohort-out awaits a second conversion-labeled cohort (OASIS-2 cdr trajectory / AIBL / NACC) embedded in the same frozen 768-d space; site-disjoint LOSO is the honest single-cohort analog. The 'fused' block is a NAIVE concat (768 imaging + 3 plasma) auto-PCA-10'd together, which under-weights the low-dim plasma signal and understates fusion; the attention-weighted late fusion in run_conversion_fusion.py is the proper multimodal number (~0.82) and still shows no CI-supported imaging gain over plasma. AUCs via probe.auc_ci_perm (grouped CV, auto PCA-10, bootstrap CI, permutation null). Frozen inference only; weights never stored.
