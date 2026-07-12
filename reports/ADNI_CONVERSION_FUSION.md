# MCI->AD Conversion — Attention-Weighted Late Fusion (+ NeuroJEPA seam)

**Cohort:** 323 baseline-MCI subjects with a conversion outcome and a complete fusion block (58 converters / 276 stable), 58 sites. Target = **conversion** (converter=1 / stable=0). Real ADNI; leakage-free site-disjoint OOF.

## Modalities (gated leave-one-out)

| Modality | Standalone AUC | Gate weight |
|---|---|---|
| imaging | AUC 0.7399 [0.6714, 0.799], p_perm=0.001 | 0.2783 |
| plasma | AUC 0.7989 [0.7315, 0.8633], p_perm=0.001 | 0.502 |
| neurojepa | AUC 0.7163 [0.6419, 0.7874], p_perm=0.001 | 0.2198 |

**Attention gate:** imaging=0.2783, plasma=0.502, neurojepa=0.2198
**Top modality (largest leave-one-out drop):** plasma

## Fusion result

- **3-modality (imaging + plasma + NeuroJEPA):** AUC 0.8209 [0.7597, 0.8776], p_perm=0.001
- **2-modality baseline (imaging + plasma, NO NeuroJEPA):** AUC 0.8201 [0.7604, 0.8763], p_perm=0.001

### Does NeuroJEPA add prognostic signal?

- Leave-one-out: removing NeuroJEPA changes fused AUC by **0.0008** (fused 0.8209 -> 0.8201 without it).
- Baseline delta: 3-modality 0.8209 vs 2-modality 0.8201 = **+0.0008** AUC.

**Verdict:** attention gate [imaging=0.28, plasma=0.50, neurojepa=0.22]; 'plasma' drives the largest leave-one-out AUC drop. No CI-supported gain over best single modality plasma (AUC 0.7989): delta=+0.0220

## Calibration (out-of-fold fused P(convert))

Brier=0.1973, ECE=0.2344, MCE=0.4625

## Honesty

ATTENTION-WEIGHTED LATE fusion (NOT a transformer): a principled numpy/sklearn softmax gate over per-modality out-of-fold P(AD) scores. Each modality's score is leakage-free and site-disjoint (identical CV machinery to the gauntlet); the gate weight is a softmax over each modality's above-chance contribution. Gate weights and the leave-one-out attribution are computed from labelled out-of-fold scores over the full slice, so they carry a small in-sample optimism (disclosed); the per-view AUC/CI/p_perm and the fused AUC use the same frozen-score bootstrap + within-site permutation null as probe.auc_ci_perm. ADNI-only decision support; depends on the gated ADNI export and is NOT outcome-validated against known AD drugs.

Conversion prediction is genuinely harder than cross-sectional diagnosis; a cross-validated, permutation-significant AUC above chance is a REAL prognostic signal, and an at-chance result is reported as such.