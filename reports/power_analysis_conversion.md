# Power analysis — stacked fusion vs plasma (MCI->AD conversion)

Current: n=498, 142 converters (28.5%). AUC fused=0.827 vs plasma=0.814 (diff +0.012). DeLong one-sided p=0.121 (SE_diff=0.0105) — NOT separable.

| cohort mult | total n | converters | power (p<0.05) |
|---|---|---|---|
| 1.0 | 498 | 142 | 0.285 |
| 1.5 | 747 | 213 | 0.398 |
| 2.0 | 996 | 284 | 0.475 |
| 3.0 | 1494 | 426 | 0.672 |
| 4.0 | 1992 | 568 | 0.758 |
| 5.0 | 2490 | 710 | 0.827 |
| 6.0 | 2988 | 852 | 0.9 |
| 8.0 | 3984 | 1136 | 0.953 |

**~710 converters (total n≈2490) for 80% power** to detect the observed +0.012 lift.

> Bootstrap resamples observed (s_plasma,s_fused,y) triples to larger cohorts at the same 28.5% converter rate, preserving the empirical fused/plasma correlation; power = P(DeLong one-sided p<0.05) assuming the observed +0.012 lift is the true effect. Stacked-fusion OOF carries mild in-sample optimism (shared across both models).
