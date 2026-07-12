# Discovery half — statistical rigor report

_Generated 2026-07-12T03:10:34.754096+00:00; prefer_offline=False; n_boot=2000, n_perm=1000, n_draws=1000._

Adds bootstrap AUC confidence intervals, BH-FDR multiple-testing control, negative controls (housekeeping decoys + a degree-matched network null), and shortlist rank-stability to the target-prioritization validation. Read-only; does not touch the referee/demo path.

## 1. Test battery — AUC with 95% bootstrap CI and BH-FDR q-value

| Test | kind | n_gold | AUC | 95% CI | perm p | BH q |
|---|---|---|---|---|---|---|
| opentargets_vs_gwas_heldout_nongenetic | clean | 15 | 0.845 | [0.783, 0.908] | 0.001 | 0.001 |
| opentargets_vs_drugs_heldout_nonclinical | clean | 9 | 0.720 | [0.576, 0.857] | 0.009 | 0.009 |
| pi4ad_vs_gwas | residually_circular | 15 | 0.869 | [0.719, 0.969] | 0.001 | 0.001 |
| opentargets_nongenetic_vs_DECOY | negative_control | 6 | 0.541 | [0.345, 0.692] | 0.361 | — |
| pi4ad_vs_DECOY | negative_control | 15 | 0.872 | [0.773, 0.943] | 0.001 | — |

**How to read it:** a *clean* test with a CI whose lower bound stays above 0.5 after FDR is genuine signal; a *negative_control* test SHOULD straddle 0.5 (that is the pass condition — decoys must not score high); the *residually_circular* row is the optimistic ceiling, not evidence.

## 2. Degree-matched network null (specificity of the STRING-RWR test)

- Observed AUC (NOVEL_2022 via RWR from KNOWN_2019): **0.587**
- Degree-matched null AUC: mean **0.489**, 95% [0.358, 0.627] over 1000 draws
- Empirical p (observed vs degree-matched null): **0.073**

This isolates propagation signal from the trivial 'novel genes are high-degree hubs' confound: if the observed AUC is not clearly above the degree-matched null, the network test carries little beyond degree.

## 3. Shortlist rank-stability

Baseline composite top-5: **MAPK1, APP, ESR1, TREM2, BIN1**

Bootstrap top-k membership frequency (universe resampled 2000×):

- MAPK1: **66%**
- APP: **65%**
- ESR1: **65%**
- TREM2: **64%**
- BIN1: **62%**

Leave-one-signal-out top-k (does the shortlist survive dropping each signal?):

- drop_pi4ad_priority: MAPK1, ESR1, APP, BIN1, BACE1
- drop_ot_assoc_heldout: MAPK1, HRAS, APP, ESR1, TREM2
- drop_net_centrality: APP, MAPK1, ESR1, TREM2, BIN1
- drop_struct_confidence: APP, MAPK1, ESR1, MAPT, TREM2

A gene that stays top-k across most bootstraps AND most signal-drops is a robust recommendation, not an artifact of one signal or one weighting.
