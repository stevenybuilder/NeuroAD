# Target Prioritization — Validation (LIVE full universe)

_Generated 2026-07-11T19:14:09.225172+00:00; prefer_offline=False, n_perm=1000, seed=0._

## Honest verdict

CLEAN non-circular signal at the full live universe: opentargets_vs_gwas_heldout_nongenetic (AUC=0.728, p=0.003). This is genuine, honest evidence the ranking surfaces independently-known AD-risk genes from out-of-evidence signal. Residually-circular (strong but NOT clean): pi4ad_vs_gwas (AUC=0.869, p=0.001) — CAVEAT: PI4AD's Priority Index integrates genetic/GWAS evidence as an input, so ranking GWAS genes high is partly circular. At/below chance (honest negative): opentargets_vs_drugs_heldout_nonclinical (AUC=0.516, p=0.432). Overall: a rigorously-filtered, wet-lab-testable HYPOTHESIS ENGINE (organoid/iPSC) — not a validated efficacy predictor. The clean held-out signal is real but is prognostic-of-relevance, not proof a target is druggable.

## Honest tests (non-circular)

| Test | Source | N univ | N gold | AUC | perm p | Circularity |
|---|---|---|---|---|---|---|
| pi4ad_vs_gwas | live | 14676 | 15 | 0.869 | 0.001 | PI4AD's Priority Index integrates genetic/GWAS evidence as an input, so ranking GWAS genes high is partly circular |
| opentargets_vs_gwas_heldout_nongenetic | live | 200 | 15 | 0.728 | 0.003 | clean (out-of-evidence) |
| opentargets_vs_drugs_heldout_nonclinical | live | 200 | 9 | 0.516 | 0.432 | clean (out-of-evidence) |

## Circular comparators (Guard 1 — optimistic, NOT honest)

| Test | Source | AUC (circular) | perm p |
|---|---|---|---|
| opentargets_vs_gwas_overall_CIRCULAR | live | 0.858 | 0.001 |
| opentargets_vs_drugs_overall_CIRCULAR | live | 0.967 | 0.001 |

## L5 PI4AD flesh-out

- PI4AD table: **14676 genes** (source=live).
- STRING-RWR from 15 GWAS seeds (15 in graph, 75 subgraph nodes, source=string_live): **8 non-seed hubs** surfaced.

  Hubs: PSEN1(r16,deg34), PSEN2(r17,deg34), APP(r18,deg52), CD2AP(r19,deg39), INPP5D(r20,deg30), MS4A4E(r21,deg29), MAPT(r23,deg29), MS4A4A(r25,deg30)

## L6 AlphaFold structural confidence (LIVE, keyless AF DB)

- 21/21 targets fetched LIVE from `https://alphafold.ebi.ac.uk/api/prediction/{accession} (keyless)`.

| Target | UniProt | resolved via | mean pLDDT | residues | source |
|---|---|---|---|---|---|
| EGFR | P00533 | uniprot_live | 75.95 | 1210 | live |
| SRC | P12931 | uniprot_live | 83.46 | 536 | live |
| GRB2 | P62993 | uniprot_live | 88.71 | 217 | live |
| AKT1 | P31749 | uniprot_live | 83.05 | 480 | live |
| CD4 | P01730 | uniprot_live | 85.24 | 458 | live |
| ACTB | P60709 | uniprot_live | 95.2 | 375 | live |
| B2M | P61769 | uniprot_live | 94.04 | 119 | live |
| TP53 | P04637 | uniprot_live | 75.05 | 393 | live |
| FN1 | P02751 | uniprot_live | 69.65 | 2477 | live |
| ITGB1 | P05556 | uniprot_live | 85.87 | 798 | live |
| FYN | P06241 | uniprot_live | 80.84 | 537 | live |
| INS | P01308 | uniprot_live | 52.9 | 110 | live |
| ACHE | P22303 | uniprot_live | 92.94 | 614 | live |
| APP | P05067 | ad_map | 67.39 | 770 | live |
| BCHE | P06276 | uniprot_live | 93.36 | 602 | live |
| GRIN1 | Q05586 | uniprot_live | 82.86 | 938 | live |
| GRIN2A | Q12879 | uniprot_live | 60.85 | 1464 | live |
| GRIN2B | Q13224 | uniprot_live | 60.7 | 1484 | live |
| GRIN2C | Q14957 | uniprot_live | 67.47 | 1233 | live |
| GRIN2D | O15399 | uniprot_live | 63.22 | 1336 | live |
| GRIN3A | Q8TCU5 | uniprot_live | 73.06 | 1115 | live |

> AlphaFold DB = free/keyless precomputed monomer structures (used here, LIVE). AlphaFold3 de-novo COMPLEX folding is the account/weight-gated product; the open MIT Boltz-2 GPU job is its license-clean substitute for the L6 complex step.

