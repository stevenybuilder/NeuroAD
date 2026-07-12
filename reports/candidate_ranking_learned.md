# Learned, calibrated candidate ranking

_Generated 2026-07-12T03:24:29.888992+00:00; prefer_offline=False; universe=598 genes (15 gold); top_n=600, n_boot=2000, n_perm=1000._

Replaces the hand-set composite weights with weights **learned** from the data (L2 logistic on rank-normalized features, predicting clean GWAS-gold membership), leave-one-out cross-validated, plus the new LINCS efficacy feature. Read-only; does not touch the referee/demo path.

> **Low-n caveat:** the gold label is ~15 genes, so this is a SENSITIVITY ANALYSIS of what the data implies — not a definitive re-weighting. Numbers ride with their uncertainty.

## Honest performance (leave-one-out cross-validated)

- **OOF AUC: 0.740** (95% CI [0.626, 0.848])
- Permutation p: **0.0010**
- Brier calibration: **0.024** (lower is better)

## Learned weights vs hand-set weights

| Feature | learned coef | hand-set weight | coverage |
|---|---|---|---|
| pi4ad_priority | +0.623 | 0.30 | 96% |
| ot_assoc_heldout | +0.877 | 0.35 | 100% |
| net_centrality | +0.709 | 0.20 | 95% |
| struct_plddt | -1.177 | 0.15 | 2% |
| lincs_efficacy | -0.890 | — (new) | 17% |

A near-zero or negative learned coefficient means the data does not support that signal's positive contribution once the others are present. Coverage < 100% marks a feature that was imputed (rank 0.5) for the genes lacking it — read its coefficient with that in mind.

## Pipeline shortlist in the learned ranking

| Gene | learned score | learned rank |
|---|---|---|
| APP | 0.0896 | 5 |
| TREM2 | 0.0474 | 85 |
| BIN1 | 0.0915 | 4 |
| BACE1 | 0.0503 | 76 |
| MAPT | 0.1093 | 2 |
| APOE | 0.0434 | 110 |
| PSEN1 | 0.0554 | 56 |
| CLU | 0.0353 | 175 |
| ESR1 | 0.1097 | 1 |

## Top-20 learned ranking

| Rank | Gene | learned score | gold? |
|---|---|---|---|
| 1 | ESR1 | 0.1097 |  |
| 2 | MAPT | 0.1093 |  |
| 3 | PIK3R1 | 0.0926 |  |
| 4 | BIN1 | 0.0915 | ✓ |
| 5 | APP | 0.0896 |  |
| 6 | ATM | 0.0879 |  |
| 7 | PLCG2 | 0.0758 | ✓ |
| 8 | SCARB1 | 0.0753 |  |
| 9 | JUN | 0.0749 |  |
| 10 | CDKN2A | 0.0747 |  |
| 11 | PICALM | 0.0728 | ✓ |
| 12 | CDKN1A | 0.0721 |  |
| 13 | CREB1 | 0.0718 |  |
| 14 | AXL | 0.0706 |  |
| 15 | CANX | 0.0697 |  |
| 16 | HIF1A | 0.0693 |  |
| 17 | APOB | 0.0688 |  |
| 18 | CD2AP | 0.0687 |  |
| 19 | FOXO3 | 0.0686 |  |
| 20 | AURKA | 0.0685 |  |
