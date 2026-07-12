# LINCS L1000 — perturbational efficacy-proxy axis

_Generated 2026-07-12T03:19:57.078585+00:00; source=live; databases=l1000_xpr, l1000_shRNA; limit=1000, n_boot=2000, n_perm=1000._

A **perturbational / efficacy** axis, orthogonal to every association signal (PI4AD, Open Targets, STRING). Queries SigCom LINCS (keyless) for genetic loss-of-function perturbations whose signature **reverses** a curated AD brain signature: a gene whose knockout reverses AD is an efficacy-relevant **inhibition** target.

> **HONEST CAVEAT.** This is an efficacy *proxy*, not efficacy. L1000 lines are (mostly) cancer cell lines, not neurons/microglia, so 'reverses an AD transcriptomic signature in a cancer line' is a weak surrogate for neuronal AD efficacy. Every hit is a hypothesis. The AD signature is a curated consensus (approximation), not a single-study DE table.

## Validation — does the efficacy proxy recover known AD biology?

| Test | kind | n_gold | AUC | 95% CI | perm p | BH q |
|---|---|---|---|---|---|---|
| efficacy_vs_gwas | clean_orthogonal | 5 | 0.583 | [0.176, 0.896] | 0.271 | 0.812 |
| efficacy_vs_drugs | clean_orthogonal | 7 | 0.372 | [0.132, 0.647] | 0.875 | 0.875 |
| efficacy_vs_novel2022 | prospective | 15 | 0.432 | [0.295, 0.574] | 0.800 | 0.875 |
| efficacy_vs_DECOY | negative_control | 8 | 0.605 | [0.404, 0.803] | 0.145 | — |

A *clean_orthogonal* test whose CI lower bound clears 0.5 after FDR would be genuine orthogonal recovery of AD biology; a null is the honest, expected outcome for a cancer-line proxy and is still useful as a weak, independent feature for the learned ranker (which can down-weight it). The *negative_control* (housekeeping decoys) must sit at chance.

## Top reversal-efficacy hypotheses (strongest AD-signature reversers)

| Gene | reversal score | # sigs | database | cell line |
|---|---|---|---|---|
| MRPL15 | 11.7987 | 2 | l1000_xpr | MCF7 |
| MEI1 | 11.2208 | 2 | l1000_xpr | MCF7 |
| CES1 | 10.7323 | 1 | l1000_xpr | MCF7 |
| PRDX2 | 10.6103 | 1 | l1000_xpr | AGS |
| CNTN1 | 10.5461 | 1 | l1000_xpr | BICR6 |
| GPR65 | 10.1898 | 1 | l1000_xpr | MCF7 |
| WARS2 | 10.0978 | 2 | l1000_xpr | PC3 |
| KRTAP5-2 | 10.0813 | 1 | l1000_xpr | BICR6 |
| EHF | 9.934 | 1 | l1000_xpr | U251MG |
| NDUFA1 | 9.8225 | 3 | l1000_xpr | A375 |
| LPA | 9.8201 | 1 | l1000_xpr | BICR6 |
| ITGA3 | 9.7703 | 4 | l1000_xpr | A375 |
| OGFOD1 | 9.7686 | 1 | l1000_xpr | PC3 |
| EPHA8 | 9.7337 | 1 | l1000_xpr | MCF7 |
| MUC20 | 9.7281 | 1 | l1000_xpr | MCF7 |

_AD signature: 24 up / 24 down genes. Curated consensus AD brain transcriptomic signature (up = reactive glia / microglial activation / complement, down = synaptic / neuronal), after Zhang et al. 2013 (Cell 153:707; TYROBP causal network), Mathys et al. 2019 (Nature 570:332; single-cell AD), and Mostafavi et al. 2018 (Nat Neurosci 21:811). APPROXIMATION — not a single-study DE table._
