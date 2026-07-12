# LINCS L1000 efficacy-proxy axis — design contract

_Status: v1 (2026-07). Live results in `reports/lincs_efficacy.{json,md}`; snapshot at
`src/neuroad/integrations/data/lincs_ad_reversal_snapshot.json`._

## Why this exists

Every other discovery-half signal is an **association** signal — it says a gene is
*linked* to AD (PI4AD priority, Open Targets association, STRING network centrality).
None tests **efficacy**: whether *perturbing* the gene would move the disease state. The
retrospective validation (`reports/target_prioritization_validation.md`) makes this
explicit — the drug-target arm is at/near chance, because association ≠ druggable
efficacy.

This axis adds the missing, mechanistically orthogonal dimension using **LINCS L1000**
perturbational transcriptomics: does a genetic loss-of-function perturbation of a gene
produce a signature that **reverses** an Alzheimer brain signature?

## Method

1. **AD signature** (`integrations/lincs.py`): a curated consensus up/down gene set —
   up = reactive glia / microglial activation / complement (GFAP, TYROBP, C1Q*, CD68,
   TREM2, …); down = synaptic / neuronal (SNAP25, SYT1, NRGN, GAP43, …). Cited to Zhang
   2013 (TYROBP causal network), Mathys 2019 (single-cell AD), Mostafavi 2018. An
   **approximation**, flagged as such.
2. **Query** the keyless SigCom LINCS Data API (`/enrich/ranktwosided`) over the
   loss-of-function databases `l1000_xpr` (CRISPR KO) and `l1000_shRNA` (shRNA KD).
   *Reversers* have the AD-up genes enriched at the bottom and AD-down genes at the top.
3. **Map** each reverser signature → its perturbed gene (`meta.pert_name`), and score
   each gene by its **strongest** reverser strength `|z-sum|` across its signatures.
   A gene whose KO reverses AD ⇒ an **inhibition** target hypothesis.
4. **Validate** the axis exactly like the rest of the pipeline: a signed connectivity
   universe (reversers `+|z-sum|`, mimickers `−|z-sum|`) scored against the held-out
   gold sets (GWAS, FDA-drug, novel-2022) with bootstrap AUC CI + permutation p +
   BH-FDR, and the housekeeping **DECOY** as a negative control.

## Honest caveats (loud, everywhere)

- **Efficacy proxy ≠ efficacy.** L1000 cell lines are (mostly) cancer lines — HT29,
  MCF7, ES2, AGS — not neurons or microglia. "Reverses an AD transcriptomic signature in
  a cancer line" is a weak surrogate for neuronal/glial AD efficacy. Every hit is a
  hypothesis to test in an AD-relevant model (iPSC neurons/microglia, organoids).
- The AD signature is a curated consensus, not a region-/cell-type-specific DE table.
- A **null** validation result is the honest expected outcome and is still useful: the
  axis is a weak, independent feature the learned ranker (Track 2) can down-weight,
  and reporting the null demonstrates the same rigor applied to every other signal.

## Frozen-seam guarantee

`integrations/lincs.py` is a standalone, offline-first adapter (never raises; degrades to
the committed snapshot). It is **not** imported by `translation.py`, `agent.py`, or any
referee/demo-path code. It feeds only the offline reports and, optionally, the Track-2
learned ranker — never `app/demo_data.json`. The demo is unaffected.

## Reproduce

```bash
# LIVE build + validate + (re)write the committed snapshot:
PYTHONPATH=src ./.venv/bin/python scripts/build_lincs_signature.py --limit 1000
# Offline (deterministic, from the committed snapshot):
PYTHONPATH=src ./.venv/bin/python scripts/build_lincs_signature.py --offline
```

Live API: SigCom LINCS, Ma'ayan Lab (Evangelista et al., Nucleic Acids Res 2022;50:W697),
`https://maayanlab.cloud/sigcom-lincs` — keyless.
