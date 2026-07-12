# Learned, calibrated ranker — design contract

_Status: v1 (2026-07). Results in `reports/candidate_ranking_learned.{json,md}`.
Module: `src/neuroad/harness/ranking_model.py`; driver: `scripts/rank_candidates_learned.py`._

## Why this exists

The shipped composite (`harness/ranking.py`) fuses the discovery signals with **hand-set
weights** (PI4AD 0.30 / OT-heldout 0.35 / STRING 0.20 / pLDDT 0.15) and **min-max**
normalization. Two weaknesses: the weights are a judgement call (not derived from data),
and min-max is outlier-sensitive. This module makes the fusion a **fitted, calibrated
model** and adds the new LINCS efficacy feature.

## Method

- **Features** (rank-normalized to [0,1] — outlier-robust, replacing min-max):
  `pi4ad_priority`, `ot_assoc_heldout`, `net_centrality`, `struct_plddt`,
  `lincs_efficacy`. Missing values are imputed to rank 0.5, and **coverage per feature is
  reported** so a mostly-imputed feature's coefficient is never over-read.
- **Label**: clean GWAS-gold membership over the Open Targets non-genetic top-N universe
  (the same held-out background the validation uses).
- **Model**: L2-regularized logistic regression. Because features share a common rank
  scale, the coefficients ARE the data-derived analogue of the hand-set weights.
- **Honest performance**: leave-one-gene-out cross-validation → out-of-fold AUC, with a
  bootstrap CI, a label-shuffle permutation p, and a Brier calibration score.

## What the live run found (and how to read it)

- OOF AUC ≈ **0.74** [0.63, 0.85], p≈0.001, Brier≈0.02 (well-calibrated).
- Largest positive learned weight = **`ot_assoc_heldout`** — the model independently
  recovers the hand-set intuition that the clean non-genetic OT signal is the workhorse.
- `lincs_efficacy` and `struct_plddt` get **negative/near-zero** learned weights — honest,
  and consistent with LINCS's null validation and pLDDT's ~2% coverage (mostly imputed).
- The multi-signal OOF AUC (0.74) does **not** beat single-feature OT-heldout alone
  (0.845 on the wider universe). Honest takeaway: like plasma p-tau217 on the predictive
  side, **one clean signal carries the discovery half**; the extra signals do not add
  cross-validated lift on this tiny gold set.

## Honest framing (low-n)

The clean gold label is ~15 genes, so this is a **sensitivity analysis** of what the data
implies about signal weighting — not a definitive re-weighting, and not a claim that the
learned model should replace the referee's default. Every number rides with its CI.

## Frozen-seam guarantee

`ranking_model.py` and `rank_candidates_learned.py` are standalone. They are **not**
imported by `translation.py`, `agent.py`, `ranking.py`, or the demo builder; the referee's
default ranking (`method="pi4ad"`) and `translate()`'s output schema are untouched. This
is a decision-support side artifact only.

## Reproduce

```bash
PYTHONPATH=src ./.venv/bin/python scripts/rank_candidates_learned.py --top-n 600   # LIVE
PYTHONPATH=src ./.venv/bin/python scripts/rank_candidates_learned.py --offline     # snapshot
```
