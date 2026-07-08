---
name: site-scanner-leakage
description: >-
  Gauntlet stage 2 (⭐ highest weight) of NeuroAD Discovery Engine. Tests whether a
  structural-MRI signal is disease biology or just which scanner/site acquired
  the scan, by pointing the SAME linear-probe head at the scanner/site label and
  comparing AUCs. Use when auditing any brain-foundation-model embedding finding
  for batch-effect leakage. Weight 25/100, star test. Headline metric: the
  subject-disjoint leakage margin. Implemented by neuroad.gauntlet.test_site_scanner.
when-to-use: >-
  Run on any multi-scanner or multi-site cohort. On single-scanner real data
  (OASIS-1/2) it returns NA — reframe as cohort/batch leakage instead. This is
  the signature adversarial test and carries the most weight.
---

# Site / Scanner Leakage (⭐)

**Adversarial question:** *Is it disease signal, or just which machine acquired
the scan?*

The heaviest gauntlet stage (weight **25**, ⭐) and the product's signature move:
**point the same reused probe head at the `site` / `scanner` label column**
(`contract.LABEL_TARGETS`) instead of the outcome, and see how well the frozen
embeddings predict acquisition. Same code, different label.

Implemented by `test_site_scanner(df, target)` in `src/neuroad/gauntlet.py`,
which delegates the measurement to `leakage_margin` in `src/neuroad/leakage.py`.

> The insight that frozen foundation-model embeddings predict scanner/site as
> well as biological outcome is **published prior art** (arXiv:2604.14441,
> *Batch Effects in Brain Foundation Model Embeddings*; arXiv:2606.09189,
> *Pretrained, Frozen, Still Leaking* — leakage margin ~0.16–0.37; PathoROB).
> Cite it via `calibration.PRIOR_ART`. Do **not** claim it.

## Exact statistic — the leakage margin

    margin = outcome_AUC − scanner_AUC

computed by `leakage_margin(df, target)`:

- `outcome_auc` = `cross_val_auc(X, y_outcome, groups=site)` — **subject/site-disjoint**.
- `scanner_auc` = `cross_val_auc(X, y_scanner, groups=None)` — deliberately NOT
  group-aware, because we *want* to expose the machine signal. Scanner is
  preferred; falls back to `site`; if neither varies, `scanner_auc = 0.5`.

A large positive margin means the outcome clearly exceeds the confound; a margin
near or below zero means the "finding" is mostly which machine took the picture.

## Verdict thresholds (`test_site_scanner`)

- **PASSED** — `margin ≥ 0.10`: outcome clearly exceeds scanner.
- **WEAKENED** — `0 < margin < 0.10`: real but confound-adjacent (outcome only
  narrowly exceeds scanner).
- **FAILED** — `margin ≤ 0`: scanner predicted as well as (or better than) the
  outcome; likely an acquisition artifact.
- **NA** — single scanner **and** single site — no acquisition confound to test.

## Stats emitted

`{"outcome_auc", "scanner_auc", "margin", "confound"}` where `confound` names the
label actually used (`"scanner"`, `"site"`, or `"none (single scanner/site)"`).
On `synthetic:KILL` the scanner AUC (~0.92) meets or exceeds the outcome AUC →
FAILED — that collapse is the demo's punchline.

## Companion instrumentation (leakage.py)

- `double_dissociation(df, target)` — projects the embedding out of the top
  scanner-discriminating LDA direction(s) and re-measures the outcome: survivors
  retain, kills collapse.
- `confound_leaderboard(df, target)` — ranks how much probe-score variance
  scanner/site, age, and sex each explain, so the scientist sees which artifact
  to fix first.

## Run it

```bash
PYTHONPATH=../../src ../../.venv/bin/python run.py                 # SURVIVOR
PYTHONPATH=../../src ../../.venv/bin/python run.py synthetic:KILL  # scanner AUC ≥ outcome
PYTHONPATH=src ./.venv/bin/python skills/site_scanner_leakage/run.py
```

## Calibration & honesty notes

- Calibrated: survivor scanner AUC ~0.64 (`CAL["site_auc_survivor"]`), KILL
  scanner AUC ~0.92 (`CAL["site_auc_kill"]`).
- On single-scanner real data (OASIS-1/2), reframe honestly as **cohort/batch
  leakage** (predict OASIS-1 vs OASIS-2 membership) rather than scanner.
- This test *bounds* leakage; it uses the same probe family it audits, so it
  cannot fully eliminate shared confounding — surface that as a caveat.
