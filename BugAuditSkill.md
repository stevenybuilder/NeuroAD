---
name: BugAuditSkill
description: Compiled audit of every bug fixed in the neuroad-referee project, each with symptom, root cause, fix, and a generalized guardrail. Read this before touching cross-validation guards, confound/nuisance controls, CLI display commands, statistical gating, or multi-cohort merges — and use the closing checklist as a pre-commit review pass.
---

# Bug Audit Skill

## Purpose

This skill captures the concrete, already-fixed bugs from the neuroad-referee
codebase so future work does not reintroduce them. Each entry records what the
bug looked like (symptom), why it happened (root cause), what the fix was, and
the generalized lesson so the same class of mistake is avoided elsewhere. The
recurring lessons are distilled into a checklist at the end.

## When to use

Consult this skill when you are:

- Feeding a subset, slice, cluster, or subgroup into stratified k-fold
  cross-validation (`cross_val_predict`, `StratifiedKFold`, any classifier that
  requires >= 2 classes per fold).
- Building or editing a CLI command whose job is to display a computed result.
- Regressing out a nuisance variable / confound / control from an outcome
  (brain age, site, age, motion, etc.).
- Gating a consequential PASS/FAIL decision on a statistic (correlation, effect
  size, AUC) — especially at small n.
- Merging or unioning records from multiple independent data sources that carry
  their own local identifiers.
- Reviewing a diff before commit — run the closing checklist.

---

## Bug 1 — Single-class / singleton-minority crash in the biomarker anchor test

- **File:** `src/neuroad/gauntlet.py` (`test_biomarker_anchor` / `_oof_scores`, ~lines 219-247)

**Symptom.** `neuroad discover oasis` raised
`ValueError: This solver needs samples of at least 2 classes in the data, but
the data contains only one class` from `_oof_scores`'s `cross_val_predict` ->
`LogisticRegression`. Whole-cohort runs were fine; the crash only appeared
during per-cluster refereeing on real cohorts, when `discovery.py`'s
`_referee_cluster` ran the gauntlet on a single k-means cluster.

**Root cause.** The guard only checked `len(classes) < 2` (are both classes
present at all?). A cluster could have both classes present but with a singleton
minority (e.g. exactly one `Converted == 1` subject). `StratifiedKFold` then
puts that lone positive in a test fold, leaving a single-class training fold,
and `LogisticRegression` rejects it. The class-count check was necessary but not
sufficient — it never checked the minority-class **count**.

**Fix.** After `classes, counts = np.unique(yo, return_counts=True)`, added:

```python
if int(counts.min()) < 2:
    return TestEvidence('biomarker_anchor', TestResult.NA,
        'minority class too small to cross-validate safely (singleton) — cannot anchor')
```

The test now degrades to `NA` (routing to a cohort with enough labelled
subjects) instead of crashing, so per-cluster refereeing on sparsely-labelled
real sub-cohorts completes gracefully.

**Guardrail.** A "both classes present" guard is NOT a "cross-validatable"
guard. Any code that feeds a subset into stratified k-fold CV must validate the
**minority-class count** (>= 2, ideally >= `n_splits`), not just the number of
classes. Guards written against the full dataset must be re-validated for every
sliced / subgroup / per-cluster invocation, because slicing is exactly what
produces degenerate label distributions.

---

## Bug 2 — `scanner-leakage` CLI computed results but printed only the header

- **File:** `src/neuroad/cli.py` (`_cmd_scanner_leakage`, ~lines 150-163)

**Symptom.** `neuroad scanner-leakage` (the real OpenBHB scanner-leakage demo)
printed only the banner/header line and then exited — no scanner AUC, no site
AUC, no verdict. The command looked like it did nothing.

**Root cause.** `_cmd_scanner_leakage` called `openbhb.real_scanner_leakage()`
and bound the result to `out`, but never printed any field of `out`. The
expensive computation ran and was silently discarded; only the static header
printed before the call produced visible output.

**Fix.** After the `if not out: return 1` guard, added prints for the computed
values: scanner AUC with n/classes, site AUC with n/classes, and the trailing
`out['message']` verdict line, all pulled from `out['detail']` /
`out['scanner_auc']` / `out['site_auc']`.

**Guardrail.** A function whose entire purpose is to display a computed result
must be exercised end-to-end (actually run it and read its stdout), not just
type-checked or unit-tested at the compute layer. "Computed but never rendered"
is invisible to tests that assert on return codes or on the underlying compute
function. Verify CLI commands by running them and reading the actual output.

---

## Bug 3 — Brain-age control fit-and-applied on the same rows over-removed signal

- **File:** `src/neuroad/gauntlet.py` (`test_brain_age`, ~lines 161-169)

**Symptom.** For the `dx_binary` outcome, the brain-age control spuriously
collapsed a real signal: the CN (cognitively-normal) subjects used to train the
brain-age regressor were also the negative class of the outcome, so controlling
for brain age on in-sample predictions removed more than generic aging and made
a genuine survivor look like it failed.

**Root cause.** Predicted brain age was produced by fitting the regressor and
predicting on the same CN rows it trained on (in-sample fit). Residualizing the
outcome against a control that was overfit to the training (CN) subjects leaks
information and over-removes signal for any outcome whose negative class **is**
the brain-age training set.

**Fix.** Compute brain age with out-of-fold `cross_val_predict` values for the
training (CN) subjects (`brain_age_all[train] = yhat_cv`), and use the plain
in-sample prediction only for subjects the regressor never trained on. The
control is therefore never fit-and-applied on the same rows.

**Guardrail.** Any nuisance-variable / confound control that is regressed out of
an outcome must be estimated **out-of-fold** on the rows it was trained on.
Fit-and-apply on the same samples leaks and biases the control toward removing
real signal — especially dangerous when the control's training set overlaps one
of the outcome classes.

---

## Bug 4 — Biomarker anchor could pass on a lucky small-n correlation

- **File:** `src/neuroad/gauntlet.py` (`_anchor_corr` / `test_biomarker_anchor`, ~lines 198-283)

**Symptom.** With few complete-case biomarker samples, a ~2-sigma correlation on
essentially noise could spuriously PASS the hard molecular gate, and conversely
a real-but-noisy anchor could FAIL purely by random seed — the pass/fail flipped
with n and seed.

**Root cause.** Gating decisions were made on the raw Pearson r magnitude, which
is high-variance at small n. A single hard gate keyed to `|r|` has no notion of
estimate uncertainty, so small samples produce both false passes and false
fails.

**Fix.** Added a minimum complete-case n floor (`_ANCHOR_MIN_N = 20`, below which
-> `NA`) and switched the gate to the **lower bound** of the two-sided 95%
Fisher-z confidence interval for r (`ci_lo`): PASS when `ci_lo >= 0.12`,
WEAKENED when `ci_lo > 0`, else FAILED. The anchor must be confidently positive,
not just point-estimate positive.

**Guardrail.** Never gate a consequential decision on a raw point estimate of a
noisy statistic. Enforce a minimum sample size, and threshold on a
confidence-interval bound (or effect size with uncertainty) so that small-n
noise cannot flip the verdict in either direction.

---

## Bug 5 — OASIS-1 / OASIS-2 subject_id collision when merging real cohorts

- **File:** `src/neuroad/data/real.py` (~line 150)

**Symptom.** Merging the OASIS-1 cross-sectional and OASIS-2 longitudinal CSVs
could produce duplicate/colliding `subject_id` keys (the contract requires
`subject_id` to be unique), which would violate the table contract or silently
mix subjects across cohorts.

**Root cause.** Two independently-numbered public cohorts share the same integer
subject-numbering scheme, so raw IDs are not globally unique once combined.

**Fix.** Namespace the IDs with `OAS1_` / `OAS2_` prefixes so keys can never
collide across cohorts, then `drop_duplicates('subject_id', keep='first')` as a
belt-and-suspenders guard before `contract.validate_table`.

**Guardrail.** When unioning records from multiple independent sources, never
trust source-local identifiers to be globally unique — namespace them with a
source prefix before the merge, and assert uniqueness afterward rather than
assuming it.

---

## Checklist (distilled recurring lessons)

Run through this before committing changes in the relevant areas:

- [ ] **Guard CV on minority-class COUNT, not just class count.** Before any
      stratified k-fold, assert `counts.min() >= 2` (ideally `>= n_splits`), not
      just `len(classes) >= 2`.
- [ ] **Re-validate every guard on sliced/subgroup/per-cluster inputs.** A guard
      that holds on the full dataset can fail on a slice — slicing is what
      creates degenerate distributions.
- [ ] **Always print computed CLI results.** If a command computes something,
      render every field the user needs and run the command end-to-end to
      confirm real stdout — return-code and compute-layer tests won't catch a
      silent drop.
- [ ] **Estimate confound/nuisance controls out-of-fold.** Never fit-and-apply a
      control on the same rows; use `cross_val_predict` for training rows.
      Extra-dangerous when the control's training set overlaps an outcome class.
- [ ] **Gate on uncertainty, not point estimates.** Enforce a minimum n floor
      and threshold on a confidence-interval bound / effect size, so small-n
      noise can't flip PASS/FAIL.
- [ ] **Namespace identifiers before merging independent sources.** Prefix
      source-local IDs, then de-dup and assert uniqueness — don't assume
      cross-source IDs are globally unique.
- [ ] **Degrade gracefully to NA instead of crashing** when input is too
      degenerate to analyze safely (singleton minority, n below floor), and
      route to a viable cohort.
