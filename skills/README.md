# NeuroAD Discovery Engine — Drop-in Adversarial Skills

Each subfolder here is a **self-contained Agent Skill** wrapping one stage of the
NeuroAD Discovery Engine gauntlet — the five ways a structural-MRI "finding" can
turn out to be an artifact instead of biology.

This is the "built to outlast the week" story: the referee is not a monolith, it
is a **composable set of falsification tests**, and adding a sixth is dropping in
a sixth folder that satisfies one small contract.

## The five stages

| Folder | Stage | Adversarial question | Weight | Function |
|---|---|---|---|---|
| `age_sex/` | Age / sex adjustment | Survives demographic covariates? | 15 | `test_age_sex` |
| `site_scanner_leakage/` | Site / scanner leakage ⭐ | Disease signal, or which machine took the scan? | 25 | `test_site_scanner` |
| `brain_age_control/` | Brain-age control ⭐ | More than generic aging / atrophy? | 25 | `test_brain_age` |
| `biomarker_anchor/` | Biomarker anchor | Backed by molecular pathology (p-tau217 / GFAP) when available? | 20 | `test_biomarker_anchor` |
| `replication/` | Replication split | Reproduces on a held-out site / cohort? | 15 | `test_replication` |

Weights sum to 100 (asserted in `src/neuroad/contract.py`). The two ⭐ tests carry
the most weight — they are the artifacts a scanner or generic aging can most
easily fake.

## How the skills compose into the gauntlet

`neuroad.gauntlet.run_gauntlet(df, claim)` simply calls the five functions above
in order and returns a `list[TestEvidence]`. Each `TestEvidence` carries a
`TestResult` (PASSED / WEAKENED / MIXED / FAILED / NA). Scoring
(`contract.robustness_score`) turns those into a 0–100 number:

- Each dimension earns a fraction of its weight — `RESULT_CREDIT`: PASSED = 1.0,
  WEAKENED / MIXED = 0.5, FAILED = 0.0.
- **NA dimensions are dropped from both numerator and denominator**, so the score
  reflects only evidence we actually have (with a completeness caveat surfaced
  elsewhere). A test that could not run never silently costs points.
- `verdict_for(score)` maps the score to a `Verdict` via `VERDICT_BANDS`
  (≥85 strong, ≥70 robust-enough-for-follow-up, ≥40 partially robust, else
  fragile). Only claims at or above `PROMOTION_FLOOR` (partially robust) may reach
  the biology step, and promotion still requires independent corroboration:
  molecular anchor when available, or leakage-clean replication when plasma is
  unavailable.

Run any single stage in isolation:

```bash
# from a skill directory
PYTHONPATH=../../src ../../.venv/bin/python run.py [dataset] [target]
# from the repo root
PYTHONPATH=src ./.venv/bin/python skills/age_sex/run.py synthetic:SURVIVOR conversion
```

`dataset` defaults to `synthetic:SURVIVOR` (a calibrated real signal) — pass
`synthetic:KILL` to watch the same test collapse an artifact. `target` defaults
to `conversion` (also accepts `dx_binary`). Datasets come from
`neuroad.data.loaders.load` (see `AVAILABLE`).

## Adding a 6th adversarial test — the contract

A new stage is a **pure function of a contract table + a target**, returning one
`TestEvidence`. That is the entire interface. Concretely:

```python
# skills/my_new_test/run.py  (and a sibling function in your own module)
import numpy as np, pandas as pd
from neuroad.contract import TestEvidence, TestResult
from neuroad.probe import point_head, cross_val_auc   # the reused probe

def test_my_new_confound(df: pd.DataFrame, target: str) -> TestEvidence:
    # 1. Guard: if the data can't support the test, return NA (never fabricate).
    X, y, groups = point_head(df, target)             # X = frozen embeddings
    if len(np.unique(y)) < 2:
        return TestEvidence("my_new_confound", TestResult.NA, "target has <2 classes")

    # 2. Compute a statistic that isolates YOUR artifact from real signal
    #    (e.g. residualize the embedding against your suspected confound and
    #     re-measure the outcome AUC).
    auc_before = cross_val_auc(X, y, groups=groups)
    # ... your control ...
    auc_after = cross_val_auc(X_controlled, y, groups=groups)
    retained = (auc_after - 0.5) / max(auc_before - 0.5, 1e-6)

    # 3. Map the statistic to PASSED / WEAKENED / FAILED against a threshold that
    #    is pinned in calibration.CAL — never a free-floating magic number.
    res = TestResult.PASSED if retained >= 0.70 else \
          TestResult.WEAKENED if retained >= 0.40 else TestResult.FAILED
    return TestEvidence("my_new_confound", res,
                        f"effect retained {retained:.0%} after my control",
                        {"auc_before": auc_before, "auc_after": auc_after,
                         "retained": retained})
```

The contract your test must satisfy (all defined in `src/neuroad/contract.py`):

1. **Signature:** `test_x(df: pd.DataFrame, target: str) -> TestEvidence`.
   `df` is a **contract-valid table** (see `METADATA_COLUMNS`, `EMBED_PREFIX`,
   validated by `validate_table`); `target` is one of `LABEL_TARGETS`.
2. **Features are always the embeddings.** Use `contract.embedding_matrix(df)` or
   `probe.point_head` — never feed a label/metadata column into the probe as a
   feature. Metadata (age, sex, site, scanner, biomarkers) are targets or
   covariates only.
3. **Return exactly one `TestEvidence(key, result, detail, stats)`** with a
   `TestResult`. Use **NA** whenever the data can't honestly support the test —
   the scorer drops NA cleanly.
4. **No fabricated numbers.** Every threshold or headline figure must trace to a
   range in `calibration.CAL` / `FACTS` (with a citation) or be computed live
   from the data. This is the "no invented science" rule the whole project rests
   on.
5. **Ship a `SKILL.md` + `run.py`** in your folder: the doc states the
   adversarial question, the exact statistic, how to read PASSED/WEAKENED/FAILED/
   NA, and the calibrated thresholds; the runner is a thin wrapper that loads a
   cohort and prints the `TestEvidence`.

To wire it into the score, register a `GauntletDimension` (key, label, question,
weight) in `contract.GAUNTLET` and add the call to `gauntlet.run_gauntlet` —
keeping the weights summing to 100 (the module asserts this). Nothing downstream
(scoring, verdict, claim card export) changes: it reads the `TestResult` and
`stats` you produced.

## How this maps to Claude Science's composable-skills architecture

The referee is **Claude as adjudicator, not just coder**. Each skill is a small,
inspectable, independently runnable unit of scientific scrutiny with a fixed
input/output contract — so Claude can (a) read the `SKILL.md` to know *when* to
apply a test and *how to interpret* it, (b) invoke `run.py` to execute exactly
one falsification step, and (c) compose the five verdicts into a single robustness
score and a promote/reject decision. New scientific scrutiny = a new skill folder
satisfying the same `TestEvidence` contract, with no change to the orchestration
or the scoring. That is what makes the gauntlet extensible past the hackathon
week: the science lives in drop-in skills, and the composition is fixed.

See `src/neuroad/contract.py` (`TestEvidence`, `TestResult`, `GAUNTLET`,
`robustness_score`, `verdict_for`), `src/neuroad/gauntlet.py` (the five tests +
`run_gauntlet`), and `src/neuroad/calibration.py` (every pinned number).
