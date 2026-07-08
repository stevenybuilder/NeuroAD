---
name: dataset-connection
description: >-
  Connect open neuroimaging tabular datasets (OASIS-1/2, OpenBHB) to the NeuroAD
  Referee embedding contract, compute REAL scanner/site leakage on healthy
  brains, and drop in new feeders with zero downstream code change. Use when
  wiring a new cohort into `loaders.load(...)`, reproducing/verifying the
  vendored open data, mapping a raw manifest into a contract table, or
  demonstrating the STAR leakage mechanic on real (not synthetic) data.
---

# Dataset Connection Skill (NeuroAD Referee)

## Purpose

Everything downstream of NeuroAD Referee (the gauntlet, the Detective clustering,
the Bridge biology, the demo UI) reads ONE frozen interface: the embedding-table
**contract** in `src/neuroad/contract.py`. Once a dataset is mapped to that
contract, swapping data sources changes exactly one thing — the table, not the
code. This skill documents the proven, bug-free ways to connect open
neuroimaging tabular data to that contract:

- Map **OASIS-1 + OASIS-2** into one contract table (`real.load_oasis`).
- Map the **OpenBHB** HuggingFace TSV mirror into one contract table
  (`openbhb.load_openbhb`).
- Compute **real scanner/site leakage** on OpenBHB healthy brains
  (`openbhb.real_scanner_leakage`, AUC ~0.89 scanner / ~0.78 site).
- Reproducibly **download/validate** the open CSV/TSV files offline
  (`scripts/download_open.py`, `scripts/download_openbhb.py`).
- **Add a new feeder** with the manifest→contract→CSV recipe and
  `gated.load_gated` for zero-downstream-change integration.

All actions run **offline, with no credentials** — the two open cohorts are
vendored under `data/real/`.

## When to use

- You want to wire a new cohort into `loaders.load(name)`.
- You need a real (not injected) demonstration of the site/scanner leakage STAR.
- You need to re-download or integrity-check the vendored open data.
- You have a per-subject feature matrix + clinical manifest and want a drop-in
  contract CSV.

Repo root referenced throughout: `/Users/stevenyang/Documents/claude-life-sciences-hack/neuroad-referee`.

## Prerequisites & Setup

```bash
cd /Users/stevenyang/Documents/claude-life-sciences-hack/neuroad-referee
source .venv/bin/activate
export PYTHONPATH=src
```

- Vendored open data lives under `data/real/`:
  - `oasis_cross-sectional.csv` (OASIS-1, 374 lines)
  - `oasis_longitudinal.csv` (OASIS-2, 437 lines)
  - `openbhb_participants.tsv` (OpenBHB, 3985 lines = 3984 data rows, TAB-separated)
- Everything here runs with **zero network access**. The download scripts are a
  reproducibility utility, not part of the demo path.

## The Embedding Contract (target shape)

Source of truth: `src/neuroad/contract.py` (`CONTRACT_VERSION = "1.0.0"`).

One row per subject. The embedding is stored as columns `emb_0 .. emb_{D-1}`.
The fixed non-embedding columns and dtypes (`METADATA_COLUMNS`):

| column       | dtype     | notes                                             |
|--------------|-----------|---------------------------------------------------|
| `subject_id` | string    | unique per row                                    |
| `dx`         | category  | `DX_LEVELS = ["CN","MCI","AD"]`                    |
| `conversion` | Int8      | MCI→AD 1/0/`<NA>` (NA if not MCI or unknown)       |
| `age`        | float64   | years                                             |
| `sex`        | category  | `SEX_LEVELS = ["M","F"]`                           |
| `site`       | category  | acquisition site / study                          |
| `scanner`    | category  | scanner model / field-strength label              |
| `amyloid`    | Int8      | 1/0/`<NA>`                                         |
| `p_tau217`   | float64   | plasma p-tau217, `<NA>` allowed                   |
| `gfap`       | float64   | plasma GFAP, `<NA>` allowed                       |
| `nfl`        | float64   | plasma NfL, `<NA>` allowed                        |
| `apoe4`      | Int8      | APOE e4 count 0/1/2, `<NA>` allowed               |

Coverage gaps (NaN / `<NA>`) are **allowed by design** — `validate_table` does
not reject them; `cohort_summary()` reports coverage.

Building blocks every feeder uses:

```python
from neuroad import contract

frame = contract.make_embedding_frame(X)   # (n, D) array -> emb_0..emb_{D-1}
frame.insert(0, "subject_id", ...)
frame["dx"] = pd.Categorical(dx, categories=contract.DX_LEVELS)
frame["sex"] = pd.Categorical(sex, categories=contract.SEX_LEVELS)
# Int8 nullable columns via pd.array([...], dtype="Int8")
# float64 marker columns via np.full(n, np.nan, dtype="float64")
contract.validate_table(frame)             # raises ContractError on violation
```

`validate_table` checks: all metadata columns present, at least one `emb_*`
column (when `require_embeddings=True`), `subject_id` unique, `dx` within
`DX_LEVELS`, `sex` within `SEX_LEVELS`.

## Dataset Registry & Dispatch

Source: `src/neuroad/data/loaders.py`. One-name dispatch:

```python
from neuroad.data import loaders

loaders.load("oasis")          # real.load_oasis("both")
loaders.load("oasis:oasis1")   # OASIS-1 only
loaders.load("oasis:oasis2")   # OASIS-2 only
loaders.load("openbhb")        # openbhb.load_openbhb()
loaders.load("synthetic:SURVIVOR")
loaders.load("synthetic:KILL")
```

`loaders.AVAILABLE` is the human-facing catalogue:
`["synthetic:SURVIVOR", "synthetic:KILL", "oasis", "oasis:oasis1", "oasis:oasis2", "openbhb"]`.

## Connecting OASIS-1 + OASIS-2

Source: `src/neuroad/data/real.py` — `load_oasis(which="both")`,
`which ∈ {both, oasis1, oasis2}`.

**Files** (under `data/real/`):
- `oasis_cross-sectional.csv` → OASIS-1 (columns `ID`, `M/F`, `Age`, `CDR`,
  `eTIV`, `nWBV`, `ASF`).
- `oasis_longitudinal.csv` → OASIS-2 (columns `Subject ID`, `Visit`, `Group`,
  `CDR`, `M/F`, `Age`, `eTIV`, `nWBV`, `ASF`).

**Row selection**
- OASIS-1: keep only CDR-labeled rows (`raw["CDR"].notna()`) — others have no dx.
- OASIS-2: reduce to one **baseline** row per subject
  (`sort_values(["Subject ID","Visit"])`, keep `Visit == 1`, drop duplicates).

**Weight-free structural embedding** (standardized, `ddof=0`):
`[nWBV, eTIV, ASF]` plus two engineered ratios:
- `nWBV_x_eTIV = nWBV * eTIV`
- `brain_vol_proxy = nWBV * eTIV / ASF`

MMSE and CDR are **deliberately excluded** from the embedding — they *define* the
labels and would leak the answer.

**Label mapping**
- `dx`: `CDR == 0 → CN`, `CDR == 0.5 → MCI`, `CDR >= 1 → AD` (`_dx_from_cdr`).
- `conversion`: OASIS-2 `Group == "Converted" → 1`, `"Nondemented" → 0`, else `<NA>`.

**Identity & framing**
- `subject_id` prefixed `OAS1_` / `OAS2_` to prevent collisions.
- `site` = pseudo-site `OASIS1` / `OASIS2` — this exposes the **cohort/batch
  leakage** framing of the STAR (each OASIS cohort is effectively single-scanner,
  so `scanner` is a single value; the ground-truth scanner-confound KILL lives in
  the synthetic harness).
- No open OASIS plasma markers → `amyloid`, `p_tau217`, `gfap`, `nfl`, `apoe4`
  are all `<NA>`.

**Verification (proves it works)**
```python
from neuroad.data import loaders
from neuroad import contract
df = loaders.load("oasis")
contract.validate_table(df)               # passes
len(df)                                    # 385
df["dx"].value_counts()                    # CN=220, MCI=122, AD=43
df["site"].value_counts()                  # OASIS1=235, OASIS2=150
len(loaders.load("oasis:oasis1"))          # 235
len(loaders.load("oasis:oasis2"))          # 150
```
Tests: `tests/test_data.py::test_load_oasis_is_contract_valid`,
`::test_load_oasis_single_cohort`, `::test_loaders_dispatch` (part of 73/73 green).

## Connecting the OpenBHB HuggingFace TSV mirror

Source: `src/neuroad/data/openbhb.py` — `load_openbhb()`.

**File**: `data/real/openbhb_participants.tsv`, **TAB-separated**
(`pd.read_csv(..., sep="\t")`), 3984 healthy-control rows.

**Weight-free structural embedding** (standardized, `ddof=0`):
`[tiv, csfv, gmv, wmv]` (total intracranial / CSF / grey-matter / white-matter
volume). **`age` is intentionally EXCLUDED** — it is a covariate the gauntlet
adjusts for, not a structural feature; folding it in would blur the "structure
alone leaks the scanner" point.

**Label mapping**
- `dx = "CN"` for **every** subject (healthy-controls cohort).
- `sex`: `female → F`, `male → M`.
- `subject_id = "BHB_" + participant_id`.
- `site = "BHB_" + integer site code` (62 acquisition sites).
- `scanner` derived from `magnetic_field_strength` via `"{:g}T"` formatting →
  labels `1.5T` / `3T`.
- Healthy-controls cohort → `conversion` and all molecular markers `<NA>`.
- Dedupe `subject_id`, then `validate_table`.

**Verification (proves it works)**
```python
df = loaders.load("openbhb")
contract.validate_table(df)               # passes
len(df)                                    # 3984
set(df["dx"].dropna())                     # {"CN"}
df["scanner"].value_counts()               # 3T=3637, 1.5T=347
df["site"].nunique()                       # 62
```
Tests: `tests/test_openbhb.py` (7 tests: contract-valid, healthy-controls-only,
multiple field strengths, no molecular markers, age-not-in-embedding, loaders
dispatch) all pass.

## Computing Real Scanner/Site Leakage

Source: `src/neuroad/data/openbhb.py` — `real_scanner_leakage(df=None)`.
Driver: `scripts/real_scanner_leakage.py`.

Point the **single reused linear probe head** (`probe.point_head` +
`probe.cross_val_auc`) at the `scanner` (field strength) and `site` targets on
OpenBHB. Because every subject is a healthy control, there is **no disease
signal** — so a high AUC is pure acquisition batch effect, *measured* (not
injected, unlike the synthetic KILL).

- CV is **non-group-aware by design** (`groups=None`): holding out the very group
  you predict is degenerate; we *want* to see the machine/site signal.
- Returns `scanner_auc`, `site_auc`, per-target `detail` (`n`, `n_classes`),
  cited `prior_art` (from `calibration.PRIOR_ART`, e.g. "Batch Effects in Brain
  Foundation Model Embeddings", arXiv:2604.14441), and a plain-language `message`.

**Run it**
```bash
PYTHONPATH=src python scripts/real_scanner_leakage.py
# writes reports/openbhb_scanner_leakage.json, exits 0
```

**Verification (proves it works)**
- scanner AUC = **0.8911** (n=3984)
- site AUC = **0.7843** (62 sites)
- Test `tests/test_openbhb.py::test_real_scanner_leakage_is_strong` asserts
  `scanner_auc > 0.7` and passes.

## Reproducing / Verifying the Open Data

Both utilities are **idempotent**, need **no login**, and default to
fill-only-missing. Flags: `--check` (validate only, never download), `--force`
(re-download even if valid).

### OASIS CSVs — `scripts/download_open.py`
Targets `oasis_longitudinal.csv` (required cols include `Subject ID`, `Group`,
`Visit`, `CDR`, `eTIV`, `nWBV`, `ASF`; `min_rows=300`) and
`oasis_cross-sectional.csv` (required cols `ID`, `M/F`, `Age`, `CDR`, `eTIV`,
`nWBV`, `ASF`; `min_rows=400`). On force/missing/invalid, fetches from no-login
raw-GitHub mirrors and accepts only a CSV that parses with the expected schema
and row count.
```bash
python scripts/download_open.py --check    # prints [OK] for both, exits 0
```

### OpenBHB TSV — `scripts/download_openbhb.py`
Target `openbhb_participants.tsv`. Validation: required columns present,
`MIN_ROWS=3000`, and **≥2 distinct `magnetic_field_strength` values** (needed for
the scanner-leakage star). On force/missing/invalid, fetches the no-login
HuggingFace mirror (`benoit-dufumier/openBHB` `participants.tsv`, Apache-2.0).
```bash
python scripts/download_openbhb.py --check # prints [OK], exits 0
```
Vendored TSV has 3985 lines (3984 data rows > 3000) with 2 field strengths
(1.5T, 3T).

**Restoring a pruned file**: on a fresh checkout that dropped a `data/real/` file,
run the matching script without `--check` (or with `--force`) to re-fetch from
the mirror; the scripts also print a git-restore reminder if every mirror fails.

## Adding a New Feeder (drop-in recipe)

The reusable pattern (documented in `notebooks/neurojepa_embeddings.ipynb`,
section 4, and implemented identically by `real.py` and `openbhb.py`):

1. Build the per-subject feature matrix `X` (n × D).
2. `emb = contract.make_embedding_frame(X)` → `emb_0..emb_{D-1}`.
3. `emb.insert(0, "subject_id", ...)`.
4. Attach `dx`/`conversion`/`age`/`sex`/`site`/`scanner`/markers with the correct
   dtypes: `Categorical` with `contract.DX_LEVELS` / `SEX_LEVELS`, `Int8`
   nullable ints, `float64` markers.
5. `contract.validate_table(frame, require_embeddings=True)`.
6. Write CSV — it drops into `gated.load_gated` with **zero** downstream code
   change.

For gated/credentialed cohorts (ADNI, OASIS-3, NACC, EPAD) use
`src/neuroad/data/gated.py`:
```python
from neuroad.data import gated
df = gated.load_gated("~/Downloads/adni_ucsf_freesurfer.csv", "adni")
```
`load_gated` accepts either an already-contract-shaped file (coerce dtypes +
validate) or a raw FreeSurfer + clinical export (mapped via `GATED_CONFIGS`), and
transparently falls back to a clearly-marked stub (`df.attrs["is_stub"] = True`)
if no real file is supplied. `GATED_NAMES = ["adni", "epad", "nacc", "oasis3"]`.

> **GATED / future — does NOT run end-to-end.** The notebook's own Neuro-JEPA
> weight path is intentionally gated (guarded pseudocode; `emb_*` are NaN
> placeholders). Only the **schema-mapping / CSV-write pattern** is the proven
> reusable action — not live embedding generation. That same dtype/column recipe
> is exactly what passes `contract.validate_table` across all 73 tests.

## Verification Checklist

Run from the repo root with the venv active and `PYTHONPATH=src`:

```bash
# 1. Open-data integrity validators (each exits 0)
python scripts/download_open.py --check
python scripts/download_openbhb.py --check

# 2. Contract validity + row counts for each loader
python -c "from neuroad.data import loaders; from neuroad import contract; \
d=loaders.load('oasis'); contract.validate_table(d); print(len(d))"        # 385
python -c "from neuroad.data import loaders; from neuroad import contract; \
d=loaders.load('openbhb'); contract.validate_table(d); print(len(d))"      # 3984

# 3. Real scanner/site leakage (scanner AUC 0.8911, site AUC 0.7843)
PYTHONPATH=src python scripts/real_scanner_leakage.py

# 4. Full test suite (73 tests, incl. test_data.py and test_openbhb.py)
pytest -q
```

Expected: both validators print `[OK]`; OASIS = 385 rows (CN=220, MCI=122,
AD=43; OASIS1=235 / OASIS2=150); OpenBHB = 3984 rows (all CN; 3T=3637 / 1.5T=347;
62 sites); scanner leakage AUC = 0.8911; `pytest -q` reports 73 passing.
