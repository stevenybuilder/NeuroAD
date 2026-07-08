# Gated-dataset STUBS — HARDCODED PLACEHOLDERS

> **These CSVs are NOT real data.** Every row is hand-written, fictional, and
> exists only so the pipeline has schema-valid input to exercise while the real
> datasets are behind registration / data-use agreements. Do **not** report any
> number computed from a stub as a scientific result.

## Why they exist

The genuinely open tabular data (OASIS-1 cross-sectional + OASIS-2 longitudinal)
is vendored at `data/real/oasis_*.csv` and needs no login. The four datasets
below require an application, a data-use agreement, or credentialed download, so
we cannot vendor them. Each is represented here by a tiny (5–7 row) stub that
**matches the frozen contract schema** (`src/neuroad/contract.py`) column-for-column.

| Stub | Real source | Access tier | Has plasma markers? |
|---|---|---|---|
| `adni_stub.csv`   | ADNI (adni.loni.usc.edu)          | gated application (LONI) | yes (p-tau217/GFAP/NfL) |
| `oasis3_stub.csv` | OASIS-3 (central.xnat.org)        | registration + DUA       | no (imaging + PET only) |
| `nacc_stub.csv`   | NACC UDS (naccdata.org)           | data-request form        | yes; **multi-center** (best future site-leakage substrate) |
| `epad_stub.csv`   | EPAD LCS (ep-ad.org / ADDI)       | gated application        | yes (preclinical biomarker anchor) |

## Drop-in replacement (zero code change)

Each stub carries the **exact contract columns**:
`subject_id, dx, conversion, age, sex, site, scanner, amyloid, p_tau217, gfap,
nfl, apoe4, emb_0..emb_k`. When real access is granted:

1. Export the real cohort into the same schema (map the source's structural
   features to `emb_0..emb_k`, standardized; map CDR/clinical status to `dx`;
   map converters to `conversion`).
2. Overwrite the corresponding `*_stub.csv` (or point the loader at the real
   file). Nothing downstream changes — the contract is the only interface.

## Loading

The first lines of each stub are `#` banner comments, so load with:

```python
import pandas as pd
df = pd.read_csv("data/real/_stubs/adni_stub.csv", comment="#")
```

`<NA>` values are encoded as empty fields (e.g. OASIS-3 has no plasma markers,
so `p_tau217/gfap/nfl` are blank — exactly as `contract.validate_table` expects).

## What is real in this repo

- `data/real/oasis_cross-sectional.csv` — real OASIS-1 (curl-verified, no login)
- `data/real/oasis_longitudinal.csv` — real OASIS-2 (curl-verified, no login)
- `src/neuroad/data/synthetic.py` — the offline synthetic harness (clearly synthetic)

Everything in **this** `_stubs/` directory is a placeholder.
