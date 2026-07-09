# ADNI swap-in runbook

When LONI/IDA downloads come back (see the open support thread — the ARC Builder /
"Study Files" workspace was hanging on "Building your workspace"), this is the
**2-minute path** from raw ADNI files to a live real-data referee run. Nothing
downstream changes — everything reads the same contract table.

The seam is already built and tested (`src/neuroad/data/gated.py`,
`scripts/adni_to_contract.py`, wired into `loaders.load("adni")`). Until a real
file exists, `loaders.load("adni")` transparently returns the clearly-marked
stub, so the pipeline runs today and **auto-upgrades** the moment you drop the
real export in.

## 1. Download these tables from IDA (Search & Download → Study Files)

Tabular only — no raw DICOM/NIfTI (the preprocessing tax rules those out for the
week; ADNIMERGE ships FreeSurfer summary columns already).

| Purpose | ADNI table | Gives |
|---|---|---|
| **Core** (required) | **ADNIMERGE** | dx, longitudinal DX (→ conversion), age, sex, SITE, FLDSTRENG (scanner), APOE4, AV45 (→ amyloid), and the FreeSurfer summary columns (Hippocampus, WholeBrain, Ventricles, Entorhinal, MidTemp, Fusiform, ICV) that become the structural embedding |
| Plasma anchor | **UGOT plasma p-tau217** | p_tau217 (the molecular anchor no open set has) |
| Plasma anchor | **ADNI plasma Simoa (GFAP, NfL)** | gfap, nfl |
| (optional) richer FreeSurfer | **UCSF FreeSurfer cross-sectional (UCSFFSX)** | only if you want ROI-level features beyond ADNIMERGE's summary set |

ADNIMERGE alone already lights up **diagnosis + site/scanner leakage + brain-age +
replication** on real data. The plasma files are what unlock the **biomarker-anchor**
gauntlet beat.

## 2. Map to the contract (one command)

```bash
cd neuroad-discovery-engine
source .venv/bin/activate && export PYTHONPATH=src

python scripts/adni_to_contract.py \
    --adnimerge ~/Downloads/ADNIMERGE.csv \
    --plasma    ~/Downloads/UGOT_PTAU217.csv \
    --plasma    ~/Downloads/ADNI_PLASMA_SIMOA.csv \
    --out       data/real/_gated/adni.csv
```

`--plasma` is optional and repeatable; each file is auto-scanned for an `RID` key
and any p-tau217 / GFAP / NfL column by name pattern. The script:

- derives **conversion** from the longitudinal DX trajectory (baseline MCI/EMCI/LMCI
  that later reaches Dementia → 1; stable MCI with follow-up → 0; else `<NA>`),
- thresholds **amyloid** from AV45 SUVR (≥ 1.11 → positive; falls back to CSF ABETA),
- joins the **plasma** markers by RID (baseline visit),
- hands off to the tested `gated.map_export` for dx banding, `emb_*`, dtypes, and
  `contract.validate_table`.

It prints a coverage report; sanity-check that `conversion` and the plasma columns
are non-zero.

## 3. Run the referee (no code change)

```bash
# CLI — 'adni' now resolves to data/real/_gated/adni.csv automatically
python -m neuroad.cli run adni "MRI embeddings predict which MCI patients convert to AD"

# or programmatically
python - <<'PY'
from neuroad.data import loaders
from neuroad import pipeline
df = loaders.load("adni")            # real if the file exists, else stub
print("real:", not df.attrs.get("is_stub"))
card = pipeline.run_referee(df, "MRI embeddings predict MCI-to-AD conversion")
PY
```

## What ADNI upgrades vs. the current (open + synthetic) demo

- **Real conversion** labels replace the synthetic conversion signal.
- **Real plasma p-tau217 / GFAP / NfL** replace the synthetic biomarker anchor —
  the one beat no open cohort can cover.
- Diagnosis, site/scanner leakage, and brain-age were **already real** (OASIS /
  OpenBHB / Neuro-JEPA); ADNI just adds multi-site scanner breadth.

## Honest gaps to keep in mind

- **amyloid** needs a threshold choice (AV45 ≥ 1.11 is the florbetapir convention;
  adjust in `derive_amyloid` if you prefer a cohort-specific cutoff).
- If a plasma file uses non-standard headers the auto-scan misses, either rename its
  marker column to `p_tau217` / `gfap` / `nfl`, or add the header to
  `_MARKER_PATTERNS` in `scripts/adni_to_contract.py`.
- Coverage will be partial (plasma p-tau217 is measured on a subset) — the product's
  completeness labels ("holds on complete subset, n=X") already handle this.
