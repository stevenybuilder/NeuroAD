# Data Access Guide — what unlocks what capability

The NeuroAD Discovery Engine runs against **one frozen interface**
(`src/neuroad/contract.py`). Every dataset below becomes a feeder by mapping its
structural features into `emb_0..emb_k` and its clinical fields into the contract
columns. Because the contract is the only seam, **swapping a real cohort in is a
file swap, not a code change** (see [Drop-in](#the-drop-in-zero-code-swap)).

This guide is deliberately honest about what needs **you** (a signature, an
account, an institutional affiliation) versus what runs **now, no login**.

---

## Lead with capability: what unlocks what

The Referee's gauntlet has five dimensions (`contract.GAUNTLET`). Two are the
⭐ heavy-weight tests — site/scanner leakage and brain-age. Here is which data
unlocks which capability, in the order you should pursue them:

| You want to run… | Cheapest data that unlocks it | Access cost |
|---|---|---|
| Offline demo, both verdicts (SURVIVOR / KILL) | `synthetic:*` (in-repo) | none |
| Real conversion + cohort/batch leakage on AD data | **OASIS-1/2** (vendored) | none |
| **Real scanner/site-leakage ⭐ NOW** + brain-age control | **OpenBHB** (vendored HF mirror) | none, healthy-only |
| Real scanner-leakage ⭐ **WITH AD labels** + FreeSurfer clustering | **OASIS-3** | ~1-week DUA, needs **you** |
| **The plasma p-tau217 biomarker anchor** on real AD | **ADNI** | LONI committee, needs **you** + institution |
| Ground-truth site-leakage ⭐ on real multi-center data | **NACC** | ~3-day request, needs **you** |
| Biomarker anchor for early/preclinical survivors | **EPAD** | ADDI application, needs **you** |

**The short version:** OpenBHB (no-login) gives you a *real* scanner-leakage and
brain-age test today. OASIS-3 (one signature, ~1 week) is the single most
valuable next unlock — it is the only easily-reachable set that pairs
multi-scanner heterogeneity with AD labels. ADNI is the biomarker-anchor
end-game but the slowest and needs an institutional affiliation.

---

## Tier 0 — runs now, no login (vendored in this repo)

### OpenBHB — real scanner/site-leakage + brain-age, TODAY
- **What you get:** 3,984 healthy T1 subjects, 60+ sites / multi-acquisition,
  CAT12/FreeSurfer-style structural features (`tiv, csfv, gmv, wmv`), a
  `siteXacq` batch column, wide age (6–88). **No disease label** (healthy-only).
- **Why it matters:** the ⭐ leakage and brain-age tests need real multi-scanner
  heterogeneity, which OASIS-1/2 (single-scanner) and IXI (raw-only) lack. This
  is the only vendored set that supplies it.
- **Access:** none. Vendored at `data/real/openbhb_participants.tsv` from the
  Apache-2.0 HF mirror `huggingface.co/datasets/benoit-dufumier/openBHB`.
- **Needs you?** No.

### OASIS-1 + OASIS-2 — real AD labels + conversion, no login
- **What you get:** OASIS-1 (436 subjects, wide age 18–96) + OASIS-2 (150
  subjects, real MCI→AD converters). Structural features `nWBV, eTIV, ASF`;
  labels from CDR/MMSE. Single-scanner (Siemens 1.5T).
- **Why it matters:** the real conversion probe, the brain-age control cohort,
  and a genuine longitudinal replication split. Single-scanner, so its leakage
  star is reframed as OASIS1-vs-OASIS2 **cohort/batch** leakage.
- **Access:** none. Vendored at `data/real/oasis_*.csv`. The canonical source is
  the OASIS DUA at `sites.wustl.edu/oasisbrains`.
- **Needs you?** No (already vendored).
- **Loader:** `neuroad.data.real.load_oasis("both")` / `loaders.load("oasis")`.

### IXI — healthy multi-site, but raw-only and 403s automation
- **What you get:** ~600 healthy T1/T2/PD subjects, 3 sites / 3 scanners, age
  20–86. CC BY-SA.
- **The honest catch:** the host (`biomedic.doc.ic.ac.uk`) **403s automated
  fetches** — a human browser download is required — and it ships **raw NIfTI
  only** (no derived volume table; you must run FreeSurfer/CAT12 yourself). For a
  no-login multi-site substrate, prefer OpenBHB, which is already vendored and
  tabular.
- **Needs you?** Yes (a manual browser download + your own preprocessing).

---

## Tier 1 — one signature / self-serve registration (needs you)

### OASIS-3 — the single most valuable gated unlock ⭐
- **Capability unlocked:** real scanner-leakage ⭐ **with AD labels** + rich
  FreeSurfer volumetric clustering. The multi-scanner (Siemens 1.5T Vision, 3T
  TIM Trio, 3T BioGraph) + AD-label combo is exactly what OpenBHB/IXI lack.
- **Step-by-step:**
  1. Go to `sites.wustl.edu/oasisbrains` and sign the OASIS **Data Use
     Agreement** (name / email / institution / intended-use). **No committee.**
  2. Wait ~**1 week** for an XNAT Central (`central.xnat.org`) invite to the
     `OASIS3` project.
  3. Export the **FreeSurfer volumetric spreadsheets** + the clinical
     **CDR/MMSE CSVs**. Bulk download via `github.com/NrgXnat/oasis-scripts`.
- **Time:** ~1 week. **Needs you?** Yes (your signature; personal email is fine).
- **Note:** there is **no lawful no-login mirror** — do **not** use dubious
  Kaggle repackages.
- **Drop-in:** `gated.load_gated(csv_path, "oasis3")`.

### MIRIAD — compact external replication check
- **What you get:** 708 serial T1 scans (46 AD / 23 CN), single-scanner, raw
  NIfTI only.
- **Step-by-step:** self-serve registration at
  `miriad.drc.ion.ucl.ac.uk/atrophychallenge`; download immediately.
- **Time:** minutes. **Needs you?** Yes (a free account). Single-scanner, so no
  leakage-star help; a compact external replication check only.

---

## Tier 2 — data-request form, short turnaround (needs you)

### NACC — the strongest real site-leakage substrate
- **Capability unlocked:** the ground-truth site-leakage ⭐ on real
  **multi-center** data — the strongest real scanner/site heterogeneity of any
  cohort here.
- **Step-by-step:**
  1. Submit the data request form at `naccdata.org`.
  2. Turnaround ~**3 business days**.
  3. Receive the **UDS tabular** data (+ optional MRI). Plasma markers
     (p-tau217/GFAP/NfL) available.
- **Time:** ~3 business days. **Needs you?** Yes.
- **Drop-in:** `gated.load_gated(csv_path, "nacc")`.

---

## Tier 3 — formal application / committee (needs you + institution)

### ADNI — the plasma p-tau217 biomarker anchor (richest, slowest)
- **Capability unlocked:** the biomarker-anchor gate on real AD data — plasma
  **p-tau217** and GFAP, FreeSurfer tables, amyloid PET, AD labels, multi-scanner.
  This is the richest cohort and the intended biomarker anchor.
- **Step-by-step:**
  1. Apply through the LONI Image & Data Archive (IDA) at `adni.loni.usc.edu`.
  2. A **committee** reviews (~**2 weeks**). You generally need an **institutional
     affiliation** — unaffiliated / personal-email applicants are usually denied.
  3. Export the **UCSF Cross-Sectional FreeSurfer** table + the **plasma
     biomarker** tables.
- **Time:** ~2 weeks + review. **Needs you?** Yes (and an institution).
- **Drop-in:** `gated.load_gated(csv_path, "adni")`.

### EPAD — biomarker anchor for early / preclinical survivors
- **Capability unlocked:** CSF + plasma + MRI with prodromal staging — the
  biomarker anchor for early-stage (preclinical/prodromal) survivors.
- **Step-by-step:** apply via the **Alzheimer's Disease Data Initiative** / AD
  Workbench (`ep-ad.org` → ADDI).
- **Time:** application-dependent. **Needs you?** Yes.
- **Drop-in:** `gated.load_gated(csv_path, "epad")`.

---

## The drop-in (zero-code swap)

Each gated cohort has a **hand-written stub** at
`data/real/_stubs/<name>_stub.csv` that matches the contract schema
column-for-column. Until you have real access, the feeder returns the stub and
**marks it clearly** so no placeholder number is ever mistaken for a result:

```python
from neuroad.data import gated

# No real file yet -> transparently returns the stub, marked as such.
df = gated.load_gated(csv_path=None, dataset="oasis3")
assert df.attrs["is_stub"] is True        # <- the honesty flag

# Once access is granted, point at the real export. Nothing else changes.
df = gated.load_gated("~/Downloads/oasis3_freesurfer.csv", "oasis3")
assert df.attrs["is_stub"] is False and df.attrs["source"] == "real"
```

`load_gated` accepts **two source shapes** (see `src/neuroad/data/gated.py`):

1. **Already in contract shape** — the file carries
   `subject_id, dx, …, emb_0..emb_k` (the stubs, or a pre-mapped export). It is
   coerced to contract dtypes and validated. A real file matching the stub
   schema is a literal file swap.
2. **A raw FreeSurfer + clinical export** — the file carries the source's own
   column names. `gated.GATED_CONFIGS[<name>]` lists, per contract column, the
   **candidate source column names** we look for, and the structural feature
   columns we standardize into `emb_*`. Diagnosis is normalized
   (CN/NL/HC→CN, EMCI/LMCI/SMC→MCI, Dementia/DAT→AD) or, if only CDR is present,
   banded (0→CN, 0.5→MCI, ≥1→AD). If your export uses headers not in the
   candidate list, either add them to the config or rename the columns — still
   zero change to any downstream module.

Every path ends by calling `contract.validate_table`, so a malformed export
fails fast and loudly rather than silently poisoning a verdict.

> **Wiring note:** `gated.load_gated` is dispatch-compatible (name → contract
> table, exactly like `loaders.load`) but is intentionally **not** wired into
> `loaders.py` here — that file is owned by another track. Wiring is a one-line
> addition there when desired.

---

## The Neuro-JEPA embedding path (documented, not shipped)

`notebooks/neurojepa_embeddings.ipynb` scaffolds the "real frozen Neuro-JEPA
structural embeddings" feeder. It is an **honest future path, not a shipped
feature**, for two reasons:

1. **Gated weights that auto-deny us.** The Neuro-JEPA weights are HF-gated
   behind an **institutional-PI agreement that auto-denies personal / gmail
   emails**, and the assistant cannot accept such an agreement on anyone's
   behalf.
2. **A preprocessing tax.** Even with weights, each raw T1 needs FSL FLIRT
   registration to MNI152 1mm, skull-stripping, and bias-field correction before
   the frozen 122M-parameter encoder runs (inference-only; fits a Colab free-tier
   T4).

The notebook documents the whole pipeline with the gated step clearly guarded as
pseudocode, so a PI-affiliated user can complete it — but nothing in the shipped
referee depends on it. The weight-free structural feeders (OASIS, OpenBHB, the
gated FreeSurfer drop-ins above) satisfy the same contract today.
