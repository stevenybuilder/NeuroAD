# Session Handoff #2 — Predictive Power, Build-Out, ADNI Imaging

_Last updated: 2026-07-11. Branch: `feat/molecule-translation-loop`. Tests: **365 passed / 2 skipped** (green)._

This continues `docs/SESSION_HANDOFF_NEUROJEPA.md`. Read that first for Layer-1
(OASIS NeuroJEPA) context and the plasma-ensemble wiring.

---

## 1. TL;DR

This session (a) hardened and exercised the **triangulated plasma ensemble on real
ADNI data**, (b) built the **MCI→AD conversion** predictive task and its **multimodal
fusion** — the genuine predictive-power advance, (c) **built out four stub pipeline
layers** (L3 fusion, L5 network propagation, L6 targeting, Output validation) plus
**L6 Boltz-2 / L3 attention / L2 FastSurfer** via multi-agent workflows, (d) delivered
an **honest scientific assessment** of what the pipeline can and cannot predict, and
(e) got the **ADNI raw T1w MRI onto Colab** (transfer problem solved via Drive+rclone),
though the **imaging embed itself is not yet complete** (blocked on a Colab runtime
numpy issue — see §7, fully recoverable).

**The headline honest result:** the pipeline has real, cross-validated predictive
power for **classification/prognosis**, and its **target-discovery half is an
unvalidated hypothesis engine** — exactly as the plan frames it.

---

## 2. Headline results (all leakage-free OOF, site-disjoint, permutation-tested)

| Task | Substrate | AUC | n | Note |
|---|---|---|---|---|
| ADNI AD-vs-CN (dx) | FreeSurfer contract (ComBat) | **0.92** | 2951 | promoted 100/100 |
| **MCI→AD conversion** | FreeSurfer structure alone | 0.644 | 1199 (412 conv) | promoted 77/100 |
| MCI→AD conversion | **plasma p-tau217 block** | **0.814** | 498 (142 conv) | strongest single modality |
| MCI→AD conversion | structure only (same 498) | 0.726 | 498 | |
| MCI→AD conversion | APOE + demographics | 0.671 | 498 | |
| MCI→AD conversion | **naive concat fusion** | 0.741 | 498 | **WORSE than plasma** (323-d structure dilutes 5-d plasma) |
| **MCI→AD conversion** | **stacked (late) fusion** | **0.827** | 498 | **best predictor** — beats plasma + concat |
| OASIS AD-vs-CN (prior session) | NeuroJEPA 768-d imaging | 0.87–0.88 | 210/150 | cross-cohort transfer 0.89 |

**Scientific takeaways (honest):**
- **Plasma p-tau217 is the dominant MCI→AD conversion predictor** (0.814), far above
  structure (0.726). Consistent with p-tau217 capturing the molecular cascade.
- **Naive concatenation fusion is counterproductive** (0.741 < 0.814): the high-dim
  structural block drowns the plasma signal. This is *why* modality-balanced fusion is
  needed — a real, non-obvious finding.
- **Stacked late fusion (combine per-modality OOF scores) is the best predictor at
  0.827** — recovers plasma + adds marginal structural lift. Above plasma on the point
  estimate (+0.012), not CI-separable at n=498 (CIs overlap). Honest: fusion helps but
  the plasma marker carries most of the signal.
- Reports: `reports/adni_conversion_card.json`, `reports/adni_conversion_multimodal.json`.

---

## 3. The predictive-power question — honest assessment

The pipeline makes **two claims** with very different validation status:

**Claim A — structural/multimodal → AD state/prognosis (L1–L4).** Real predictive
power beyond the training set: OOF cross-validated, permutation-tested, FDR-corrected,
and **holds across independent cohorts** (OASIS-1/2 + cross-cohort transfer 0.89; ADNI
promotes). Not brain-age; beats classical atrophy. MCI→AD conversion at 0.83 (fused) is
a genuine, clinically-meaningful prognostic signal on 498–1199 subjects. *Caveat:* the
OASIS imaging pooled AD-vs-CN rests on ~41 AD (wide CIs) until ADNI imaging lands.

**Claim B — AD biology → drug-target prioritization (L5/L6/Output).** **Not
outcome-validated.** The `harness/validation.py` harness (built this session) shows the
target ranking is **at/below chance against held-out known-AD-target gold sets** — the
only "significant" signal is circular (Open Targets scores derived from the same
evidence). This is **not a sample-size problem** — it's that computational retrospective
validation of drug discovery is near-impossible with ~15 known GWAS genes / ~9 drug
targets. The honest validation is **prospective wet-lab (organoid/iPSC)**, which the
plan itself specifies. **Position the discovery half as a rigorously-filtered hypothesis
engine, NOT a validated efficacy predictor.**

**One-line framing for the demo/writeup:** *"A cross-cohort-validated structural-MRI +
plasma AD classifier/prognostic model feeding a provenance-honest, wet-lab-testable
hypothesis engine"* — not *"an AI that predicts AD drug targets."*

---

## 4. Pipeline layer status (updated)

| Layer | Status | This session |
|---|---|---|
| Input: Raw MRI | built-unvalidated | ADNI T1w now downloaded (590 scans, on Drive) |
| L1 NeuroJEPA (768-d) | built+validated (OASIS) | ADNI imaging embed **in progress/blocked** (§7) |
| L2 U-Net segmentation | **scaffold built** | `structural_segmenter.py` (parse aseg + honest degrade) + `scripts/fastsurfer_volumes_colab.py`; **GPU run deferred** (research → FastSurfer) |
| Tabular feed (plasma+FS) | built+validated | **triangulated plasma executed on real data** (3 assays, Lilly fix → depth-3); +`ab42_40`,`pct_ptau217` |
| L3 Multimodal fusion | **built** | `integrations/fusion.py`: fitted concat head + **attention-weighted late fusion** + ablation/attribution + calibration + imaging seam |
| L4 Refinement (referee) | built+validated | unchanged (mature) |
| L5 PI4AD multi-omics | **built** | real STRING v12 subgraph + RWR/heat-diffusion propagation in `pi4ad.py`, wired to `translation.py` |
| L6 Molecular targeting | **built (open substitute)** | `integrations/boltz.py` BoltzClient (MIT Boltz-2, complex+affinity) — **snapshot empty, GPU fold deferred**; AF3 stays gated/unwired |
| Output prioritization | **validation built** | `harness/validation.py` — outcome-validation harness vs gold sets (honest negative result) |
| **NEW: MCI→AD conversion** | **built+validated** | `run_adni_conversion.py` (0.644) + `run_adni_conversion_multimodal.py` (**0.827 fused**) |

---

## 5. Files created / changed this session

**New scripts:**
- `scripts/run_adni_conversion.py` — MCI→AD conversion card (structure, 0.644).
- `scripts/run_adni_conversion_multimodal.py` — **multimodal conversion (0.827 stacked)**.
- `scripts/adni_colab_dicom_to_embed.py` — one-shot Colab driver: unzip→dcm2niix per
  IMAGEUID→skull-strip→NeuroJEPA embed→push CSV to Drive (detached-run ready).
- `scripts/organize_adni_downloads.py` — local DICOM→NIfTI organizer (crosswalk-driven).
- `scripts/build_adni_image_manifest.py`, `scripts/run_adni_crosscohort.py` (prior turn,
  + 768-d guard added), `scripts/boltz_fold_colab.py`, `scripts/fastsurfer_volumes_colab.py`,
  `scripts/build_string_subgraph.py`.

**New src modules:** `integrations/fusion.py`, `integrations/boltz.py`,
`integrations/string_ppi.py`, `integrations/structural_segmenter.py`,
`harness/validation.py`, `integrations/data/{string_ppi_subgraph.csv,string_snapshot.json,boltz_snapshot.json}`.

**Edited src:** `data/plasma_ensemble.py` (Lilly long-format fix → 3-assay triangulation),
`contract.py` (p_tau217 docstring + EXTENDED_BIOMARKER_COLUMNS), `integrations/pi4ad.py`
(+propagation), `harness/translation.py` (+network hubs, +boltz wiring),
`integrations/alphafold.py` (note), `data/real.py` (volumes join), `data/gated.py`.

**Data (gitignored, local):** `data/real/_gated/adni.csv` **rebuilt with triangulated
plasma** (union 1605, triangulated 957, +ab42_40/pct_ptau217).
`data/real/_manifests/adni_ptid_rid_crosswalk.csv` + `ida_imageids_*.txt` pick-lists.

**Reports (tracked):** `adni_conversion_card.json`, `adni_conversion_multimodal.json`,
`ADNI_CONVERSION.md`, updated survivor cards.

**Nothing committed** — all changes are in the working tree on `feat/molecule-translation-loop`.

---

## 6. The ADNI MRI transfer — SOLVED (operational learnings)

- **True uplink is ~161 Mbps** (Cloudflare test); the `colab upload` CLI is throttled to
  **~3 Mbps** (base64-through-kernel) and **fails on multi-GB files** (chunk timeout).
  Do NOT use `colab upload` for large data.
- **Working path: Google Drive + rclone.** User uploads zips to a Drive folder (browser),
  `rclone config` (leave client_id/secret BLANK — a fake value → `invalid_client`), copy
  `~/.config/rclone/rclone.conf` to the runtime, `rclone copy` Drive→/content over
  Google's backbone (**8.5 GB in ~21 s**). Headless Drive *mount* fails (interactive
  OAuth); Drive API also needs interactive auth — **rclone is the reliable route**.
- ADNI Drive folder: `1Qd754tBNX-CfkjYG_fztdjVszIbdM8Jh` (2 zips: 5.47 GB + 3.10 GB).
- The 590 DICOM series are the **exact FreeSurfer-anchor IMAGEUIDs** (crosswalk):
  **503 CN + 87 AD, all 3T, all plasma-linked.**

---

## 7. IN PROGRESS / BLOCKED: ADNI imaging embed

**State:** zips are safe on Drive. The detached-driver run **crashed at `import pandas`**
— the fresh Colab runtime came up with a **broken numpy** (`numpy._core._multiarray_umath`
import fails, version mismatch), before our code executed. Runtime was released.

**To resume (fresh session):**
1. `colab start --gpu t4` → SID.
2. Re-stage small files + `rclone.conf` + `hf_token.txt` (from repo `.env`) to `/content`
   (see `scripts/adni_setup_detached.sh` in scratch, or the upload loop in this handoff's
   history). Install rclone on runtime (`curl https://rclone.org/install.sh | sudo bash`).
3. `rclone --config /content/rclone.conf copy gdrive: /content --drive-root-folder-id 1Qd754tBNX-CfkjYG_fztdjVszIbdM8Jh --include "*.zip"`.
4. **Fix the numpy issue FIRST**: `colab exec -c "import subprocess; subprocess.run('pip install -q numpy==2.0.2 pandas', shell=True)"` (or pin to match the preinstalled stack), verify `import pandas` works, THEN run the driver.
5. Run **detached** (survives socket drops): `colab exec -c "import subprocess; subprocess.Popen(['python','-u','/content/adni_colab_dicom_to_embed.py'], stdout=open('/content/run.log','w'), stderr=subprocess.STDOUT)"` and poll `/content/run.log`.
6. Driver pushes `adni_neurojepa_embeddings.csv` to Drive on completion; pull it to
   `data/real/adni_neurojepa_embeddings.csv`, then `PYTHONPATH=src ./.venv/bin/python scripts/run_adni_crosscohort.py`.

**Known Colab dep hazard (from `colab-gpu-cli` skill):** the embed's `install_deps()`
can downgrade torch/numpy. Install with `--no-deps` and KEEP Colab's torch; consider
pinning numpy before importing pandas. This is the main thing to get right on resume.

**Note:** this embed is NOT blocking the core deliverable — MCI→AD multimodal (0.827)
and everything else is done without it. It adds a 3rd imaging cohort + tightens Claim A.

---

## 8. Recommended next steps (prioritized)

1. **Resume the ADNI imaging embed** (§7) — fix numpy, re-run detached → cross-cohort
   result + fold the 768-d imaging embedding into the L3 fusion seam (`attention_fusion(df, imaging_embedding=...)`) for a true imaging+plasma conversion model.
2. **Honesty-audit the 3 newest builds** (L6 Boltz-2, L3 attention, L2 FastSurfer) — they
   self-report tests-pass (365 green) but deserve a correctness/honesty review like the
   prior build-out got. Check `honesty_note`s in `tasks/w756j7s8a.output`.
3. **L6 real fold** (optional): run `scripts/boltz_fold_colab.py` on GPU to populate
   `boltz_snapshot.json` — lights up real complex/affinity targeting.
4. **L2 real volumes** (optional): run `scripts/fastsurfer_volumes_colab.py` on OASIS.
5. **Do NOT oversell Claim B** — keep target prioritization framed as hypothesis-generation
   for wet-lab; the validation harness proves it's not yet outcome-predictive.

---

## 9. How to reproduce the key results

```bash
cd neuroad-discovery-engine
# rebuild the triangulated-plasma contract (needs ../download gated CSVs):
PYTHONPATH=src ./.venv/bin/python scripts/build_adni_contract.py
# MCI->AD conversion (structure):
PYTHONPATH=src ./.venv/bin/python scripts/run_adni_conversion.py
# MCI->AD conversion (multimodal — the 0.827 result):
PYTHONPATH=src ./.venv/bin/python scripts/run_adni_conversion_multimodal.py
# full suite:
PYTHONPATH=src ./.venv/bin/python -m pytest -q      # 365 passed / 2 skipped
```

_Data files (`*embeddings*.csv`, `_gated/adni.csv`, volumes, weights, tokens) are
git-ignored and local-only. Reports under `reports/` are tracked._
