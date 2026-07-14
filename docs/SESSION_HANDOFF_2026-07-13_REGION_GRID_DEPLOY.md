# Session Handoff — 2026-07-13 · Region-conditioned probe + precompute grid + DEPLOYED

Continue from here. **This session's work is COMMITTED, PUSHED, and DEPLOYED live.**

---

## TL;DR — what shipped (all live)

**Live URL:** https://neuroad-demo-31043195041.us-central1.run.app (revision `neuroad-demo-00013-k48`, 100% traffic)

1. **Anchor drives the science** (was cosmetic). FE sends the chosen anchor; backend routes mechanism + anchor-congruent lead (amyloid→APP, p_tau217→MAPT, gfap→TREM2) + organoid readout. PI4AD panel unchanged (honest).
2. **Region-conditioned probe** — new `adni:roi` FreeSurfer named-ROI cohort (45 Desikan-Killiany regions). A brain region in the hypothesis conditions the **whole referee** (naive AUROC + 5-gauntlet + leakage) via ONE seam. Real per-region AUROCs follow the AD Braak gradient (hippocampus 0.88 → precuneus 0.71 → pallidum 0.51 chance-control).
3. **Real metadata + biomarkers on adni:roi** — age/sex 100%, site/scanner 99.9%, amyloid 59%, apoe4 91%, gfap/nfl 25%, conversion 41%, **plasma p-tau217 30%** (Fujirebio). Full gauntlet runs → strong regions PROMOTED (→ translation), weak ones honestly KILLED.
4. **Latency 66–82s → <10ms** — precompute-grid cache + gated the unrendered ~40s grounding layer off the live path. Live compute only on a genuine cache miss.

---

## Git + deploy state

- **Branch:** `fullcircle-drawer-polish` (remote pushed). **PR #1** open → base `main`. https://github.com/stevenybuilder/NeuroAD/pull/1
- **Commits this session:** `f3c6cea` (anchor + region-conditioning + cache) and `2d75479` (45 regions + metadata join + Fujirebio p-tau217). Parent `4d1eccd`.
- **Deployed:** `neuroad-demo-00013-k48`. **Rollback:** `gcloud run services update-traffic neuroad-demo --region us-central1 --project project-flash-490419 --to-revisions=neuroad-demo-00012-7d7=100`
- **Do NOT force-push main** (diverged from a teammate's line). Keep the branch+PR flow.
- **Parallel-track files — NEVER commit** (owned by another local track, modified but not by me): `README.md`, `app/demo_data.json`, `data/registry.yaml`, `docs/DEMO_SCRIPT.md`, `docs/FRAMING.md`, `reports/*`, `src/neuroad/scoring.py`. Also non-mine untracked: `app/knowledge_base.json`, `scripts/prove_live.py`.

---

## Architecture (how it works now)

**Anchor** → FE `onInvestigate` POSTs `anchor` (only when the composer chose one; default entry stays cohort-dominance). `server.py` reads it → `orchestrator.investigate(..., anchor)` → `_run_supervised` → `pipeline.run_referee(..., anchor)` → `_translate(card, df, anchor)` → `translation.translate(mechanism, df, anchor=...)`. Maps: `_ANCHOR_MECHANISM`, `_ANCHOR_LEAD`, `_ANCHOR_READOUT` in `translation.py`; `bridge._route(df, anchor)`. FE wet-lab lead reads `tx.top_target` (not `ranked_targets[0]`).

**Region** → `pipeline.run_referee` seam (right after claim parse, before `_naive_effect`): `region.extract_region(claim.claim_text, df)` → `contract.restrict_to_region(df, cols)`. Because every consumer reads features via `contract.embedding_matrix` (emb_ prefix), subsetting emb_* ONCE conditions the whole pipeline — no probe/gauntlet edits. **Only `adni:roi` carries `df.attrs['region_columns']`** (built in `freesurfer_roi.py`); every other cohort → region is a silent no-op. `server.py _enrich_case` ALSO restricts df (so the FE header `leakage_margin.outcome_auc` is region-specific — this was a real bug, fixed).

**Cache** (`app/investigate_cache.py`) — key `dataset|target|region|anchor|want_api`. `target` from `_infer_target` (df-free), `region` from `_region_for_key` (loads df ONLY for region-capable datasets, so combat hits stay fast). Hit → `personalize()` returns the real cell with the user's hypothesis text. Miss → compute live + back-fill. mtime-reload picks up a warm job's writes. Warmed by `scripts/warm_investigate_cache.py`.

**Grounding gate** — `_real_case(..., live=True)` (server path) skips `include_grounding` (~40s LOO attribution, NOT rendered by the FE — the attribution line is FE `pathwayLabel`). Offline bake keeps it.

---

## Key files (all mine, committed)

- `src/neuroad/harness/translation.py` — anchor maps + translate(anchor=)
- `src/neuroad/claude/bridge.py` — `_route(df, anchor)`
- `src/neuroad/pipeline.py` — region seam + `_region_attribution` + anchor thread
- `src/neuroad/harness/orchestrator.py` — anchor thread
- `src/neuroad/contract.py` — `Claim.region/region_columns` + `restrict_to_region`
- `src/neuroad/harness/region.py` — deterministic region extractor (45-region alias table)
- `src/neuroad/data/freesurfer_roi.py` — `adni:roi` loader (emb_i = ROI volume; dynamic REGION_ORDER from the csv)
- `src/neuroad/data/loaders.py` — `adni:roi`/`adni:freesurfer`/`adni:fsx` branch
- `app/server.py` — anchor payload, cache fast-path, `compute_investigate`, region restrict in `_enrich_case`
- `app/build_demo_data.py` — `_real_case(live=)` grounding gate
- `app/investigate_cache.py` + `app/investigate_cache.json` (105 cells) — the grid
- `scripts/build_adni_roi_table.py` — ETL: DATADIC-parse 45 ROIs + metadata join (age/sex from PTDEMOG.rda, site/scanner MRIMETA, p_tau217 Fujirebio pT217_F, amyloid ADSP, ab42_40 C2N, apoe4 APOERES, gfap/nfl Fujirebio, conversion DXSUM). Writes `data/real/_gated/adni_roi.csv` (gitignored) + local crosswalk.
- `scripts/warm_investigate_cache.py` — grid preloader

---

## Run locally

```
cd neuroad-discovery-engine
kill -9 $(lsof -ti:8080); sleep 2
set -a; source .env; set +a          # ANTHROPIC_API_KEY -> live Claude
NEUROAD_N_BOOT=200 NEUROAD_N_PERM=200 PYTHONPATH=src .venv/bin/python -m app.server &
# demo: http://localhost:8080/neuroad.html
```
Restart after backend edits. Rebuild ROI cohort: `PYTHONPATH=src .venv/bin/python -m scripts.build_adni_roi_table`. Re-warm grid (after ANY cohort/engine change, so cache matches live): clear `app/investigate_cache.json` then `PYTHONPATH=src .venv/bin/python -m scripts.warm_investigate_cache` (~14 min for 105 cells). **Warm in a shell WITHOUT NEUROAD_N_BOOT leaking** only if you want full 1000-iter rigor; the deployed grid used N_BOOT=200 (matches the live path).

Deploy: use the `deployment` agent (owns Cloud Run mechanics + secret safety). Verified commands are in the agent's memory / prior handoff. Pre-flight: `gcloud meta list-files-for-upload | grep -iE '\.env|secret|token|_manifests|crosswalk|download/'` MUST be empty; `adni_roi.csv` + `investigate_cache.json` MUST be present.

---

## Data facts (important, verified this session)

- **GCS bucket `gs://neuroad-adni-project-flash-490419` is IMAGING-ONLY** (NeuroJEPA embeddings + raw T1.nii.gz + crosswalks). No plasma/clinical there. All plasma/clinical came from LOCAL `../download/` + `../download (1)/` raw LONI exports + the `ADNIMERGE2.tar.gz` R package (read via `pyreadr`, which IS installed in the venv).
- **plasma p-tau217 source = UPenn Fujirebio `pT217_F`** (~1593 subjects, real pg/mL). This is what BOTH `adni:combat` (via `scripts/build_adni_contract.py`: `p_tau217 = pT217_F`) and now `adni:roi` use. The Lilly MSD600 assay is a thin sub-study (~278) — do NOT use it as the primary. Combat's "p_tau217" is confirmed real *plasma* p-tau217, NOT CSF p-tau181 (an earlier false-alarm flag was retracted).
- Broad tables inside `ADNIMERGE2.tar.gz`: `FNIHBC_BLOOD_BIOMARKER_TRAJECTORIES.rda` (multi-assay plasma p-tau217, 393-subj deep sub-study), `UPENNBIOMK_ROCHE_ELECSYS.rda` (CSF p-tau181, ~1650 — a DIFFERENT biomarker, do not relabel), `PTDEMOG.rda`/`ADSL.rda` (age/sex).

---

## STILL OPEN / TODO (none block the live deploy)

1. **Full ~450 grid** — only region×{dx_binary,conversion}×none (90 cells) is warmed. The region×**anchor** multiplication (×5) and cohort strata (APOE4/amyloid split) are the designed ~450–1000 cells (validated manifest: 25 curated regions × 8 anchors × outcomes). Warming all is a ~2.5hr offline batch; ship partial + live fallback, or run the batch in CI. Region cell compute is ~5–15s each (full gauntlet).
2. **Live-compute fallback layers (speed cache MISSES)** — NOT built. Design: (a) stage-level memoize the gauntlet on `(cohort,target,region)` — it does NOT depend on anchor, so a miss reuses it + recomputes only the ~0.6s translation; (b) precompute the cohort PCA/standardization basis; (c) `n_reps` budget lever for the live path. Note: N_BOOT is NOT the lever (gauntlet time is flat vs N_BOOT); cost is base CV fitting (`_naive_effect` ~11s, gauntlet ~15s on 2951×323).
3. **Dynamic decision tree** — the FE `nodes` array (`neuroad.html:1938`) is a STATIC scaffold; `_applyCaseToTree` only overwrites node subtitles. Make branch topology/labels derive from the real gauntlet outcomes (which confounds fired/survived, routed mechanism, real AUC). FE work, honesty-safe.
4. **Story-card scroll overlap** — ranked-targets card overlaps the card below on canvas scroll (removing the red box shortened it but layout wasn't reworked). Chrome automation extension WORKED this session (drive it to verify).
5. **ICV Blocker A** — the displayed headline AUROC (`leakage_margin.outcome_auc`) is RAW ROI volume, head-size inflated (bigger heads → bigger ROIs). `icv` IS shipped as a covariate; residualize it in the outcome-AUC path (`leakage.py`/`point_head`) to make the headline ICV-adjusted, OR label it "raw-volume." One substantive stat decision.
6. **`adni:roi` not in the `AVAILABLE` datasets list** (`loaders.py:141`) — cosmetic; queries work (loader resolves it). Add for a clean `/api/health`.
7. **FRAMING.md §14 wording** — says `/api/investigate` "recomputes live per request, not from a cache." With the grid cache this is inaccurate for a cache HIT (it's a precomputed real result). User chose the HYBRID posture (precompute + live fallback) and wanted NO UX/badge changes. §14 is parallel-track-owned — flag to that track to reconcile the doc; do not edit the FE provenance/UX.
8. **p-tau217 coverage on adni:roi** could rise from 30%→~47% by joining Fujirebio pT217_F at NEAREST-visit (not baseline-only) + triangulating C2N/Lilly, matching `build_adni_contract.py`'s `_plasma_nearest`.

See prior handoff `docs/SESSION_HANDOFF_2026-07-13_LATENCY_AND_HONESTY.md` for the earlier latency/anchor context.
