# Session handoff — 2026-07-13 — Deploy the live SFG imaging backend to Cloud Run

## TL;DR for the next session
The MRI-viz integration is **done, verified locally, and deployed**. `main` on GitHub is now our
integration work; the Cloud Run demo is live with the engine + Claude. **The one remaining task:
the brain-viz (MRI volumes, registration heatmap, silent-failure-guard live flags) is CACHED-ONLY
on the deployed link** because the SFG imaging backend was never containerized/deployed. Next task =
**deploy the SFG backend as a second Cloud Run service and point the main service at it.**

## Current state (all verified)
- **Branch / GitHub main:** `main` = `e648f19` (our full integration, force-pushed as a clean
  superset). Local work branch: `integrate-mri-viz` (same tip). Old main preserved as
  **`sid_fmri_features` = `fdfc476`** on origin (backup, do not touch).
- **Cloud Run (live):** https://neuroad-demo-31043195041.us-central1.run.app
  - Revision `neuroad-demo-00017-f8x`. Rollback: `gcloud run services update-traffic neuroad-demo
    --region us-central1 --to-revisions=neuroad-demo-00016-ssf=100`
  - Verified live: `/api/health` claude_live:true; `/api/ask` live opus-4-7; ADNI datasets live;
    demo_data.json rebuilt from engine.
  - **`/api/sfg/health` → 503** (no SFG service on Cloud Run — the gap to close).
- **Uncommitted WIP (leave alone):** app/claude_science.html, demo-animations/*, scene0b_validation.html.
- **Local SFG backend** is running on :8091 (started via the run_demo.sh uvicorn line); demo server :8080.

## What this session did (commits on main, newest first)
- `e648f19` Drop the mis-aligning MNI attention overlay from the default Brain-data view (clean MRI;
  per-finding heatmaps stay in the full report). *(The orange-wash bug: a bundled MNI-space ROI mask
  `/scans/roi_*.nii.gz` painted over a native-space real scan.)*
- `cb3ddcf` Registration heatmap works WITHOUT SynthStrip: classical `weak_strip` fallback in
  `mri_visualizations/backend/sfg/checks/registration.py`; IXI reference registered (copied
  data/_ixi_fallback → data/ixi, gitignored); frontend cohort prefers a Guy's IXI.
- `277a2c8` Cut MRI viewer latency ~10× : `/api/sfg/volume` serves a cached factor-2 downsample
  (`_preview_volume` in sfg/server.py, nibabel slicer preserves world space); frontend unmounts the
  drawer viewer while a modal is open.
- `8a3c8b9` Brain-viz core fix: cohort derived from `/api/sfg/scans` (env-agnostic real ids, was
  hardcoded to teammate's 10006/ADNI_10006), real default backdrop (`backdropUrl()`), BrainViewer
  separates WebGL-init (fatal) from volume-load (non-fatal) so a bad scan reads "Scan volume
  unavailable" not a false "needs WebGL2", ranked-targets node defaults to Brain data tab,
  `run_pipeline` tolerant of missing scan_ids.
- `761b79f` Entry-screen anchor-biomarker switcher (amyloid/p-tau217/GFAP/NfL) via onEntryInvestigate.
- `fbcb900` Retire the Claude·ask conversation rail (matches teammate's dock-only layout); honesty
  fix: guard-report caption gated on !cached.
- `3caf8bf` Merge teammate MRI-viz work (main fdfc476): guard full-report popup, heatmap fix, 3D
  markers, auto QC-summary, canvas notes/pins/dock + per-scan Claude agent, /api/sfg-summary route.

## THE NEXT TASK — deploy a live SFG imaging backend to Cloud Run

### Why it's not there
- `cloudbuild.backend.yaml` + `Dockerfile.backend` build ONLY the main engine. Nothing references
  `mri_visualizations`. The main server proxies `/api/sfg/*` → `SFG_BACKEND` env
  (`app/server.py:233`, default `http://127.0.0.1:8091`). On Cloud Run that env is unset → localhost
  → 503 → the frontend falls back to the cached snapshot (`sfgStatus:'cached'`, badged, no live
  volumes/heatmaps).
- The SFG backend (`mri_visualizations/backend/sfg`, FastAPI/uvicorn) was only ever a LOCAL dev
  service (started by run_demo.sh).

### What "live on the deploy" requires (proposed plan — validate before executing)
1. **Containerize the SFG backend.** New `mri_visualizations/Dockerfile` (or extend the build):
   base python3.12, `pip install fastapi "uvicorn[standard]" nibabel numpy scipy scikit-image
   "pydantic>=2" SimpleITK`, COPY `mri_visualizations/backend` + the scan data, CMD
   `uvicorn sfg.server:app --host 0.0.0.0 --port $PORT` (Cloud Run injects $PORT).
2. **Ship the scan data to that PRIVATE image** (NOT GitHub): fixtures (33M), a CURATED ADNI subset
   (currently 99M — trim to the ~6 scans the cohort uses), IXI reference (45M, public), and
   optionally pre-warm the downsampled cache (7.5M). ADNI stays de-identified (no PTIDs) per the
   deploy-data policy; a private Cloud Run image is the sanctioned place for the compute subset.
   Consider generating the ds2 downsample cache at build time so first-load is instant.
3. **Deploy as a 2nd Cloud Run service** (e.g. `neuroad-sfg`), internal/private ingress.
4. **Point the main service at it:** set `SFG_BACKEND=https://<neuroad-sfg-url>` on `neuroad-demo`
   (redeploy or `gcloud run services update neuroad-demo --set-env-vars SFG_BACKEND=...`).
5. **Verify:** `/api/sfg/health` → 200 on the live URL; open a node → Brain data → MRI renders (the
   ds2 volumes are ~2.3MB so ~4s); Full report → registration heatmap renders live.

### Gotchas / decisions for the next session
- **SynthStrip is NOT installed** and should stay out (torch ~2GB). Registration + skull-strip
  already fall back to the classical stripper — keep it that way; don't add torch to the image.
- **Cohort is env-agnostic** now (derived from `/api/sfg/scans`), so it will work with whatever
  scans ship in the image without code changes.
- **Downsample cache** (`config.CACHE_DIR/downsampled`) is generated on first request; pre-warm it
  in the image build for snappy first load, or accept a one-time ~0.5s per scan.
- **Honesty:** the deployed SFG must set `sfgStatus:'ok'` (live) only when the run truly succeeds;
  the cached snapshot fallback must stay badged `cached`. The registration heatmap is a controlled
  synthetic-misalignment DEMO on the IXI reference — keep it framed as such (it already is).
- **Data-safety:** GitHub push must never include secrets/weights/gated-data/PTIDs. ADNI compute
  subset → PRIVATE Cloud Run image only. Verify `.gitignore` before any push.
- **Automation flakiness (for anyone browser-testing):** the entry "Investigate" click often needs
  a second click after page load (focus timing). The app itself is fine.

## How to run locally (to reproduce/verify brain-viz)
`./run_demo.sh` (starts SFG :8091 + demo :8080). First run needs the venv + IXI fetch — see the
run_demo.sh header. Then open http://localhost:8080/ ; hard-reload after code changes.

## Reference
- Deploy mechanics: use the `deployment` agent (owns Dockerfile.backend, secrets, demo_data rebuild).
- Full plan/history of the integration: /private scratchpad INTEGRATION_PLAN.md (this session).
