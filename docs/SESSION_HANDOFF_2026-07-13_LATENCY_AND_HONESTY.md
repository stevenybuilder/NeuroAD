# Session Handoff — 2026-07-13 · Live latency + anchor honesty + declutter

Continue from here in a new session. Everything below is **local + uncommitted** unless stated. Demo runs at **http://localhost:8080/neuroad.html**.

---

## TL;DR of what shipped this session (all UNCOMMITTED, local only)

1. **Live `/api/investigate` latency: 44s → ~25s** (the `investigate()` core is now **10s**). Backend changes.
2. **Anchor now honors the user's chosen anchor with REAL stats** (amyloid → "anchored to amyloid positivity · Δ +0.35, n=1131"). Frontend.
3. **Removed the `anchor:'gfap'` node hardcodes** — that was the bug making Repurposing show GFAP on the default path. Anchor now reflects the chosen anchor consistently.
4. **Declutter:** all SNAPSHOT/REAL chips, the offshoot banner, "template protocol" verbosity, STRING-PPI caveat, cohort chip, the **red "Decision-support only" box**, and HuggingFace/`HF_ACCESS.md` text all removed. Live **Open Targets** grounding added to the wet-lab card.

Nothing is deployed. Cloud Run live revision is `neuroad-demo-00012-7d7` — **STALE** (predates all of the above).

---

## Files I changed this session (vs. the rest, which are parallel-session — DO NOT clobber)

**Mine (this session's fixes):**
- `app/neuroad.html` — all frontend fixes below.
- `app/server.py` — sets `NEUROAD_N_BOOT/N_PERM=200` (live budget); passes `xcard` into `_enrich_case`→`_investigate_block`; a `[PROFILE]` log line (line ~412, remove before ship if noisy).
- `app/build_demo_data.py` — `_investigate_block(..., xcard=None)` reuses the precomputed card instead of re-running investigate; `_real_case` reuses the gauntlet's leakage instead of recomputing.
- `src/neuroad/probe.py` — `N_BOOT/N_PERM` env-configurable (default 1000, `max(50, ...)`).
- `src/neuroad/gauntlet.py` — `test_replication` uses `N_BOOT` instead of hardcoded 1000.
- `src/neuroad/pipeline.py` — `run_referee(..., use_claude=True)` gates the 4 Claude calls (adjudicate/biology/review/narrate).
- `src/neuroad/harness/orchestrator.py` — `_run_supervised(..., use_claude)`; `investigate` passes `use_claude=api`.
- `src/neuroad/data/loaders.py` — `@functools.lru_cache` on `load()` (loads each cohort once per process).

**Parallel-session files — MODIFIED but NOT by me; leave them, they belong to another local track:**
`README.md`, `app/demo_data.json`, `data/registry.yaml`, `docs/DEMO_SCRIPT.md`, `docs/FRAMING.md`, `reports/*`, `src/neuroad/translation.py`, `src/neuroad/scoring.py`.

---

## Latency: root cause + what fixed it (important context)

Profiled the 42s call. It is **pure compute, not Claude/network on the drawer path**. Three duplicated/oversized costs:
1. **Bootstrap/permutation at 1000×1000** (~12,000 logistic refits) — dialed to 200 for the live path via `NEUROAD_N_BOOT/N_PERM`, set in `app/server.py` BEFORE any neuroad import. **Offline `build_demo_data.py` (run without those env vars) keeps 1000-iteration rigor, so baked-in numbers are unchanged.**
2. **The handler ran the whole investigation TWICE** — `_enrich_case`→`_investigate_block` called `orchestrator.investigate` a second time. Now reuses the handler's `xcard`.
3. **The cohort was reloaded per request** + leakage recomputed — now cached (`loaders.lru_cache`) and reused from the gauntlet.
4. **Live Claude narration** added ~30s when the key is present — now gated on `api` (drawer path `api=False` skips it; the Ask rail keeps live Claude).

Remaining ~15s is in `_real_case` case-enrichment (not yet chased). Next optimization target if you want <15s: profile `_real_case` (`app/build_demo_data.py:1338`) — likely more reused-vs-recomputed opportunities. Frontend already raised the fetch timeout 40s→120s (`neuroad.html` `onInvestigate`) so the call lands.

**Caveat:** if `NEUROAD_N_BOOT` leaks into the shell env, `build_demo_data.py` would bake 200-iter numbers. Regenerate `demo_data.json` only in a shell WITHOUT that env var. (Current `demo_data.json` is fine.)

---

## Frontend (`app/neuroad.html`) — key edits + where

- **`_anchorLine(anc, bm)` helper** + `_activeAnchor()` — the "anchored to plasma X" finding line now uses `anchorFocus || dominant_biomarker` and pulls matching real stats (amyloid uses `amyloid_delta/_n`; plasma markers use `_r/_n`; graceful p-tau217 fallback). Wired into the ranked-targets sigHeader and `proteinDataEl`.
- **`anchorChipLabel(node)`** — node-aware (but the `anchor:'gfap'` node hardcodes were REMOVED, so it's effectively `anchorFocus || dominant_biomarker`).
- **`OTGrounding` React component** (after `StructViewer`) — live Open Targets fetch (tractability + AD association), `OT_ENSEMBL` crosswalk for the 6 targets, renders null on failure (never faked). Wired into the wet-lab `experiments` card.
- **`StructViewer._lockWheel()`** — earlier scroll-zoom fix (wheel stays in the 3Dmol viewer). AlphaFold 3D viewer is **click-gated** in the Protein-data tab (click a target).
- Removed: `_storyBody` banner (now `return kids`), cohort chip in `storyShell`, the `tx.caveat` red box, all srcBadge/REAL/SNAPSHOT chips.

---

## STILL OPEN / TODO

1. **Card overlap on scroll** (user screenshot) — removing the red box shortens the card; not specifically reworked. Check the story-card layout on the canvas.
2. **Visual verification blocked** — the Chrome automation extension is wedged (`vendor.js: No Listener` errors, NOT our code). Verify by hand in the browser, or restart the extension.
3. **Deploy + PR (held all session):**
   - **Branch:** local is on `main` at `4d1eccd`. Remote branch `fullcircle-drawer-polish` == `4d1eccd` (pushed earlier). **All this session's work is uncommitted on top.** The PR was **never created** (a `gh pr create` heredoc-quoting error) — create it with `--body-file`, not an inline heredoc.
   - `main` **diverged** from the teammate's remote line (protein-tab + workspace polish); do NOT force-push. Keep using the branch+PR flow.
   - **Cloud Run deploy** (private image ships de-identified ADNI; GitHub stays clean). Use the `deployment` agent. Verified-safe commands from this session:
     ```
     gcloud builds submit --config cloudbuild.backend.yaml --project project-flash-490419 .
     gcloud run deploy neuroad-demo --image us-central1-docker.pkg.dev/project-flash-490419/cloud-run-source-deploy/neuroad-demo:backend \
       --region us-central1 --project project-flash-490419 --allow-unauthenticated \
       --set-secrets ANTHROPIC_API_KEY=anthropic-api-key:latest --set-env-vars ROOT_PAGE=claude_science.html --port 8080 --memory 1Gi --cpu 1 --quiet
     ```
   - Before building: `gcloud meta list-files-for-upload | grep -iE '\.env|secret|token'` must be empty (`.gcloudignore` handles it).

---

## How to run / verify locally

- **Demo server** (currently running, pid was `69838`, on :8080). Restart it after backend edits:
  ```
  cd neuroad-discovery-engine
  kill -9 $(lsof -ti:8080); sleep 2
  set -a; source .env; set +a          # loads ANTHROPIC_API_KEY -> live Claude on the Ask rail
  NEUROAD_N_BOOT=200 NEUROAD_N_PERM=200 PYTHONPATH=src .venv/bin/python -m app.server &
  ```
  (server.py `setdefault`s the budget too, so the explicit prefix is belt-and-suspenders.)
  - Frontend edits need NO restart (server serves `neuroad.html` fresh per request).
- **SFG QC backend** on :8091 (pid was `37493`) — leave it; `mri_visualizations/.venv-sfg`.
- **Syntax check frontend:** extract the `<script>` block containing `ensureSfgFlags` (lines ~1527..next `</script>`) → `node --check`.
- **Time the live call:** `time curl -s -X POST http://127.0.0.1:8080/api/investigate -H 'Content-Type: application/json' -d '{"hypothesis":"amyloid drives cortical thinning","dataset":"adni:combat","seed":0}' -o /dev/null -w "%{time_total}s\n"` (~25s; the `[PROFILE]` server log shows the `investigate()` core at ~10s).

---

## Verify checklist (do this in-browser)

1. Anchor a hypothesis on **amyloid** → after the live call lands (~25s), the finding line flips p-tau217 → **"anchored to amyloid positivity · Δ +0.35, n=1131."**
2. **Repurposing** node on the default path → anchor chip is **p-tau217** (not GFAP).
3. Wet-lab card → **"Open Targets · live"** block with real APP data + Open Targets/Literature links.
4. No SNAPSHOT/REAL chips; no red "Decision-support only" box; no `HF_ACCESS.md` text.
5. Protein-data tab → click a target (e.g. APP) → **AlphaFold 3D structure** renders; scroll-zoom stays in the viewer.

See also prior handoff `docs/SESSION_HANDOFF_2026-07-12_FULLCIRCLE.md` for the full-circle flow (`b6c48a0`).
