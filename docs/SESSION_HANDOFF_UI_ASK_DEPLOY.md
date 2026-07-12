# Session Handoff — scan notes, Brain-data viz, Ask Claude LLM, Cloud Run deploy

Branch `feat/molecule-translation-loop` (pushed to origin `stevenybuilder/NeuroAD`).
**Suite: 472 passed / 2 skipped.** Everything below is committed + pushed + verified
on localhost. The ONLY unfinished item is the Cloud Run backend deploy (recipe at
the bottom — it's one rebuild + one deploy away; the build blocker is already fixed).

## Shipped this session (committed, newest first)
- `24b2beb` `.gcloudignore`: stop excluding requirements.txt/pyproject.toml/*.md (backend build ctx).
- `700e31e` Dockerfile.backend + cloudbuild.backend.yaml: enable live Ask Claude on Cloud Run backend.
- `77316e9` **Ask Claude fix** — real repo-grounded LLM. `server.py` adds `POST /api/ask`
  (model **claude-opus-4-7**, assistant persona, grounded in the passed `case` +
  `hypothesis_registry.json`, hedged, cites numbers; live iff `ANTHROPIC_API_KEY` set,
  else honest offline notice). `neuroad.html` `onChatSend` now POSTs `/api/ask` with
  `{question, hypothesis, case:heroCase()}` and renders the answer + `Claude · <model>`
  badge. Added `ROOT_PAGE` env (server serves NeuroAD at `/` on localhost, Claude
  Science at `/` on Cloud Run). VERIFIED LIVE on localhost with the key: answered a
  "top target" question with APP/priority 8.60, p-tau217 anchor r=+0.49 n=876, STRING
  APP-SORL1 0.999, AlphaFold pLDDT 67.4, hedged as hypothesis-not-validated.
- `a469ef6` **Scan-note polish** (research-driven: /deep-research 102 agents + local audit).
  Empty-state disclosure (no panel until a note exists); pins get a 2px surface keyline
  ring + white numeral (distinct from warm heatmap / blue crosshair); dbl-click
  preventDefault+stopPropagation (no NiiVue conflict); trimmed instruction line; notes
  store the view they were pinned in + render pins only in that view; stable per-note
  numbering; pins Hide/Show toggle + count (default visible); hoverable pins.
- `2a7c4a4` **Location-pinned scan notes** (item approved via mockup). Double-click the
  MRI scan drops a numbered pin + animated sticky editor; Enter files it into a "Scan
  notes" subsection under the scan; pin persists (dimmed). Reuses `state.notes`/`saveNotes`.
- `3c66be7` Removed the **Notes tab**; replaced the FABRICATED Brain-data viz
  (clusterPlot/dataTable — seeded-RNG "silhouette 0.42" + random S#### rows) with the
  real `discoveryFigure` (Artifacts) + a node-dynamic `regionEffectTable` (Brain data);
  relocated the upload into Brain data.
- Earlier (prior sub-session): `acb4813` registry-driven /start prefill; `4b90b37` Summary+
  Artifacts honesty fixes; `bcca31c` Summary+Artifacts tabs; heatmap pop.

## Secrets / key (DONE, do not redo)
- `ANTHROPIC_API_KEY` is in `.env` (gitignored, NOT tracked — verified). Real `.env`
  never committed; only `.env.example` is tracked.
- Secret Manager: secret **`anthropic-api-key`** created (version 1) in project
  `project-flash-490419`; Secret Manager API enabled; `roles/secretmanager.secretAccessor`
  granted to `31043195041-compute@developer.gserviceaccount.com`.

## Run locally
- Backend WITH live Ask Claude (needs the key in env):
  `export ANTHROPIC_API_KEY=$(grep '^ANTHROPIC_API_KEY=' .env | cut -d= -f2-)`
  `PORT=8096 PYTHONPATH=src ./.venv/bin/python -m app.server`
  NeuroAD at `/`, Claude Science at `/start`, `/api/ask` live.
- (A server may already be running on 8096 from this session; kill+restart if stale.)

## LAST STEP — finish the Cloud Run backend deploy (neuroad-demo)
Goal (user chose "live everywhere"): switch `neuroad-demo` from static nginx to the
Python backend so `/api/ask` is live on the Cloud Run link. Build blocker (files
excluded from context) is FIXED in `24b2beb`. Remaining, in order:

1. Build the backend image (Cloud Build; docker isn't local):
   `gcloud builds submit --config cloudbuild.backend.yaml --project project-flash-490419 .`
   (Two earlier builds failed only because .gcloudignore excluded src/ then
   requirements.txt/pyproject.toml/*.md — both now un-ignored. If it fails again,
   read the tail for the next missing COPY source and un-ignore it.)
2. Deploy the revision with the key secret + Claude-Science routing:
   ```
   gcloud run deploy neuroad-demo \
     --image us-central1-docker.pkg.dev/project-flash-490419/cloud-run-source-deploy/neuroad-demo:backend \
     --region us-central1 --project project-flash-490419 --allow-unauthenticated \
     --set-secrets ANTHROPIC_API_KEY=anthropic-api-key:latest \
     --set-env-vars ROOT_PAGE=claude_science.html \
     --port 8080 --memory 1Gi --cpu 1 --quiet
   ```
3. Verify the live link: https://neuroad-demo-31043195041.us-central1.run.app
   - `/` = Claude Science; `/neuroad.html` = product; scan notes (localStorage) work.
   - Ask Claude rail should now answer live (POST `/api/ask`). `curl .../api/health`
     should show `claude_live: true`.
   - Expected degradations on Cloud Run (fine, frozen-demo behavior): `/api/investigate`
     falls back to `demo_data.json` (embeddings not in image), `/api/sfg/*` offline.
   - Rollback if needed: `gcloud run services update-traffic neuroad-demo --to-revisions=<prev>=100`.

Note: the prior static routing (Claude Science at `/`) is preserved via `ROOT_PAGE`.
Do NOT touch the concurrent session's files: `docs/FRAMING.md`,
`scripts/run_conversion_embed_colab.py`, `scripts/analyze_ad_expansion.py`,
`mri_visualizations/claude_science_flow/`.
