# Session Handoff — Demo hardening (converters, transformer, testing)

Branch `feat/molecule-translation-loop`. **Suite: 472 passed / 2 skipped.** This session
hardened the demo toward REAL data + added the Claude-Science entry flow + a data-science
rigor pass. A separate concurrent session owns the conversion embed / LONI pulls / AD
expansion — do NOT touch `scripts/run_conversion_embed_colab.py`, `scripts/analyze_ad_expansion.py`,
`mri_visualizations/claude_science_flow/`.

## Preference locked (memory): REAL data preferred
See memory `neuroad-real-data-preferred`. Default demo happy-path = a PREFILLED hypothesis that
still runs the real feeders + logic layer; a new typed hypothesis must be accommodated by the
backend (`POST /api/investigate` → `orchestrator.investigate`) which loads the decision tree +
fields dynamically from real data. Never surface synthetic as real.

## Shipped this session (committed, newest first)
- `a18549c` **Strip synthetic from frontend + wire live real data + rigor fixes.**
  - `demo_data.json` rebuilt from REAL feeders only: SYNTHETIC HARNESS 16→0; synthetic:SURVIVOR/KILL
    substrate removed; SURVIVOR/KILL now name REAL ADNI referee cases (3T survivor vs field-strength kill).
  - `neuroad.html`: hardcoded fabricated literals removed (fake "Head motion" Δ, SYT1/SNAP25,
    "1,864 participants", branch AUCs); Investigate now POSTs `/api/investigate` and overlays the live
    ExperimentCard. `server.py` `/api/investigate` returns a rich additive `result["case"]`
    (tests, cohort, leakage_margin, score, verdict, translation, tree).
  - Rigor (all additive, no headline AUC moved): age/sex-residualized primary AUC (neurojepa adj **0.764**,
    conversion adj **0.669**); fold-honest `combat_cv_auc` (**0.718**) + whole-cohort leakage caveat;
    label-shuffle test; site-disjoint biomarker_anchor; fold-honest residualizers; BH-FDR on gauntlet
    p-values; OOF confound_leaderboard. Files: `src/neuroad/{harmonize(data/),gauntlet,leakage,pipeline,probe}.py`.
  - **KNOWN LIMITATION:** the tree topology/colors are still the fixed narrative (survivor=green/killed=gray)
    with real DATA values — NOT recomputed from `tests[].result` (honest recolor would flip 3/5 branches for
    real ADNI, breaking "UI identical"). Decide later whether to make topology data-driven.
- `297fcb8` Claude Science: declutter dataset banner (drop icons/chips/sub-header) + prefill hypothesis field (editable).
- `3b9aeee` Claude Science: pulsing dataset-connect step ("Pulling…"/"Researching…"/"Connecting to NeuroAD…"
  shimmer + ✓dataset chips) → clean dataset banner → Open-in-NeuroAD (`?h=&ds=`). Keyword→dataset match.
  Spec: `docs/DATASET_FRONTEND_DEMO.md`. File: `app/claude_science.html` (self-contained, `/start`).
- `918650a` `docs/SCIENTIFIC_RIGOR_AND_FRONTEND_AUDIT.md` — NeuroJEPA (Haoxu Huang, **arXiv:2606.14957**) rigor
  + confound verdict + frontend audit that drove the above fixes.
- `5743da2` `adni:conversion` feeder (334 MCI, 58 pMCI/276 sMCI) + leave-one-site-out conversion arm
  (`scripts/run_conversion_loso.py`): imaging 0.718 < plasma 0.810 — plasma dominates conversion too.
- `8b3bb41` Green checkpoint (L2–L6 layers, discovery-half, MRI-QC gallery).

## Key science facts (for the writeup)
- NeuroJEPA = 3D ViT-Base-MoE, 768-d, V-JEPA-2 latent prediction, pretrained on 1.55M **internal NYU**
  clinical scans (not ADNI). Frozen+attentive-probe is author-sanctioned. Their eval does LESS confound
  control than ours (random patient splits, no ComBat, no age/sex adj). Do NOT quote their absolute AD AUROC
  or claim to beat it (they fine-tune; we freeze).
- Confound verdict: **site is controlled in the estimate** (site-disjoint GroupKFold); **age/sex are only
  tested post-hoc (15%), not deconfounded** — now MITIGATED by the new `value_adjusted` AUC (surface it in UI/report).
- Plasma p-tau217 dominates AD-vs-CN (~0.93) AND conversion (0.81 > imaging 0.72). Field-honest, not a weakness.

## NEXT STEPS (pending)
1. **Brain-viz "pop"** — workflow `wski0apyg` running (attention heatmap overlay + BLUE crosshair on click +
   one minimal SFG-anomaly context line). Refs: `mri_visualizations/attention_heatmap_mockup.png`,
   teammate's `/Users/stevenyang/Downloads/context/IMPLEMENTATION.md` (Silent-Failure-Guard: "run classical
   checks across scans to detect anomalies before using data, inject as context to the hypothesis model").
   Verify live in browser (NiiVue) + commit. Minimal/clean, few words.
2. **Artifacts + Summary tab** — design APPROVED as blueprint (Option A), v2 mockup sent (Summary tab first;
   Artifacts = typed provenance rows: ≤3 reasoning bullets, ≤2 datasets, pipeline-layer trail from
   `pipeline.txt` L1–L6, a LIVE driving-analysis figure; upload/add-your-own bundled into the Notes tab).
   NOT built yet — implement on `neuroad.html` result card after brain-viz. Bind to the live `result["case"]`
   (real gauntlet/translation) + the dynamic clustering (`demo_data.json` discovery/discovery_real).
3. **Downloadable self-contained HTML** for the teammate — package `neuroad.html` (inline `demo_data.json`
   + `vendor/niivue.umd.js` as data), or ship `claude_science.html` (already self-contained). IN PROGRESS.
4. **`app/hypothesis_registry.json`** (uncommitted) — demo-drivable hypothesis→real-dataset→cited-verdict
   registry. Aligns with "prefilled + dynamic real data." Decide: wire into claude_science prefill / commit.
5. Surface the new fold-honest numbers (`value_adjusted`, `combat_cv_auc`) in the UI/report.
6. Transformer-on-Colab (real vkola ncomms2025) — DEFERRED per user until the other session's Colab finishes.

## Uncommitted working tree (as of handoff)
Mine, pending: `app/hypothesis_registry.json` (item 4). Concurrent session (LEAVE): `docs/FRAMING.md`,
`docs/DATASET_FRONTEND_DEMO.md` (used by item, harmless), `scripts/{run_conversion_embed_colab,analyze_ad_expansion}.py`,
`reports/ad_expansion_analysis.json`, `mri_visualizations/claude_science_flow/`, `.conv_prefix.txt`.
Run suite: `PYTHONPATH=src ./.venv/bin/python -m pytest -q`. Demo: `PORT=8095 PYTHONPATH=src ./.venv/bin/python -m app.server` (neuroad.html at `/`, claude_science at `/start`).
