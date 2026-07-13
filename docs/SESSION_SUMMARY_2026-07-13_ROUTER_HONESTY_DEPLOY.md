# Session Summary — 2026-07-13 · Router, honesty caption, /start sync, deploy

Autonomous overnight run against the 6-item handoff
(`SESSION_HANDOFF_2026-07-13_AUROC_ROUTING_LLMJUDGE.md`). All items executed end-to-end.
Live demo redeployed. Branch `fullcircle-drawer-polish`.

## What shipped (done + verified)

### P1 — Flagship 0.922 honesty audit → SAFE (data-qa-power agent)
- The flagship **Diagnosis AUROC 0.922 (AD vs CN, adni:combat)** is **NOT field-strength-inflated**.
  Decisive proof: the effect holds **within each field strength** (3T-only 0.925 n=1193, 1.5T-only 0.935
  n=422; both within ±0.05 of the pooled), where a field-strength confound is impossible.
- Full writeup: `docs/P1_FLAGSHIP_AUDIT.md`. `score=100` is legitimate for this cell (all 5 gauntlet tests
  pass; brain-age IS available on the ComBat cohort — the renorm-overclaim was the OLD 3T survivor, which
  does not ship).
- **Caveat found + fixed:** the UI was proving the de-confound with the *wrong optimistic number* —
  "scanner AUC 0.37 · ruled out" is a whole-cohort-ComBat artifact (the honest fold-honest residual is
  ~0.65). Replaced with the within-field-strength invariance everywhere it appeared (see Honesty below).

### P4 — LLM-as-judge router (Sonnet 5)  ✅
- New `src/neuroad/claude/router.py`: `route_target(text, df)` → normalized-text routing cache
  (`app/router_cache.json`) → Sonnet-5 enum-constrained call on miss → keyword `_infer_target` backstop.
  Never raises; offline path is byte-identical to the old regex.
- **One canonical router** feeds BOTH `claim_parser._fallback` AND `investigate_cache._infer_target`, so the
  cache key's target can never diverge from the engine's routed target.
- Kills the `predicts → conversion` misroute: e.g. "p-tau217 predicts hippocampal atrophy in preclinical AD"
  now routes to **dx_binary** (0.92), not conversion (0.64). The neuroad.html chip "p-tau217 predicts
  hippocampal atrophy" now shows 0.92 too.
- Golden set (`tests/data/router_golden.jsonl`, 58 items, stratified) + CI eval (`tests/test_router.py`):
  always-on no-regression + keyword-baseline floors; a gated live eval (`ANTHROPIC_API_KEY` +
  `NEUROAD_ROUTER_EVAL=1`) that asserts LLM ≥ keyword per class and materially better on the adversarial
  collision bucket. **Live eval PASSED.** Full suite: **476 passed, 3 skipped.**
- `app/router_cache.json` pre-warmed (`scripts/prewarm_router.py`) over the flagship + demo chips + seeds so
  no demo click pays a live classify call.

### P3 — Stuck "Ask Claude" follow-up  ✅
- Root cause: `onChatSend` cleared the grow timer but not `reasoning`, freezing the last verb
  ("Pruning confounded branches…") when a follow-up was sent mid-grow.
- Fix: a follow-up now cancels the intro animation (both timers + `_playing`) and clears the overlay.
- **Verified** by simulating the exact mid-grow stuck state in the live UI: overlay clears to null, pending
  timer does not leak.

### P2 — /start ↔ localhost sync  ✅
- Added the flagship as the FIRST **lead** entry in `app/hypothesis_registry.json`
  (`ad_vs_cn_combat_flagship`, `dataset.name: adni:combat`, `claim.target: dx_binary`, cites
  `reports/adni_dx_3T_survivor.json` + `P1_FLAGSHIP_AUDIT.md`, real n=1615 / 462 AD / 1153 CN). Did **not**
  clobber the curated neurojepa rigor entry (kept in place).
- Verified: `/start` `PREFILL` == the flagship text exactly and `realDatasetFor` → `adni:combat`;
  neuroad.html `entryValue` == the same text and `_investigateDataset()` → `adni:combat`. Both surfaces now
  run the same hypothesis + cohort + **0.922**.

### Honesty caption (the P1 caveat)  ✅
- Engine: `src/neuroad/leakage.py` now computes real **within-field-strength invariance**
  (`leakage_margin.field_strength_invariance`: per-stratum outcome AUROC + ±0.05 equivalence-band flag),
  additive — the score/margin path is untouched. Guarded to ~2 CV fits per outcome cell (no warm slowdown).
- Frontend (`app/neuroad.html`): the scanner de-confound now shows **"field-strength invariant · 3T 0.93 /
  1.5T 0.94"** (real engine numbers) instead of "scanner AUC 0.37 · ruled out"; the survivor reasoning bullet
  leads with the within-strata story. **Every finding AUROC is task-labeled** via one helper
  `_findingAurocLine()` → "Diagnosis AUROC 0.92 [0.91–0.94] · p=0.005 (AD vs CN)", identical on the story
  card and the Protein-data tab (single source of truth). Branch-node sublabels read "AD vs CN · AUC 0.93".
- **UX-review Finding B (also fixed):** the standalone "Kill reason — scanner confound" teaching block still
  rendered the forbidden 0.37 with a self-contradictory "scanner predicts better" narrative. Now data-driven:
  when the outcome beats the scanner it shows the honest within-strata ruling-out (no 0.37, no
  contradiction). Verified in the live UI.

### P6/P7 — Re-warm + browser QA  ✅
- Full grid re-warm: `app/investigate_cache.json` = 217 cells, each dx_binary/conversion cell now carries
  `field_strength_invariance`. Flagship serves as a **fast cache hit** (~0.28s), dx_binary, 0.922.
- Browser QA (Claude-in-Chrome): flagship 0.92; AlphaFold auto-loads on the Protein-data tab (APP, P05067);
  lead gene named; task labels + honest de-confound render; no app JS errors (only the benign extension
  "No Listener").

### P8 — Deploy  ✅
- Live URL: **https://neuroad-demo-31043195041.us-central1.run.app**
- Final live revision: **`neuroad-demo-00015-cfl`** (includes the kill-block honesty fix; two deploys this
  session: `00014-dgz` then `00015-cfl`).
- Rollback targets: `neuroad-demo-00014-dgz` (this session, pre-kill-block-fix) and `neuroad-demo-00013-k48`
  (prior known-good).
- Verified against the LIVE URL: `claude_live:true`; flagship `dx_binary` / `outcome_auc 0.922` /
  `field_strength_invariance` present / cache hit; neuroad.html serves the "Scanner confound — ruled out"
  honest block (no 0.37).
- Private image ships investigate_cache + router_cache + de-identified gated ADNI tables; ANTHROPIC_API_KEY
  via Secret Manager; live Claude preserved (`claude_live:true`). Pre-flight clean: 0 PTIDs in the upload
  set, no secrets/manifests in the build context. demo_data.json rebuilt from the real engine path (carries
  field_strength_invariance at first paint), NOT committed.

## ⚠️ Needs your attention (not acted on autonomously)

### 1. PUBLIC repo exposes LONI-DUA manifests (pre-existing, flagged not fixed)
`github.com/stevenybuilder/NeuroAD` is **PUBLIC**. `data/real/_manifests/*.txt` (ADNI image-ID lists) and
`scripts/build_cohort_crosswalk.py` are tracked and already pushed — a `.gitignore` gap (it covers
`_gated/` and `adni*.csv` but NOT the `ida_imageids_*.txt` files, which the .gitignore's own comment says are
"under the LONI DUA — Never commit"). I did **not** rewrite history or change repo visibility autonomously.
Recommended: make the repo private, and/or `git rm --cached` those paths + close the .gitignore gap + purge
from history, and confirm with your DUA obligations.

### 2. A mid-session commit swept in parallel-track files
Commit `a95c171` ("…latest pipeline WIP", authored by you at 07:13, already pushed) committed the
parallel-track files (`app/demo_data.json`, `README.md`, `data/registry.yaml`, `docs/DEMO_SCRIPT.md`,
`docs/FRAMING.md`, `reports/*`, `src/neuroad/scoring.py`) that the handoff says to keep off the branch. If a
teammate owns those, they may need reconciling. My own commits (13c1191, a72518a) touched only my files.

## Deferred UX-review items (documented, not applied — need your call)
The `ux-design-reviewer` scored the flagship and I applied only the honesty-critical Finding B. The rest are
FLAG/POLISH judgment calls left for you (keeping the UI minimal, per your instruction):
- **BLOCK-A — persistent trust chrome:** provenance / "Claude · live" badge / "Real ADNI" data-mode pill are
  legible only after opening a node's Artifacts tab, not on the always-on canvas. Reviewer wants a persistent
  top-bar cluster + bottom rail. This is a new UI element → your design call.
- **Finding C — naive-vs-adjusted drift:** node sublabels show 0.93 (naive_effect) while drawers show 0.92
  (site-disjoint). Both real, legitimately different. Cleanest fix: put 0.92 on the node too and let 0.93
  appear only in the before→after; or label the node "0.93→0.92 adj".
- Lower: Artifacts-tab AUC density (default-collapse frozen-model/driving panels); Reasoning tab duplicates
  the inline card; mini-map occludes the candidate card; GFAP anchor-provenance label; wheel-zoom appeared
  unbound; 0.90/0.91 age-sex rounding drift; "lead target · PI4AD #18" reads oddly; MRI attentive-probe uses
  a reserved state hue.

## Files changed this session (mine)
NEW: `src/neuroad/claude/router.py`, `app/router_cache.json`, `scripts/prewarm_router.py`,
`tests/test_router.py`, `tests/data/router_golden.jsonl`, `docs/P1_FLAGSHIP_AUDIT.md`, this file.
EDITED: `src/neuroad/claude/claim_parser.py`, `app/investigate_cache.py`, `src/neuroad/leakage.py`,
`app/neuroad.html`, `app/hypothesis_registry.json`, `scripts/warm_investigate_cache.py`,
`app/investigate_cache.json` (re-warmed).
Committed on `fullcircle-drawer-polish`: `13c1191` (caches), `a72518a` (kill-block honesty).
`app/demo_data.json` + `reports/*` rebuilt for the image but left uncommitted (parallel-track).
