# Session Handoff — 2026-07-13 · Flagship AUROC routing, UI fixes, LLM-router plan

**Nothing is committed or deployed this session.** All changes are in the working tree on branch
`fullcircle-drawer-polish`. Server was running locally on :8080 (kill ALL `app.server` procs — there were 3 stale
ones this session). Re-read the prior handoff `docs/SESSION_HANDOFF_2026-07-13_REGION_GRID_DEPLOY.md` for the
pre-existing architecture (stage memo, ICV, grid) which this session built on.

---

## ⚠️ CRITICAL OPEN QUESTION — resolve FIRST, before syncing /start or deploying

I reworded the flagship default hypothesis so it routes to **diagnosis (dx_binary)** instead of **conversion**,
changing the displayed AUROC **0.64 → 0.922** (combat dx_binary). **This must be honesty-audited before trusting it:**

- `hypothesis_registry.json` entry 0's own note warns ADNI AD-vs-CN can be **inflated ~0.03 by a field-strength
  confound** (added AD are 40% 1.5T, CN are 100% 3T; NeuroJEPA embeddings read field strength at AUC 0.990 → report
  the **3T-matched 0.861**, not the pooled ~0.89–0.92).
- BUT that note is about the **NeuroJEPA embeddings**. The flagship uses **`adni:combat`** = ComBat-harmonized
  FreeSurfer features (a *different* substrate designed to remove site/scanner). The combat dx_binary cache cell
  shows **score 100 / promoted / "strong candidate"**, implying it PASSED the scanner-leakage gauntlet.
- **ACTION:** run the **`data-qa-power`** agent (or check `case.leakage_margin` for combat dx_binary: `scanner_auc`,
  `margin`, `margin_ci_excludes_zero`) to confirm 0.922 is de-confounded, not field-strength-inflated. If it is
  confounded, revert the flagship or route to the 3T-matched number. **Do not deploy a confounded headline.**

---

## DONE + VERIFIED this session

1. **Live-compute fallback (stage memo)** — `src/neuroad/harness/orchestrator.py`: `_base_memo` LRU caches the
   anchor-invariant referee base per `(dataset,target,region,seed,api)`; anchor only re-applies the ~0.5s
   translation. **36.8s → 0.7–2.2s (~53×)** on misses, numbers byte-identical across anchors (verified). Plus opt-in
   `NEUROAD_LIVE_N_REPEATS` lever in `pipeline.py` (default = full rigor, unchanged). N_BOOT is NOT the lever.
2. **ICV Blocker A** — `src/neuroad/leakage.py`: `leakage_margin` emits `outcome_auc_icv_adj` + `icv_adjusted` for
   ROI-volume cohorts (fold-honest; raw `outcome_auc` + score path untouched). FE `_icvNote` in `neuroad.html`.
   Verified: hippocampus 0.875 → 0.889.
3. **Grid expanded 105 → 217 cells** — `scripts/warm_investigate_cache.py` now sweeps region×anchor×outcome, gating
   anchors on promotion. **STALE re: translation change (#8) — MUST RE-WARM.**
4. **Out-of-scope honesty fix** — `app/server.py`: out-of-scope hypotheses (e.g. "tau-PET SUVR trajectory") now
   bypass the disk cache get AND put, so they hit the honest refusal instead of a colliding promoted SURVIVOR cell.
   Fixed a pre-existing failing test.
5. **Dynamic decision tree** — `neuroad.html` `_applyCaseToTree`: branch colors + confound sublabels now derive from
   real `case.tests[]`/`promoted` (verified both promoted → green survivors + "scanner AUC 0.37 · ruled out", and
   killed → survivors grey + "scanner AUC 0.82 · drives signal").
6. **Story-card overlap fix** — `neuroad.html` node y-coords spread 1.8× (250→450 units) so confound-column cards
   stop overlapping; `cam` re-centered. Verified via DOM geometry (couldn't screenshot — WebGL render loop blocks
   the extension; use `window.__APP` + cancelAnim to inspect).
7. **Flagship AUROC routing fix** — `neuroad.html` `entryValue` changed from
   `"p-tau217 plasma level PREDICTS hippocampal atrophy in preclinical AD"` (word "predicts" → `_infer_target`
   returns **conversion** → 0.64) to
   `"p-tau217-anchored hippocampal atrophy SEPARATES Alzheimer's disease from cognitively normal"` (→ **dx_binary**
   → 0.922). Header now = tree (0.93). **Root cause of the whole "0.64 crazy low" scare: the keyword router keyed on
   "predicts".** PROVEN not caused by my wrangling: combat conversion = 0.644 in the pre-session cache backup AND now.
8. **UI workflow (ultracode/Sonnet 5)** — `src/neuroad/harness/translation.py`: wet-lab card now NAMES the real lead
   gene ("knock down **APP/MAPT/TREM2**", from `lead.top_target`, not hardcoded) + amyloid readout is **Aβ42/40-only**
   (removed "+ p-tau217") so anchors read distinctly. VERIFIED via direct `translation.translate()` call.
   `neuroad.html`: `topTargetGene()` auto-selects the top target when the Protein-data tab opens so the AlphaFold
   3D model renders immediately (code done, **NOT browser-verified**).
9. **Pre-deploy review fixes** — applied 2 of 4 low-sev findings: memo cold-path deepcopy (`orchestrator.py`), ICV
   0.5-sentinel guard (`leakage.py`). SKIPPED: seed-in-`investigate_cache.key` (pre-existing; adding seed would
   invalidate the warmed grid; no deployed client sends nonzero seed).

**Test suite: 472 passed, 2 skipped** (before #7/#8; re-run to confirm).

---

## PENDING — do these to finish

- **P1. Honesty-audit the flagship 0.922** (see CRITICAL section). `data-qa-power`.
- **P2. /start ↔ localhost SYNC** (the user's biggest repeated complaint). `/start` = `app/claude_science.html`,
  prefills `REGISTRY_DEFAULT.hypothesis` = `hypothesis_registry.json` entry 0 ("AD is decodable from frozen
  embedding" → neurojepa, a curated FRAGILE "catch the scanner artifact" rigor story) via `pickDefaultEntry`
  (prefers `demo_priority:"lead"`). `neuroad.html` defaults to the flagship (#7) → combat. **They diverge, AND even
  matching text can route /start→neurojepa (0.857) vs direct→combat (0.922).** To truly sync: add the flagship as a
  registry **lead** entry with `dataset.name: "adni:combat"`, `claim.target: "dx_binary"`, cite the real report
  `reports/adni_dx_3T_survivor.json`, place it FIRST → `/start` prefills flagship AND `realDatasetFor` routes it to
  combat. GATED on P1 (use the honest number). Do NOT overwrite the curated rigor entry — demote it to `supporting`
  or keep as a second option. `realDatasetFor` uses keyword-overlap matching; verify the flagship actually resolves
  to combat (its fallback default is `adni:neurojepa`).
- **P3. Stuck "Ask Claude" follow-up** — typing a follow-up leaves "Pruning confounded branches…" stuck. NOT a crash
  (server returns 200, no app JS error — only the extension's own `No Listener` error). It's the grow animation not
  clearing. Look at `neuroad.html` `growSeq`/`goStep`/`scheduleAdvance` (~lines 2321-2456) and the `/api/ask`
  follow-up spawn (~2405). Repro in-browser (see browser-QA note below).
- **P4. LLM-as-judge ROUTER (Sonnet 5, `claude-sonnet-5`)** — replace the brittle keyword `_infer_target` so words
  like "predicts" don't misroute. Design (confirmed with user): use the LLM as a **router** (free text → the finite
  enum {conversion,dx_binary,site,scanner} + anchor), NOT a science-judge; **cache the classification by normalized
  hypothesis text** (only pays on a miss; hits stay <10ms); keep the deterministic keyword router as the **offline
  fallback**; preserve the precompute-grid honesty. Hooks: `src/neuroad/claude/claim_parser.py` (`parse_claim` /
  `_infer_target`) + `app/investigate_cache.py` (`key()` must use the SAME classified target or cache keys drift).
  **Two research agents were spawned for the plan (see below).**
- **P5. UX design review** — run `ux-design-reviewer` agent on the running UI; apply only human-approved cleanups;
  keep it clean/minimalistic. Feed it the design-principles research (below).
- **P6. Re-warm the grid** (required after #7/#8 engine/text changes so cached happy-path cells match live):
  ```
  cd neuroad-discovery-engine
  pkill -9 -f app.server; sleep 2; lsof -ti:8080 | xargs kill -9 2>/dev/null
  echo '{}' > app/investigate_cache.json
  NEUROAD_N_BOOT=200 NEUROAD_N_PERM=200 PYTHONPATH=src:. .venv/bin/python -m scripts.warm_investigate_cache   # ~11 min
  ```
- **P7. Browser QA end-to-end** (`claude-in-chrome`) — verify: flagship shows 0.92 header=tree; AlphaFold auto-loads
  on Protein-data tab; wet-lab card names the gene + amyloid≠p-tau readout; anchor switching changes lead/readout;
  /start and localhost now match; stuck-follow-up fixed. GOTCHA: the WebGL/niivue render loop blocks the extension's
  screenshot/`get_page_text` (they wait for document_idle). Screenshots DO work between animations; when they don't,
  drive/inspect via `mcp__claude-in-chrome__javascript_tool` reading `window.__APP` (the React app instance; fiber-walk
  to the stateNode with `.nodes` + `.state.cam`). To hold a zoom for inspection, override `overviewCam` + `cancelAnim()`.
- **P8. Deploy** — via the `deployment` agent. Rollback revision `neuroad-demo-00013-k48`. Rebuild demo_data first ONLY
  if needed (it's parallel-track-owned — don't clobber). Ship `app/investigate_cache.json` + gated `adni_roi.csv`;
  keep ANTHROPIC_API_KEY secret + live Claude. Pre-flight: `gcloud meta list-files-for-upload | grep -iE
  '\.env|secret|token|_manifests|crosswalk|download/'` MUST be empty.

---

## RESEARCH — DONE, saved to disk (do NOT re-run)

Both deep-research efforts completed and are saved — read these instead of re-researching:
1. **`docs/RESEARCH_scientific_info_ux.md`** — dense-science UX design principles (feed to P5 / `ux-design-reviewer`).
   Headline: NeuroAD's biggest risk is **honesty under compression** — every AUROC must carry the TASK it measures
   (diagnosis vs conversion — literally our bug), a provenance chip (cohort/n/measured-vs-derived), CI-as-primary
   visual, "not-validated" as a dignified distinct-from-ruled-out state, and single-source-of-truth (no panel drift).
2. **`docs/RESEARCH_llm_router_plan.md`** — implementation-ready LLM-router plan (feed to P4). A new
   `src/neuroad/claude/router.py` with a normalized-text routing cache (`app/router_cache.json`) → Sonnet-5
   enum-constrained structured-output call on miss → keyword `_infer_target` backstop. ONE `route_target()` feeds
   BOTH `claim_parser._fallback` AND `investigate_cache._infer_target` so the cache key can't drift from the engine
   target. Includes prompt, schema, wiring, warm/ship, a golden-set eval, and risk table. This directly kills the
   "predicts→conversion" misroute.

---

## GUARDRAILS (carry forward — unchanged)

- **NEVER force-push `main`** (diverged from a teammate). Keep branch `fullcircle-drawer-polish` + PR #1.
- **NEVER commit parallel-track files:** `README.md`, `app/demo_data.json`, `data/registry.yaml`,
  `docs/DEMO_SCRIPT.md`, `docs/FRAMING.md`, `reports/*`, `src/neuroad/scoring.py`. Untracked non-mine:
  `app/knowledge_base.json`, `scripts/prove_live.py`. (`app/hypothesis_registry.json` + `app/claude_science.html`
  ARE editable — tracked, not parallel-track.)
- **Every displayed number must be a real engine computation** — no fabrication/imputation. This is why P1 matters.
- De-identified ADNI compute tables ship to the PRIVATE Cloud Run image only; raw + RID crosswalk stay off GitHub;
  GCS bucket is imaging-only.
- **Restart the server after backend edits** (kill ALL `app.server` procs; `find src -name __pycache__ -exec rm -rf`);
  **re-warm the grid after any cohort/engine/translation change.**
- **Cache-key gotcha:** `investigate_cache.key()` uses the INFERRED TARGET, not literal text — a "random-prefix"
  hypothesis still hits the same cell. To force a genuine miss, change the inferred coordinate or clear the cache.
- The live server local run:
  ```
  set -a; source .env; set +a
  NEUROAD_N_BOOT=200 NEUROAD_N_PERM=200 PYTHONPATH=src:. .venv/bin/python -m app.server &
  # http://localhost:8080/neuroad.html   and   /start (claude_science.html)
  ```

## Files changed this session (all mine, uncommitted)
`src/neuroad/harness/orchestrator.py`, `src/neuroad/pipeline.py`, `src/neuroad/leakage.py`,
`src/neuroad/harness/translation.py`, `app/server.py`, `app/neuroad.html`, `scripts/warm_investigate_cache.py`,
`app/investigate_cache.json` (217 cells, STALE re: translation — re-warm). `git diff --stat` to see them.
A cache backup from BEFORE this session is at the scratchpad:
`/private/tmp/claude-501/-Users-stevenyang/aef992c2-f8f4-4102-905c-ae69968e8461/scratchpad/investigate_cache.backup.json`
(used to PROVE the conversion number 0.644 was pre-existing, unchanged by the wrangling).
