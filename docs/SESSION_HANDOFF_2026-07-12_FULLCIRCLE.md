# Session handoff — full-circle protein flow + demo hardening (2026-07-12)

Owner-session summary for whoever picks up `neuroad.html` / the demo next. Read this
before editing. Multiple sessions share this repo — **stage explicit paths, never `git add -A`.**

## TL;DR — what shipped this session
- **Committed** (`b6c48a0`, `app/neuroad.html` only): the **full-circle ranked-targets flow**
  (gate → per-protein AlphaFold cards) + AUROC/CI finding-level header + tab-row wrap fix.
- **Uncommitted, other sessions' territory** (do NOT commit unless you own them): `app/demo_data.json`,
  `src/neuroad/gauntlet.py`, `README.md`, `app/build_demo_data.py`, `data/registry.yaml`,
  `reports/*`, `docs/*`. Plus untracked `app/knowledge_base.json`, `app/structures/*.cif`,
  `app/vendor/3Dmol-min.js`.

## Run / view the demo
```bash
cd neuroad-discovery-engine
set -a; . ./.env; set +a           # loads ANTHROPIC_API_KEY -> live Claude
PORT=8080 PYTHONPATH=src ./.venv/bin/python -m app.server &
# open http://localhost:8080/         (neuroad.html — the judge-facing demo)
#      http://localhost:8080/start     (claude_science.html — Claude entry / workspace card)
```
`/api/health` shows `claude_live:true` when the key is loaded. `POST /api/investigate` is the live
referee; the prefilled submit also fires `/api/sfg/run` (MRI-QC panel) — both return 200.
**Gotcha:** a running Python server does NOT reload changed modules — after backend edits, restart it.
Verification agents (`demo-seam-guardian`) `pkill -f app.server` on start, which kills your :8080 —
restart it after they run.

## The full-circle flow (committed) — code map in `app/neuroad.html`
Everything renders in the **right-rail drawer** (search `data-drawer`; `state.pinned`, `state.drawerTab`).
- **Tabs** built in `buildTabs(node)` (`defs.push([...])`); the **`'protein'` / "Protein data"** tab is
  added only when `node.story==='proteins'`. `drawerContent(node)` switch routes `t==='protein'` →
  `proteinDataEl(node)`.
- **Default tab**: `onNodeClick` sets `drawerTab: ok?(n.story==='proteins'?'protein':'summary'):'reason'`
  and resets `selectedTarget:null`. So pinning the ranked-targets node lands on the **gate**.
- **`proteinDataEl(node)`** (search the method) is the whole flow:
  - **Finding-level header** ("Measured on the imaging finding"): `AUROC + [CI] + p` from
    `case.leakage_margin` (`outcome_auc`/`outcome_ci`/`outcome_p_perm`) + `anchored to plasma p-tau217`
    from `tests.find(biomarker_anchor).stats` (`ptau217_r`/`ptau217_n`). Shown **ONCE**.
  - **Gate**: `translation.ranked_targets.map(row)` — each row is a `<button onClick=selectTarget(gene)>`,
    top candidate (`i===0`) highlighted green, shows PI4AD `priority_score` + bar.
  - **Per-protein card** (`if(selectedTarget)`): back button (`clearTarget`), gene + TOP CANDIDATE badge,
    `evidence_note`, `priority_score` + `rank`, and `StructViewer` for that gene.
- **State/methods**: `selectedTarget` in state; `selectTarget=(g)=>`, `clearTarget=()=>` near `toggleStruct`.
- **`StructViewer`** (class, search `VENDORED_STRUCTURES`): loads `/structures/<GENE>.cif` per gene
  (all 6 present), colors by pLDDT (b-factor), gentle spin, **graceful fallback** to an AlphaFold link on
  WebGL/fetch failure. Vendored viewer at `app/vendor/3Dmol-min.js` (self-contained, CSP-safe, no CDN).

## HONESTY RULES (do not violate)
- **AUROC/CI is FINDING-LEVEL, shown once.** Never a per-protein AUROC — targets carry PI4AD
  `priority_score`/`rank` only.
- **Every number traces to the case** (`leakage_margin`, `tests[].stats`, `translation.ranked_targets`).
  No hardcoding. The GWAS ranking-validation `AUC 0.728` lives only in
  `reports/target_prioritization_validation.json` — to surface it, plumb into `demo_data.json` first
  (do not hardcode in UI).
- The `UNIPROT` map + AlphaFold entry URLs in `proteinDataEl` are stable identifiers (fine to hardcode);
  they are not scientific numbers.

## Other work this session (context; mostly other files)
- **Overclaim fixes (committed earlier by another session's flow)**: L1 node labels real substrate
  (Morph vs JEPA) + surfaces real NeuroJEPA evidence; L3 fusion note = honest surrogate (NOT the
  ncomms2025 transformer); discovery shows the REAL 0-promotable result (never synthetic ARI=1.0);
  baked-vs-live provenance honest + cohort-aware; ranked rows show shortlist 1–6 + PI4AD # as evidence;
  F5 fusion node gates on real fusion (empty `{}` no longer truthy); kill cards end on a "Do this next"
  remediation; hero "100/100" reframed as "a robustness score, not a confidence level; provisional"
  (NOT de-rated — it is a genuine 5/5; the NA-renorm loophole is already capped at 84 in `scoring.py`).
- **Amyloid anchor** (`src/neuroad/gauntlet.py` + baked into `demo_data.json`, UNCOMMITTED): real
  binary positivity-enrichment anchor Δ +0.35, point-biserial r 0.42, CI ≥ 0.30, n=1131 — closes the gap
  where the survivor narrative claimed amyloid enrichment but never computed it.
- **Biomarker+hypothesis composer** (`neuroad.html`, committed with prior work): collapsible inline widget
  in the Ask-Claude rail (biomarker dropdown incl. amyloid + hypothesis → real `/api/investigate`);
  region is OUTPUT-only attribution, never a fake input.
- **Knowledge base** for Ask-Claude: `app/knowledge_base.json` (untracked) injected as the authoritative
  first block in `server.py` `_build_ask_context()` — canonical data-scale/AUROC/discovery facts so
  Ask-Claude stops improvising numbers. Includes the "sample size = subjects, never voxels" rule and the
  killer one-liner / full-circle templates.
- **Pipeline trail** (`_trailEl`, committed): labels enlarged (name 9→14px, ids 10→14px) and made specific:
  Embedding · AD probe · Fusion · Gauntlet · Ranking · Structure (was JEPA/Probe/Fusion/Refine/PI4AD/Target).
- **Canvas node labels** enlarged to 14px (sub 12px).
- **`app/claude_science.html`** (`/start`, other-session file): workspace card — removed bottom note,
  right-side `.dsmeta` text now `--ink` (black), not gray.

## Open items / known issues (for the next session)
1. **Autopilot is dead code on `neuroad.html`** — `onTogglePlay`/`onNext`/`goStep` exist but are NOT wired
   to any button/keyboard/minimap (the working ▶ tour lives in `app/index.html`). If the demo relies on the
   guided walkthrough, wire it; otherwise the interactive click-through is the intended flow.
2. **Deploy**: use the `deployment` agent (GitHub + Cloud Run). Safety model: GitHub stays clean of
   secrets/weights/gated data/PTIDs; the PRIVATE Cloud Run image DOES ship de-identified ADNI tables so
   `/api/investigate` runs live in prod. Pre-flight already confirmed: `hf_token.txt`, `.env`,
   `data/real/_gated/` are git-ignored; new assets (`app/structures/`, `app/vendor/`, `knowledge_base.json`)
   are NOT ignored and will ship. Ship `Dockerfile.backend` (the live backend), not the static one.
3. **`0.728` GWAS validation** is not yet in the frontend — would need plumbing into `demo_data.json`.

## Verification agents (project-registered)
- `demo-seam-guardian` — drives the browser, checks number-sync + no console errors + no synthetic-as-real.
  Best after any neuroad.html/demo_data change. (It restarts the server; re-serve :8080 after.)
- `ux-design-reviewer` — visual/design audit vs the ZUI/Dana-Cho canon (human-in-the-loop; can run
  report-only). `overclaim-auditor` — claim↔evidence honesty referee. `data-qa-power` — stats/confound audit.
</content>
