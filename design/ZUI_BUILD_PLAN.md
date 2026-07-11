# Plan: ZUI Decision-Tree — Prezi-style zooming, hierarchy-by-proximity, provenance story blocks

## Context

NeuroAD Discovery Engine is a Claude-for-Life-Sciences hackathon build, reframed as **"an AlphaFold for Alzheimer's neuroimaging"**: it uses fMRI + biomarker signals to kill confounded hypotheses, surface strong ones, and **translate survivors into ranked candidate proteins, known compounds, and testable wet-lab experiments**. The current product is a dense, dark 12-panel "workbench" (audited as incoherent/"frankenstein"); the agreed direction is a light, Claude-Sciences-themed **zooming decision-tree canvas**.

The user asked to develop and emphasize two Prezi-derived properties — **ZUI (zoom/pan)** and **hierarchy-by-proximity** — with two requirements: **zoom-back** must be first-class, and **provenance** must ride on Prezi-style **"story blocks."** The user reported **slight motion disorientation** in the prototype and asked how Prezi keeps ZUI while mitigating it.

**Decisions locked with the user:**
- **Surface:** evolve the existing prototype at `scratchpad/sketch/index.html` (the light "AlphaFold" build served locally) into a **standalone `zui.html`**. Leave the real app untouched as a shippable fallback; migrate into the product later.
- **Scope:** build the **full feature** (all phases below) — BUT **first** produce an explicit **Design System spec** (colors, components, system, visual elements, fonts) for **ingestion into Claude design** before implementation begins. This is deliverable #1.

## Intended outcome

- Turn the prototype's *illustrative* candidates into **real, citable data** (backend translation layer is ~80% built but unrendered).
- Make the canvas **calm / non-disorienting** via research-backed smooth zoom.
- Keep the audit's non-negotiable: **trust/provenance/honesty never recede** — delivered *spatially*, attached to each claim.

---

## PART A — Design System spec (deliverable #1, ingest into Claude design first)

Produce `neuroad-discovery-engine/design/DESIGN-SYSTEM.md` (+ a `tokens.css`/`tokens.json`) enumerating the below, then review/ingest before building Part C. Values are the current prototype's, tuned to the audit's "one system" fixes (6-step type scale, one bar primitive, semantic-only color).

### Color tokens (light "Claude paper" theme)
| Token | Hex | Role |
|---|---|---|
| `--paper` | `#F5F4EE` | canvas background (warm paper) |
| `--paper2` | `#EFEDE4` | recessed fills, bar tracks |
| `--card` | `#FFFFFF` | cards, story blocks, rails |
| `--edge` / `--edge2` | `#E6E3D8` / `#D9D5C7` | borders (hairline / stronger) |
| `--ink` / `--ink2` / `--mute` | `#26241F` / `#5B584F` / `#8A877D` | text: primary / secondary / muted |
| `--clay` / `--clay-soft` / `--clay-deep` | `#C96442` / `#F0DACF` / `#A94E30` | **Claude brand accent** + candidate-output nodes |
| `--green` / `--green-edge` / `--green-soft` | `#12936A` / `#1BA97A` / `#DCF0E7` | **surviving / pass** state |
| `--gold` / `--gold-edge` | `#C08415` / `#D69A1F` | **newly-iterated offshoot** |
| `--gray` / `--gray-edge` | `#B0ABA1` / `#CBC7BC` | **killed / pruned** (dashed, de-emphasized) |

**Semantic rule (from audit):** green/amber-gold/gray/clay carry *state only*; never decorative. Node color = one glance of provenance at overview zoom. Real vs synthetic uses green (real) vs a muted amber pill; live vs offline uses the green-dot Claude badge.

### Typography
- Faces: `--sans` = system UI stack (`-apple-system, "Segoe UI", Inter, Roboto…`); `--mono` = `SFMono/Menlo/Consolas`. Mono reserved for **numbers/data only** (AUCs, pLDDT, scores); labels/titles in sans.
- 6-step scale (replaces the ~20 ad-hoc sizes): `11 / 12 / 13 / 15 / 22 / 34`. All-caps only for true section labels (`.dlabel`) with `.1em` tracking.

### Spacing / shape / elevation
- Radius: 8 (controls), 10–14 (cards). Border: 1px hairline `--edge`. Shadows: soft, low-opacity warm (`rgba(40,36,28,.05–.10)`), reserved for floating layers (rails, prompt, story blocks).
- One **`.bar` primitive** (track `--paper2` + fill by semantic color) replaces the ~8 bespoke meters in the old app.

### Component inventory (each spec'd with states)
1. **Trust HUD — top**: wordmark `NeuroAD` + tagline "AlphaFold for Alzheimer's neuroimaging"; Claude live/offline badge (green dot); data-mode segmented control (Synthetic demo / Real ADNI · OASIS); Overview/home button.
2. **Trust HUD — bottom**: verdict/provenance rail (`.stamp` state pill, `.caveat` honesty text, `.num` mono metric) + Replay.
3. **Entry card** (Claude-Sciences replica): kicker, H1 value prop, sub, prompt textarea + clay "Investigate" button, example-hypothesis chips.
4. **Canvas**: SVG `#cam` group; wheel-zoom / drag-pan.
5. **Nodes** (5 roles): `root`, `gold` offshoot, `green` survivor, `gray` killed, `clay` **candidate-output** (larger, hero) — circle + label + sub + optional icon. LOD states: dot / dot+label / dot+inline story block.
6. **Edges**: solid semantic (survivor/iterate/translate) + gray **dashed** (killed); optional edge label ("iterate", "translate", "routed from p-tau217").
7. **Story blocks** (reusable, provenance-bearing): `cohort/dataset`, `confound-check` (`.check/.mark/.q`), `kill-reason`, `candidate-target` (`.rank/.bar` proteins, `.exp` experiments, `.illus` disclaimer).
8. **Provenance chip**: per-leg pills — dataset badge, real/synthetic, biomarker anchor (p-tau217/GFAP), prior-art citation, live/offline, per-source `SNAPSHOT|LIVE|SURROGATE`.
9. **Collapsible chat rail** + persistent "Claude · ask" handle.
10. **Node drawer** (secondary pin target).
11. **Mini-map** (new): corner overview + viewport rectangle.
12. **Guided-path controls** (new): prev/next, step counter, play/pause, "Resume tour" pill.

### Motion
- One camera path type: Van Wijk–Nuij swoop, constant perceived velocity, interruptible (see Part B). Content fades (story blocks) ease ~150ms; camera never uses CSS transitions. `prefers-reduced-motion` → cut/cross-fade.

---

## PART B — Interaction model (design thinking)

### Semantic zoom (LOD) — emphasize the ZUI
- **Overview (k<0.75):** whole tree as colored dots; read the *shape* (how many died/survived/produced candidates).
- **Mid (0.75–1.6):** dot + label + one-line status.
- **Detail (k>1.6):** node's **story block materializes inline on the canvas** (not only the drawer). Hysteresis on thresholds prevents flicker.

### Hierarchy by proximity — layout IS the outline
Root center-left; offshoots nested near parent; each survivor's candidate output nested immediately right; dead branches short-stubbed near parent and de-emphasized. `x` strictly increases with depth; top lineage entirely above bottom lineage (enables the monotonic tour). Claude authors coordinates.

### Story blocks for provenance
Each block carries its own provenance chip inline; provenance renders at every LOD (dot color → one-line badge → full pill row) plus a persistent HUD summary. Provenance is spatially bound to the claim it supports.

### Zoom-back + the "path"
- Persistent **Overview/home** (Esc / background click) → smooth fit to whole tree.
- **Mini-map** shows viewport within the tree.
- **Guided path**: ordered narrative (root → killed stub → survivor → candidates), steppable by arrows, auto-advance with dwell, **free-deviate-and-resume**. Layout separated from narrative sequence; the chat extends the path (a follow-up spawns a new gold branch).

### Disorientation mitigation (research-grounded)
Root cause in the prototype: camera reversed direction repeatedly and per-segment CSS easing restarted velocity each hop. Fixes:
1. **Van Wijk & Nuij smooth zoom** — constant *perceived* velocity, zoom-out→pan→zoom-in swoop, position+velocity continuous, interruptible.
2. **Monotonic path** (left→right, top branch before bottom; no ping-pong).
3. **Zoom to bigger targets** (frame whole subtrees) + keep offshoots close.
4. **Dwell** on each beat; honor `prefers-reduced-motion`.
5. **Overview anchor + mini-map** for context maintenance.
Basis: Van Wijk & Nuij (constant-velocity interruptible zoom); Manitoba HCI (animated transitions ~2× faster, fewer errors); Prezi practical rules.

---

## PART C — Implementation

### Track 1 — Frontend: evolve `scratchpad/sketch/index.html` → `sketch/zui.html`
Reuse the sketch's SVG `#cam` group, node model, deterministic timeline, and `?p=` capture mode. All camera motion funnels through **one** code path (`flyTo`/`view`/`apply`) so calm/interruptible/reduced-motion behavior is guaranteed everywhere.

- **Camera (recommend hand-rolled, no d3):** port `d3.interpolateZoom` math (~40–60 lines, Van Wijk–Nuij) into `interpolateZoom(a,b)→{fn(t),S}`; drive a single rAF loop `flyTo`/`tick`/`cancelFly`. Convert between sketch `view={s,tx,ty}` and Van Wijk `(cx,cy,w)`: `w=W/s`, `cx=(W/2-tx)/s`, `cy=(H/2-ty)/s`. **Delete the CSS `transition:transform` on `#cam`** (the velocity-reset culprit, sketch line ~65). Duration = Van Wijk arc length × global `SPEED`. Interruptibility: on new input capture the current interpolated `view` as the next start. `prefers-reduced-motion` → set `view` directly.
- **Semantic-zoom LOD:** `applyLOD()` at end of `apply()`/`tick`; set `svg.dataset.lod` (CSS shows/hides label/sub); mount story blocks as **`foreignObject`** inside `#cam` only when `level===2` AND node in viewport (frustum cull), pooled per node id. Guard on "did level/viewport change" to avoid per-frame stutter.
- **Layout:** keep Claude-authored coords (not d3-hierarchy); add `role`, `branchId`, `parent`, `lod2Side`; helpers `treeBBox()` / `subtreeBBox(branchId)` for home-fit and subtree framing.
- **Story blocks:** replace the per-`kind` `openDrawer` if/else (sketch line ~312) with a `STORY_BLOCKS` registry keyed by role; reuse existing `.check/.rank/.bar/.exp/.dlabel/.illus` CSS; wrap each with a provenance chip; drawer stays as pin target.
- **Nav shell:** `goHome()`=`flyTo(fit(treeBBox))` on Esc + background click + Overview button; mini-map (corner SVG from `NODES` + live viewport rect); `TOUR=[step…]` (monotonic, per-step `dwell`, deviate-and-resume via `flyTo` from live `view`). Repurpose the existing grow-in intro; keep `?p=` sampling the tour deterministically.
- **Risk:** `foreignObject` can blur at high zoom in some browsers → fall back to native SVG text/rect blocks if needed. Prototype `foreignObject` first.

### Track 2 — Data: surface the real translation + derive a `tree` in `build_demo_data.py`
`translate(mechanism, df)` (`src/neuroad/harness/translation.py:156`) already returns `TranslationLead.to_dict()` with `ranked_targets / top_target / structure / repurposing / wet_lab_experiment / provenance / caveat`, offline-first, and is wired to `card.translation` (`pipeline.py:210`, serialized `contract.py:278`). The only gap is surfacing.

- **Engine path** — in `_real_case` (`build_demo_data.py:~821`, near the `card.to_dict()` copies ~926) add: `if d.get("translation"): case["translation"] = d["translation"]`.
- **Fallback path** — add `_static_translation(mechanism)` calling `translation.translate(mechanism, df=None)` (offline), with a `_CALIBRATED_TRANSLATION` constant mirroring the 3 snapshots as last resort; call it in the scaffold loop (`~782`) for `promoted` cases.
- **Detective** — in `_cluster_payload` (`~1148`), route mechanism per cluster from its biomarker effect sizes (gfap>p_tau217 → glial else amyloid_cascade), attach `translation` for promoted clusters.
- **Normalized `tree` object** (new, derived in the build so the UI consumes one shape): `tree.{root,nodes[],edges[]}` where nodes = `hypothesis` (root; story = `investigate.kill_criteria/spec`) → 5 `gate` nodes (from `tests[]`; `state`=result; site_scanner gate carries `leakage_margin`+`confound_leaderboard`; replication gate carries `seed_sweep`) → `verdict` node (`score/verdict/leakage_margin/honesty_rung/double_dissociation` + courtroom/caveats) → terminal `candidate` node (`case.translation` verbatim). **KILL cases dead-end with no candidate node** (honest punchline).
- **Schema:** bump `meta.schema` → `1.1.0`; add `meta.translation_note` and `meta.molecular_sources_unverified: true` (PI4AD/TxGNN/AlphaFold/Jasodanand not yet in `docs/CITATIONS_VERIFIED.md` — tracked follow-up). Field names mirror `TranslationLead.to_dict()` exactly; `priority_score/rank/mean_plddt` may be `null` → render "unranked", never 0.
- **Provenance honesty:** per-leg source badges (`ranked_targets[].source`, `structure.source`, `repurposing[].source`, `provenance` roll-up); pin `translation.caveat` verbatim to the candidate header; label the `verdict→candidate` edge "routed from plasma p-tau217 → amyloid_cascade" (translation is from the biomarker, **not** the imaging embedding). Keep multimodal_transformer orphaned; do not imply it drove routing.

### Phasing (full feature, calmest-first; each phase demoable, Phase 0 is safe fallback)
0. **Kill the nausea** — Van Wijk camera + monotonic tour + reduced-motion in `zui.html` (same visuals, calm).
   Data: 2-line `_real_case` copy + rebuild `demo_data.json` (proves real translation flows; `grep -c pi4ad demo_data.json > 0`).
1. **Semantic-zoom LOD** — dots↔labels; verify shape-reading at overview.
2. **Inline story blocks + provenance** — registry, `foreignObject` mount/cull, chips at all LODs, HUD summary. Data: derive `tree` + candidate story block.
3. **Nav shell** — mini-map, home/zoom-back, guided path (arrows/counter/auto-advance/resume).
4. **Real-data binding** — `zui.html` consumes the regenerated `tree`/`translation` (real PI4AD/AlphaFold/compounds), not hand-typed labels; per-leg provenance badges; pinned caveat.
5. **Detective + polish** — unsupervised phenotype parallel tree (`discovery.points[]/clusters[]` → candidate nodes); substrate toggle re-authors the tree; mini-map drag; `?p=` capture wired to tour.

---

## Verification
- **Design system:** render `tokens.css` swatches + a component gallery page; confirm 6-step type scale, semantic-only color, one `.bar` primitive; ingest `DESIGN-SYSTEM.md` into Claude design and confirm it round-trips before building Part C.
- **Camera calm:** serve `zui.html` locally; headless-screenshot the guided tour at sampled `?p=` positions; assert monotonic camera path (x non-decreasing, no y reversals); toggle `prefers-reduced-motion` and confirm cuts (no fly). Manual: run the tour end-to-end and confirm no direction ping-pong / dwell on each beat.
- **Semantic zoom:** script wheel-zoom through k bands; assert `svg.dataset.lod` transitions with hysteresis and story blocks mount only at LOD2 in-viewport (≤3 live).
- **Real data + honesty:** rebuild `demo_data.json`; assert the synthetic SURVIVOR carries real `translation` (APP / AlphaFold P05067 pLDDT 67.38 / repurposing NCTs) with correct `source` tags; assert KILL case has no candidate node; confirm `translation.caveat` and per-leg badges render and never recede; confirm `meta.molecular_sources_unverified` surfaces a footnote.
- **Fallback parity:** run build with engine present vs absent; assert the `translation` block is byte-identical.

## Critical files
- `scratchpad/sketch/index.html` → evolve into `sketch/zui.html` (camera `focusOn/apply/CAMK` ~L233–262, `NODES` ~L181–190, `openDrawer` ~L312, LOD CSS ~L61–79).
- `neuroad-discovery-engine/app/build_demo_data.py` (`_real_case` ~821/926, `fallback_demo_data` ~736, scaffold loop ~782, `_cluster_payload` ~1148; add `tree` transform + `_static_translation`).
- `neuroad-discovery-engine/src/neuroad/harness/translation.py` (reuse `translate()`/`TranslationLead` verbatim — no change).
- `neuroad-discovery-engine/src/neuroad/contract.py` (`ClaimCard.to_dict` ~278 already serializes `translation`).
- `neuroad-discovery-engine/app/demo_data.json` (regenerated; new `translation` + `tree`; bump `meta.schema`).
- New: `neuroad-discovery-engine/design/DESIGN-SYSTEM.md` + `tokens.css`/`tokens.json` (deliverable #1).
- Follow-up: `docs/CITATIONS_VERIFIED.md` (+ `data/registry.yaml`, `docs/DATA_PROVENANCE.md`) — add the 4 molecular sources.
