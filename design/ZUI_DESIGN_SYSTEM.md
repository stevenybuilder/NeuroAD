# NeuroAD — ZUI Decision-Tree Design System

> Handoff doc for Claude design. Self-contained: product framing, visual tokens, typography, component inventory (with states), interaction model, and motion rules for the zooming decision-tree canvas. Ingest this before generating or refining UI.

---

## 1. Product & theme framing

**NeuroAD Discovery Engine** — "an **AlphaFold for Alzheimer's neuroimaging**." It uses fMRI + biomarker signals to **kill confounded hypotheses, surface strong ones, and translate survivors into ranked candidate proteins, compounds, and testable wet-lab experiments.**

The UI is a **light, warm "Claude Sciences paper" zooming canvas** (ZUI). A typed hypothesis becomes the root of a decision tree that builds itself; branches die (gray), survive (green), iterate (gold), and terminate in candidate-target outputs (clay). The user zooms/pans; detail reveals itself by proximity.

**Design north star:** calm, one focal point at a time, plain language, and **trust/provenance that never recedes** — provenance is spatially bound to each claim.

---

## 2. Color tokens

Light theme, warm paper. Hex + role. **Color is semantic only — never decorative.**

| Token | Hex | Role |
|---|---|---|
| `--paper` | `#F5F4EE` | canvas background (warm paper) |
| `--paper2` | `#EFEDE4` | recessed fills, bar tracks |
| `--card` | `#FFFFFF` | cards, story blocks, rails |
| `--edge` | `#E6E3D8` | hairline border |
| `--edge2` | `#D9D5C7` | stronger border / input outline |
| `--ink` | `#26241F` | primary text |
| `--ink2` | `#5B584F` | secondary text |
| `--mute` | `#8A877D` | muted text / captions |
| `--clay` | `#C96442` | **Claude brand accent**; candidate-output nodes; primary button |
| `--clay-soft` | `#F0DACF` | candidate node fill; accent-soft backgrounds |
| `--clay-deep` | `#A94E30` | candidate label text / accent-on-light |
| `--green` | `#12936A` | **surviving / pass** state |
| `--green-edge` | `#1BA97A` | survivor edges |
| `--green-soft` | `#DCF0E7` | pass pill background |
| `--gold` | `#C08415` | **newly-iterated offshoot** |
| `--gold-edge` | `#D69A1F` | offshoot edges |
| `--gray` | `#B0ABA1` | **killed / pruned** (de-emphasized) |
| `--gray-edge` | `#CBC7BC` | killed edges (dashed) |

### Semantic color rules
- **State palette** (green / gold / gray / clay) encodes tree state only. At overview zoom, node color is the one-glance status: green=survived, gold=iterated, gray=killed, clay=produced candidates.
- **Real vs synthetic:** real data = green accents; synthetic/illustrative = muted amber pill + explicit caveat.
- **Live vs offline Claude:** green-dot badge = live; muted = offline/template.
- Do **not** reuse a state hue for an unrelated category (a past failure — scatter dots colliding with verdict colors). Categorical data (e.g. embedding clusters) must use a separate ramp (blues/violets/teal) sharing no hue with the state palette.

---

## 3. Typography

- **Sans face:** system UI stack — `-apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, Helvetica, Arial, sans-serif`. Used for all labels, titles, body.
- **Mono face:** `ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`. **Reserved for numbers/data only** — AUCs, pLDDT, scores, margins, IDs. Never for labels.
- **Type scale (6 steps):** `11 / 12 / 13 / 15 / 22 / 34` px. Use this ramp everywhere; no ad-hoc sizes.
  - 34 — hero metric / entry H1 emphasis
  - 22 — entry H1 / section hero
  - 15 — node labels, body, input
  - 13 — secondary body, story-block rows
  - 12 — badges, chips, captions
  - 11 — section labels (all-caps), fine print
- **All-caps** only for true section labels (`.dlabel`), with `.1em` letter-spacing. Everything else sentence case.

---

## 4. Spacing, shape, elevation

- **Radius:** 8 px (controls, pills), 10–14 px (cards, story blocks, prompt).
- **Border:** 1 px hairline `--edge` default; `--edge2` for inputs/emphasis.
- **Elevation:** soft warm shadows only on floating layers — `0 1px 2px rgba(40,36,28,.04)` (chips), `0 6–14px 24–44px rgba(40,36,28,.05–.10)` (rails, prompt, mounted story blocks). Flat everywhere else.
- **Rhythm:** use whitespace and border-weight to *group* (the audit fix). The one focal element (active node / verdict) gets more breathing room than supporting elements.

### The one bar primitive
A single `.bar` component replaces all bespoke meters: track = `--paper2`, fill = semantic color, optional tick. Sizes via modifier (sm/md). Use it for every magnitude (confound effect, PI4AD score, pLDDT, stability). Never reinvent per-panel.

---

## 5. Component inventory (with states)

### Chrome (always present — trust must never recede)
1. **Trust HUD — top bar**
   - Wordmark `Neuro` + clay `AD`; sub-line tagline "AlphaFold for Alzheimer's neuroimaging".
   - **Claude badge** — green dot + "Claude · live" | muted "offline template".
   - **Data-mode segmented control** — "Synthetic demo" / "Real ADNI · OASIS" (real = green).
   - **Overview / home button** (returns to full-tree zoom).
2. **Trust HUD — bottom rail**
   - `.stamp` state pill (e.g. "Candidates ranked" / "No candidates yet"), `.caveat` honesty line, `.num` mono metrics. Leads with the payoff ("3 protein targets · 2 compounds · 4 experiments — candidates to test, not confirmed").
   - **Replay** control.

### Entry (Claude-Sciences replica)
3. **Entry card** — kicker (clay, uppercase), H1 value prop, sub, **prompt** (white card, textarea + clay "Investigate →" button), **example-hypothesis chips**. On submit → full transition into the canvas.

### Canvas
4. **Canvas** — SVG with a transformed camera group; wheel-zoom, drag-pan.
5. **Nodes** — 5 roles, each = circle + label + sub + optional icon:
   - `root` (clay ring, white fill) — the hypothesis.
   - `gold` — newly-iterated offshoot.
   - `green` — surviving branch.
   - `gray` — killed branch (faded, `.dead`, opacity ~.55).
   - `clay` **candidate-output** — larger/hero, clay-soft fill, `⌾` icon; the terminal payoff.
   - **LOD states:** dot only (overview) → dot+label+sub (mid) → dot + **inline story block** (detail).
6. **Edges** — solid semantic curve (survivor/iterate/translate) or **gray dashed** (killed); optional edge label ("iterate", "translate", "routed from p-tau217 → amyloid_cascade").

### Story blocks (reusable, provenance-bearing)
7. Mounted **inline on the canvas** at detail zoom (and pinnable to a drawer). Each carries a provenance chip.
   - **cohort/dataset** — n, sites, real/synthetic, license.
   - **confound-check** — the 5 gauntlet tests (`.check` rows: ✓/✗/— mark, label, plain question, per-test `.bar`).
   - **kill-reason** — the winning confound + AUC comparison that killed the branch.
   - **candidate-target** — ranked proteins (`.rank` + `.bar`, PI4AD score), AlphaFold structure (pLDDT gauge + model link), repurposing compounds (`.exp`, with ClinicalTrials.gov NCT), wet-lab experiment text, `.illus` disclaimer.
8. **Provenance chip** — per-leg pills, rendered at every zoom level (node color at overview → one-line badge at mid → full pill row at detail):
   - dataset badge (REAL ADNI / REAL OASIS / SYNTHETIC HARNESS)
   - real vs synthetic
   - biomarker anchor (p-tau217 / GFAP / "no anchor")
   - prior-art citation pill (clickable)
   - live vs offline Claude
   - per-source tag on each candidate leg: `SNAPSHOT` / `LIVE` / `SURROGATE`
   - **Non-dismissible caveat** pinned to candidate header: "Decision-support only … NOT a validated target, and never derived from the imaging embedding."

### Navigation & conversation
9. **Collapsible chat rail** (left) + **persistent "Claude · ask" handle** (never fully hidden). Expanding pushes the canvas (re-frames the tree), doesn't cover it. A follow-up question spawns a new gold branch.
10. **Node drawer** (right) — secondary "pin this block" target.
11. **Mini-map** (corner) — miniature tree + live viewport rectangle ("where am I").
12. **Guided-path controls** — prev/next, step counter ("2 / 6"), play/pause, "Resume tour" pill.

---

## 6. Interaction model (what design must support)

- **Semantic zoom (level-of-detail):** zoom changes *what is shown*, not just scale.
  - Overview (zoomed out): tree as colored dots — read the *shape* (how many died/survived/produced candidates).
  - Mid: dot + label + one-line status.
  - Detail (zoomed in): the node's story block materializes inline. Use hysteresis on thresholds to avoid flicker.
- **Hierarchy by proximity:** spatial layout IS the outline. Root center-left; offshoots nested near parent; each survivor's candidate output nested immediately to its right; dead branches short-stubbed near parent and de-emphasized. `x` increases with depth; the top lineage sits entirely above the bottom lineage.
- **Zoom-back is first-class:** persistent Overview/home (Esc / background click) always returns to the whole tree; mini-map maintains orientation.
- **Guided "path":** a deliberate camera tour (root → killed stub → survivor → candidates), steppable and auto-advancing, that the user can freely deviate from and resume. Layout is separate from narrative sequence.

---

## 7. Motion & disorientation rules (critical — the prototype felt slightly nauseating)

**Cause to avoid:** camera direction reversals (up/down/up) and per-segment easing that restarts velocity each hop.

**Rules:**
1. **One camera path type — Van Wijk & Nuij smooth zoom:** constant *perceived* velocity; the "swoop" (zoom-out → pan → zoom-in); position **and** velocity continuous (no pops); interruptible (re-plan from current state on new input).
2. **Monotonic path:** move left→right, finish the top branch before the bottom; never ping-pong.
3. **Zoom to bigger targets:** frame whole subtrees, not tiny nodes; keep related nodes close (short distance = gentle swoop).
4. **Dwell:** arrive, hold on the beat (~1.4 s), then move; fewer transitions.
5. **Respect `prefers-reduced-motion`:** cut/cross-fade instead of fly.
6. **Content vs camera:** story blocks may fade (~150 ms ease); the *camera* never uses CSS transitions (that was the velocity-reset bug).

Research basis: Van Wijk & Nuij constant-velocity interruptible zoom; Manitoba HCI (animated transitions ~2× faster, fewer errors when smooth); Prezi practical rules (no reversals, close objects, big targets, slow pacing).

---

## 8. Accessibility & honesty checklist
- Provenance/verdict/caveat visible at every zoom level; never collapsed away.
- Real vs synthetic and live vs offline always legible before any number is read.
- `prefers-reduced-motion` fully honored.
- Numbers precise (mono), framing human (sans, plain language) — e.g. "Disease signal, or just which machine acquired the scan?" not "scanner-AUC leakage margin ≤ 0".
- Candidate outputs always tagged as hypotheses to test, not validated results.

---

*Reference implementation (visual source of truth): the light "AlphaFold" prototype in `scratchpad/sketch/index.html`. This document is the design contract to evolve it into the full ZUI feature.*
