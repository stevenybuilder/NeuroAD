# NeuroAD Demo Animations — Shared Design Contract (READ FIRST)

You are building ONE of four **dynamic, self-animating** scenes for the NeuroAD Discovery
Engine demo video. These are NOT static PNGs — they are living HTML/CSS/JS animations the
founder will screen-record while narrating. Match the look-and-feel of the existing product
exactly, and take visual inspiration from the Opus-4.7 hackathon winner ("Medkit") whose
signature move was an **animated React-Flow-style pipeline** on a warm cream dot-grid canvas.

---

## 0. NON-NEGOTIABLE OUTPUT RULES

1. Deliver **one standalone `.html` file** — all CSS and JS inline in `<style>`/`<script>`.
   No external requests of any kind (no CDNs, no web fonts, no remote images). System fonts only.
2. It must be **full-viewport** (fills the browser window), designed to look great when
   screen-recorded at **1920×1080 (16:9)**. Center the composition; add generous margins.
3. It must **auto-play on load**, run a clean choreographed sequence, then **loop seamlessly**
   (or hold on a satisfying final beat for ~2.5s and restart). No user action required to see it.
4. Add a single small, unobtrusive **"↻ Replay"** control (bottom-right, `--mute` color, quiet)
   so the founder can re-trigger the sequence on demand. Nothing else interactive is required.
5. Fully honor `prefers-reduced-motion: reduce` — replace fly/slide/particle motion with
   simple cross-fades and show the final composed state.
6. Vanilla JS only. Prefer `requestAnimationFrame`, CSS transitions/keyframes, and SVG.
   Absolutely no libraries. Keep it robust — it must run by just double-clicking the file.
7. Put a `<title>` and a short comment header naming the scene.

---

## 1. COLOR TOKENS (use these EXACT hex values — the product's real palette)

```
--paper:#F5F4EE   /* canvas background, warm paper */
--paper2:#EFEDE4  /* recessed fills, bar tracks, card insets */
--card:#FFFFFF    /* cards, panels */
--edge:#E6E3D8    /* hairline border */
--edge2:#D9D5C7   /* stronger border / input outline */
--ink:#26241F     /* primary text */
--ink2:#5B584F    /* secondary text */
--mute:#8A877D    /* muted text / captions */
--clay:#C96442      /* CLAUDE BRAND ACCENT; primary; candidate/output nodes */
--clay-soft:#F0DACF /* accent-soft fill */
--clay-deep:#A94E30 /* accent text on light */
--green:#12936A     /* SURVIVING / pass */
--green-edge:#1BA97A
--green-soft:#DCF0E7
--gold:#C08415      /* newly-iterated / offshoot */
--gold-edge:#D69A1F
--gold-soft:#FBF2DD
--gray:#B0ABA1      /* killed / pruned (de-emphasized) */
--gray-edge:#CBC7BC
```

**Semantic color law (critical):** color encodes STATE only, never decoration.
green = survived/pass, gold = iterated/new, gray = killed/pruned, clay = Claude & final output.
Do NOT introduce new decorative hues. If a scene needs a *categorical* ramp (e.g. embedding
clusters), use desaturated blues/violets/teal that share no hue with the state palette — but
prefer to avoid it. Background is always warm paper, never white-white, never dark.

## 2. TYPOGRAPHY

```
--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
```
- Sans for ALL labels/titles/body. **Mono ONLY for numbers/data** (AUCs, scores, r-values, n,
  pLDDT, IDs, percentages). Never mono for words.
- Type scale, use ONLY these px sizes: **11 / 12 / 13 / 15 / 22 / 34** (you may go up to ~44/56
  for a single hero scene title since these are full-screen recording assets — keep it tasteful).
- ALL-CAPS only for true section/kicker labels, with `letter-spacing:.1em`. Everything else
  sentence case. Titles are weight 700, letter-spacing ~-.02em.

## 3. SHAPE / ELEVATION / RHYTHM

- Radius: 8px controls/pills, 12–16px cards/panels.
- Border: 1px hairline `--edge` default; `--edge2` for emphasis. Colored state borders use the
  matching `-edge` token.
- Shadows are soft & warm, ONLY on floating layers:
  `0 1px 2px rgba(40,36,28,.04)` (chips) … `0 8px 30px rgba(40,36,28,.10)` (cards/panels).
  Flat elsewhere. Calm, lots of whitespace, ONE focal point at a time.
- The page background should carry the product's subtle radial glow:
  `background:#F5F4EE; background-image:radial-gradient(1100px 620px at 60% 24%,#FBFAF6 0%,#F5F4EE 62%);`
- **Dot-grid texture** (the winner's signature, and it reads as "engineering canvas"): overlay a
  faint dot grid using a CSS radial-gradient background, e.g.
  `background-image:radial-gradient(rgba(40,36,28,.05) 1px, transparent 1px); background-size:26px 26px;`
  layered under content. Keep it very subtle (~5% ink).

## 4. THE "PIPELINE CARD" PRIMITIVE (emulate the winner's React-Flow node)

Reusable node/card the winner used, restyled to OUR palette. Each card:
- White fill, 1px `--edge` border, 14px radius, soft shadow, ~200–240px wide.
- **Kicker pill** top-left: tiny (11px) ALL-CAPS label in a soft pill (e.g. STAGE 1, FROZEN,
  SKILL, DATA) — pill background = the soft token of that card's state color, text = deep token.
- **Node-handle dot** top-right: a small 8px ring/dot (the React-Flow "port" signature).
- **Centered glyph**: a simple line/emoji-free SVG or unicode glyph (◇ ◆ ⌾ ⊹ ✦ ⟳ △ ▣ ⬡ etc.) in
  the state color; keep it geometric and quiet.
- **Title** (15px, weight 700, `--ink`) + **subtitle** (12–13px, `--ink2`, one short line).
- **Connectors**: thin (2.6px) curved SVG paths between cards; solid for pass/flow, gray dashed
  (`stroke-dasharray:7 7`) for killed. Optional small edge label (12px sans).
- Optional tiny mono metric on the card (e.g. `AUC 0.86`, `n=590`, `r=0.62`) — mono, `--ink`.

## 5. MOTION RULES (the product's "calm, never nauseating" doctrine)

- **Constant perceived velocity**, gentle easing (`cubic-bezier(.4,0,.2,1)` or ease-in-out).
  No jarring pops, no direction reversals, no bounce/overshoot on the camera.
- **Sequential reveal**: reveal cards/stages left→right (or around the ring) with ~180–260ms
  stagger. Each element fades+rises ~8–12px into place. Then **dwell** ~1.2–1.6s on the composed
  state before looping.
- **Data-pulse particles**: a small glowing dot (state color) travels ALONG each connector path
  (use SVG `<animateMotion>` or JS path sampling / `offset-path`) to show flow lighting up the
  next node. This is the single most important "alive" detail — every scene that has a flow
  should have travelling pulses.
- Content may fade (~150ms). Keep total loop length ~10–16s so it's easy to record a clean take.
- Respect `prefers-reduced-motion` (see rule 5 in §0).

## 6. TONE / HONESTY (the product is a rigor-first scientific tool)

- Plain-language framing, precise mono numbers. Never overclaim: candidate outputs are
  "hypotheses to test," not validated results. Where a scene shows the final target, include a
  quiet caveat line ("Decision-support candidate — not a validated target").
- The vibe is **calm, confident, warm, credible** — "Claude Sciences paper," an AlphaFold for
  Alzheimer's. NOT flashy, NOT neon, NOT dark-mode dashboard. Think elegant, editorial, alive.

## 7. STRUCTURE EACH SCENE SHOULD FOLLOW

- A small **kicker** (ALL-CAPS clay) + **scene title** (big, bold) top-left or top-center.
- The **animated composition** as the focal point (center).
- A one-line **caption** at the bottom (`--mute`, 13px) — the narration hook for that scene,
  mirroring how the winner captioned each diagram (e.g. "One frozen model. Three questions. The
  disagreement is the discovery.").
- The quiet **↻ Replay** control bottom-right.
- Optionally a tiny **NeuroAD wordmark** top-corner (`Neuro` ink + `AD` clay) for brand.

Build the most beautiful, alive, on-brand version of your assigned scene you can. This is the
founder's hero reference asset — precision and polish matter more than feature count.

---

## 8. THE WINNER'S VISUAL GRAMMAR (what to emulate — described from his real frames)

The founder specifically loved three of the Medkit demo's visuals. Replicate their *structure,
composition, and motion*, but re-skinned into OUR palette (see §9 translation rule). What they look like:

**REF A — Horizontal pipeline ("What happens during one case."):**
- A `RUN-TIME` kicker pill top-left, then a big bold title, then a single horizontal row of ~5
  equal cards connected by short straight arrows (`→`) between them, on a faint dot-grid canvas.
- Each card: rounded ~18px, a **crisp dark outline (~2.5px)** with a **hard offset drop-shadow**
  down-right (a "sticker" look — shadow is a solid darker shape, not a blur). A **white kicker
  pill** (dark outline) top-left naming the card's role (TRAINEE / REAL-TIME / STATE / PURE FN /
  OPUS 4.7). A **small node-handle dot** (colored, dark ring) top-right. A **centered line-icon**
  (star, mic, brain, sparkle, stethoscope). A **bold title** low-left and a **2-line subtitle**.
- Cards use distinct warm fills per role (cream, peach, mint, yellow, peach).

**REF B — Radial agent topology ("Sub-rules feed in. Sessions stream out. Pulses are real tokens of work."):**
- Bold title across the top. A **central circle** ("MANAGED AGENT / Opus 4.7 / medkit-attending")
  wrapped in a **dotted/ticked ring** (like a radar/gauge bezel) with a soft inner highlight.
- A **left column** of 4 stacked cards (POLICY / CONTRACT / HARD RULE / FRAMEWORK) under a white
  header pill "SUB-RULES", and a **right column** of 4 cards (SESSION / SKILL / SKILL·LOOP …)
  under a header pill "SESSIONS".
- **Curved bezier connectors** run from each left card into the hub and from the hub out to each
  right card, each connector in that card's category color, terminating in a **small colored
  connection dot** at the card edge. Motion: little **pulses travel along the connectors** toward
  and away from the hub — "real tokens of work."

**REF C — Title / onboarding page ("The clinic that lets you make every mistake before they count."):**
- A big soft **rounded-bubble wordmark** centered, a one-line tagline inside a **rounded pill**,
  a chunky **"▸ Tap to begin"** button, and a "press [space] to continue" hint below.
- Warm gradient sky, a few **playful floating doodles** (sun, clouds, sparkle, star, heart) idly
  bobbing, and a gentle **wave/hill** shape along the bottom edge. Cheerful, calm, inviting.

## 9. TRANSLATION RULE (winner's charm → OUR brand — DO NOT copy his skin)

Keep his **composition, card grammar, radial topology, dot-grid, header pills, connection dots,
and travelling-pulse motion**. But render them in the NeuroAD ZUI system:
- Palette = OUR tokens (warm paper, clay/green/gold/gray/ink) — NOT his blue/peach/mint/pink.
  Where he used 4–5 distinct card colors for *categories*, use our SEMANTIC colors meaningfully
  (green=survivor/pass, gold=iterate/new, clay=Claude/output, gray=killed, ink/neutral=input),
  or a restrained soft-tinted set drawn only from our tokens. Never neon, never random hues.
- Borders: you MAY use a slightly bolder card border than the default hairline to echo his
  "sticker" cards — but keep it tasteful and warm (e.g. 1.5–2px in `--edge2`/`--ink` at low
  opacity, soft offset shadow `0 6px 0 rgba(40,36,28,.06)` + `0 10px 24px rgba(40,36,28,.10)`).
  Lean elegant/editorial, not cartoon. When in doubt, favor OUR calmer, softer look.
- Type: our system sans (no rounded display font). Mono for numbers only.
- Tone: his product is a warm cartoon sandbox; OURS is a rigor-first "Claude Sciences paper."
  Stay calm, credible, precise, honest. Warmth yes; whimsy sparingly.

## 10. SCENE ROSTER (the founder relaxed the count — build a cohesive SUITE)

Each scene is its OWN standalone file per §0. All five share this contract so they feel like one
system. Your assignment names which scene you build. Content facts come from the NeuroAD pipeline
(see the `PIPELINE_FACTS` block in your task prompt — use those exact stages/numbers, don't invent).

- **scene0_title.html** — NeuroAD title/opener (emulates REF C, our palette). Hero wordmark
  `Neuro` (ink) + `AD` (clay), tagline "An AlphaFold for Alzheimer's neuroimaging," a sub-line
  ("Kill weak hypotheses. Surface strong ones. Accelerate the experiment."), a quiet "▸ Begin
  investigation" pill, and a few tasteful floating doodles in our palette (brain outline,
  molecule/hexagon, sparkle, DNA tick). Calm entrance animation, gentle idle bob, soft paper
  gradient + faint bottom wave. This opens the demo video.
- **scene1_pipeline.html** — "The Discovery Pipeline" (emulates REF A). Horizontal card flow of
  the pipeline stages with kicker pills, node dots, centered glyphs, arrow connectors, dot-grid.
  Cards reveal left→right with stagger; a travelling data-pulse runs the connectors and lights
  each card as it arrives. THE tech-stack/architecture asset.
- **scene2_flywheel.html** — "The Discovery Flywheel." A rotating circular loop of the discovery
  cycle (Hypothesis → Falsify/gauntlet → Survivor → Biomarker route → Ranked target → Experiment
  → sharper Hypothesis). Center hub "Kill weak · surface strong." A sweeping highlight + pulse
  orbits continuously; each station lights as the sweep passes. Momentum metaphor. The founder
  loved the winner's flywheel — make this the signature loop.
- **scene3_claude.html** — "Orchestrated by Claude" (emulates REF B radial topology). Central hub
  "CLAUDE · Opus orchestrator." LEFT column feeds IN (tools/inputs: Neuro-JEPA embeddings,
  clinical table, referee stats, PI4AD omics, AlphaFold/Boltz, literature RAG). RIGHT column
  streams OUT (artifacts: ranked protein target, falsifiable experiment card, provenance/audit
  trail). Curved colored connectors with pulses travelling in and out = "every tool call is a
  real token of work." Caption ties in BUILD-time Claude too: "Built in days with Claude Code
  auto-mode, subagents, skills, and Claude Design." THE how-I-used-Claude asset.
- **scene4_probe.html** — "One Probe, Three Questions" (the core scientific insight). A frozen
  embedding vector streams into ONE probe head; the same head is pointed at THREE label columns —
  Q1 Disease (AD vs CN), Q2 Scanner (site ID), Q3 Protein (p-tau217). Three gauges/bars fill with
  mono AUCs; then the DISAGREEMENT resolves a verdict: "predicts disease AND p-tau217 but NOT the
  scanner → real biology" (green) vs "predicts disease AND the scanner → artifact, killed" (gray).
  The intellectual "aha" of the whole engine.

You may add ONE tasteful extra flourish if it strengthens your scene, but do not bloat it.

