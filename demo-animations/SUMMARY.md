# NeuroAD Demo Animations — Build Summary

*Built overnight, autonomously, while you slept. Everything below is done and verified.*

**TL;DR:** You now have **5 dynamic, on-brand animated scenes** + **two ways to play them**
(a single double-clickable file, and a folder player). They translate the Medkit hackathon
winner's visual grammar (animated pipeline cards, radial agent topology, flywheel, title page)
into *our* warm "Claude Sciences paper" ZUI palette. Open **`neuroad_demo_animations_standalone.html`**
and screen-record.

---

## What to open first

- **`neuroad_demo_animations_standalone.html`** — one file, double-click, no server, all 5 scenes
  embedded. Tabs / arrow keys / number keys 1–5 to switch; press **`h`** to hide the bar and record
  a scene full-bleed. **This is your main deliverable.**
- `neuroad_demo_animations.html` — same player but loads scenes from `scenes/` (handy while iterating).
- `scenes/scene*.html` — the five scenes as individual standalone files (open/record any one directly).
- **Record target: 1920×1080.** Every scene auto-plays, loops, and honors `prefers-reduced-motion`.

> These are **live HTML animations** (rAF loops, CSS keyframes, travelling-pulse connectors), not
> static images. A scene only looks still if its browser tab is backgrounded (browsers pause
> animation there) — it plays the moment the tab is focused, i.e. always, while you record it.

---

## The 5 scenes — what each is for

| # | Scene | What it shows | Demo beat | Emulates (Medkit ref) |
|---|-------|---------------|-----------|------------------------|
| 0 | **Title / opener** | `NeuroAD` hero, "An AlphaFold for Alzheimer's neuroimaging", "Kill weak · Surface strong · Accelerate the experiment", real-cohort chips (ADNI/OASIS/EPAD), floating science doodles, soft wave. | **Cold open** before you speak. | His warm title page (ref C) |
| 1 | **Discovery Pipeline** | 7 stage cards (Input → Frozen Neuro-JEPA → One Probe/3 Questions → Referee → Biomarker Bridge → PI4AD → Output) with kicker pills, node-handle dots, glyphs, mono metrics, arrow connectors, a travelling data-pulse, and a dashed **KILLED** offshoot at the Referee. | **Tech stack / architecture** — walk the pipeline. | His animated card pipeline "What happens during one case" (ref A) |
| 2 | **Discovery Flywheel** | A rotating ring: Hypothesis → Falsify → Survivor → Route → Rank → Experiment → *sharper* Hypothesis, around a clay hub "Kill weak · surface strong," with a sweeping highlight + return arrow. | **The flywheel** — why it compounds. | The flywheel you loved |
| 3 | **Orchestrated by Claude** | Central **CLAUDE · Opus orchestrator** hub in a dotted tick-ring; 6 "TOOLS IN" cards feed curved pulse-connectors into it; 3 "ARTIFACTS OUT" cards stream out; caption on build-time Claude use. | **How I used Claude** (both runtime orchestration *and* Claude Code / subagents / Claude Design). | His radial "Managed Agent — pulses are real tokens of work" (ref B) |
| 4 | **One Probe, Three Questions** | Frozen embedding heatmap → one probe head swapping its label column → 3 gauges (Disease green / Scanner gray "near chance" / p-tau217 green) → verdict: **REAL BIOLOGY** vs a **KILLED** scanner artifact. "The disagreement is the discovery." | **The core scientific "aha."** | His clear engineering-explainer style |

Scenes 1 (architecture) and 3 (how-I-used-Claude) are your two core asks; 2 (flywheel), 0 (title),
and 4 (core insight) round it into a full suite you can pull from in any order.

---

## How this was built (end-to-end, with ultracode)

1. **Studied the reference.** Read your `medkit_video_analysis_and_demo_guide.md`, stepped through the
   Medkit YouTube demo in-browser (captured his animated pipeline + product UI), and pulled the exact
   ZUI design tokens from the live app (`app/zui.html`) so the scenes match your product 1:1. Your
   three reference screenshots (radial topology, card pipeline, title page) set the card grammar.
2. **Wrote a strict design contract** (`DESIGN_CONTRACT.md`) + **canonical facts file**
   (`PIPELINE_FACTS.md`) so every scene is on-brand, accurate, and translates his charm into *our*
   palette (semantic clay/green/gold/gray, warm paper, honesty-first) — **not** his pastels.
3. **Ran a multi-agent workflow (ultracode):** judge-artifact research → **5 scenes built in parallel**
   by separate high-effort agents → an integrator wired them into the player. 7 agents, 0 errors.
4. **Verified every scene in-browser** (composition, palette, legibility, encoding) and **ran the
   `ux-design-reviewer`** against our ZUI / Dana Cho canon.
5. **Applied the review fixes** (below), then generated the single-file standalone build.

### What the judge research said (informed the scenes)
- Show the product **working, live, early**; the demo *is* the pitch. One polished end-to-end flow beats
  five half-features. → the Pipeline + Probe scenes make the flow legible in seconds.
- Prove Claude is doing something a wrapper can't: **orchestration, verification, provenance**. → the
  Claude topology scene + the Referee/"kill weak" framing throughout.
- Polished-demo visual techniques that read as high-production: **travelling-pulse connectors, sequential
  reveal, dot-grid engineering canvas, consistent semantic color, calm smooth motion** — all used here.
- Antipatterns avoided: dense one-shot diagrams, buzzword veneer, jittery cuts, overclaiming.

### Design-review fixes applied
- **scene1:** fixed a missing `<meta charset>` that mojibaked the dashes/arrows; nudged two sub-scale
  font sizes onto the type scale.
- **scene2:** enlarged the flywheel (tightened viewBox + taller SVG) to fill the frame and made the
  station labels/metrics legible at recording distance.
- **scene3:** neutralized the input-tool cards so state colors (green/gold/clay) aren't used
  decoratively — now *neutral inputs flow into the clay Claude hub, meaningful artifacts flow out*
  (on-canon); fixed a possible pulse flash at the SVG origin.
- **scene4:** reserved a caption gutter and gave the caption a product-style paper rail so the verdict
  cards never crowd it; nudged two sub-scale fonts.
- **scene0:** removed an off-brand middot in the kicker; trimmed 7 floating doodles to 5 for calm.

---

## Honesty / accuracy notes (so the demo stays defensible)
- All AUC / r / rank numbers in scene 4 are **explicitly labelled "illustrative — they demonstrate the
  decision logic, not a validated result."** They show the *method*, not a specific finding.
- Scene 1's output card and scene 4 carry the caveat: candidate outputs are **hypotheses to test, not
  validated targets.** This matches your project's honesty doctrine.
- Stage/model names and numbers (Neuro-JEPA 1.5M scans frozen, PI4AD 0–10, p-tau217 r ≥ 0.2, ~6-week
  iPSC/organoid readout, HRAS/MAPK1, APP/ESR1 calibration) are taken straight from your pipeline docs.

---

## File inventory (`demo-animations/`)
```
neuroad_demo_animations_standalone.html   ← open this (single file, all scenes embedded)
neuroad_demo_animations.html              ← folder-based player (loads scenes/)
scenes/scene0_title.html … scene4_probe.html   ← 5 standalone scenes
README.md                                 ← how to open / record
DESIGN_CONTRACT.md                        ← the design system the scenes follow
PIPELINE_FACTS.md                         ← canonical names/numbers used
build_standalone.py                       ← regenerates the standalone if you edit a scene
SUMMARY.md                                ← this file
```

## Suggested next steps (optional, when you're back)
- Record each scene at 1920×1080 (Screen Studio gives the cinematic zoom/cursor polish the winner used).
- If you want a scene tweaked (copy, timing, an extra stage), each is a small self-contained file —
  edit and re-run `python3 build_standalone.py`.
- Consider a 6th "Referee gauntlet" close-up if you want to dwell on the falsification story; the
  current 5 already cover architecture, Claude, flywheel, title, and the core insight.
