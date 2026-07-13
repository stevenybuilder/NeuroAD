# NeuroAD — Demo Animations

Five standalone, on-brand animation scenes for the NeuroAD demo, plus a single
master "player" page for presenting and recording them.

## The 5 scenes

Each file in `scenes/` is fully standalone: all CSS/JS is inline, no external
requests, system fonts only. Every scene **auto-plays, loops, and honors
`prefers-reduced-motion`** (motion collapses to a static end-state for viewers
who request reduced motion).

| # | File | Shows | Narration beat it supports |
|---|------|-------|----------------------------|
| 0 | `scenes/scene0_title.html` | **Title / opener** — NeuroAD wordmark, tagline, the "why". | Cold open — sets the frame before you speak. |
| 1 | `scenes/scene1_pipeline.html` | **Discovery pipeline** — data → compute → results flow. | **Architecture / tech stack** — walk the pipeline end-to-end. |
| 2 | `scenes/scene2_flywheel.html` | **Discovery flywheel** — how each result feeds the next hypothesis. | **The flywheel** — why this compounds over time. |
| 3 | `scenes/scene3_claude.html` | **Orchestrated by Claude** — Claude driving the pipeline. | **How I used Claude** — the orchestration layer. |
| 4 | `scenes/scene4_probe.html` | **One probe, three questions** — a single probe answering multiple questions. | Payoff — the concrete "so what" of the engine. |

## Open the master player

**Easiest — one file, double-click:** **`neuroad_demo_animations_standalone.html`**.
All five scenes are embedded inside it (as base64 data-URIs), so it needs **no
server and no other files** — just double-click and present. This is the one to
copy to another machine or hand off.

**Folder version:** **`neuroad_demo_animations.html`** loads the scenes from
`scenes/` via `<iframe src=…>`. It also opens by double-click over `file://` in
Chrome; if a browser ever blocks the local iframes, either use the standalone file
above or serve the folder (`python3 -m http.server` in this directory).

Both render one scene at a time in a full-bleed `<iframe>` behind a slim on-brand top bar.

Controls:

- **Tabs** (Title · Pipeline · Flywheel · Claude · Probe) — click to jump; active tab is clay.
- **Prev / next** — the `‹` `›` buttons.
- **Left / Right arrows** (or Space) — step between scenes.
- **Number keys 1–5** — jump straight to a scene.
- Switching scenes **reloads the iframe**, so each animation restarts from the top — good for clean takes.

## Record a single scene full-bleed

1. Open the master and switch to the scene you want.
2. Press **`h`** (or the "Hide bar" button) to hide the top bar — the scene fills the whole window with no chrome. Press `h` again to bring the bar back.
3. Alternatively, open the scene file directly (e.g. `scenes/scene1_pipeline.html`) for a chrome-free window from the start.

**Record target: 1920×1080** (16:9, 1080p). Size the browser window / capture region to 1920×1080 before recording.

## Notes

- `neuroad_demo_animations_standalone.html` is the fully self-contained build (scenes inlined, zero external requests). If you edit a scene, regenerate it with `python3 build_standalone.py`.
- All scenes are standalone and reduced-motion-aware — safe to open or embed individually.
- Every scene animates via `requestAnimationFrame` / CSS keyframes. Note: browsers pause animation in a **backgrounded/hidden tab**, so a scene may look blank until its tab is focused — it starts as soon as the tab is visible (i.e. always, when you're recording it).
- Palette: `--paper #F5F4EE`, `--card #FFFFFF`, `--edge #E6E3D8`, `--clay #C96442` (NeuroAD ZUI), system font stack. Mono is used for numbers only.
