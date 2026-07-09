# Frontend Polish Spec — visual contract for WAVE 3

This is a **spec, not code.** It defines exact targets, timings, colors, and choreography
for the WAVE-3 animation pass on `app/index.html`. Design does **not** edit `index.html`;
frontend implements this. All references are to the current single-file app.

**Global rules**
- Every animation honors `prefers-reduced-motion: reduce` by **snapping to the final
  state** (the app already has an instant-render path at `runGauntlet()` line ~4280 —
  extend it, don't fork it).
- All timers go through `S.timers` and are cleared by `clearTimers()` (line 4245) so the
  autopilot `Esc`/`Space` cancel works.
- Numeric text uses the existing tabular-mono (`.num` / `font-variant-numeric:tabular-nums`)
  so digits don't jitter mid-count.
- Palette tokens (do not invent colors): `--red #ff5d6c` / `--red-d #2c1015`,
  `--green #37d69a` / `--green-d #0f2b22`, `--amber #f2b53b`, `--accent #5bc8ff`,
  `--accent2 #7aa2ff`, `--ink #e7eef6`, `--ink3 #8ea3b8`, `--star #ffd166`,
  scanner colors `--sc0 #4c9aff` `--sc1 #f2a93b` `--sc2 #22c3a6`.

---

## 1. Count-ups synced to the ~1.1s needle glide

The needle (`#needle`) glides via CSS `transition` on `left` (~1.1s ease) when
`setNeedle(score)` is called in `renderVerdict(true)` (line 3872). The count-ups must be
choreographed **around that 1.1s glide** so the number resolves and the needle "seats"
as one landing.

**Easing:** hand-rolled `easeOutCubic` (`p => 1 - Math.pow(1-p, 3)`) driven by
`requestAnimationFrame`. One shared helper `countUp(el, from, to, ms, fmt)`.

| Element (id) | From | To (data path) | Duration | Fires at | Format |
|---|---|---|---|---|---|
| Naive AUC big (`#…` line 3826) | `0.50` | `curCase().naive_effect.value` (KILL 0.87) | **900ms** | cold open / claim render | 2-dp, no `+` |
| Verdict score (`#vScore` line 3873) | `0` | `curCase().score` (KILL 40, SURVIVOR 80) | **1000ms** | same tick as `setNeedle()` | integer |
| Leakage margin (`#mv` line 3861) | `0.00` | `curCase().leakage_margin.margin` (KILL −0.08) | **800ms** | with pair-bar race (§5) | signed 2-dp |

**Sync contract for the verdict landing (the payoff beat):**
- `t0` = the moment `renderVerdict(true)` runs (inside the `endT` timeout, line 3872).
- At `t0`: start `#vScore` count-up (1000ms) **and** call `setNeedle(score)` (1100ms glide)
  simultaneously.
- The 100ms difference is intentional: the **number resolves first (1000ms), then the
  needle clicks into place (1100ms)** — a two-stage "computed → committed" landing.
- The verdict-slam stamp (§3) fires at `t0 + 1000ms` (when the number is done), so the
  stamp lands *on* the final number, not a mid-count value.
- `#vScore` color is set to `verdictColor(c.verdict)` at count start (already at line 3874)
  so the digits count up already in the verdict color.

---

## 2. Scatter morph choreography (color-by-outcome → color-by-scanner)

`toggleColor()` (line 4311) currently hard-swaps via `drawScatter()`. Replace the swap
with a **~700ms rAF tween**. The seeded points **never move** (positions are fixed by
`scatterPoints(sp)`); what animates is **color, ellipses, and centroids**.

**Data behind the reveal (why it works):** on the KILL, `scatter.outcome_gap = 0.6` but
`scatter.scanner_gap = 3.0` — so the outcome coloring produces two near-coincident,
near-round ellipses, while the scanner coloring produces two ellipses **pulled far apart
and elongated along the scanner axis.** That contrast IS "the machine is the axis."

**Three things animate, in one 700ms tween (`t: 0→1`, easeOutCubic):**

1. **Point color crossfade.** Each point has a start color (outcome: `#ff7a8a`/`#4cc2ff`)
   and a target color (scanner: `SC_COLORS[scanner]`). Lerp RGB by `t`. (Positions stay
   put.)

2. **1σ ellipses — reshape AND rotate.** The current `drawScatter` computes axis-aligned
   ellipses (`sdx`, `sdy` separately, no rotation) — **upgrade to a rotated covariance
   ellipse** so the "dominant ellipse rotates" reads:
   - Per active group, compute the 2×2 screen-space covariance `[[cxx,cxy],[cxy,cyy]]`,
     eigendecompose → orientation angle `θ`, major axis `a`, minor axis `b`, center
     `(mx,my)`.
   - Cache the **outcome-grouping** ellipse params (2 groups) and the **scanner-grouping**
     ellipse params (2 groups). Tween each of `(mx, my, a, b, θ)` from start→target.
   - Tween `θ` along the **shortest angular path** (normalize Δθ into `[-π/2, π/2]` since
     an ellipse is π-symmetric) so it rotates the short way, not a full spin.
   - Render with `g.ellipse(mx, my, a, b, θ, 0, 2π)` (fill `globalAlpha .12`, stroke `.5`,
     matching current lines 3659–3661).

3. **Ringed centroids slide.** The centroid markers (lines 3673–3678) tween their
   `(x, y)` from the outcome-centroids (two nearly-overlapping dots at cloud center) to
   the scanner-centroids (two dots pulled apart along the scanner axis). This is the
   literal "centroids slide apart" motion.

**Group-count mismatch:** outcome grouping = 2 classes; scanner grouping = `n_scanners`
(2 here). When counts differ across substrates, pair groups by nearest start-centroid and
fade in/out any unmatched group's ellipse via `globalAlpha` over the same 700ms.

**Legend + caption** (`renderScatterMeta`, line 3918) crossfade opacity over the same
700ms so the label change tracks the morph.

**Reduced motion:** skip the tween, call the existing `drawScatter()` once at the target
coloring.

---

## 3. Verdict rubber-stamp slam + panel color flood (REFUSED vs PROMOTED)

Fires on `S.phase === "done"`, at `t0 + 1000ms` (after the score count-up, §1). Targets
the verdict panel and the promo chip (`#vPromo`, line 3877).

**Stamp (`#vPromo`) keyframe (~450ms):**
- `transform: scale(1.6) rotate(-5deg)` → `scale(1) rotate(0)`, `cubic-bezier(.2,.9,.2,1)`.
- `opacity 0 → 1`; one-shot `box-shadow` flash that fades to none.
- A subtle 1px letter-spacing settle so it reads as an ink stamp hitting paper.

**Stamp text + colors:**
| Verdict | promoted | Stamp text | Text color | Chip bg |
|---|---|---|---|---|
| partially robust (KILL, 40) | false | **✕ REFUSED** | `--red` | `--red-d` |
| robust enough for follow-up (SURVIVOR, 80) | true | **✓ PROMOTED** | `--green` | `--green-d` |
| capped promotion (illustrative — no live case in demo_data) | true | **▲ PROMOTED (capped)** | `--amber` | `--amber-d` |

(Keep the existing secondary line "PROMOTED TO BIOLOGY" / "BLOCKED — no biology" beneath.)

**Panel color flood (the refuse-vs-promote contrast — the thesis):**
- Flood the **verdict panel** border + a radial background wash:
  - REFUSED → border `--red`, wash `radial-gradient` from `--red-d` fading to panel base.
  - PROMOTED → border `--green`, wash from `--green-d`.
  - Capped → `--amber` / `--amber-d`.
- Flood animates in over ~350ms starting with the stamp, then **settles to a calmer
  resting tint** (border stays colored, wash drops to ~35% strength) so the panel isn't
  glaring for the rest of the take.
- One-shot **screen-edge pulse**: a full-viewport inset box-shadow in the verdict color,
  ~250ms, opacity `0→.5→0` — a single "gavel" beat. Extends the existing `.flash`
  pattern (line 402). Skipped under reduced motion.

---

## 4. CI-band visual (uncertainty as craft, not omission)

A horizontal confidence band that **snaps in beneath a fragile headline number**, reading
"the interval clears the line." Two placements:

**(a) Real leakage AUC (Beat 4, the climax).** Under the `AUC 0.96` headline in the real-
evidence panel (`realEvBody`, ~line 4040). Axis = the same **0.5 → 1.0** scale as the
pair-bar (`pairBarHTML`, line 3842).
- Draw a thin band (height ~6px) spanning `[ci_lo, ci_hi]` from `demo_data` `[backend]`,
  color `--accent` at `.5` alpha, with a **center tick** at the point estimate in solid
  `--accent`.
- A `0.5 chance` guide line at the left. If `ci_lo > 0.5`, render a small green
  `--green` chip: **"95% CI excludes chance"**. Show the permutation-null `p` chip next to
  it: **"p < …"** `[backend]`.

**(b) Leakage margin (Beats 2 & 5).** Under `#mv`. Axis is **0-centered** (margin can be
±). Band spans `[margin_ci_lo, margin_ci_hi]` `[backend]`.
- If the band **excludes 0** and is negative → color `--red`, chip **"margin CI excludes
  zero — leakage"** (KILL). If it straddles 0 → color `--amber`, chip **"CI includes zero —
  thin"** (SURVIVOR, margin +0.06). If excludes 0 positive → `--green`.

**Snap-in animation (~500ms):** the band **grows from its center tick outward** to
`[lo,hi]` (scaleX from 0 at the point estimate), easeOutCubic; the chip fades in at the
end. Reduced motion: render at full width instantly.

**Honesty note:** if the backend has not yet emitted CI fields, the frontend must **hide
the band** (not fake it) and the caption "CI excludes chance" must not appear — the band
is only shown when real `ci_lo`/`ci_hi`/`p` are present in `demo_data`.

---

## 5. Leakage pair-bar race (the money shot)

`pairBarHTML` (line 3842) currently writes all three bars at once. Stage them so the
**scanner bar visibly races the outcome bar.**

Timeline (from `t0` when the leakage row resolves):
1. `t0`: **outcome** bar (`.pb-fill.out`, `--accent`) fills `0 → outcome_auc%` over
   **500ms** easeOutCubic. Its value counts up in sync.
2. `t0 + 400ms`: **scanner** bar (`.pb-fill.scan`, `--red`) fills `0 → scanner_auc%` over
   **500ms**. On the KILL this bar **overtakes** the outcome bar (0.95 > 0.87) — let it
   visibly pass; on the SURVIVOR it stalls just under (0.68 < 0.74).
3. `t0 + 950ms`: the **margin gap strip** (`.pb-gap`) snaps in between the two bar ends,
   `pos` → `--green`, `neg` → `--red`.
4. `t0 + 1000ms`: the **margin value** (`#mv` / `.pb-val`) — on a **negative** margin,
   flash red (opacity `1→.4→1` + scale `1→1.15→1`, ~300ms, twice); on a **positive** margin,
   a single green pulse.

Widths use the existing `toW(a)=clamp((a-0.5)/0.5)` transform (line 3843), so the axis
stays `0.5 chance → 1.0 AUC`. Reduced motion: single-shot render (current behavior).

---

## 6. Seed-sweep needle-fan (verdict stability)

New panel/overlay on the verdict meter. Consumes `demo_data.seed_sweep` `[backend]`:
`{ scores:[…20], mean, std, flip_rate, promotion_line }`.

**Choreography (~2s total, Beat 6):**
1. **Fan out (0–700ms):** render **20 ghost needle lines** across the meter, one per seed
   score (`left: score%`), each `--accent` at `.35` alpha, ~1px. Stagger their appearance
   ~20ms apart so they "fan" open. This shows the spread honestly.
2. **Hold (700–1200ms):** all 20 visible; a faint bracket spans `[min, max]`.
3. **Collapse (1200–1900ms):** ease each ghost toward the **mean** score, converging into
   a **tight cluster**; the bracket shrinks to `[mean−std, mean+std]`. Cluster color goes
   `--green` if `flip_rate === 0` (all seeds same side of `promotion_line`), else `--amber`.
4. A caption/readout: **"stable across 20 seeds · σ=<std> · <flips> flips"**.

If any seed crosses `promotion_line`, the crossing needles stay `--amber` through the
collapse (honest: don't hide a flip). Reduced motion: render the final collapsed cluster +
bracket + readout with no fan/collapse motion.

---

## 7. Cross-references (so frontend can wire fast)

| Spec section | Existing code to extend | Autopilot fn (from DEMO_SCRIPT) |
|---|---|---|
| §1 count-ups | `renderVerdict` 3857, `setNeedle` 3840 | `countUpNaiveAUC`, `verdictSlam` |
| §2 scatter morph | `drawScatter` 3632, `toggleColor` 4311 | `toggleColor('scanner')` |
| §3 stamp + flood | `#vPromo` 3877, `.flash` 402, `.promo` 219 | `verdictSlam` |
| §4 CI band | `pairBarHTML` axis 3842, `realEvBody` 4040 | `snapCIBand` |
| §5 pair-bar race | `pairBarHTML` 3842, `.pb-fill` 413 | `pairBarRace` |
| §6 seed-fan | `#needle` / meter, new overlay | `seedSweepFan` |

**Do not** animate stale content: §4/§6 depend on WAVE-2 backend fields
(`ci_lo/ci_hi/p_perm`, `seed_sweep`). If absent from `demo_data`, hide those elements
rather than fabricate — the whole pitch is a referee that doesn't overclaim.
