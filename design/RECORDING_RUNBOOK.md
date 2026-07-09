# Recording Runbook — the 3-minute NeuroAD take (one pass)

A human records this. The workbench drives itself: the **▶ Cinematic Demo** autopilot
plays a deterministic, byte-identical beat sequence. You supply the voiceover and hit
one button. Follow this top-to-bottom.

> **Honesty note baked into the take**
> - The cold open is a **KILL badged `SYNTHETIC HARNESS`** — we lead with a *refusal* of a
>   plausible-looking classifier, not a discovery.
> - The only asserted "result" is the **REAL OpenBHB / Neuro-JEPA scanner-leakage** finding.
> - The SURVIVOR is an explicitly-badged **gate MECHANIC**, not a discovery.
> - The Claude badge shows whatever is true — **`● LIVE CLAUDE`** only if you record with an
>   API key set (see Setup step 2); otherwise **`○ OFFLINE TEMPLATE`**, and the autopilot's
>   own sub-caption says so. **Do not claim "live" on camera unless the badge reads LIVE.**

---

## 0. What the on-screen autopilot actually does (read first)

The implemented `▶ Cinematic Demo` autopilot in `app/index.html` runs a **compressed ~92-second**
timeline (0:00 → 1:32), *not* the literal 3:00 wall-clock in `docs/DEMO_SCRIPT.md`. The **beat
order, captions, and numbers match the script**; the animation is just tighter. To fill a 3-minute
video you pace the (longer) DEMO_SCRIPT voiceover *over* the 92s of animation and add a short
title-card intro + a closing hold on the export tray. The beat table below lists the **real**
on-screen timestamps so your narration lands on the right visual.

---

## 1. Environment / window

- **Browser:** Chrome (clean profile — no extensions, no bookmarks bar). Zoom **100%**.
- **Resolution:** record at **1920×1080**. Full-screen the browser window; presentation mode
  (below) adds **+12%** scale so mono microtext survives 1080p/H.264 compression.
- **OS motion:** ensure **Reduce Motion is OFF** (macOS: System Settings → Accessibility → Display →
  Reduce motion **off**). With it on, every count-up / scatter-morph / stamp / seed-fan snaps to
  its final state and the take reads flat.
- **Audio:** record VO live or as a separate track (script in §5).

## 2. Serve over http (never `file://`) + optional LIVE Claude

`file://` blocks the `fetch("demo_data.json")` (CORS), so the workbench would fall back to stale
embedded JSON. Always serve over http.

```bash
cd /Users/stevenyang/Documents/claude-life-sciences-hack/neuroad-discovery-engine

# (OPTIONAL, for the ● LIVE CLAUDE badge — strongest Claude-Use beat)
#   Set a real key, then regenerate the payload so meta.claude.live = true.
#   Skip this block to record the honest OFFLINE path (badge reads ○ OFFLINE TEMPLATE).
export ANTHROPIC_API_KEY=sk-ant-...          # your key
python -m neuroad.cli demo                   # rebuilds app/demo_data.json + reports/ live

# Serve the app directory on a fixed port
python3 -m http.server 8000 --directory app
```

Leave that server running. Verified: `GET /index.html`, `GET /`, and `GET /demo_data.json` all
return **200** on port 8000.

## 3. Open the exact cold-open URL

```
http://localhost:8000/?sub=synthetic&case=KILL&color=outcome&present=1&autoplay=1
```

Params (all read at boot):
- `sub=synthetic` → **SYNTHETIC HARNESS** substrate.
- `case=KILL` → Case B · KILL loaded.
- `color=outcome` → scatter colored by outcome (so the morph-to-scanner reveal has somewhere to go).
- `present=1` → **Presentation Mode** on (+12% scale). Equivalent to pressing **`P`**.
- `autoplay=1` → autopilot auto-starts **600 ms** after load.

**If you want to control the start yourself:** drop `autoplay=1`, and instead **click the
`▶ Cinematic Demo` button** (top control bar) on your cue. Same result, on your beat.

## 4. Start / stop

- **Start:** the single action is **click `▶ Cinematic Demo`** (or `autoplay=1` starts it for you).
- **Cancel back to manual:** press **`Esc`** or **`Space`** — clears all timers at the current frame.
- **Presentation mode toggle:** **`P`** (or the `⛶ Present` button).

---

## 5. Beat table — real on-screen timestamps · caption · voiceover

`t` = seconds from the autopilot roll (the moment the caption band lights up). Captions are the
**exact on-screen lower-third**; VO is what you say. Numbers are what the engine emits into
`app/demo_data.json` (verified this pass).

### 0:00 — Cold open, badged SYNTHETIC KILL
- **Screen:** KILL claim card. Naive probe **AUC counts up 0.50 → 0.87** (`t≈1.2s`, 900 ms ease).
  `SYNTHETIC HARNESS` badge top-left the whole beat.
- **Caption:** `A finding that looks publishable. AUC 0.87. Watch the referee.`
  · sub: `SYNTHETIC HARNESS · a planted classifier, by design`
- **VO:** "This looks like a real finding — conversion decoded from frozen brain-MRI embeddings at
  AUC 0.87. A lab could spend a quarter chasing it. Watch our referee take ten seconds — and notice
  it's *our own* synthetic test case."

### 0:06 — Run the gauntlet
- **Screen:** `runGauntlet()`. Rows tick; age/sex fails, the ⭐ Site/scanner leakage row runs.
- **Caption:** `Same probe, different label — it reads the SCANNER better than the disease.`
- **VO:** "Same frozen embeddings, now pointed at the scanner label instead of the disease."

### 0:11 — The money shot: pair-bar race
- **Screen:** outcome bar fills to **0.87**, then ~400 ms later the **scanner** bar races *past* it
  to **0.95**; margin strip snaps in and **flashes red at −0.08** (margin CI excludes zero).
- **Caption:** `Same probe, different label — it reads the SCANNER better than the disease.`
- **VO:** "Point it at the scanner and it scores 0.95 — better than it predicts conversion. The
  leakage margin is negative."

### 0:15 — Scatter morph (color by scanner)
- **Screen:** scatter **morphs** outcome→scanner: the two scanner centroids slide apart, the
  dominant 1σ ellipse rotates to lie along the scanner axis.
- **Caption:** `The dominant axis of variance is the machine, not the biology.`
- **VO:** "Recolor the same points by scanner and the separation *is* the machine. The disease
  coloring was noise."

### 0:20 — Verdict slam: REFUSED
- **Screen:** verdict **rubber-stamp slam** — score counts up to **40/100**, needle glides, panel
  floods **red**, stamp reads **REFUSED**; biology zone stays 🔒 locked.
- **Caption:** `REFUSED · 40/100 · leakage margin −0.08. The gate stays shut.`
- **VO:** "Age fails, the molecular anchor fails, the scanner test fails. Even with a middling
  score, the corroboration gate stays shut. Refused — a quarter saved in ten seconds."

### 0:25 — Claude adjudicates, then critiques itself
- **Screen:** Courtroom + Reviewer **auto-expand** (no chevron click). Prosecution/defense/judge,
  then the Reviewer argues *against* the verdict. **Claude badge** visible.
- **Caption:** `Claude argues prosecution, defense, and judge — then argues against itself.`
  · sub (offline path): `● transcript · offline template (no API key) · honestly badged`
- **VO:** "Claude runs the courtroom — prosecution, defense, judge — then does the thing a real
  referee does: it argues against its *own* verdict and lists why you still shouldn't trust it."
  *(If you recorded with a key and the badge reads `● LIVE CLAUDE`, say "and this is a live,
  captured transcript, not a template.")*

### 0:34 — The REAL finding (climax): the model leaks the scanner
- **Screen:** real-evidence panel spotlighted. **Structural features → scanner AUC 0.89 on 3,984
  healthy OpenBHB brains, 62 sites, no disease.**
- **Caption:** `REAL DATA · 3,984 healthy brains, no disease. The encoder leaks the scanner.`
- **VO:** "Here's the real one. On thirty-nine hundred healthy brains with *no* disease, structural
  features predict which scanner took the picture at AUC point eight-nine."

### 0:37 — CI band snaps in on the Neuro-JEPA number
- **Screen:** **frozen Neuro-JEPA embeddings → scanner AUC 0.96 (PCA-10)**; a bootstrap **95% CI
  band snaps in** and visibly excludes chance (0.5). (Raw 768-d 0.998 shown as p≫n, flagged.)
- **Caption:** `frozen Neuro-JEPA · AUC 0.96 (PCA-10) · 95% CI excludes chance`
- **VO:** "The foundation model's *own* embedding leaks scanner field strength at point nine-six.
  Bootstrap interval clears chance; the permutation null confirms it."

### 0:44 — Reproduce from a clean clone
- **Screen:** terminal chip types **`$ neuroad reproduce-finding`** → prints **AUC 0.958,
  95% CI [0.91, 1.00], permutation p = 0.001** from the checked-in PCA-10 fixture.
- **Caption:** `Reproducible from a clean clone · $ neuroad reproduce-finding`
- **VO:** "And you can reproduce it from a clean clone in one command — no gated weights."

### 0:52 — The promotion-gate MECHANIC (badged synthetic)
- **Screen:** back to SYNTHETIC HARNESS · SURVIVOR; `runGauntlet()` at `t≈0:54`. ⭐ leakage row
  amber — margin only **+0.06** (outcome 0.74 vs scanner 0.68). Biomarker anchor **p-tau217
  r = +0.40 (95% CI lower +0.20)** opens the gate; score climbs to **80/100** (`robust enough
  for follow-up`), panel floods **green**, stamp **PROMOTED**, biology zone 🔓 unlocks. Every
  number badged `SYNTHETIC HARNESS`.
- **Caption:** `MECHANIC DEMO (synthetic): a thin imaging margin, rescued by a molecular anchor.`
  · sub: `SYNTHETIC HARNESS · illustrating the gate — not a claimed result`
- **VO:** "This is how the gate is *supposed* to work — and to be clear, this cohort is synthetic,
  built to demonstrate the mechanic. A thin imaging margin alone stays blocked; only a molecular
  anchor whose confidence interval clears zero opens the biology gate. We show the machine; we
  don't pretend it's a discovery."

### 1:08 — Seed-sweep stability
- **Screen:** **20 ghost needles fan out** across the meter, hold, then collapse into a tight
  cluster around the mean.
- **Caption:** `Verdict stable across 20 seeds — not a single-seed accident.`
- **VO:** "And it's not one lucky seed. Twenty reruns — the refusal never flips, and the survivor
  sits in a tight band." *(Honest: the KILL verdict is 0/20 flips; the SURVIVOR promotes on 19/20.)*

### 1:18 — KILL-vs-SURVIVOR split
- **Screen:** `openSplit()` shows the two verdicts side by side (REFUSED 40/100 vs PROMOTED 80/100).
- **Caption:** `Open-source. Reproducible. A referee that kills bad claims — including its own.`
- **VO:** "One referee, two verdicts — it refuses the plausible-looking one, including our own."

### 1:25 — Export tray + close
- **Screen:** export tray slides up (claim card, evidence ledger, reviewer report — all badged);
  `$ neuroad demo`.
- **Caption:** `Open-source. Reproducible. A referee that kills bad claims — including its own.`
- **VO:** "Everything you saw is open-source, offline, and one command. A scientific referee that
  refuses plausible-looking claims — including ours — and shows its work. That's NeuroAD."

### 1:32 — Autopilot ends, controls restored.
*(To hit a full 3:00: open on a ~10 s title card, keep VO deliberate through each beat, and hold on
the export tray while you finish the closing line.)*

---

## 6. Pre-flight checklist (run every one before you hit record)

- [ ] **URL bar shows `http://localhost:8000/...`** — NOT `file://`.
- [ ] **No console errors.** Open DevTools → Console: it should be clean (no red). Confirm
      `demo_data.json` returned **200** in the Network tab (no CORS fallback to embedded JSON).
- [ ] **Claude badge decision made & honest.** Badge reads either `● LIVE CLAUDE · <model>` (only if
      you set `ANTHROPIC_API_KEY` and rebuilt in step 2) or `○ OFFLINE TEMPLATE · offline-template`.
      Your VO must match what the badge says.
- [ ] **Presentation Mode active** — masthead visibly larger (+12%). `present=1` in URL or press `P`.
- [ ] **Cold-open frame correct:** `SYNTHETIC HARNESS` substrate, **Case B · KILL**, scatter
      **color-by-outcome**, gauntlet **un-run**, naive-AUC card showing the pre-count-up value.
- [ ] **Reduce Motion OFF** at the OS level (else all animations snap).
- [ ] **Window 1920×1080**, browser zoom 100%, clean profile (no extension chrome).
- [ ] **Dry run once:** click `▶ Cinematic Demo`, watch all ~92 s end-to-end, confirm every panel
      renders (pair-bar race, scatter morph, REFUSED stamp, courtroom+reviewer expand, real-evidence
      + CI band, reproduce chip, SURVIVOR PROMOTED, seed-fan, split, export tray). Then **reload the
      URL** for a clean take.
- [ ] **Recorder ready:** 1080p, 30/60 fps, cursor hidden if possible, audio armed.

## 7. If something looks wrong

- **Stale numbers / embedded fallback:** you're on `file://` or the server isn't running from `app/`.
  Re-serve with `--directory app` and reload the http URL.
- **Everything snaps, no animation:** OS Reduce Motion is on. Turn it off and reload.
- **Cold open shows OASIS/SURVIVOR:** the URL params didn't take — re-open the full URL in §3.
- **Autopilot double-fires / desync:** press `Esc`, reload the URL, start fresh (don't click the
  button twice).
- **Badge says OFFLINE but you wanted LIVE:** you didn't rebuild with the key set — redo step 2's
  optional block, restart the server, reload.
