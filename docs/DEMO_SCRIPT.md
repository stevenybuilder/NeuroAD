# Demo script — 3 minutes, beat by beat (deployed ZUI, live Claude, honest spine)

> **This script narrates the DEPLOYED product** at
> `https://neuroad-demo-31043195041.us-central1.run.app/` — the Claude Science
> entry (`app/claude_science.html`) handing off into the ZUI (`app/neuroad.html`),
> running against **real cohorts** with **live Claude** (`/api/health` →
> `claude_live: true`, model `claude-opus-4-7`). It replaces the retired script that
> narrated the offline `app/index.html` autopilot with a `SYNTHETIC HARNESS`
> cold-open — a *different* surface. Nothing here is synthetic-as-real: the entry
> refuses to load any `synthetic:*` dataset (`claude_science.html:150`).

**The thesis (say it, don't bury it):** *NeuroAD is a scientific referee whose
credibility IS the product. Watch it kill a confounded hypothesis on real data,
promote only what survives, and — live on camera — watch Claude reason about the
result, grounded in this exact investigation.*

**Rubric map (why each beat exists):**
- **Claude-Use (25%)** — Beats 1, 4, 7 show *live* Claude: `/api/investigate?api=true`
  grows the tree, `/api/ask` answers grounded follow-ups (the money beat), and the
  `/api/orchestrate` tool-runner sequences the pipeline. Health badge proves it's live.
- **Impact (25%)** — real multi-site ADNI (2,951 subjects, real plasma p-tau217) and a
  real, cited leakage problem measured on the foundation model itself.
- **Depth (20%)** — the leakage-honest gauntlet + the reproducible OpenBHB finding
  (`neuroad reproduce-finding`), 472 passing tests.
- **Demo (30%)** — one continuous, calm ZUI take; trust chrome legible throughout.

---

## ⚠️ TWO DECISIONS TO LOCK BEFORE FILMING

1. **The promoted-survivor number.** The ADNI SURVIVOR card currently renders
   **score 100 / "strong candidate"** because its `brain_age` control is
   `not_available` and renormalizes out (75/75), and the honesty cap only covers a
   missing *molecular* anchor, not a missing brain-age test (`scoring.py:82-84`).
   **Do NOT narrate "100 / strong candidate" on camera.** Before recording, either:
   - **(a) Approve the one-line cap fix** so any renormalized-out star test caps the
     score to ≤84 → the card renders **"robust enough for follow-up"** (recommended;
     also fixes it everywhere, not just on camera), OR
   - **(b) Make `brain_age` run on ADNI** (real age exists for all 2,951 subjects) so
     a 5/5 card earns its score honestly, OR
   - **(c) Film the OASIS survivor** as the promoted example — it already renders a
     correct **84 / "robust enough for follow-up."**
   Beat 5 below is written to the honest framing and is safe under (a) or (c).

2. **Which result is the money shot** — the **OpenBHB scanner-leakage reproduction**
   (Beat 6: 100% real data + math, the product thesis, reproducible in one command) is
   the recommended climax. The clean **ADNI AD-vs-CN FreeSurfer AUC 0.935** is the
   honest disease result and rides as corroboration (Beat 5). Both are real; only one
   leads. This script leads with the reproduction.

All numbers below are what the engine emits / the live endpoints return. Do not
hand-type them — the ZUI reads them from the payload.

---

## Entry state (cold-open frame)

Open on **`/`** — the **Claude Science workspace** (`claude_science.html`). Real
reference cohorts are shown as "the researcher's workspace"; the prefilled hypothesis
is the real, cited default from `hypothesis_registry.json` (maps a real hypothesis →
a real loadable cohort → a cited verdict). The **Claude · live** badge is visible.

---

## Beat table (timestamp · on-screen action · lower-third caption · VO)

Lower-third = the fixed caption band at the bottom. Keep each caption ≤ ~12 words.
Captions are **exact text**; VO is the spoken track.

### BEAT 1 — 0:00–0:25 · The workspace: a real hypothesis, live
- **Action:** On the Claude Science entry, the dataset-connect shimmer resolves to
  **real cohorts** (ADNI · OASIS · OpenBHB). The prefilled real hypothesis sits in the
  field. Hit **Investigate →**; the view transitions into the ZUI canvas.
- **Lower-third:** `A researcher's workspace — real cohorts, a real hypothesis.`
- **Sub-caption (mono):** `Claude · live · claude-opus-4-7 · real ADNI/OASIS/OpenBHB`
- **VO:** "This isn't a slide deck — it's a workspace. Real Alzheimer's cohorts are
  connected, and we hand Claude a real hypothesis to investigate. Watch it build the
  argument."

### BEAT 2 — 0:25–0:50 · The tree grows from a live investigate call
- **Action:** `POST /api/investigate` (live) on **real ADNI — 2,951 subjects, 72
  sites, real plasma p-tau217**. The decision tree grows from the root hypothesis:
  **green** branches survive the confound gauntlet, **gray** ones are killed, **clay**
  nodes are candidate targets. Semantic zoom holds one focal branch at a time.
- **Lower-third:** `One hypothesis → a tree the engine grows and prunes.`
- **Sub-caption:** `Real ADNI · 2,951 subjects · 72 sites · anchor p-tau217 (measured)`
- **VO:** "From one hypothesis, the engine grows a tree on nearly three thousand real
  subjects. Green survived the gauntlet. Gray was killed. And it shows you *why* — not
  just a score."

### BEAT 3 — 0:50–1:20 · The referee kills a confound (the thesis)
- **Action:** Zoom into the gray **"Kill reason — age confound"** node. Its card:
  **Hypothesis AUC 0.92** vs **Age-only baseline 0.90**, the honest bar pair, and the
  verdict strip **"Confound ≥ signal → branch killed. Pruned from candidate
  generation."** Killed branches stay visible (gray, dashed) — the refusal is on the
  record, not hidden.
- **Lower-third:** `Chronological age explains the atrophy. No disease signal left.`
- **Sub-caption:** `Killed branches stay visible — the refusal is on the record.`
- **VO:** "Here's the referee at work. This branch looked good — until age alone
  explained the atrophy as well as the biomarker did. Confound beats signal, so the
  branch is killed and pruned. A team that shows you what it *threw out*."

### BEAT 4 — 1:20–1:50 · LIVE Claude, grounded — the money beat (Claude-Use)
- **Action:** In the **Claude · ask** rail, type **"which protein was killed and
  why?"**. This fires **`POST /api/ask` (live)**; Claude answers grounded in *this
  investigation's* case + registry — naming the ranked shortlist (**APP rank 18, ESR1
  61, MAPT 151, APOE 185, PSEN1 492**) and correcting the premise (no *protein* was
  killed; BACE1 is unranked, `candidate_only`, not failed). A new **gold "Claude ·
  follow-up"** branch spawns from the answer.
- **Lower-third:** `Ask a follow-up — Claude answers, grounded in this run.`
- **Sub-caption:** `● Claude live · /api/ask · answer grounded in the case, not canned`
- **VO:** "And this is a live model call. I ask which protein was killed — and Claude,
  grounded in *this* investigation, corrects me: no protein was killed, the gauntlet
  ran on the imaging signal, and it hands back the ranked shortlist with its own
  caveats. Every follow-up spawns a new branch. Claude isn't narration here — it's
  reasoning about the result in real time."

### BEAT 5 — 1:50–2:15 · A survivor is promoted (honest framing)
- **Action:** Move to a **green survivor** whose gauntlet passes on real data —
  age/sex ✓, site/scanner ✓, **plasma p-tau217 anchor ✓ (real, measured)**,
  replication ✓ — with the **brain-age control not evaluated on this cohort** (shown as
  `—`, not hidden). Verdict pill: **"robust enough for follow-up."** Clay candidate
  nodes unlock (ranked targets → AlphaFold pLDDT → repurposing).
  **[Per DECISION 1: film only after cap fix (a), or film the OASIS survivor (c).
  Do NOT show "100 / strong candidate."]**
- **Lower-third:** `Survives 4/5 checks — robust enough for follow-up.`
- **Sub-caption:** `Real p-tau217 anchor passed · brain-age control not evaluated`
- **VO:** "When a hypothesis *survives* — passing on age, on scanner, on a real plasma
  p-tau217 anchor, and replicating — it's promoted to *robust enough for follow-up*.
  Not 'proven.' The card even flags the one control it couldn't run. Then it translates
  the survivor into ranked, testable targets."

### BEAT 6 — 2:15–2:45 · The reproducible finding (Depth climax)
- **Action:** A terminal chip types **`$ neuroad reproduce-finding`** → recomputes,
  live from a checked-in PCA-10 fixture of **96 real healthy brains**, that frozen
  **Neuro-JEPA embeddings predict the scanner at AUC 0.958**, **95% CI [0.907, 0.997]**,
  **permutation-p 0.001** — the band visibly clears chance (0.5). Corroborating large-n
  figure fades in: **structural → scanner AUC on 3,984 healthy OpenBHB brains, 62
  sites.**
- **Lower-third:** `Real data: the frozen model predicts the SCANNER, not the disease.`
- **Sub-caption:** `neuroad reproduce-finding · AUC 0.958 [0.907,0.997] · p=0.001 · n=96`
- **VO:** "Here's the finding this whole referee exists to catch — and it's the one
  number we assert as a result. On ninety-six real healthy brains, no disease, the
  frozen foundation model predicts which *scanner* took the scan at point-nine-six.
  Bootstrap interval clears chance; permutation null confirms it. Reproducible from a
  clean clone in one command."

### BEAT 7 — 2:45–3:00 · Claude orchestrates + close on the honesty chrome
- **Action:** Brief cut to the **`/api/orchestrate`** trace — Claude sequences the real
  tool chain (`describe_cohort → referee_hypothesis → prioritize_targets →
  protein_structure → repurposing`), **tools decide, Claude never overrides a verdict.**
  Pull back to the ZUI: the **Claude · live** badge, the **Real ADNI/OASIS** provenance
  pills, and a bottom rail: **472 passing tests · open repo · reproducible.**
- **Lower-third:** `Claude sequences; the tools decide; the verdict is the engine's.`
- **Sub-caption:** `● Claude live · 472 tests green · open-source · one command`
- **VO:** "Claude also runs the pipeline as an orchestrator — it sequences the tools,
  but it never overrides a verdict. Everything you saw is live, open-source, and
  reproducible: a scientific referee that kills bad claims — including its own — and
  shows its work. That's NeuroAD."

---

## Filming checklist (do before the take)
- [ ] `curl .../api/health` shows `claude_live: true` (the live badge depends on it).
- [ ] DECISION 1 resolved — survivor card renders ≤84 / "robust enough for follow-up"
      (cap fix), OR the take uses the OASIS survivor. **No "100 / strong candidate" on camera.**
- [ ] DECISION 2 resolved — reproduction leads (this script) unless changed.
- [ ] Human in-browser dry run on `/`: type a *novel* hypothesis, confirm the numbers
      change (real `/api/investigate`, not a canned snap), confirm the `/api/ask`
      follow-up returns a live grounded answer, check the console for errors.
- [ ] Trust chrome legible in every frame judges see: Claude live badge, real/synthetic
      pill, verdict, caveat — visible at overview zoom, not only on node detail.
- [ ] Every on-screen number traces to a real payload field (no hand-typed figures).

## What changed from the previous script (for reviewers)
- **Retargeted from `index.html` (offline synthetic autopilot) to the DEPLOYED
  `claude_science.html → neuroad.html` ZUI** running on real cohorts with live Claude.
- **Removed the `SYNTHETIC HARNESS` cold-open** — the deployed demo shows only real
  data; no synthetic cohort appears on camera.
- **Elevated live Claude to a hero beat** (`/api/ask` grounded follow-up, Beat 4) plus
  the orchestrator trace (Beat 7) — directly serving the 25% Claude-Use axis the demo
  previously under-showed.
- **Replaced the "100 / strong candidate" survivor** with the honest "robust enough for
  follow-up" framing, gated on the CAP-1 fix decision (removes the flagship overclaim).
- **Kept** the reproducible OpenBHB leakage climax, the CI/permutation rigor, the
  honest live-vs-offline badge, and the "kills its own claims" thesis.
