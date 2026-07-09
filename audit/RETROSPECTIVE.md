# NeuroAD — Ultracode Pipeline Retrospective

A multi-agent audit → align → execute → review → fix → declutter run over the NeuroAD
Discovery Engine hackathon submission. This documents (1) what was successfully built and
why it worked, and (2) what turned out to be bugs / not-ideal and what had to be cleaned up.

Judging rubric it was optimized against: **Impact 25% · Claude Use 25% · Depth & Execution 20% · Demo 30%.**

---

## 1. What was successfully built — and what made it succeed

### Delivered
- **Engine science upgrade** (`src/neuroad/probe.py`, `gauntlet.py`, `leakage.py`)
  - Automatic PCA/whitening front-end to the probe, triggered only in the p≫n regime, fit
    *inside* each CV fold (no test-fold leakage). Collapsed the inflated raw-768-d
    Neuro-JEPA scanner AUC of **0.998 → a defensible 0.94 (PCA-10)**.
  - Bootstrap/DeLong 95% CIs + stratified (within-site) label-permutation nulls on the
    headline leakage AUC and the leakage margin. Near-threshold verdicts reframed as
    "margin CI excludes zero" instead of bare point-estimate cutoffs.
  - Two-way brain-age control (strict predicted-brain-age AND Franke/Gaser brain-age GAP),
    so an over-aggressive control can no longer silently kill a real signal.
- **Honesty reframe** — every synthetic number badged `SYNTHETIC HARNESS` at the source;
  `claim.yaml` turned from an unbadged "strong candidate / 88 / promoted" into a badged
  synthetic SURVIVOR with a margin-CI-includes-zero note; "Proteins confirm it" removed;
  all six prior-art citations verified live (all resolved).
- **Reproducibility** — `neuroad reproduce-finding` regenerates the leakage AUC
  (0.958, 95% CI [0.907, 0.997], perm p=0.001) from a checked-in PCA-10 fixture, offline,
  no gated weights. `neuroad demo` regenerates all reports + `demo_data.json` deterministically.
- **Demo UI** (`app/index.html`) — deterministic "Cinematic Demo" autopilot, count-ups,
  scatter color-morph, verdict rubber-stamp, courtroom auto-expand, seed-sweep fan,
  leakage pair-bar race, CI bands. Then a declutter pass: gauntlet collapsed to a
  hover-reveal progress strip, hero PCA+Verdict hierarchy, one primary CTA.
- **CI / packaging** — GitHub Actions (offline `demo` + `reproduce-finding` smokes),
  Makefile, `[claude]` optional extra. 133 tests pass.
- **Judge verdict** — final go/no-go: **GO**, all 9 adversarial checks pass.

### What made it work (the mechanics worth reusing)
1. **Read-only audit first, execute later.** Four domain auditors (science, data-science,
   engineering, design) ran with zero write access. This surfaced the load-bearing insight
   — *both "hero" results fail the tool's own gauntlet* — before a single line was changed,
   so the whole build was reframed around honesty instead of polishing a doomed narrative.
2. **A cross-discussion + alignment step.** Each domain reacted to the other three's
   findings; an aligner synthesized one rubric-weighted, owner-assigned plan and pushed
   low-ROI work to an explicit out-of-scope list. This is what turned four wish-lists into
   one sequenced plan.
3. **Disjoint file ownership per wave.** Parallel executors never shared a file within a
   wave (frontend owned `index.html` only; backend owned `src/**`+`reports/**`; design owned
   `design/**`; qa owned `.github/**`+`tests/**`). This made concurrent edits safe *without*
   git worktrees and without merge conflicts.
4. **Hard gates in the sequence.** No report regeneration until the engine science landed
   and changed the numbers; frontend animated only the final `demo_data.json`; citations
   verified before any doc was finalized. These gates prevented animating stale numbers.
5. **Agents verify with real commands.** Every executor ran `pytest` / the CLI / `curl` and
   reported the actual result, not a claim of success.
6. **Agents never touched git; the main loop committed to a branch.** The working tree
   stayed reviewable throughout, and the session's delta was snapshotted onto a branch to
   protect it from a second session editing the same tree.
7. **Adversarial "clone-the-repo / pause-the-frame" reviewer.** A dedicated skeptic caught
   the two fabrication-liabilities that a rubric-by-rubric review missed (see §2).
8. **Resume-on-failure.** When agents died (schema retry cap; 529 overload), the cached
   prefix replayed for free and only the failed + downstream agents re-ran.

---

## 2. What turned out to be bugs / not-ideal — and the cleanup

| Issue | Root cause | Cleanup |
|---|---|---|
| **Unbadged promoted survivor in `claim.yaml`** | The most-clickable export shipped `verdict: strong candidate / score 88 / promoted: true` with no synthetic badge — the exact confounded claim the referee is supposed to refuse. | Reframed to a badged synthetic SURVIVOR (`SYNTHETIC HARNESS`, `synthetic: true`), verdict "robust enough for follow-up", with an explicit note that its leakage-margin CI includes zero. |
| **Demo script told the presenter to claim an offline transcript was live** | `DEMO_SCRIPT.md` BEAT 3 said "● LIVE CLAUDE … a live, captured transcript, not a template" while `live_transcript.json` was `live:false, is_placeholder:true`. On-camera fabrication of the flagship Claude beat. | Flipped the badge/caption/VO to "● OFFLINE (template)"; badge only reads LIVE when a real call sets the flag. |
| **Number drift** | Engine science fixes changed outputs (synthetic KILL 15/fragile → 40/"partially robust"), but demo captions, the embedded `file://` fallback in `index.html`, and the docs hardcoded the old values. | Backend regenerated all artifacts; frontend re-synced `index.html` (incl. the embedded fallback, now byte-identical to `demo_data.json`); design re-synced all docs. Final verify diffed all consumer surfaces — they agree. |
| **Two heavy audit agents crashed** | The initial `FINDINGS_SCHEMA` was deeply nested with `additionalProperties:false` + many required fields; long outputs became invalid JSON and hit the StructuredOutput retry cap (5). | Lighter schema (fewer required nested fields, no additionalProperties lock) + explicit output-length discipline; resumed with the two good audits replaying from cache. |
| **529 Overloaded mid-run** | Transient server overload killed the two round-2 Wave-B agents and cascaded the verify into a retry-cap error. | Resumed the workflow; cached agents replayed, only the 3 failed re-ran successfully. |
| **Two sessions on one working tree** | A second Claude chat committed to `main` (UI cleanup + a `harness/`/`director-agent-pipeline` feature) while this session's workflow edited uncommitted files in the same tree. | Snapshotted this session's delta onto a branch (`ultracode-pipeline`); verified no overlap reverted the other session's code (only 2 regenerable report files touched). **Unresolved: the two branches still need a deliberate merge.** |
| **Autonomous agents can't screenshot** | `claude-in-chrome` requires interactive browser selection, unavailable to background subagents. | Agents verified via `curl` + DOM/`node --check`; the main loop did the visual verification and fine-tuning. |
| **Cached reviews read stale** | Stage-3 reviews were cached from before the Wave-A fixes, so a "still broken" finding was already fixed. | Treated the **live final verify** as authoritative, not the cached rubric reviews. |

### Residual / open items (not blocking the GO)
- Cosmetic "~0.93" phrasing in the `reproduce-finding` tail / Makefile comment vs the printed 0.958.
- The FM-embedding leakage finding is demonstrated on healthy brains only; it never meets a
  real disease cohort end-to-end (OASIS-3/ADNI is the documented roadmap, deliberately out of scope).
- `ultracode-pipeline` branch ↔ `main` merge is still pending.
