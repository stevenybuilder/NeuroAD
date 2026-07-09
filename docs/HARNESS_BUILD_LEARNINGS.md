# Harness Build Learnings — NeuroAD Stage-2

Canonical record of what we built, what worked, what broke, and the reusable
method behind it. Distilled from two Director-of-Product audits: a **build
audit** (the code — what shipped, what's ambiguous, what's buggy) and a
**process audit** (the method — the multi-agent pipeline that produced it).
Written blunt on purpose. If you are starting the next build, read this first,
then invoke `skills/harness-build-learnings/`.

---

## Executive summary

The lane is solid and honestly demoable: **120 tests green** (up from 74),
`investigate` runs **offline-deterministic**, the policy fallback is
**byte-identical** with policy present or absent (proven by diff, not just
asserted), and the one real overclaim bug is **fixed and verified**.

It worked for one structural reason: **an adversarial judge-simulation ran
FIRST, and every downstream agent was pointed at closing the single gap it
named.** That turned "make it better" into one falsifiable thesis — *convert the
weakest axis (real discovery / institutional fit) into the demo's hero* — that
appears verbatim across the science, biostatistics, and blueprint docs. Two
mechanisms let 10 parallel agents touch shared, high-blast-radius files without
drift or breakage: a **shared VERIFIED-facts grounding block** and a
**constant-fallback invariant** on every externalization.

The sharpest lesson: the one defect that shipped landed in the code that
enforces the project's own headline value (honesty) — the honesty ladder
overclaimed in its own guard. **Grounding kills fact-drift; it does not kill a
wrong semantic mapping.** Budget a separate adversarial verification pass for the
logic that operationalizes your core claim.

---

## Part 1 — Build audit (the code)

### 1a. SUCCESSFUL — built well, works

- **Policy-loader constant-fallback pattern** (`src/neuroad/harness/policy.py`).
  Three accessors (`table` / `thresholds` / `brief`), each a hardcoded
  transcription of a live `src/neuroad` constant. `thresholds()` degrades
  **key-by-key** (overlays only well-formed numeric keys it already knows); `_num`
  rejects bool/None so a poisoned doc value falls through; parse failures are
  never cached. **Empirically proven byte-identical** in both directions (rename
  `policy/` away, re-run, `diff` the emitted report — clean) and unit-asserted
  (`test_policy.py`). Strongest piece in the lane.
- **`investigate()` flow** (`orchestrator.py:488`). parse→route→referee→gate→
  card→guard, each step serializable and offline by default. Provenance records
  mode, dataset, pre-registered `expected_direction` + `kill_criterion`, and the
  anchor-gate decision — auditable, defensible output.
- **Honesty ladder correctly *caps*** (`compute_honesty_rung`, `orchestrator.py:399`).
  Cumulative: KILL fails the site/scanner STAR test and stops at rung 2
  (`stable_cluster`); SURVIVOR clears anchor + promotion and reaches rung 4
  (`severity_anchored`). Verified live on both cases.
- **Biomarker-anchor HARD GATE** (`apply_biomarker_anchor_gate`, `orchestrator.py:320`).
  Blocks promotion only when `promoted and status==FAILED`; NA is neither
  credited nor condemned. Unit-covered, correct on the KILL run.
- **Honesty guard word-boundary handling** (`honesty_guard`, `orchestrator.py:95`).
  Walks the full rendered card including dynamic Claude side-artifacts, uses
  `\bcure\b` so "secure/accurate" don't false-trip.
- **discovery_router** — deterministic, serializable `RouteDecision`;
  "novel-language-wins" precedence is intentional and tested; unknown target
  falls back to a valid `LABEL_TARGET`; no crash on empty string.
- **Offline determinism** — identical score/rung/novelty across repeated runs
  (`test_investigate_is_deterministic_offline` + repeated manual runs).

### 1b. NEEDED MORE CLARITY — built but ambiguous/underspecified

- **Two honesty ladders coexist; they were never actually reconciled.**
  `experiment_card.py:42` defines `HONESTY_LADDER` with one vocabulary
  (`artifact-suspected … replication-ready`) and `default_honesty_rung` emits it;
  `orchestrator`/`policy` use a **different** 5-word vocabulary
  (`raw_pattern … externally_replicated`). In the shipping path the orchestrator
  always passes an explicit rung, so `default_honesty_rung` is bypassed — but it
  is still live and still tested, so any future caller of `build_experiment_card()`
  that omits `honesty_rung` silently gets the *other* vocabulary. The
  "reconciliation" is really "one path shadows the other." **Fix:** delete/redirect
  `default_honesty_rung` to the policy ladder, or document it as a separate contract.
- **Rung vocabulary is clustering-flavored but applied to supervised runs.**
  `stable_cluster` (rung 2) is stamped on a *named-contrast* SURVIVOR/KILL run,
  where the code actually means "materially above-chance effect." "cluster" reads
  wrong for a supervised probe. Cosmetic but user-facing.
- **Honesty-guard denylist is narrower than its docstring implies.**
  `FORBIDDEN_OVERCLAIMS` is 5 specific bigrams; the brief says never say
  "discovery"/"proof"/upgrade the noun, yet standalone `proven`, `proof`,
  `discovery`, `definitive`, `breakthrough`, `confirmed/established biomarker` are
  **not** caught. No current template emits those, so it's thinner
  defense-in-depth than advertised. Clarify intent or broaden the list.
- **Rung 5 (`externally_replicated`) is unreachable — wired for a runner that
  doesn't exist.** `external_replication`/`external_cohort` are only *read*, never
  written anywhere in `src/` or `tests/`. Safe direction (can't overclaim) and the
  intended design, but state it: no cross-cohort path exists yet.
- **Harness annotations are not surfaced in the UI.** `app/index.html` has zero
  references to `honesty_rung`/`novelty_class`. Expected (separate lane), but the
  L5 investigate output currently reaches only CLI/reports. Conscious integration
  TODO, not an oversight.

### 1c. STRAIGHT-UP BUG — real defects

- **`externally_replicated` overclaim — FIXED, verified.** Rung 5 now requires an
  explicit `provenance['external_replication'|'external_cohort']` flag that a
  single-dataset `investigate()` never sets. SURVIVOR (anchor+replication passed,
  promoted STRONG) stamps `severity_anchored`, not `externally_replicated`. The
  old bug — treating an internal held-out split as external replication — cannot
  recur. **Status: fixed.** This is the single most important defect in the build:
  it landed in the code enforcing the project's core anti-overclaim value.
- **Misleading threshold in the HARD-GATE caveat — OPEN, minor.**
  `apply_biomarker_anchor_gate:345-347` says *"its 95% CI lower bound is at/below
  {ci_pass:.2f}"* = 0.12, but the block only fires on `status==FAILED`, and FAILED
  means CI lower ≤ `ci_weak` = 0.0 (0.0–0.12 is WEAKENED and is *not* blocked). The
  number implies the wrong decision line on a promotion-blocking event. **Open,
  low severity.**
- **Router substring false-positive `"site"` — OPEN, cosmetic.**
  `NAMED_CONTRAST_KEYWORDS` contains bare `"site"`, which matches inside
  "oppo**site**"/"web**site**". Cannot mis-route (novel-pattern always wins,
  named-contrast is default anyway) — only pollutes `rationale`/`signals` strings.
  **Open, cosmetic.**

**No bug found in:** the byte-identical offline path, the anchor-gate promotion
flip, the guard's coverage of its 5 listed phrases, determinism, or the
KILL-caps-below-SURVIVOR invariant.

---

## Part 2 — Process audit (the method)

### 2a. Director roles ranked by signal

| Rank | Role | Why high-signal | Reuse verdict |
|---|---|---|---|
| 1 | **Judge-simulation (adversarial pre-mortem)** | Scored 74/100 with per-axis gaps and named the exact credibility cliff; its 3 highest-ROI moves became the whole plan ~1:1. | **Always run first.** Highest-leverage single agent. |
| 2 | **Biostatistics compute-costing** | Only director that verified against disk state; turned "get more data" into a hard gate via power math + a CU budget → made GO/NO-GO decidable and the spend approvable. | **Reuse.** Grounds the plan and sharpens the human spend checkpoint. |
| 3 | **Engineering policy-layer design** | Delivered the literal build spec: layer table + concrete config samples + a **file→change→risk table with per-file fallback**. Lowest-ambiguity handoff. | **Reuse.** Force a risk-rated file-change table, not prose. |
| 4 | **Director of Science (north star)** | Decisive, not a survey — collapsed option space to one choreography, issued explicit marching orders. | **Reuse**, but only *after* audit/science/competitive inputs exist. |
| 5 | **Reference research (Anthropic + Databricks)** | Gave the reusable architectural *shape* (governed semantic layer, dual-consumption, FM-as-substrate). | **Reuse for vocabulary + template; time-box it — it over-produces.** |
| 6 | **Frontier-science red-lines** | Citations that became the honesty rubric and anti-overclaim boundaries. | **Reuse** as the "what-not-to-claim" source. |
| 7 | **Competitive/white-space scan** | Sharp positioning; least load-bearing on the build (changed no architecture). | **Reuse for the pitch, not the build.** |

**Pattern:** the two most load-bearing roles were the two that were *adversarial
and empirically grounded*. Survey/positioning roles were least load-bearing.
Weight the pipeline toward "audit + cost against reality," not "survey the
landscape."

### 2b. Did the shared grounding block work? — Yes, measurably

- **Zero fact-drift across 7 parallel agents.** Rubric weights, the compute pool,
  the hardware rate, the exact data inventory, and the CU budget appear
  *identically* across every doc; the blueprint's compute table is a byte-match to
  the biostatistics doc. That consistency is the grounding block's fingerprint.
- **It forced verification over assumption.** The biostatistics director
  re-checked disk state and *corrected a fear in the grounding note itself*.
  Grounding gave a substrate concrete enough to be checked and refined, not just
  parroted.
- **It made human checkpoints crisp** — a clean "approve ~4 CU (0.2%)" instead of
  a hand-wave.
- **The limit:** grounding eliminated *factual* drift but did **not** prevent the
  *semantic/logic* bug. **Grounding kills hallucination; it does not kill
  misimplementation.** Budget a separate verification pass for logic.

### 2c. Reference architectures — load-bearing, not decorative

- The **"governed semantic layer / dual-consumption from one file"** idea became
  the literal design of `policy/*.yaml`, read by *both* deterministic code *and*
  Claude's system prompt — the single biggest architectural decision, straight
  from the reference.
- The **Skills `SKILL.md` Markdown-plus-YAML-frontmatter** pattern was adopted
  verbatim as the policy-doc format.
- **"Trusted badge + shown reasoning + provenance"** → the UI trust primitives.
- **"FM is a substrate, not the brain"** → the L0 invariant (model frozen behind
  the harness).
- **Where it was decorative:** the reference generated more architecture than
  shipped (extra skills, MCP connectors, an RL next-experiment selector — all
  correctly cut/never built). **Rule:** mine references for the 1–2 primitives
  that map to your actual gap; treat the rest as a menu, not a spec.

### 2d. Build techniques ranked by leverage

1. **"Keep tests green + preserve the offline-deterministic fallback" as a
   universal invariant.** Highest-leverage technique in the build. Every
   externalization shipped with a hardcoded constant fallback to prior behavior,
   so 10 agents could modify shared high-blast-radius files safely. 74→120 tests;
   `investigate` byte-identical offline. **Reuse verbatim.**
2. **File-disjoint lane assignment.** Ownership by file path made parallel commits
   collision-free.
3. **Dependency-ordered phases as an explicit DAG.** The integration point ran
   only once its inputs existed; no agent blocked on a missing dependency.
4. **Structured-output schemas everywhere** (capability→tool, file→change→risk,
   layer→status→home tables) → synthesis is a merge, not a rewrite.
5. **Explicit cut-lines with a stated cut order** (demo-critical / depth-critical
   / nice-to-have; "first to cut: X → Y → Z") → no mid-build scope thrash.
6. **Pre-registered kill criteria** — commit the falsification condition before
   the run.

### 2e. Human-in-loop checkpoints ranked by load-bearingness

1. **AskUserQuestion before irreversible compute spend** — the canonical "stop
   before an irreversible external action" gate; the hero-vs-fallback pivot hinges
   on it.
2. **Pre-commit to the hero-vs-fallback trigger** — a decision-hygiene gate that
   makes a Day-2 empirical readout binding in advance so nobody relitigates it Day 4.
3. **Parallel-commit lane confirmation** — collision avoidance that also routes
   credential-bearing work to the human.
4. **Held commits + token-from-env-only confirmation.**

**Rule:** put human gates only at irreversible boundaries (spend money, branch the
demo, write shared history, touch credentials). Make reversible steps autonomous.

### 2f. What wasted effort (honest)

- **New "symmetric" skills** that duplicated what the policy layer already did —
  correctly first-to-cut. "Make it symmetric with existing structure" is a weak
  reason to build.
- **Architecture over-production** — ~10–20% of director output never on the
  critical path. Not harmful *because cut-lines were explicit*, but a tighter brief
  ("spec only what's demo- or depth-critical") would trim it.
- **Positioning depth** beyond what the build needed — ROI is in the pitch.

### 2g. The caught bug — the sharpest process lesson

The honesty ladder stamped the top rung on single-cohort runs by treating an
internal held-out split as external replication; fixed to cap at
`severity_anchored`. What it teaches about the *process*:

- **The defect landed in the code that enforces the project's core value.** The
  highest-risk code is always the code that operationalizes your headline promise
  — that is where a subtle mapping error does the most reputational damage.
- **Grounding facts don't catch it** — every number was verified; the bug was a
  semantic mapping (held-out split ≠ external cohort).
- **A post-build verification pass caught it — keep that pass.** The guard was
  tested for its output strings ("never says 'validated biomarker'") but not for
  its *inputs* ("can any input reach a rung it shouldn't?"). **Add an explicit
  adversarial test of the honesty/guard layer itself to the standard checklist.**

---

## Part 3 — Reusable patterns, ranked by leverage

1. **Adversarial judge-simulation FIRST.** Score the current state against the real
   rubric with a hostile judge agent before any planning. Its named single-biggest
   gap becomes the whole build's objective function.
2. **Constant-fallback invariant on every externalization.** Any refactor that
   moves logic (constants → config, code → policy docs) ships with a hardcoded
   fallback to prior behavior + green tests. Lets many agents touch shared files
   safely and preserves an offline/deterministic guarantee.
3. **One shared VERIFIED-facts block, grounded against actual artifacts** (disk,
   rubric, real cost numbers). Require at least one director to *re-verify* it
   against reality, not just consume it.
4. **A cost-and-power director** that turns "do more" into a hard, approvable gate
   (power math + compute budget + GO/NO-GO) so the human spend-checkpoint is a
   clean yes/no.
5. **File-disjoint lanes + dependency-ordered DAG + explicit cut-lines with a
   stated cut order.** The parallelism-safety triad.
6. **Structured-table outputs from every agent** so synthesis is a merge, not a
   rewrite.
7. **Human gates only at irreversible boundaries** (spend, commit history,
   credentials, demo-branch decisions); pre-commit binding decisions to prevent
   later thrash.
8. **Mine reference architectures for 1–2 primitives that map to your gap;** treat
   the rest as a menu.
9. **Adversarially test the code that enforces your headline value — its inputs,
   not just its output strings.**
10. **Decisive north-star synthesis** that collapses option space to one
    choreography — but only *after* the audit/science/competitive inputs exist.

**One-line distillation:** *Audit adversarially first, ground every agent in
re-verified reality, externalize behind constant fallbacks, parallelize on
disjoint files with explicit cut-lines, gate humans only at irreversible
boundaries — and adversarially test the code that enforces your core claim,
because grounding won't catch a wrong mapping there.*
