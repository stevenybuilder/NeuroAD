---
name: harness-build-learnings
description: >-
  The durable method behind the NeuroAD Stage-2 build — how a multi-agent Claude
  Code pipeline shipped a demoable harness (74→120 green tests, offline-
  deterministic, byte-identical policy fallback, one honesty bug caught before
  the demo). Packages the director-agent pipeline shape, the shared VERIFIED-
  facts grounding technique, the policy-layer + constant-fallback pattern, file-
  disjoint parallel lanes, and where to place human-in-loop compute/commit gates.
  Use at the START of any hackathon/grant/product build that will fan out into
  parallel Claude agents against a known rubric. Full write-up:
  docs/HARNESS_BUILD_LEARNINGS.md.
when-to-use: >-
  Invoke before planning a new multi-agent build, before parallelizing agents
  over shared files, or when deciding how to externalize logic into config/policy
  while keeping an offline-deterministic guarantee. Also use as a pre-demo
  checklist: did we adversarially test the code that enforces our headline claim?
---

# Harness Build Learnings

The reusable method that made the NeuroAD Stage-2 build work. This is the parent
skill; two sub-skills automate its two highest-leverage pieces:

- `skills/director-agent-pipeline/` — ground a build in judging criteria + verified
  facts BEFORE implementing.
- `skills/policy-layer-fallback/` — externalize logic into policy docs read by BOTH
  code and Claude, behind constant fallbacks, with a byte-identical guarantee.

Read `docs/HARNESS_BUILD_LEARNINGS.md` for the full audit and evidence.

## The one thesis

**Audit adversarially first, ground every agent in re-verified reality,
externalize behind constant fallbacks, parallelize on disjoint files with
explicit cut-lines, gate humans only at irreversible boundaries — and
adversarially test the code that enforces your core claim, because grounding
won't catch a wrong mapping there.**

## The pipeline shape (run in this order)

1. **Adversarial judge-simulation FIRST.** A hostile judge agent scores the
   current state against the *real* rubric and names the single biggest gap. That
   gap becomes the whole build's objective function. (In our build the judge's "3
   highest-ROI moves" became the plan ~1:1.) → see `director-agent-pipeline`.
2. **Grounding + costing pass.** Build one **shared VERIFIED-facts block** and
   require at least one agent (a "cost-and-power" director) to *re-verify it
   against disk/reality* and turn "do more" into a hard, approvable GO/NO-GO gate
   with real numbers. → see `director-agent-pipeline`.
3. **Architecture pass emits a file→change→risk table** with a per-file fallback
   strategy — not prose. Lowest-ambiguity handoff to builders.
4. **Decisive north-star synthesis** collapses options to one choreography and
   issues explicit per-agent marching orders — only *after* steps 1–3 exist.
5. **Parallel build on file-disjoint lanes**, dependency-ordered, behind constant
   fallbacks. → see `policy-layer-fallback`.
6. **Adversarial verification pass** on the code that enforces your headline value.

## The five techniques

### 1. Shared VERIFIED-facts grounding block
One block of ground-truth numbers (rubric weights, budget, hardware rates, exact
data inventory) that every agent consumes. It eliminated cross-agent fact-drift
(our docs were byte-identical on every number). **Require one agent to re-verify
it against actual artifacts, not just quote it.** Limit: grounding kills
hallucination, NOT misimplementation — a wrong semantic mapping passes right
through. Budget a separate logic-verification pass.

### 2. Policy-layer + constant-fallback pattern
Externalize logic (constants → config, code → policy docs) into files read by
BOTH deterministic code AND Claude's prompt. Every externalization ships with a
hardcoded transcription of the live constant as fallback, degrading **key-by-key**,
never caching a parse failure, rejecting malformed values. Prove it **byte-
identical** with the policy dir present vs renamed away (`diff` the output), and
unit-assert the same. This is what let 10 agents touch shared high-blast-radius
files safely. → `policy-layer-fallback`.

### 3. File-disjoint lanes + dependency DAG + cut-lines
Assign ownership by file path so parallel commits never collide. Sequence phases
as a DAG so the integration point runs only after its inputs exist. Pre-declare
cut-lines with a stated cut order (demo-critical / depth-critical / nice-to-have)
so time pressure never triggers scope thrash. Every agent emits structured tables
so synthesis is a merge, not a rewrite.

### 4. Human-in-loop gates only at irreversible boundaries
Put `AskUserQuestion` gates ONLY where an action is irreversible or hard to
reverse: spending money/compute, branching the demo (hero-vs-fallback), writing
shared commit history, touching credentials. Make reversible steps autonomous.
Pre-commit branching decisions as *binding* (e.g. "the Day-2 readout decides;
nobody relitigates Day 4") to prevent later thrash.

### 5. Adversarially test the code that enforces your core claim
The one bug that shipped landed in the honesty guard itself — it overclaimed in
the tool built to prevent overclaiming. Test its **inputs**, not just its output
strings: "can any input reach a state/label it shouldn't?" Add this to the
standard pre-demo checklist.

## Pre-demo checklist (steal this)

- [ ] Did an adversarial judge score us against the *real* rubric first?
- [ ] Is there ONE shared facts block, and did someone re-verify it against disk?
- [ ] Does every externalization have a constant fallback + a proven byte-identical
      offline path?
- [ ] Are lanes file-disjoint, phases dependency-ordered, cut-lines pre-declared?
- [ ] Are human gates only at irreversible boundaries?
- [ ] Did we adversarially test the INPUTS of the code that enforces our headline
      value (not just its output strings)?
- [ ] Tests green, offline-deterministic, no coexisting duplicate vocabularies for
      the same concept?

## Anti-patterns we hit (don't repeat)

- **Two vocabularies for one concept coexisting** because one path shadows the
  other instead of a real merge — a latent footgun for any future caller.
- **"Make it symmetric with existing structure"** as a reason to build — produced
  low-value scaffolding that was correctly first-to-cut.
- **Architecture over-production** from reference-mining — mine references for the
  1–2 primitives that map to your gap; treat the rest as a menu.
- **Docstrings that oversell** a curated list (a 5-item denylist described as
  "refuses any forbidden overclaim").
