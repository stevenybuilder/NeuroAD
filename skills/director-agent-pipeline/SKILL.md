---
name: director-agent-pipeline
description: >-
  Ground a hackathon / grant / product build in its real judging criteria and in
  re-verified facts BEFORE writing implementation code, using a fan-out of
  specialized "director" research agents synthesized into one falsifiable build
  thesis. The two most load-bearing directors are adversarial and empirically
  grounded — a hostile judge-simulation that scores against the real rubric, and
  a cost-and-power director that verifies against disk and turns "do more" into
  an approvable GO/NO-GO gate. Produces a shared VERIFIED-facts block, a scored
  gap analysis, and a file→change→risk build spec. Use at the very start of a
  build, before parallelizing implementers.
when-to-use: >-
  Invoke when you have a known rubric/spec and need "what's the state, what are
  the gaps, what's the plan" before building. Especially for competition or
  review deadlines where the objective function is external and fixed. Do NOT use
  once implementation is underway — this is the planning front-end. Pairs with
  skills/policy-layer-fallback (the safe way to then execute in parallel).
---

# Director-Agent Research → Plan Pipeline

Turn a diffuse "make it better" into ONE falsifiable thesis, grounded in verified
reality, before any implementer touches code. The pattern that produced the
NeuroAD Stage-2 plan: the judge-simulation's "3 highest-ROI moves" became the
entire build ~1:1.

**Core rule:** the two most load-bearing agents are the two that are *adversarial
and empirically grounded*. Survey/positioning agents are least load-bearing —
fund them for the pitch, not the build.

## Procedure

### Step 1 — Judge-simulation FIRST (highest leverage)
Spawn a hostile judge agent. Give it the **real rubric verbatim** (exact axes +
weights) and the current artifact. Require it to:
1. Produce a **scored** result per axis (e.g. `Impact 25 / ClaudeUse 25 / Depth 20 /
   Demo 30`), not prose.
2. Name the **single biggest credibility gap** (the "if a judge pokes here, we
   die" cliff).
3. Emit **3 highest-ROI moves**, ranked, each tied to the axis it lifts.

The named gap becomes the build's objective function. Every later agent is
pointed at closing it. If this step produces only vibes, re-run it with a harsher
persona and the literal scoring sheet.

### Step 2 — Build the shared VERIFIED-facts block
Collect the ground-truth numbers every downstream agent must agree on: rubric
weights, compute/$ budget, hardware rates, exact data inventory, deadlines. Put
them in ONE block that every agent's prompt includes verbatim. This is what
prevents cross-agent fact-drift (in our build, every doc was byte-identical on
these numbers).

### Step 3 — Cost-and-power director (second highest leverage)
Spawn one agent whose only job is to **verify the facts block against reality**
(read the actual disk state / data / configs — do not assume) and turn "do more"
into a hard gate:
- **Power/feasibility math** → is the ask even achievable? (e.g. n=61 → CI
  half-width 0.257 → forced negative; n≈235 → 0.129 → success possible.)
- **A real budget** in the real unit (compute-units, dollars, hours) → an
  approvable number, not a hand-wave.
- **An explicit GO/NO-GO** the human can approve with one yes/no.
This agent is allowed — expected — to *correct the facts block* from what it finds
on disk. That correction is the signal the grounding is real, not parroted.

### Step 4 — Domain red-lines with citations
Spawn a science/domain agent that produces the **"what we must NOT claim"** list
with citations. This becomes the honesty rubric and the anti-overclaim
boundaries — what keeps the pitch credible instead of performative.

### Step 5 — Architecture director emits a file→change→risk table
Not prose. A table: `file | change | risk | fallback strategy`, plus concrete
config samples. Lowest-ambiguity handoff to implementers. Reuse the same
structured-table discipline for every agent so synthesis is a merge, not a
rewrite.

### Step 6 — (Optional) reference-mining, time-boxed
If mining reference architectures, extract only the **1–2 primitives that map to
your named gap**. Treat everything else as a menu, not a spec — reference-mining
reliably over-produces (~10–20% of output never ships).

### Step 7 — Decisive north-star synthesis (LAST)
Only now spawn the synthesis agent. It collapses the option space to **one
choreography** ("ship A; C is its Act I; B is a supporting panel") and issues
explicit per-agent marching orders. A north-star agent with nothing yet to
converge is low-value — it must run after steps 1–6 exist.

## Human gate (mandatory, placed here)
After Step 3, before any irreversible spend, run one `AskUserQuestion`:
"Approve <budget> (<% of pool>) on <resource>?" This is the canonical
"stop before an irreversible external action" checkpoint, and the hero-vs-fallback
pivot usually hinges on it. Also pre-commit the branch trigger as *binding*
("the Day-2 readout decides; no relitigating Day 4").

## Output artifacts
- A **scored gap analysis** with one named objective function.
- A **shared VERIFIED-facts block** (re-verified against disk).
- A **file→change→risk build spec** with per-file fallbacks.
- A **red-lines / do-not-claim list** with citations.
- One **north-star choreography** with per-agent orders.

## Failure modes to avoid
- Judge agent that scores politely → useless. Make it hostile and rubric-literal.
- Facts block nobody re-verifies → parroted numbers that may be wrong.
- North-star run too early → survey mush instead of a decision.
- Grounding does NOT catch semantic/logic bugs — schedule a separate adversarial
  verification pass on the code that enforces your core claim (see
  skills/harness-build-learnings).
