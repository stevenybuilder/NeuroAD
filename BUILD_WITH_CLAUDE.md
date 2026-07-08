# Built with Claude Code — the multi-agent, contract-first build

NeuroAD Discovery Engine was built the way it runs: Claude is not just the coder, it is
the reasoning engine inside the product *and* the orchestration method behind the
repo. This document is the "creative Claude use" evidence.

## 1. Contract-first, then parallel agents

We froze one file before writing any feature: `src/neuroad/contract.py` — the
embedding-table schema, the gauntlet dimensions and weights, the verdict bands,
and the `ClaimCard` artifact. A companion `src/neuroad/calibration.py` pinned
every on-screen number to a literature range with a citation ("no fabricated
science" gate).

With the interface frozen, the build fanned out into **file-disjoint Claude Code
agents that ran in parallel** — they could not collide because every module codes
against the contract, never against another agent's code:

| Agent | Module | Owns |
|---|---|---|
| M1 | `probe/gauntlet/leakage/scoring/detective` | the referee engine |
| M2 | `data/` | synthetic harness, OASIS adapter, loaders, stubs |
| M3 | `claude/` | claim parser, courtroom, narrator, bridge, reviewer |
| M4 | `app/` | offline visual workbench |
| M5 | `pipeline/cli` + docs | orchestration, packaging, this write-up |

Each agent built its module to be **independently importable and unit-testable**
using small contract-valid fixtures — so a module lands and its tests pass even
before its siblings exist. The orchestration layer (this agent's `pipeline.py`)
imports siblings lazily and wraps the whole Claude layer in safe fallbacks, so a
half-finished tree still runs.

## 2. Claude as the adjudicator (inside the product)

The Claude reasoning layer is `src/neuroad/claude/`. Every module speaks live
Anthropic API when `ANTHROPIC_API_KEY` is set and falls back to a **deterministic
template** otherwise, so the demo is fully offline and reproducible.

- **Courtroom** (`courtroom.adjudicate`): a **Prosecution** subagent argues the
  signal is an artifact, a **Defense** subagent argues it is real biology, and a
  **Judge** subagent renders the verdict with reasoning. Each makes a
  *consequential* decision — this defeats the "Claude is decoration" failure mode.
- **Reviewer** (`reviewer.review`): a peer-review agent that argues **against**
  the final verdict — flags the proxy brain-age control, p-tau217 missingness,
  and "partially robust ≠ robust." A referee that referees itself.
- **Bridge** (`bridge.propose_biology`): survivors only → one biomarker-routed
  mechanism + one falsifiable experiment + falsification criteria.
- **Claim parser** and **narrator** turn a plain-language hunch into a structured
  `Claim` and the final card into plain-language verdict prose.

`pipeline.run_referee` chains these so the consequential Claude steps only fire
for *promoted* survivors, and never crash a run if the layer is missing/offline.

## 3. Gauntlet stages as drop-in Agent Skills

Each gauntlet stage is packaged as a drop-in Agent Skill (a `SKILL.md` pack) and
exposed through a one-command `neuroad` CLI / `/referee` slash command — so the
tool is "built to outlast the week": a scientist can invoke a single stage, or
the whole referee, without us in the room.

## 4. Why this is the honest version

We ran Claude's own "does this already exist?" test against July-2026 literature
and it told us the leakage insight is **published prior art**. So we repositioned:
cite arXiv:2604.14441 / 2606.09189 / PathoROB openly, drop "co-scientist" and
"discovery platform," and own the parts nobody shipped — the runnable referee,
the closed loop, the biomarker gate, and Claude-as-adjudicator. Using Claude to
red-team our own novelty claim is itself part of the build story.
