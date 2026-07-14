# Built with Claude Code — the multi-agent, contract-first build

NeuroAD Discovery Engine was built the way it runs: Claude is the intent router,
orchestrator, and Q&A rail inside the product — over a deterministic referee — *and*
the multi-agent method behind the repo. This document is the "creative Claude use"
evidence.

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
| M2 | `data/` | real cohort loaders (OASIS/ADNI), adapters, seeded test fixtures |
| M3 | `claude/` | router, claim parser, courtroom, narrator, bridge, reviewer |
| M4 | `app/` | offline visual workbench |
| M5 | `pipeline/cli` + docs | orchestration, packaging, this write-up |

Each agent built its module to be **independently importable and unit-testable**
using small contract-valid fixtures — so a module lands and its tests pass even
before its siblings exist. The orchestration layer (this agent's `pipeline.py`)
imports siblings lazily and wraps the whole Claude layer in safe fallbacks, so a
half-finished tree still runs.

## 2. Claude inside the product — three live roles, a deterministic referee

The referee's verdict is deterministic arithmetic (`contract.py`) — **Claude never
sets a score.** That is the honesty contract (`_client.model_badge` reports
`referee: deterministic`). Claude runs live (when `ANTHROPIC_API_KEY` is set) in
three real, consequential roles:

- **Intent router** (`claude/router.py`, Claude Sonnet) — the LLM-as-judge. It
  classifies a free-text hypothesis into the finite target enum
  {conversion, dx_binary, site, scanner} via an enum-constrained structured call,
  with a keyword classifier as the backstop on a cache hit or offline. It picks
  *which* precomputed cell is read — never a number — and fixes the
  "predicts → conversion" misroute. One canonical `route_target` feeds both the
  engine target and the cache key, so they can't drift.
- **Orchestrator** (`harness/agent.py`, Claude Sonnet) — a tool-runner that
  sequences the engine's tools toward a goal (`/api/orchestrate`); it drives the
  pipeline but can never override a verdict.
- **Ask Claude** (`/api/ask`, Claude Opus) — a grounded Q&A rail that answers a
  scientist's follow-ups and drills into the decision tree, grounded in the live
  case + a curated knowledge base; with no key it returns an honest offline notice,
  never a fabricated answer.

The rest of `claude/` — the courtroom (prosecution/defense/judge), reviewer,
bridge, narrator, and claim parser — is **deterministic Python** synthesized from
the gauntlet's real stats, not an LLM call. The engine refuses to launder a
template as a Claude verdict.

## 3. Gauntlet stages as drop-in Agent Skills

Each gauntlet stage is packaged as a drop-in Agent Skill (a `SKILL.md` pack) and
exposed through a one-command `neuroad` CLI / `/referee` slash command — so the
tool is "built to outlast the week": a scientist can invoke a single stage, or
the whole referee, without us in the room.

## 4. Why this is the honest version

We checked the leakage insight against July-2026 literature and it is **published
prior art**. So we repositioned: cite arXiv:2604.14441 / 2606.09189 / PathoROB
openly, drop "co-scientist" and "discovery platform," and own the parts nobody
shipped — the runnable referee, the closed loop, the biomarker gate, and Claude as
router + orchestrator over a deterministic engine.
