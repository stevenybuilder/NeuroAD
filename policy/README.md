---
policy: policy_layer_readme
layer: L3
kind: brief
consumers: [code, claude]
version: 1.0.0
schema_version: 1
last_verified: "2026-07-08"
mirrors:
  - src/neuroad/gauntlet.py
  - src/neuroad/calibration.py
  - src/neuroad/contract.py
  - src/neuroad/leakage.py
  - src/neuroad/claude/bridge.py
  - src/neuroad/claude/claim_parser.py
---

# L3 Policy Layer — declarative domain knowledge, read by both consumers

This directory holds the **declarative domain knowledge** of NeuroAD Discovery
Engine: the priors, thresholds, staging frameworks, verdict bands and parse
contract that the pipeline reasons with. It sits **below** the deterministic
code (`src/neuroad`, L1) and the Claude personas (L2) as a shared, editable
**source of policy** — change a threshold's meaning here and in the mirrored
constant together, and both consumers stay in agreement.

Nothing in this directory *executes*. It is knowledge, not logic.

## The dual-consumption pattern

Every file serves **two readers at once** through one format — YAML front-matter
(or a top-level `meta:` block) for machines, Markdown / structured YAML body for
Claude:

| Consumer | Reads | For |
|----------|-------|-----|
| **Deterministic code** (the policy loader, gauntlet, bridge fallback) | the **YAML tables & thresholds** | fallback values identical to the running constants, so behavior never changes if a live value goes missing |
| **Claude prompts** (Claim Parser, Referee, Bridge personas) | the **Markdown briefs & prose** | domain reasoning: which artifact to blame, how to hedge, which mechanism to route to |

This mirrors the existing `skills/*/SKILL.md` convention (YAML front-matter +
Markdown body), extended so the same file can also carry a machine-loadable
data table.

### File → consumer map

| File | `kind` | Primary code consumer | Primary Claude consumer |
|------|--------|-----------------------|-------------------------|
| `confound_priors.yaml` | table | gauntlet threshold fallback | "which confound killed it" reasoning |
| `biomarker_routing.yaml` | table | `bridge._fallback` mechanism build | BRIDGE routing guidance |
| `atn_framework.md` | brief | anchor-gate n/CI constants (front-matter) | AT(N)+I staging & anchor eligibility |
| `verdict_rubric.md` | brief | verdict-band fallback (front-matter) | verdict wording + hedged-language rule |
| `novelty_rubric.md` | brief | `novelty_class` / `honesty_rung` vocab | honesty-ladder self-assessment |
| `hypothesis_schema.yaml` | table | claim-parse validation + offline router | pre-registered kill-criterion format |

## Frozen-constant rule (why the values are transcribed, not invented)

Every threshold in this layer is a **transcription** of a live constant in
`src/neuroad`, stamped inline with its `source:`. The policy files exist so the
loader has a defensible fallback and Claude has readable context — **not** to
introduce new numbers. If a value here ever disagrees with its mirrored constant,
the code is authoritative and the policy file has a bug. Load-bearing sources:

- retained bands → `gauntlet._SURVIVOR_RETAINED (0.70)`, `_KILL_RETAINED (0.40)`
- anchor gate → `gauntlet._ANCHOR_CI_PASS (0.12)`, `_ANCHOR_CI_WEAK (0.0)`, `_ANCHOR_MIN_N (20)`
- replication → `test_replication` (PASS `>= 0.65`, WEAK `>= 0.58`)
- verdict bands / floor → `contract.VERDICT_BANDS`, `PROMOTION_FLOOR (>= 40)`
- gauntlet weights / credit → `contract.GAUNTLET`, `RESULT_CREDIT`
- CAL ranges → `calibration.CAL`
- mechanisms → `bridge._MECHANISMS`
- parse schema → `claim_parser._SCHEMA`, `_GROUPS`, `_infer_target`

## Shared front-matter / `meta` schema (the loader contract)

Both `.md` front-matter and the `.yaml` top-level `meta:` block use one schema so
the policy loader can enumerate this directory uniformly:

```yaml
policy: <stable-slug>          # unique id for the file
layer: "L3"                    # always L3 here
kind: "table" | "brief"        # table = code-first YAML data; brief = Claude-first prose
consumers: [code, claude]      # who reads it
version: <semver>              # policy-file version
schema_version: 1              # version of THIS front-matter schema (bump on shape change)
last_verified: "YYYY-MM-DD"    # date the mirrored constants were last reconciled
mirrors: [ <src/... paths> ]   # the source-of-truth modules this file transcribes
```

The loader keys files by `policy`, dispatches by `kind` (`table` → parse the YAML
body/`meta`-sibling data keys; `brief` → surface the Markdown to the prompt),
and can assert `schema_version == 1` before trusting the shape.
