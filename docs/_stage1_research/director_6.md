I have enough grounding. Here is the architecture and build plan.

---

# NeuroAD Discovery Engine — HARNESS Architecture & Stage-2 Build Plan

**Director of Engineering decision doc. Grounded in the existing repo (`src/neuroad/`, `contract.py`, `pipeline.py`, `cli.py`, `claude/*`, `skills/`, `calibration.py`). Not a greenfield — this is a thin, high-leverage layer bolted on top of what already passes 73/74 tests.**

## 0. The one-sentence framing

> The engine you have today is a **referee**. The harness turns it into a **discovery instrument a named researcher drives from a two-sentence hypothesis** — frozen NeuroJEPA gives the predictive baseline, deterministic domain modules + externalized *policy docs* enforce AD neuroscience, ML surfaces a candidate pattern, and Claude adjudicates against the *same policy documents* the code reads. The output is never "a biomarker"; it is a **refereed candidate + one falsifiable experiment**, with its honesty rung stamped on it.

The Databricks analogy the user wants is precise: **`policy/` is the governed semantic layer.** One source of domain truth (confound priors, AT(N) staging, biomarker→mechanism routing, verdict/novelty rubrics), consumed *both* by the deterministic "SQL-like" layer (gauntlet thresholds, routing tables) *and* by the AI layer (Claude's system prompt). Today those rules are scattered across `calibration.py` constants, `bridge._MECHANISMS`, `contract.VERDICT_BANDS`, and `_client.REFEREE_SYSTEM`. Stage 2 **externalizes them into declarative documents** and makes both consumers read the same file.

---

## 1. ARCHITECTURE — the layers

Bottom-up. Everything from L0–L4 **already exists**; L3 (policy) and L5 (hypothesis orchestrator) are the new work, plus rewiring L1/L4 to read L3.

| Layer | What it is | Status | Repo home |
|---|---|---|---|
| **L0 Frozen substrate** | NeuroJEPA 768-d structural-MRI embeddings, frozen inference only (CC-BY-NC-ND). The "decent predictive baseline." | exists | `contract.py` table, `data/*`, `scripts/*_embed.py` |
| **L1 Deterministic domain modules** | The 5-test gauntlet, leakage double-dissociation, confound leaderboard, calibrated facts. Pure pandas/numpy, offline. Encodes *how a neuroimaging finding fakes you out*. | exists | `gauntlet.py`, `leakage.py`, `calibration.py` |
| **L2 ML discovery / validation** | Supervised probe (point one head at any label) + unsupervised Detective (KMeans/GMM/HDBSCAN + bootstrap-Jaccard stability) + robustness scoring. The SL/SSL. | exists | `probe.py`, `detective.py`, `discovery.py`, `scoring.py` |
| **L3 Policy layer** ⭐NEW | Declarative domain-knowledge documents: confound priors, biomarker→mechanism routing, AT(N) framework, verdict rubric, novelty/honesty rubric, hypothesis schema. The governed semantic layer. | **build** | `policy/` + `harness/policy.py` loader |
| **L4 Claude reasoning / adjudication** | claim-parser, courtroom (prosecution/defense), bridge (mechanism), reviewer (argue against verdict), narrator. Offline deterministic fallbacks. | exists — **rewire to read L3** | `claude/*`, `_client.py` |
| **L5 Hypothesis entry point** ⭐NEW | `investigate("<2-sentence hypothesis>", dataset)` → one orchestrated pass → an Experiment Card. The researcher-facing personalization surface. | **build** | `harness/orchestrator.py`, `cli.py investigate` |

### Data flow: hypothesis → experiment card

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  RESEARCHER (real user: a computational / translational AD scientist)     │
 │  types 1–2 sentences:                                                     │
 │  "I think there's a temporal-atrophy pattern in MCI converters that       │
 │   isn't just brain-age, and it should track plasma GFAP not p-tau217."    │
 └───────────────────────────────┬─────────────────────────────────────────┘
                                  ▼
  ① STRUCTURED CLAIM      claim_parser + policy/hypothesis_schema.yaml
                          + policy/novelty_rubric.md
     → Claim{ target, group_a/b, covariates,
              novelty_class, expected_direction, prestated_kill_criterion }
                                  ▼
  ② DISCOVERY             harness/discovery_router → {supervised probe  |  Detective}
     (aim the instrument) novel-pattern hypotheses route to the SSL Detective;
                          named-contrast hypotheses point the head at a target
     → candidate imaging pattern  (effect size / phenotype cluster + stability)
                                  ▼
  ③ REFEREE               gauntlet(5 tests) + leakage(double-dissociation,
                          confound leaderboard) + policy/confound_priors.yaml
                          + policy/verdict_rubric.md
     → robustness 0–100 → verdict {fragile … strong} ; promote/reject
                                  ▼
  ④ CORROBORATION GATE    biomarker anchor when available; leakage-clean replication otherwise
     (imaging → molecule) + policy/biomarker_routing.yaml
     → AT(N) profile + dominant-marker routing (p-tau217/amyloid | GFAP | NfL)
                                  ▼
  ⑤ EXPERIMENT CARD       bridge + reviewer + narrator (all reading L3 policy)
     → mechanism hypothesis + ONE falsifiable next experiment
       (named cohort, target N, expected direction, kill criterion)
       + novelty_class + honesty_rung + caveats + evidence ledger
```

Claude sits at ①④⑤ as **adjudicator**, not coder; L1/L2 do the deterministic math; **L3 is read by both.** The whole chain has an offline deterministic fallback at every Claude call (existing `pipeline.py` guarantee — preserved).

---

## 2. DOMAIN KNOWLEDGE AS POLICY DOCS

**Format decision.** One convention, matching the *existing* `SKILL.md` pattern so it reads as deliberate, not ad-hoc: **Markdown with a YAML frontmatter block.** The frontmatter is the machine-readable contract the deterministic code parses; the Markdown body is the rationale Claude injects into its system prompt. Pure lookup tables (routing, confound priors) get a `data:` block in frontmatter or a sibling `.yaml`. This gives the exact **dual-consumption** the Databricks semantic-layer / Claude-skills analogy calls for: code reads the table, the model reads the prose, from **one file**.

**How the harness consumes them.** `harness/policy.py` is the loader (the semantic-layer client):

- `policy.table("biomarker_routing")` → dict the deterministic `bridge.py` uses.
- `policy.thresholds("confound_priors")` → floats `gauntlet.py` uses instead of magic numbers.
- `policy.brief("verdict_rubric")` / `policy.brief("novelty_rubric")` → Markdown bodies composed into `_client.REFEREE_SYSTEM` so **Claude adjudicates against the documented policy, not vibes.**
- **Every accessor has a hardcoded fallback** to the current constants (`calibration.CAL`, `bridge._MECHANISMS`, `contract.VERDICT_BANDS`). If `policy/` is absent or malformed, the demo runs byte-identically. This is how the offline-deterministic guarantee stays intact while we externalize.

### The six policy documents

| File | Encodes | Consumed by (code) | Consumed by (Claude) |
|---|---|---|---|
| `policy/confound_priors.yaml` | Per-confound prior magnitude + flag threshold + mitigation (scanner, site, age, sex, brain-age, ICV) | `gauntlet.py` thresholds, `leakage.confound_leaderboard` interpretation | courtroom prosecution framing |
| `policy/biomarker_routing.yaml` | Dominant-marker → mechanism → cohort/N/direction/kill routing table | `bridge._route` / `_MECHANISMS` | bridge mechanism prompt |
| `policy/atn_framework.md` | AT(N)(+I) staging; what molecular evidence anchors an imaging finding | anchor gate eligibility | bridge + anchor reasoning |
| `policy/verdict_rubric.md` | Verdict bands, promotion floor, hard-gate rule, hedged-language constraints | `scoring` / `contract.verdict_for` (bands sourced here) | `_client.REFEREE_SYSTEM` |
| `policy/novelty_rubric.md` | Candidate taxonomy + the 5-rung honesty ladder (candidate→refereed→anchored→replicated→clinical) | novelty honesty guard (assert every card is stamped) | claim_parser + narrator |
| `policy/hypothesis_schema.yaml` | Parse contract: target enum, discovery-mode selector, required Claim fields, default covariates | `claim_parser`, `discovery_router` | claim_parser prompt |

### Concrete example A — `policy/confound_priors.yaml`

```yaml
id: confound_priors
version: 1.0.0
consumed_by: [neuroad.gauntlet, neuroad.leakage, neuroad.claude.courtroom]
# Prior belief about how strongly each nuisance variable can COUNTERFEIT a
# structural-MRI disease signal, the statistic that measures it, and the
# threshold at which the referee flags it. Thresholds trace to calibration.CAL /
# published prior art (arXiv:2604.14441, arXiv:2606.09189) — NOT invented.
confounds:
  scanner_site:
    prior: high            # brain-FM embeddings predict site as well as outcome
    statistic: leakage_margin
    flag_at: 0.16          # published leakage-margin lower bound
    weight: 25             # ⭐ star test
    mitigation: residualize embeddings against scanner-predicting directions
    citation: arXiv:2606.09189
  brain_age:
    prior: high
    statistic: effect_retained_after_brainage_control
    flag_at: 0.40          # <40% retained => generic aging, FAILED (CAL.kill_retained)
    weight: 25             # ⭐
    mitigation: regress out predicted brain-age gap before re-scoring
    citation: FACTS.brain_age_gap
  age_sex:
    prior: medium
    statistic: effect_retained_after_covariate_adjust
    flag_at: 0.40
    weight: 15
    mitigation: partial out age + sex covariates
  icv_headsize:
    prior: medium          # NEW confound the harness adds beyond today's 5
    statistic: effect_retained_after_icv_adjust
    flag_at: 0.50
    weight: 0              # candidate 6th gauntlet test — nice-to-have
    mitigation: adjust for estimated total intracranial volume (eTIV)
```

### Concrete example B — `policy/biomarker_routing.yaml`

```yaml
id: biomarker_routing
version: 1.0.0
consumed_by: [neuroad.claude.bridge]
# Externalizes bridge._MECHANISMS. Which plasma marker dominates the cluster/
# contrast separation decides the mechanism, cohort, and pre-stated kill line.
routes:
  amyloid_cascade:
    dominant_markers: [p_tau217, amyloid]
    label: amyloid-cascade (tau-driven)
    atn_profile: "A+ T+ (N+)"
    cohort: ADNI-3 / EPAD
    target_n: 120
    expected_direction: probe score tracks plasma p-tau217 (r~0.30-0.55), enriched in amyloid+
    kill_criterion: no p-tau217 correlation (r<0.2) on complete-case subset
    citation: FACTS.ptau217
  glial:
    dominant_markers: [gfap]
    label: neuroinflammatory / glial (reactive astrogliosis)
    atn_profile: "A± T- N± I+"
    cohort: ADNI-3 GFAP subset / memory-clinic plasma-GFAP cohort
    target_n: 100
    expected_direction: probe score tracks GFAP more strongly than p-tau217
    kill_criterion: GFAP association no stronger than p-tau217, or vanishes after amyloid adjust
    citation: FACTS.gfap
  vascular:
    dominant_markers: [nfl]
    label: vascular / axonal
    atn_profile: "N+ (suspected non-AD pathophysiology)"
    cohort: NACC / EPAD with plasma NfL + FLAIR WMH
    target_n: 100
    expected_direction: probe score tracks NfL + white-matter-hyperintensity burden
    kill_criterion: NfL and WMH show no association with probe score
    citation: FACTS.nfl
```

Note both files literally re-encode logic that today lives as Python dicts/constants — so Stage 2 is a **refactor into a governed layer**, low-risk, with the old constants as the fallback path.

---

## 3. MAP TO REPO — exactly what to add / extend

### New package `src/neuroad/harness/`

- `harness/__init__.py`
- `harness/policy.py` — loader + typed accessors (`table`, `thresholds`, `brief`, `route`), YAML/frontmatter parse, schema validation, **fallback to `calibration.CAL` / `bridge._MECHANISMS` / `contract.VERDICT_BANDS`** when a doc is missing. No network.
- `harness/discovery_router.py` — `route(claim, df) -> ("supervised", target) | ("unsupervised", detective_cfg)`. Reads `hypothesis_schema.yaml`. Novel-pattern / "phenotype" / "subtype" hypotheses → Detective; named contrasts → probe.
- `harness/experiment_card.py` — `ExperimentCard` = thin wrapper over `contract.ClaimCard` adding `novelty_class`, `atn_profile`, `honesty_rung`, `discovery_provenance` (mode, stability, ARI if planted). `to_dict()` merges ClaimCard's dict + these. Keeps the frozen contract almost untouched.
- `harness/orchestrator.py` — **the entry point.** `investigate(hypothesis: str, dataset, mode="auto") -> ExperimentCard`: parse → route → discover → `pipeline.run_referee` (reused as-is) → anchor/AT(N) stamp → bridge/reviewer/narrator → assemble + honesty-guard.

### New top-level `policy/`

- The six docs from §2 + `policy/README.md` (mirrors `skills/README.md`: "domain knowledge as drop-in governed documents; add a 7th by dropping a file").

### New skills (mirror the existing 5)

- `skills/hypothesis_intake/` (SKILL.md + run.py) — wraps claim parsing → Claim.
- `skills/novelty_triage/` (SKILL.md + run.py) — classifies candidate + assigns honesty rung.

### Extend existing files (minimal, backward-compatible)

| File | Change | Risk |
|---|---|---|
| `contract.py` | Add 3 **optional** `ClaimCard` fields (`novelty_class`, `atn_profile`, `honesty_rung`) w/ empty defaults; extend `to_dict()`. Bump `CONTRACT_VERSION`→`1.1.0`. | Backward-compatible; existing tests pass. |
| `bridge.py` | `_route`/`_MECHANISMS` read `policy.table("biomarker_routing")`; keep dict as fallback. | Low — fallback preserves behavior. |
| `gauntlet.py` | Read flag thresholds from `policy.thresholds("confound_priors")`; keep `calibration.CAL` fallback. | Low. |
| `_client.py` | Compose `REFEREE_SYSTEM` + `policy.brief("verdict_rubric")` + `policy.brief("novelty_rubric")`. | Low — prompt-only. |
| `claim_parser.py` | Enrich `Claim` with `novelty_class`/`expected_direction`/`kill_criterion` per `hypothesis_schema.yaml`; deterministic fallback keeps working. | Low. |
| `cli.py` | Add `neuroad investigate "<hypothesis>" <dataset>` → `orchestrator.investigate` → print + `_write_reports`. | Isolated new subcommand. |
| `app/build_demo_data.py` + `app/index.html` | Add an "Investigate" panel that renders the hypothesis→ExperimentCard flow. | Demo surface. |

**Offline-deterministic guarantee — how it's preserved:** every new policy read has a constant fallback; `orchestrator.investigate` runs entirely on synthetic/OASIS cohorts with the existing Claude offline templates; no policy file requires the network; `build_demo_data.py` keeps emitting a byte-stable JSON. The `demo` subcommand path is untouched.

---

## 4. STAGE-2 BUILD TASK LIST (~5 days, Jul 8 → Jul 13 9pm ET, team ≤2)

Sequenced; file-disjoint lanes marked **[A]/[B]** so two people (or two agents) run in parallel. Flags: **🎯 demo-critical**, **🔬 depth-critical** (Claude-Use/Depth judging), **✨ nice-to-have**.

### Day 1 — Policy layer foundation
- **T1 [A] 🎯** Author the 6 `policy/*.{md,yaml}` docs (§2) + `policy/README.md`. File-disjoint; only needs a 30-min frontmatter-schema agreement first.
- **T2 [B] 🎯** `harness/policy.py` loader + accessors + **constant fallbacks** + `tests/test_policy.py` (missing-file → fallback path). Depends on T1 schema only.

### Day 2 — Hypothesis entry point (the headline)
- **T3 [A] 🎯** `contract.py` optional ClaimCard fields + `to_dict`; `harness/experiment_card.py`. Bump CONTRACT_VERSION.
- **T4 [B] 🎯** `harness/discovery_router.py` (supervised vs Detective) + unit test.
- **T5 [A+B] 🎯** `harness/orchestrator.py investigate()` wiring parse→route→discover→`run_referee`→anchor→bridge→card. Integration point. Depends T2/T3/T4.

### Day 3 — Wire policy into deterministic + Claude layers (the "harness" story)
- **T6 [A] 🔬** `bridge.py` reads `biomarker_routing.yaml` (fallback `_MECHANISMS`). *Depth-critical, not demo-blocking.*
- **T7 [B] 🔬** `gauntlet.py` reads `confound_priors.yaml` thresholds (fallback `CAL`). File-disjoint from T6. *Depth-critical, not demo-blocking.*
- **T8 [A] 🔬** `_client.py` injects verdict/novelty briefs; `claim_parser.py` enriched Claim (novelty_class, expected_direction, kill_criterion).
- **T9 [B] ✨** `skills/hypothesis_intake/` + `skills/novelty_triage/` folders.

### Day 4 — CLI, demo surface, honesty guard
- **T10 [A] 🎯** `cli.py investigate` subcommand + report writing.
- **T11 [B] 🎯** `app/build_demo_data.py` + `app/index.html` **Investigate panel** — the money shot: a 2-sentence hypothesis → Detective surfaces a candidate phenotype → gauntlet refutes/survives → biomarker anchor → experiment card, on the OASIS-1 cohort and/or the planted-phenotype Detective run (which recovers a *novel candidate pattern honestly*).
- **T12 [A] 🎯** Novelty honesty guard: assert every `ExperimentCard` carries `novelty_class` + `honesty_rung`; test that no rendered output ever says "proven/validated biomarker." This is the anti-overclaim contract the whole pitch rests on.

### Day 5 — Tests, docs, buffer
- **T13 [A+B] 🎯** `tests/test_harness.py` — end-to-end `investigate` offline (no API key), router coverage, policy-fallback coverage. Target: keep the suite green (74→~85 tests).
- **T14 [A] 🎯** `docs/HARNESS.md` (this architecture) + update `README.md` + `docs/DEMO_SCRIPT.md` with the investigate flow.
- **T15 [B] ✨** If time: use the colab-gpu-cli to embed ~174 more labeled OASIS-1 subjects (headroom is real per grounding) → strengthens "findings you trust." Pure upside; cut first if slipping.

### Critical path & cut lines
- **Demo-critical (must ship):** T1, T2, T3, T4, T5, T10, T11, T12, T13, T14.
- **Depth-critical (ship if at all possible — this IS the "policy-doc harness" narrative for Claude-Use/Depth judging):** T6, T7, T8.
- **First to cut under time pressure:** T15, then T9, then T11's polish (keep the panel functional, drop animation).

### Why this scores
- **Impact (25%) / Gladstone "advance the field":** a translational AD researcher drives novel-biomarker triage from two sentences; the tool outlasts the week because domain knowledge is drop-in `policy/` docs, not buried code.
- **Claude Use (25%):** Claude adjudicates against the *same governed policy documents the deterministic code reads* — a genuine semantic-layer + skills pattern, not "Claude as chatbot."
- **Depth (20%):** externalizing constants into a policy layer with fallbacks, an AT(N)-aware anchor, and a novelty/honesty ladder is clearly past the first idea.
- **Demo (30%):** the Investigate panel shows SSL surfacing a *candidate novel pattern*, then honestly refereeing and anchoring it — impressive **and** trustworthy, exactly the "novel biomarker done honestly" the vision demands.

**Files referenced (all absolute):** `/Users/stevenyang/Documents/claude-life-sciences-hack/neuroad-discovery-engine/src/neuroad/{contract,pipeline,cli,calibration,gauntlet,discovery,leakage,probe}.py`, `.../src/neuroad/claude/{_client,claim_parser,bridge,courtroom,reviewer,narrator}.py`, `.../skills/`, `.../app/{index.html,build_demo_data.py}`. **New:** `.../src/neuroad/harness/{policy,discovery_router,orchestrator,experiment_card}.py`, `.../policy/*.{md,yaml}`, `.../skills/{hypothesis_intake,novelty_triage}/`, `.../tests/{test_policy,test_harness}.py`, `.../docs/HARNESS.md`.
