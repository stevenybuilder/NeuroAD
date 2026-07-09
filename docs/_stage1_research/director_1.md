I have enough to synthesize. Here is the research deliverable.

---

# Harness-on-top-of-NeuroJEPA: Patterns from Anthropic + Databricks

Research synthesis to inform the "Databricks-style harness" for the NeuroAD Discovery Engine. All claims cited to primary sources.

---

## PART A — What ANTHROPIC puts in its agent/data harnesses

### A1. Building Effective Agents — the workflow/agent distinction and the 7 patterns
Anthropic's core teaching is to **separate deterministic orchestration (workflows) from model-driven autonomy (agents)**, and to prefer the former until autonomy demonstrably pays off.

- **Workflows** = "LLMs and tools orchestrated through predefined code paths"; **Agents** = "LLMs dynamically direct their own processes and tool usage." "Workflows offer predictability and consistency for well-defined tasks, whereas agents are the better option when flexibility and model-driven decision-making are needed at scale."
- The **augmented-LLM** building block = LLM + retrieval + tools + memory.
- Five workflow patterns directly relevant to your gauntlet: **prompt chaining** (fixed subtasks with programmatic "gates" for validation), **routing** (classify input → specialized branch; "separation of concerns"), **parallelization** (sectioning + voting), **orchestrator-workers** (central LLM decomposes → delegates → synthesizes), and **evaluator-optimizer** (one LLM generates, another critiques in a loop — "when clear evaluation criteria exist and iterative refinement provides measurable value").
- Three implementation principles: **maintain simplicity**, **prioritize transparency** ("explicitly showing the agent's planning steps"), and craft a clean **agent-computer interface** (tool docs + testing, "poka-yoke" designs). "Add complexity only when it demonstrably improves outcomes."
- Verification: sandboxed testing, guardrails, and for code, "automated testing verifies functionality but human review remains crucial."

Source: https://www.anthropic.com/engineering/building-effective-agents

### A2. Multi-agent research system — orchestrator + isolated subagents + LLM-as-judge + citation agent
- **Lead/orchestrator agent** analyzes the query, writes a **plan to memory** (survives the 200K-token truncation boundary), and **spawns 3–5 subagents in parallel**, each with its **own context window, tools, and trajectory**. "Each subagent needs an objective, an output format, guidance on the tools and sources to use, and clear task boundaries."
- **Separation of concerns**: "distinct tools, prompts, and exploration trajectories… reduces path dependency and enables thorough, independent investigations." Subagents return **condensed 1–2K-token summaries** to the lead.
- **Effort scaling** is encoded as an explicit heuristic (simple fact = 1 agent/3–10 calls; complex = 10+ agents).
- **Verification stack**: an **LLM-as-Judge** scores outputs 0.0–1.0 against a **rubric** (factual accuracy, citation accuracy, completeness, source quality, tool efficiency); a dedicated **CitationAgent** attaches every claim to a specific source; plus deterministic safeguards (retry logic, checkpoints, production tracing) and human eval for edge cases.

Source: https://www.anthropic.com/engineering/multi-agent-research-system

### A3. Effective context engineering — keep the context tight, retrieve just-in-time
- Goal: "the smallest set of high-signal tokens that maximize the likelihood of your desired outcome."
- **System prompts at the "right altitude"**: not brittle hardcoded logic, not vague guidance — "specific enough to guide behavior, flexible enough to provide strong heuristics." Organize with XML/Markdown sections.
- **Tools**: self-contained, minimal overlap, unambiguous.
- **Curated canonical examples** over exhaustive edge cases ("examples are the pictures worth a thousand words").
- **Just-in-time retrieval**: hold lightweight identifiers (paths, stored queries, links), load at runtime = progressive disclosure.
- **Compaction** (summarize preserving decisions/open bugs), **structured note-taking/memory** outside the window, and **sub-agent context isolation** returning distilled summaries.

Source: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

### A4. Agent Skills — the skill / prompt / MCP model
- A **Skill** = a directory with a `SKILL.md` (YAML `name` + `description`) plus bundled resources and **deterministic scripts**. **Three-level progressive disclosure**: (1) metadata in system prompt at startup, (2) full SKILL.md loaded if relevant, (3) linked files/scripts loaded "only as needed" — "a well-organized manual: table of contents → chapters → appendix."
- Skills "capture and share **procedural knowledge**" and "encode both instructions and deterministic code for reliable, repeatable operations," turning general agents into specialists.
- The stack model: **MCP connects Claude to data/systems; Skills teach Claude what to do with it; prompts are the atomic per-turn instruction.** "MCP for connectivity, Skills for procedural knowledge" — complementary, not exclusive.

Sources: https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills · https://claude.com/blog/skills-explained

### A5. Claude Science — the concrete life-sciences harness (your closest reference design)
- A **generalist coordinating agent** with access to **60+ curated skills and connectors** pre-configured per domain (genomics, single-cell, proteomics, structural biology, cheminformatics). "Specialist agents query and synthesize across sources so you don't have to navigate them individually." Agents can **spin up other agents**, including user-created specialists.
- Foundation-model layer wired in via **NVIDIA BioNeMo Agent Toolkit skills** (Evo 2, Boltz-2, OpenFold3) — i.e. **domain FMs exposed as skills/tools**, not the reasoning layer itself.
- **Provenance is first-class**: "every output carries an **auditable history of how it was made**, so you can validate and reproduce the results." Natural-language plain-language entry point; multi-step research producing refinable artifacts (figures/manuscripts).

Sources: https://www.anthropic.com/news/claude-science-ai-workbench · https://claude.com/product/claude-science · https://blogs.nvidia.com/blog/claude-science-bionemo-agent-toolkit/

---

## PART B — What DATABRICKS puts in its data harness

### B1. Data Intelligence Platform + Mosaic AI = a compound system, not one model
Databricks explicitly frames the goal as **compound AI systems** that **combine deterministic logic + retrieval + ML + foundation models**, all under one governance/lineage plane. Components:
- **Unity Catalog** — single source of truth for data *and semantics*, governance, versioning, lineage, access control.
- **Delta tables** — trusted, governed, streaming-updatable data substrate.
- **Mosaic AI** — Foundation Model APIs, Model Serving (custom + FM), **Vector Search** (embeddings → real-time retrieval), the **Agent Framework**, **AI Gateway** (govern/monitor model access), and **MLflow** for tracking + LLM evaluation.

Sources: https://www.databricks.com/blog/mosaic-ai-build-and-deploy-production-quality-compound-ai-systems · https://www.tredence.com/blog/architecting-ai-agents-with-databricks-from-vector-search-to-foundation-models

### B2. Genie — natural-language entry point with SHOWN reasoning + provenance + trust badge
This is the single most transferable idea for your UX.
- Plain-language question → Genie returns **SQL + results + visualization**, and **"with every response provides an explanation of how it arrived at the answer"** — its thought process, the **trusted assets it used**, and the SQL logic, for diagnosis.
- **"Trusted" badge**: when Genie runs a governed UDF/metric, the result is stamped Trusted, signaling confidence.
- **Agent mode** does multi-step reasoning + **hypothesis testing**: "creates and refines a research plan, running multiple SQL queries, learning from each result, and iterating until it has enough evidence."

Sources: https://www.databricks.com/blog/aibi-genie-now-generally-available · https://www.databricks.com/product/genie/agents

### B3. How Databricks ENCODES domain knowledge (the key extract)
Genie spaces encode domain expertise in explicit, layered, human-curated config — **not** in model weights:
1. **Metadata & vocabulary** — table/column descriptions, **synonyms** ("bridge business jargon and technical column names"), value dictionaries.
2. **Relationships** — explicit join definitions with cardinality (prevents double-counting).
3. **Example SQL queries** — demonstrate "how to handle complex logic — tricky calculations, specific filters, re-used multi-step aggregations," with **usage guidelines specifying *when* to apply each**.
4. **Trusted assets** — **UDFs** for "logic requiring no variation in the underlying formula" (deterministic), surfaced with the Trusted badge.
5. **Metric views** — shared business logic so "every query relies on the same approved logic" (single source of truth).
6. **General instructions** — narrative context that "explains entities and relationships **without dictating specific SQL behavior**" — used only when the structured tools are insufficient (the "right altitude" idea again).
7. **Human-in-the-loop maintenance** — "the most successful Genie Spaces are actively monitored, maintained, and improved in partnership with the users they serve," with SME feedback loops and benchmark validation.

Source: https://www.databricks.com/blog/data-dialogue-best-practices-guide-building-high-performing-genie-spaces

---

## PART C — THE REUSABLE PATTERN (both companies, one shape)

Both harnesses are the **same layered compound system**:

```
Natural-language hypothesis  ──►  Coordinating agent (planner, right-altitude prompt)
                                        │  decomposes, scales effort, saves plan to memory
                    ┌───────────────────┼─────────────────────────┐
        Deterministic domain modules   ML layer          Foundation-model layer
        (encoded knowledge:            (SL/SSL/RL         (frozen embeddings / FM
         rubrics, policy docs,          probes,            exposed as a *tool/skill*,
         config, "trusted" UDFs)        detective)         never the reasoning core)
                    └───────────────────┼─────────────────────────┘
                              Evaluator/critic loop (LLM-as-judge + rubric)
                                        │
                     SHOWN REASONING + PROVENANCE (every claim → source,
                     "Trusted" badge on deterministic results, auditable history)
```

Invariants worth copying verbatim:
1. **The FM is a *substrate/tool*, not the brain.** Databricks wires FMs behind Model Serving; Claude Science wires Evo2/Boltz behind skills; **you wire frozen NeuroJEPA behind a probe tool.** The value-add is the *harness around* the FM.
2. **Domain knowledge lives in editable, versioned config — not weights.** Synonyms, metric views, example queries, UDFs, rubrics, general instructions. Human-curated, SME-maintained.
3. **Deterministic where the formula must not vary; ML where patterns must be learned; LLM only for orchestration + adversarial reasoning + narration.**
4. **Trust is a UI primitive.** Show the plan, show the SQL/test, badge the deterministic results, attach every claim to a source (CitationAgent / Trusted badge / auditable history).
5. **Verification is a first-class subsystem** (evaluator-optimizer + LLM-as-judge + rubric + human edge-case review), separated from generation.

---

## PART D — Concrete architecture for the NeuroJEPA harness

Mapping the pattern onto your existing pieces (probe + Detective + 5-test gauntlet + Claude prosecution/defense):

**1. Natural-language hypothesis entry point (your "Genie")**
- Claude **claim-parser** (you have it) → normalizes a 1–2 sentence hypothesis into a structured **HypothesisSpec** {target phenotype, label column(s), cohort, expected direction, falsifiable prediction}. Mirror Genie's plain-language-in, structured-plan-out.

**2. Coordinating agent (orchestrator-workers)**
- Right-altitude system prompt (heuristics, not brittle rules). Writes the **investigation plan to a memory/notes file** (persist provenance, survive context limits). Scales effort: quick probe vs. full Detective sweep.

**3. Deterministic domain-knowledge modules (the "Trusted UDF" layer)** — encode AD/neuro knowledge as *code + config*, badge-able:
- `age_sex_confound.py`, `scanner_leakage.py`, `brain_age_gap.py`, `biomarker_anchor.py`, `replication.py` — your 5 gauntlet tests as deterministic UDFs (fixed formulas, no LLM variation → "Trusted" badge in the UI).
- A **neuro semantic layer** (`config/semantic.yaml`): label-column synonyms (CDR/MMSE/"conversion"), cohort join/leakage rules (site → field strength), known-confound registry (age, sex, scanner, ICV), directionality priors (hippocampal↓ in AD).

**4. ML layer (SL/SSL/RL)**
- **SSL/unsupervised Detective** (KMeans/GMM/HDBSCAN + bootstrap-Jaccard) = novel-phenotype discovery on frozen 768-d embeddings.
- **SL probes** (linear/MLP) pointed at label columns = calibrated baselines.
- (RL/active-learning, stretch) a **next-experiment selector** that ranks which additional OASIS-1 subjects to embed for max discriminative gain — honest, cheap, and demo-able.

**5. Foundation-model layer (NeuroJEPA, frozen, as a tool)**
- `neurojepa_embed` exposed as a **skill/tool** (à la BioNeMo skills), CC-BY-NC-ND compliant: frozen inference only, embeddings never published. It supplies the substrate; it is *not* the reasoner.

**6. Evaluator / critic subsystem (evaluator-optimizer + LLM-as-judge)**
- Claude **prosecution vs. defense** agents (you have them) = the evaluator-optimizer loop, each an isolated subagent with its own context.
- A **judge rubric** (`config/referee_rubric.md`) scoring each candidate 0–1 on: confound-robustness, scanner-leakage-freedom, biomarker-anchor strength, replication, effect size — mirroring Anthropic's LLM-as-judge dimensions. Verdict = survivor / rejected.
- A **CitationAgent analogue**: every verdict sentence links to the specific test output (p-value, Jaccard, AUC) that supports it.

**7. Shown reasoning + provenance (the trust UI)**
- The plan, each test's numeric output, the "Trusted" badge on deterministic tests, and an **auditable run history** (à la Claude Science) rendered in `app/index.html`. Survivors emit a **mechanism hypothesis + one falsifiable experiment** — honesty by construction (candidate, not proven biomarker).

---

## PART E — How domain knowledge should be encoded (the deliverable checklist)

Copy Databricks' layering + Anthropic's Skills format:

| Knowledge type | Encoding | Precedent |
|---|---|---|
| Fixed neuroscience formulas (brain-age gap, ICV correction, leakage tests) | **Deterministic Python UDFs**, badge-able "Trusted" | Genie trusted UDFs / metric views |
| Referee standards (what makes a phenotype survive) | **Rubric doc** `referee_rubric.md` with 0–1 dimensions | Anthropic LLM-as-judge rubric |
| Cohort/label semantics, synonyms, confound registry, directionality priors | **`semantic.yaml` config** | Genie synonyms + metric views |
| "When to run which test / how hard to push" | **General-instructions policy doc** at the *right altitude* (heuristics, not hardcoded control flow) | Context-engineering "right altitude" + Genie general instructions |
| Reusable procedures (embed a cohort, run the gauntlet, draft a next-experiment) | **Agent Skills** (`SKILL.md` + scripts, progressive disclosure) | Anthropic Agent Skills / Claude Science 60+ skills |
| Model access (NeuroJEPA, cohorts) | **MCP/tool connectors**, frozen-inference wrapper | MCP-for-connectivity; BioNeMo skills |
| Every result | **Provenance record** (claim → numeric evidence → source) | CitationAgent / Genie Trusted badge / Claude Science auditable history |

**One-line thesis for the pitch:** *NeuroJEPA is the substrate (a tool behind the harness), exactly as Databricks treats foundation models and Claude Science treats BioNeMo. The novel, judgeable value is the harness — deterministic AD-knowledge UDFs + a semantic/policy config layer + SSL discovery + an adversarial LLM referee with a scored rubric — that turns a researcher's one-sentence hypothesis into a provenance-tracked candidate biomarker plus its next falsifiable experiment.*

### Sources
- Anthropic, Building effective agents — https://www.anthropic.com/engineering/building-effective-agents
- Anthropic, How we built our multi-agent research system — https://www.anthropic.com/engineering/multi-agent-research-system
- Anthropic, Effective context engineering for AI agents — https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Anthropic, Equipping agents for the real world with Agent Skills — https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- Claude, Skills explained (skill vs prompt vs MCP) — https://claude.com/blog/skills-explained
- Anthropic, Claude Science AI workbench — https://www.anthropic.com/news/claude-science-ai-workbench · https://claude.com/product/claude-science
- NVIDIA, BioNeMo Agent Toolkit in Claude Science — https://blogs.nvidia.com/blog/claude-science-bionemo-agent-toolkit/
- Databricks, Mosaic AI / compound AI systems — https://www.databricks.com/blog/mosaic-ai-build-and-deploy-production-quality-compound-ai-systems
- Databricks, AI/BI Genie GA — https://www.databricks.com/blog/aibi-genie-now-generally-available · Genie Agents — https://www.databricks.com/product/genie/agents
- Databricks, Best-practices guide for Genie Spaces (domain-knowledge encoding) — https://www.databricks.com/blog/data-dialogue-best-practices-guide-building-high-performing-genie-spaces
- Databricks agentic architecture (Vector Search → FMs) — https://www.tredence.com/blog/architecting-ai-agents-with-databricks-from-vector-search-to-foundation-models