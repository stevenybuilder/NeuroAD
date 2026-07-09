# NeuroAD Discovery Engine — Consolidated Stage-2 Blueprint
### Chief of Staff synthesis · One plan, four directors · Deadline: Jul 13, 9pm ET

---

## 1. NORTH STAR

NeuroAD is **not a detector and not a treatment tool — it is a discovery-and-referee accelerator** that turns a researcher's two-sentence hypothesis into a stability-vetted, confound-screened, severity-anchored *candidate* imaging phenotype, plus the single experiment that would confirm or kill it. We do not race the FDA-cleared plasma p-tau217 assays on amyloid detection (a solved-enough problem we would lose at on 61 subjects); we fill the gap plasma cannot — topography, the N-axis, subtype heterogeneity, rate-of-change — which is exactly where the 2024 NIA-AA criteria retain imaging and where the Gladstone "advance the field" bar lives. The winning demo is **"the referee catches its own false discovery, then finds a real one"**: a live two-act run on real frozen Neuro-JEPA embeddings that kills a great-looking cluster as scanner noise, then surfaces a real survivor that anchors to CDR severity and ships with a named confirmatory cohort. This single choreography carries all four judging axes — Claude on the critical path (Claude Use), a produced lead not just a rejection (Impact/Gladstone), grad-level statistics shown live (Depth), and a hero result that is *real* with synthetic demoted to a labeled control (Demo).

---

## 2. THE DEMO WE'RE BUILDING

**One demo, three moves, driven from a hypothesis the researcher types.**

**Entry point (Claude on the critical path).** The researcher types:
> *"I think there's a non-hippocampal atrophy pattern in this cohort that tracks disease severity beyond normal aging. Show me if it's real or if it's scanner noise."*

Claude parses this into a structured `HypothesisSpec`: target = unsupervised phenotype in frozen embedding space; contrast = CDR/diagnosis; confounds to prioritize = age, sex, scanner/field-strength; **pre-registered falsification criterion** = "reject if scanner-AUC > threshold OR bootstrap-Jaccard < 0.6 OR anchor CI lower-bound crosses zero." The tool commits to what would kill the finding *before* it runs, and that spec **parameterizes the actual run** (which label column the anchor uses, which confounds the gauntlet prioritizes, the accept/reject thresholds). This is the current build's single biggest gap and we close it.

**What the judge sees, in order:**

1. **Hypothesis-in → plan-out card.** The structured spec + its pre-registered kill criteria, on screen, before any math runs.
2. **Act I — the trap, sprung on real data.** The Detective clusters real embeddings; the 5-test gauntlet ticks live. A promising cluster lights **red on the scanner-leakage double-dissociation** (LDA scanner directions on the 6-site OpenBHB split, scanner-AUC ≈ 0.90). Verdict: **KILLED — scanner artifact.** Claude's Prosecution/Defense narrate *why* over the same numeric evidence. We caught our own false discovery — more convincing than any accuracy number.
3. **Act II — the survivor.** The same machinery, pointed at the real disease-bearing OASIS-1 cohort, surfaces a phenotype that passes bootstrap-Jaccard stability (≥0.6), passes age/sex confound control, passes the scanner gauntlet, and anchors to CDR severity (Fisher-z CI lower bound excludes 0). Verdict meter climbs to **SURVIVOR — candidate**.
4. **The Gladstone close — one card.** Surviving candidate + mechanism *hypothesis* (never fact) + the single falsifiable next experiment: *"replicate in ADNI/EPAD; anchor cluster membership to plasma p-tau217 (AD-specificity) vs NfL (generic neurodegeneration); confirmatory n and cohort named."*

**The wow line:**
> **"Watch it kill our best-looking result because it was scanner noise — then trust the one it lets through."**

**The honest claim boundary (verbatim, self-flagged in the demo):**
> "This is **not a new biomarker.** It is a **stability-vetted, scanner-invariant candidate imaging phenotype**, discovered unsupervised from frozen foundation-model embeddings, that survived a five-test falsification gauntlet and anchors to an independent severity axis. External replication and a plasma anchor are the stated next experiment — here is the exact cohort to run it."

**Red lines we flag ourselves:** candidate ≠ biomarker; symptomatic/MCI range only, not preclinical; not "better than plasma / replaces PET"; single-site labeled cohort stated explicitly; "internally stability-vetted, external replication is the next experiment," never "validated/reproducible"; mechanism always as *hypothesis*; effect sizes + bootstrap CIs, no p-value theater on n=8 cells.

**Fallback if no cluster survives all 5 tests on real data (this is the thesis, not a failure):** ship the *method* as the result — synthetic `tau_hot` phantom (ARI=1.0) as a labeled calibration positive control, the real OpenBHB scanner kill, and the real OASIS cohort with the referee *honestly rejecting* an underpowered candidate with the power math on screen and the exact cohort+n that would make it decidable. Wow line survives: *"…then trust the one it lets through (or trust it to tell you n isn't there yet)."*

---

## 3. DATA + COMPUTE PLAN

### GO / NO-GO: **GO on the honest novel-candidate demo, conditional on one afternoon of embedding work.**

At n=61 the referee's own math forces a KILL or null anchor (structurally underpowered — see power table). At n≈235 a moderate cluster↔severity anchor can survive. That is the difference between a demo whose hero is a *rejection* and one whose hero is a *survivor*. **The 61→235 embedding job is the pivot, and it costs ~1 CU.** The 61 embeddings we already hold are enough to execute the full demo honestly if the fetch stalls — do not let the stretch block the afternoon run.

### Datasets

| Dataset | n | Role | Status |
|---|---|---|---|
| OASIS-1 embeddings | 61 (36 CN / 17 MCI / 8 AD) | Disease-bearing discovery cohort (Act II) | ✅ on disk |
| OpenBHB embeddings | 96 (6-site, 2 field strengths) | Scanner-leakage gauntlet (Act I kill) | ✅ on disk |
| OASIS-1 `cross-sectional.csv` | 235 CDR-labeled | Anchor labels + embedding target list | ✅ on disk |
| **OASIS-1 raw (the ~174 gap)** | +174 → **~235** | The power injection (Tier-1 lever) | **Fetch Day 1** |

**Why OASIS-1 and not a new cohort:** we already hold CDR labels for all 235; we only lack their embeddings. Same site, same scanner, same label semantics — pure power injection, **no new confound, no new DUA, no cross-cohort harmonization.** OASIS-1 ships a T88-registered, skull-stripped, gain-field-corrected volume (`*_masked_gfc.img`), which is exactly what `neurojepa_embed.py` expects — **we skip FreeSurfer and MNI registration entirely.** Download → point manifest at `*_masked_gfc.img` → run the existing script.

**Deferred / named as "next experiment" only:** IXI (marginal over OpenBHB — skip); MIRIAD (second disease cohort, but different scanner needs harmonization we can't validate in 5 days — name as replication cohort, do NOT claim replication); ADNI/OASIS-3/NACC/EPAD (gated stubs — the honest confirmatory cohort, naming ADNI for the plasma p-tau217 anchor IS the Gladstone close).

**Data red line:** labeled cohort is single-site OASIS-1 (1.5T). Every claim says "confound-robust to the extent the 6-site OpenBHB test exercises it; the labeled cohort is single-site." No cross-scanner generalization claim on the disease cohort.

### Compute — UNIT BUDGET (of ~1799 CU)

Frozen inference, ~174 volumes, batch 1 — the workload is latency-trivial. **Use T4** (≈1.8 CU/hr); L4/A100/H100 are strictly wasteful. Do all download/preprocessing on CPU runtime or before GPU attach; attach T4 only for the inference loop; `colab stop` the instant `embeddings.csv` is downloaded.

| Task | GPU | GPU-hrs | CU |
|---|---|---|---|
| Embed 174 OASIS-1 (main run) | T4 | ~0.5 | **~1** |
| Re-run / debug buffer | T4 | ~1.0 | **~2** |
| (Stretch) embed MIRIAD ~46 | T4 | ~0.3 | **~1** |
| **Total planned burn** | | | **~4 CU (0.2% of 1799)** |

Even a 10× overrun is <2% of budget. **The constraint is wall-clock and correctness, not units** — reserve headroom, never reach for A100.

### Power (why Tier-1 is mandatory)

Anchor = Fisher-z CI lower bound must clear 0; SE(z) = 1/√(n−3).
- **n=61:** 95% half-width ≈ 0.257 → need r ≳ 0.25 to exclude 0; CDR-1 cell = 8 → clusters ~4–12 → bootstrap-Jaccard almost certainly < 0.6 → **dies at test 1.** Honest outcome: KILL or null.
- **n≈235:** half-width ≈ 0.129 → r ≳ 0.13 excludes 0; CDR cells 135/70/28 clear Jaccard 0.60. Moderate r≈0.3 at 80% power needs n≈85 — **235 clears comfortably, 61 does not.**

### Method spine (pre-registered, in order)

**Discovery:** first **residualize age + sex + ICV(eTIV) out of every embedding dimension** before clustering (else clusters recover aging, not disease). Reduce 768-d → PCA ~10–20 comps (UMAP for viz only, never statistics). Consensus across KMeans + GMM + HDBSCAN, k∈2–4. **Validation — the 5-test gauntlet (deterministic "Trusted" UDFs):** (1) bootstrap-Jaccard stability, kill if <0.60; (2) age/sex confound; (3) scanner-leakage double-dissociation, the live Act-I kill; (4) brain-age residual (regress out *predicted brain age*, not the gap); (5) severity anchor, Fisher-z CI lower bound excludes 0. **FDR control:** pre-register the single primary contrast (cluster↔CDR); everything else is exploratory; Benjamini–Hochberg across the k-sweep; report corrected q, not cherry-picked min. **No RL** — no environment, no non-circular reward; it would be capability theater the depth judges see through. SSL (frozen NeuroJEPA) is the substrate; SL (one linear probe) is the anchor + leakage instrument — deliberately linear (a linear probe leaking scanner is a *stronger* indictment).

---

## 4. HARNESS ARCHITECTURE

**The engine today is a *referee*. The harness turns it into a *discovery instrument a researcher drives from two sentences*.** The Databricks analogy is precise: `policy/` is the **governed semantic layer** — one source of domain truth consumed *both* by the deterministic "SQL-like" layer (gauntlet thresholds, routing tables) *and* by the AI layer (Claude's system prompt). Today those rules are scattered across `calibration.py` constants, `bridge._MECHANISMS`, `contract.VERDICT_BANDS`, `_client.REFEREE_SYSTEM`. Stage 2 externalizes them into declarative documents both consumers read.

| Layer | What it is | Status | Home |
|---|---|---|---|
| **L0 Frozen substrate** | NeuroJEPA 768-d embeddings, frozen inference (CC-BY-NC-ND) | exists | `contract.py`, `scripts/*_embed.py` |
| **L1 Deterministic domain** | 5-test gauntlet, leakage double-dissociation, confound leaderboard | exists | `gauntlet.py`, `leakage.py`, `calibration.py` |
| **L2 ML discovery/validation** | Supervised probe + unsupervised Detective + bootstrap-Jaccard | exists | `probe.py`, `detective.py`, `discovery.py`, `scoring.py` |
| **L3 Policy layer** ⭐NEW | Declarative domain-knowledge docs (the governed semantic layer) | **build** | `policy/` + `harness/policy.py` |
| **L4 Claude reasoning** | claim-parser, courtroom, bridge, reviewer, narrator | exists — **rewire to read L3** | `claude/*`, `_client.py` |
| **L5 Hypothesis entry point** ⭐NEW | `investigate("<hypothesis>", dataset)` → Experiment Card | **build** | `harness/orchestrator.py`, `cli.py` |

**Data flow:** hypothesis → ① structured Claim (claim_parser + `hypothesis_schema.yaml` + `novelty_rubric.md`) → ② discovery router (novel-pattern → Detective; named contrast → probe) → ③ referee (gauntlet + leakage + `confound_priors.yaml` + `verdict_rubric.md`) → ④ independent corroboration gate (biomarker anchor when available; leakage-clean replication when plasma is unavailable) → ⑤ Experiment Card (bridge + reviewer + narrator, all reading L3). **Claude sits at ①④⑤ as adjudicator, not coder; L1/L2 do the math; L3 is read by both.** Every Claude call keeps its existing offline deterministic fallback.

**Policy docs (Markdown + YAML frontmatter, matching the existing `SKILL.md` pattern — dual-consumption from one file):**

| File | Encodes | Code consumer | Claude consumer |
|---|---|---|---|
| `confound_priors.yaml` | Per-confound prior + flag threshold + mitigation | gauntlet thresholds, leakage interpretation | prosecution framing |
| `biomarker_routing.yaml` | Dominant-marker → mechanism → cohort/N/direction/kill | `bridge._route` | bridge prompt |
| `atn_framework.md` | AT(N)(+I) staging, molecular anchoring | anchor gate eligibility | bridge + anchor |
| `verdict_rubric.md` | Verdict bands, promotion floor, hedged-language rule | `contract.verdict_for` | `REFEREE_SYSTEM` |
| `novelty_rubric.md` | Candidate taxonomy + 5-rung honesty ladder | honesty guard | claim_parser + narrator |
| `hypothesis_schema.yaml` | Parse contract, discovery-mode selector, required Claim fields | claim_parser, discovery_router | claim_parser prompt |

**`harness/policy.py`** is the loader: `policy.table(...)` → dicts for deterministic code; `policy.thresholds(...)` → floats replacing magic numbers; `policy.brief(...)` → Markdown composed into Claude's system prompt. **Every accessor has a hardcoded fallback** to today's constants (`calibration.CAL`, `bridge._MECHANISMS`, `contract.VERDICT_BANDS`) — if `policy/` is absent or malformed, the demo runs byte-identically. This is how the offline-deterministic guarantee survives externalization.

**Repo additions:** new `src/neuroad/harness/{policy,discovery_router,orchestrator,experiment_card}.py`; new `policy/*.{md,yaml}` + README; new `skills/{hypothesis_intake,novelty_triage}/`. **Backward-compatible extensions:** `contract.py` gains 3 optional `ClaimCard` fields (`novelty_class`, `atn_profile`, `honesty_rung`), bump CONTRACT_VERSION → 1.1.0; `bridge/gauntlet/_client/claim_parser` rewired to read L3 with constant fallbacks; `cli.py` gains `investigate` subcommand; `app/` gains an Investigate panel.

---

## 5. STAGE-2 EXECUTION PLAN (Jul 8 → Jul 13, team ≤2)

**Two file-disjoint lanes: [A] Harness/Engineering, [B] Data/Biostats.** Flags: 🎯 demo-critical · 🔬 depth-critical · ✨ nice-to-have.

### Day 1 — Real data + policy foundation (unblock everything)
- **[B] 🎯 D1-DATA:** OASIS-1 raw fetch (`scripts/fetch_oasis1_raw.py` — diff 235-list vs 61 embedded, download only the gap `*_masked_gfc.img`) → build manifest (`subject_id, image_path, age, sex, cdr, dx, site, scanner`) → **T4 embed 174 via colab-gpu-cli** → concat 61+174 → dedupe on `participant_id` → `contract.validate_table`. **~1 CU, ½ day.** HF_TOKEN from env only; weights never written to repo (gated-weights-compliance).
- **[A] 🎯 T1:** Author 6 `policy/*.{md,yaml}` + `policy/README.md` (30-min frontmatter-schema agreement first).
- **[A] 🎯 T2:** `harness/policy.py` loader + accessors + **constant fallbacks** + `tests/test_policy.py` (missing-file → fallback).

### Day 2 — Honesty checkpoint + hypothesis entry point
- **[B] 🎯 D2-STATS:** Residualize (age+sex+ICV) + Detective on real ~235; run full 5-test gauntlet; **read out whether a cluster survives all 5.** This is the honesty checkpoint that decides hero-vs-fallback. A visible rejection is a feature, not a failure.
- **[A] 🎯 T3:** `contract.py` optional ClaimCard fields + `to_dict`; `harness/experiment_card.py`.
- **[A] 🎯 T4:** `harness/discovery_router.py` (supervised vs Detective) + unit test.
- **[A] 🎯 T5:** `harness/orchestrator.py investigate()` — parse→route→discover→`run_referee`→anchor→bridge→card. Integration point; depends T2/T3/T4.

### Day 3 — Wire policy into both layers (the "harness" story)
- **[A] 🔬 T6:** `bridge.py` reads `biomarker_routing.yaml` (fallback `_MECHANISMS`).
- **[B] 🔬 T7:** `gauntlet.py` reads `confound_priors.yaml` thresholds (fallback `CAL`) — file-disjoint from T6.
- **[A] 🔬 T8:** `_client.py` injects verdict/novelty briefs; `claim_parser.py` enriched Claim (novelty_class, expected_direction, kill_criterion) — **this is the "hypothesis parameterizes the run" close.**
- **[B] ✨ T9:** `skills/hypothesis_intake/` + `skills/novelty_triage/`.

### Day 4 — CLI, demo surface, honesty guard
- **[A] 🎯 T10:** `cli.py investigate "<hypothesis>" <dataset>` + report writing.
- **[B] 🎯 T11:** `app/build_demo_data.py` + `app/index.html` **Investigate panel** — the money shot: 2-sentence hypothesis → Detective surfaces candidate → gauntlet Act-I kill / Act-II survivor → biomarker anchor → experiment card. Wire the `discovery_real_oasis` block (currently `build_demo_data.py` never touches the disease cohort). Real is hero, synthetic `tau_hot` badged as labeled positive control.
- **[A] 🎯 T12:** Novelty honesty guard — assert every `ExperimentCard` carries `novelty_class` + `honesty_rung`; test that no rendered output ever says "proven/validated biomarker." The anti-overclaim contract.

### Day 5 — Tests, docs, hardening
- **[A+B] 🎯 T13:** `tests/test_harness.py` end-to-end `investigate` offline (no API key), router + policy-fallback coverage. Keep suite green (74 → ~85).
- **[A] 🎯 T14:** `docs/HARNESS.md` + `README.md` + `docs/DEMO_SCRIPT.md` with the investigate flow; red-line audit; offline demo pack.
- **[B] ✨ T15:** Stretch — embed MIRIAD ~46 (pure upside, cut first if slipping).

### Critical path & cut lines
- **Must ship:** D1-DATA, D2-STATS, T1, T2, T3, T4, T5, T10, T11, T12, T13, T14.
- **Ship if at all possible (the policy-harness narrative for Depth/Claude-Use):** T6, T7, T8.
- **First to cut under time pressure:** T15 → T9 → T11 polish (keep panel functional, drop animation).

---

## 6. DECISIONS THE USER MUST MAKE (before Stage 2 starts)

1. **Compute-spend authorization (needed today).** Authorize **~4 CU on Colab T4** (0.2% of ~1799) for the Day-1 embedding of the 174 OASIS-1 subjects, plus a debug buffer. This is the single gate that converts the demo hero from a *rejection* to a *survivor*. **Recommend: APPROVE — it is the highest-leverage, lowest-cost action available.**
2. **Dataset scope confirmation.** Confirm: (a) OASIS-1 only for the labeled cohort (no new gated DUAs attempted); (b) MIRIAD embedded *only* as Day-5 stretch and *never* claimed as replication; (c) ADNI/EPAD named as the confirmatory next-experiment cohort in the closing card. **Recommend: CONFIRM all three.**
3. **Hero-vs-fallback trigger.** Agree the Day-2 honesty checkpoint is binding: if a real cluster survives all 5 tests → real-survivor hero; if not → method-as-result fallback with power math on screen. Either path ships an honest, defensible demo. **Recommend: pre-commit to this rule now so no one relitigates it Day 4.**
4. **Parallel-commit / collision coordination.** The user is committing in parallel. Assign lanes by file ownership: **Lane [A]** owns `src/neuroad/harness/*`, `policy/*`, `contract.py`, `claude/*`, `cli.py`, `app/*`; **Lane [B]** owns `scripts/*`, `data/real/*`, `gauntlet.py`, `src/neuroad/detective.py`, the Colab work. Confirm which lane the user personally holds and which files are off-limits to the agents during that window. **Recommend: user takes [B] data/Colab (needs their credentials), agents take [A].**
5. **HF token handling.** Confirm `HF_TOKEN` lives in the Colab environment only, never in the repo (gated-weights-compliance). **Recommend: CONFIRM before any embed run.**

---

## 7. RISKS + MITIGATIONS

| Risk | Severity | Mitigation |
|---|---|---|
| **Overclaiming** ("new biomarker," "detects preclinical," "better than plasma," "validated/reproducible," causal mechanism as fact) | High — kills Gladstone credibility | Red lines self-flagged **in the demo**; T12 honesty guard asserts every card carries `honesty_rung` and blocks any "proven/validated biomarker" string; `novelty_rubric.md` 5-rung ladder stamps each output; mechanism always "*hypothesis*." |
| **Preprocessing tax stalls the fetch** | Medium | OASIS-1 `*_masked_gfc.img` is already T88-registered + skull-stripped → **no FreeSurfer, no MNI reg.** If it still stalls, the 61 embeddings already on disk execute the full demo honestly (method-as-result fallback). Do the download on CPU runtime before GPU attach. |
| **n=61 underpowered → no survivor** | Medium | Tier-1 embed to n≈235 clears the power threshold (~1 CU). If it fails, fallback ships the referee honestly rejecting an underpowered candidate with power math on screen — a quantified rejection with a named fix *is* the Gladstone contribution. |
| **Claude reads as decorative** (judge audit's Claude-Use gap) | Medium | `HypothesisSpec` **parameterizes the run end-to-end** (T8) — Claude picks target/contrast/confounds and pre-registers the kill criterion that actually drives thresholds. Not narration. |
| **Time overrun in 5 days** | Medium | Clear cut lines (§5): T15→T9→polish cut first; depth-critical T6/T7/T8 protected; demo-critical path is small and integration-tested by Day 5. |
| **Parallel-committer collisions** | Medium | File-disjoint lanes [A]/[B] (§6.4); every policy read has a constant fallback so a half-landed `policy/` file never breaks the demo; `demo` subcommand path untouched; CONTRACT_VERSION bump is backward-compatible. |
| **Compute waste on wrong GPU** | Low | T4 only; `colab stop` the instant results download; never leave a session warm; A100/H100 explicitly forbidden for frozen inference. |
| **Scanner leakage in the real result** (the field's central anxiety) | Designed-for | This *is* Act I — the scanner double-dissociation on 6-site OpenBHB is the live kill; a candidate that survives it is the whole point. |

**The whole ballgame:** convert the weakest axis — real discovery / Gladstone fit — into the demo's hero, using rigor we already have, on real embeddings we already hold, for ~4 compute units.
