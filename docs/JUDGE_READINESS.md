# NeuroAD — Judge-Readiness Handoff & Next Actions

**Purpose:** resume the submission push in a fresh session. **Deadline: 2026-07-13 21:00 ET.** Judging: Impact 25% · Claude-Use 25% · Depth 20% · Demo 30% + Gladstone "advance the field" award. Deliverables: 3-min demo video + open repo/notebook + 100–200 word summary.

> Multiple Claude sessions are working this repo in parallel. **Stage only explicit paths, never `git add -A`.** Lanes below say who owns what.

---

## 1. State (what's committed)

This session ("harness + demo" lane) landed 8 clean commits on `main`:
- L3 **policy layer** + hypothesis-driven **`investigate()` harness** (`src/neuroad/harness/`, `policy/`)
- Stage-2 blueprint (`docs/STAGE2_BLUEPRINT.md`), Colab runbook (`docs/COLAB_RUNBOOK.md`), build-learnings, reusable skills (`skills/{harness-build-learnings,director-agent-pipeline,policy-layer-fallback}/`)
- Honesty-vocabulary unification + honesty-guard hardening
- **Investigate hypothesis-entry + plan-out panel** in `app/index.html` (the money-shot: type a hunch → Claude parses a spec + pre-registered kill criteria + 5-rung honesty ladder **before** any math)
- Reproducible **`notebooks/investigate_walkthrough.ipynb`** (offline, 2-act SURVIVOR/KILL) — the repo/notebook deliverable ✅

Backend: frozen NeuroJEPA embeddings → probe + Detective → 5-test referee gauntlet → biomarker gate → experiment card. **133 tests green.** Offline demo is byte-identical (embedded `#embedded-data` == `app/demo_data.json`), deterministic, zero JS errors.

Parallel sessions also landed: an "honest reframe" recalibration, engine science (`probe.py`, `pipeline.py`, `leakage.py`), a **Cinematic Demo autopilot**, deploy setup (`Dockerfile`, `nginx.conf`), and ADNI/reproduce work.

---

## 2. VERDICT: engine ready, video NOT — 6 blockers

All small/localized. **B2 is a live credibility risk.** Lanes: **[ME]**=app/index.html+my src · **[DATA]**=demo_data.json/build_demo_data.py · **[DOCS]**=docs/ · **[AUTOPILOT]**/**[HERO-DATA]**=another session's active files.

| # | Blocker | Fix | Anchor | Eff | Axis | Lane |
|---|---------|-----|--------|-----|------|------|
| **B1** | Autopilot renders payoffs OFF-SCREEN (verdict slam, scatter morph, SURVIVOR mechanic, seed-fan all fire below fold at 16:9; `html.present{zoom:1.12}` worsens it). **Highest Demo leverage.** | `scrollIntoView({block:'center'})` on each beat's target el; reset scroll-to-top before Act II. | `index.html` ~5397 `AUTOPILOT`, :5367, :5384, :464 | S | Demo | **[AUTOPILOT]** |
| **B2** | **HERO OVERCLAIMS:** OASIS SURVIVOR prints "strong candidate"/**100** at honesty **rung 3** — score hits 100 only because renormalization DROPS the unrun biomarker gate. Violates own `novelty_rubric`+`verdict_rubric`. | Downgrade word to "robust enough for follow-up"; annotate meter "(4/5 tests; molecular gate unrun)". | `demo_data.json` `substrates.oasis.cases.SURVIVOR`; meter label in `index.html`; renorm in `scoring.py` | S | Impact/Gladstone | **[HERO-DATA]** |
| **B3** | Synthetic KILL reads "partially robust"/40 despite FAILED ⭐scanner-star (replication donates 15pts on a scanner-failed case). | Star-fail language cap → "fragile / likely artifact", OR feature the OASIS KILL (already correct: fragile/38) as the autopilot kill. | `demo_data.json` `substrates.synthetic.cases.KILL`; `scoring.py:99-101` | S | Impact | **[HERO-DATA]** |
| **B4** | Free-text Investigate: typed hypothesis Jaccard-snaps to a frozen case; plan-out never changes; numbers silently answer a different contrast → Claude looks decorative. | Re-derive visible spec (`target`, populations, novelty) from typed text (reuse `claim_parser._infer_target`/`_novelty_guess`), OR disclose "maps to nearest contrast: MCI→AD." | `index.html` ~4620 `doInvestigate`, :4559 `matchHypothesisToCase`, :4571 `renderPlanOut` | M | Claude+Impact | **[ME]** ✅ safe |
| **B5** | Unverifiable number `METHODS.md:60` ("margin ~0.16–0.37") flagged for deletion by own QA still ships. | Delete it. **Verify all 4 arXiv URLs resolve live before recording.** | `docs/METHODS.md:60`; `docs/CITATIONS_VERIFIED.md` | S | Impact | **[DOCS]** |
| **B6** | Script + live caption say "judge" but courtroom has none (prosecution+defense only). `DEMO_SCRIPT.md` stale (15 beats to t=180 vs shipped autopilot t=92). | Caption → "prosecution and defense — the meter is the judge"; reconcile script to shipped ~92s. | `index.html:5404`; `docs/DEMO_SCRIPT.md` | S | Demo | **[ME]**+**[DOCS]** |

---

## 3. High-ROI polish (after blockers)

1. **Add Investigate beat-0 + Detective beat to the autopilot** [AUTOPILOT] — type a hypothesis → hold ~4s on plan-out (pre-registered falsifiers) → then gauntlet; add `spotlightDetective()` after real-evidence (the "2/2 clusters = scanner, 0 promotable, ARI 1.0" climax). Also adds the ~90s the video needs to reach 3:00 (autopilot is currently ~1:32).
2. **Make Claude run live once** — rebuild with `ANTHROPIC_API_KEY` set, store real output per case, flip badge to `● LIVE CLAUDE`. On-camera proof Claude isn't decoration.
3. **Fix inverted number hierarchy in real-data panel** — narrated hero is `AUC 0.96` (Neuro-JEPA, 96 brains) but giant headline is `AUC 0.89` (structural, 3,984); CI `[0.91,1.00]` sits under 0.89. Make 0.96 dominant; split the t=34 caption so 3,984-structural isn't attributed to "the encoder."
4. **Cold-open count-up as hero** — collapse/scroll the Investigate panel in Act I so `#naiveBig` 0.50→0.87 is the first-6-seconds hero.
5. **Verdict-meter `70`/`85` label collision** — `.s70` overlaps `.s85 strong` → renders "70⁄85 strong" in every verdict frame (`index.html:229-231`).
6. **Policy-layer depth proof** — add a `tests/test_policy.py` case that mutates a doc threshold and asserts a verdict/rung changes (today it only proves the layer changes nothing). Surface per-threshold provenance in plan-out.
7. Close `openSplit` modal before `showExportTray`; surface KILL kill-criterion on the docket; hide/merge duplicate OASIS cases or badge substrate switch loudly.

---

## 4. Don't regress (already strong)

Offline/determinism guarantee (embedded==demo_data.json byte-identical; any DATA rebuild must re-pass this + 133 tests + no `Math.random`/`Date`/external fetch). Honesty framing (SYNTHETIC HARNESS / OFFLINE TEMPLATE badges, CC-BY-NC-ND line, 10–13 caveats/card, self-adversarial reviewer, biomarker HARD GATE). Individual visuals (verdict panel, Detective, real-evidence, KILL-vs-SURVIVOR split) are demo-grade. Claude architecture is genuinely deep — score it high.

---

## 5. n=61 → 235 decision: **RECORD BEFORE it lands**

The video hero (OASIS AD-vs-CN **structural**, n=263) does NOT depend on the n=61 embedding cohort. At n=61 the real-embedding path is underpowered → can only honestly KILL. Landing 235 (via `docs/COLAB_RUNBOOK.md`, ~4 CU) *could* flip a real-embedding case to a survivor (the dream climax) but may still KILL. Treat as **re-record-only-if** it lands clean by ~July 12, yields a promotable survivor, rebuilds deterministically, and re-passes every gate. Otherwise ship the current cut. Do not gate the deadline on a late Colab fetch.

---

## 6. Recommended next actions (in order)

1. **[ME/safe] B4** — make the plan-out respond to typed hypothesis text (only clearly-uncontested blocker).
2. **[COORDINATE] B1 + B2** — the two highest-value fixes, but in the autopilot + hero-data sessions' files. Assign an owner before editing; B2 is the honesty must-fix.
3. **[DOCS] B5 + B6** — delete the flagged number, verify arXiv URLs, fix the "judge" caption, reconcile `DEMO_SCRIPT.md` to the shipped autopilot.
4. **Polish** #1–#3 above (autopilot beat-0 + Detective; live-Claude; number hierarchy).
5. **Write the 100–200 word summary** (submission requirement — not yet drafted).
6. **Record** the 3-min video (autopilot + VO), then decide on the 235 re-record.

**Full source reviews** (4 judge lenses + synthesis): workflow `wf_023c7d62-c95` transcript. **Blueprint:** `docs/STAGE2_BLUEPRINT.md`. **Notebook:** `notebooks/investigate_walkthrough.ipynb`.
