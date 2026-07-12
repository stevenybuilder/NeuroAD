# Overclaim Audit — Summary + Artifacts result-card tabs (`app/neuroad.html`, commit `bcca31c`)

**Auditor:** overclaim-auditor · **Date:** 2026-07-12 · **Scope:** ONLY the new
`summaryEl` / `artifactsEl` drawer tabs and their helpers (`_valence`, `_aucStat`,
`_confoundStat`, `_anchorStat`, `_pipelineTrail`, `_trailEl`, `_reasoningBullets`,
`_datasetRows`, `discoveryFigure`) in `app/neuroad.html`. Static code + `app/demo_data.json`
audit; browser not driven (another session owns the tab).

**Headline:** The number-binding is mostly clean — the helpers read real fields off
`heroCase()` and no CAL target is laundered. But there are **two real honesty defects**:
(1) the "Driving analysis" figure renders a **synthetic planted-cluster recovery run**
(`discovery`, tau_hot *promoted*, ARI=1.0) as if it were the real unsupervised discovery,
while the actual real-cohort discovery (`discovery_real`) found **0 promotable phenotypes**;
and (2) on a **killed/pruned node**, both tabs still show the **SURVIVOR's** AUC / Confounds /
Anchor chips and emit a comparative bullet with the survivor's *positive* margin under
"Branch killed" framing — an internal contradiction that reads as "the killed branch scored
AUC 0.93."

---

## Findings, ranked by severity

### F1 — SYNTHETIC-AS-REAL · `discoveryFigure` renders the planted-recovery run as "Driving analysis" — HIGH
- **File:line:** `app/neuroad.html:2722-2723` (binds `data.discovery`), rendered under section
  `"Driving analysis"` at `:2813`; label `"Discovery clustering · unsupervised"` at `:2741`.
- **As rendered:** three tidy clusters — **tau_hot · n=150 · AUC 0.70 · promoted**, age_atrophy
  (flagged), scanner_artifact (flagged); footer `kmeans · k=3 · sil 0.063`; note text includes
  "Recovery vs planted ground truth: ARI=1.0, AMI=1.0."
- **Evidence:** `demo_data.json` has **two** discovery blocks.
  - `discovery` (the one the figure binds to): `silhouette 0.063`, `k=3`, **`ari 1.0 / ami 1.0`**,
    three clusters of **exactly n=150 each**, `tau_hot` `promoted:true`. The `note` literally says
    *"Recovery vs planted ground truth."* Balanced n, a perfect ARI vs *planted* ground truth, and
    named phenotypes are the signature of a **synthetic planted-cluster benchmark**, not a cohort discovery.
    (Tag: my-inference, high confidence.)
  - `discovery_real`: `silhouette 0.078`, `k=2`, no ARI/AMI, ragged n (19, 77), **0 promotable**,
    both clusters flagged as scanner/site artifacts, no phenotype names. This is the real-cohort
    result — and it found nothing promotable. (Tag: asserted-by-data.)
- **Why it's a problem:** the hero case is `adni:combat` (real). The Artifacts tab's "Driving analysis"
  should show what actually drove the real cohort. It instead shows a synthetic run where `tau_hot`
  **is promoted**, contradicting the honest real result (0 promotable). The rendered `note` does say
  "planted ground truth," which is a partial hedge, but the framing ("Driving analysis" / "unsupervised
  discovery" / a promoted phenotype with an AUC) presents tier-2 synthetic output as the engine's real
  discovery. This is the project's #1 forbidden move (synthetic-as-real).
- **Verdict:** **SYNTHETIC-AS-REAL.**
- **Minimal fix:** bind `discoveryFigure` to `data.discovery_real` (`app/neuroad.html:2723`:
  `data&&data.discovery_real`). If the synthetic recovery run must stay visible, badge it explicitly
  (e.g. `SYNTHETIC · planted-cluster recovery`) and drop the "Driving analysis" header — do not label it
  unsupervised *discovery* of the cohort.

### F2 — MISLEADING · killed/pruned node shows the SURVIVOR's chips + a false comparative bullet — HIGH
- **File:line:** `summaryEl` chips `app/neuroad.html:2761-2765`; helpers `_aucStat` `:2642`,
  `_confoundStat` `:2647`, `_anchorStat` `:2651`; comparative bullet `_reasoningBullets` `:2699-2701`.
  Reachable: gray/kill nodes (`:1749` `story:'kill'`, `:1754` `story:'kill2'`, and `role:'gray'`
  nodes) pin through `buildTabs` (`:2999`) → `summaryEl`/`artifactsEl`.
- **As rendered (on a killed node):** verdict banner correctly reads **"Branch killed / Pruned from
  candidate generation — kept visible for audit"** and the score badge is suppressed — but the three
  chips still show **AUC 0.93 [0.91–0.94] · Confounds 4/5 · Anchor 0.99**, and the top "Why killed"
  bullet reads *"The scanner-only classifier (AUC 0.37) matches or beats the hypothesis (AUC 0.92) —
  margin 0.55 [0.52–0.58]."*
- **Evidence:** `heroCase()` (`:1841`) never returns a KILL case — it returns `card.case` or
  `substrates[…].cases.SURVIVOR`. So `_aucStat/_confoundStat/_anchorStat` and the bullet's
  `leakage_margin` always describe the **SURVIVOR** (adni SURVIVOR: outcome_auc 0.922, scanner_auc 0.374,
  **margin +0.549**). `_valence` (`:2636`) flips only the *framing* off the pinned node, not the data.
  The killed bullet template says the confound *"matches or beats"* the hypothesis, but it is fed a
  **+0.549** margin where outcome (0.92) clearly beats scanner (0.37) — a self-contradictory statement.
- **Why it's a problem:** a viewer pinning "Scanner-site artifact / killed" sees strong survivor stats
  (AUC 0.93, 4/5 confounds cleared, anchor 0.99) attached to a *killed* branch, plus a bullet whose
  numbers contradict its own "matches or beats" claim. Reads as "the killed branch scored AUC 0.93."
- **Verdict:** **MISLEADING** (the comparative bullet is additionally **factually inverted**).
- **Minimal fix:** on `val.killed`, suppress the AUC/Confounds/Anchor chip row and the comparative
  bullet entirely (they belong to the survivor), OR gate them behind `!val.killed` the same way the
  score badge already is (`scoreTxt` at `:2755`). At minimum, do not render the "matches or beats"
  bullet when `heroCase().leakage_margin.margin >= 0`.

### F3 — HEDGE-NEEDED · adni "strong candidate" / **100/100** surfaced verbatim — MEDIUM
- **File:line:** `summaryEl` verdict `app/neuroad.html:2754` (`c.verdict`), score badge `:2755,:2773`.
- **As rendered:** verdict banner **"strong candidate"** + score **"100/100"** for the promoted node.
- **Evidence:** `demo_data.json` `substrates.adni.cases.SURVIVOR`: `verdict:"strong candidate"`,
  `score:100`. Per the project's own `policy/verdict_rubric.md:13,48` the word "strong candidate" is
  the valid band for score 85–100, so — unlike the OASIS instance `JUDGE_READINESS.md` B2 describes —
  the *word* here is rubric-consistent (adni's biomarker_anchor gate is **run and passed**, effect 0.99).
  **However**, the perfect **100** is reached by the same NA-drop mechanism B2 names: the STAR
  `brain_age` test is `not_available`, and per `verdict_rubric.md:64` *"NA dimensions are dropped from
  both numerator and denominator"* — so a STAR gate is unrun-and-dropped, and the four remaining tests
  all pass → 100. `JUDGE_READINESS.md:31` (B2): *"HERO OVERCLAIMS … 'strong candidate'/**100** … score
  hits 100 only because renormalization DROPS the unrun … gate. Violates own novelty_rubric+verdict_rubric.
  Fix: … annotate meter '(4/5 tests; molecular gate unrun)'."* Also note `replication:"passed"` here is
  an **internal held-out split**, which `novelty_rubric.md` (rung 5) says *does NOT* count as external
  replication — the verdict does not claim external replication, so this is a caveat, not a violation.
- **Mitigation already present:** the Confounds chip (`:2763`) does show **"4/5 · 1 n/a"** in the same
  drawer, so the unrun test is disclosed nearby — the UI is more honest than raw B2 feared.
- **Verdict:** **HEDGE-NEEDED** (not a clean rubric violation, but a renorm-inflated perfect 100 on a
  single frozen cohort surfaced as the headline).
- **Minimal fix:** annotate the score badge inline, e.g. `100/100` → `100/100 · 4/5 tests` or append
  "(brain-age gate n/a)", so the perfect score is not read as full-gauntlet certainty.

### F4 — HEDGE/PRECISION · AUC point estimate and CI come from two different computations — LOW
- **File:line:** `_aucStat` `app/neuroad.html:2645`; reasoning bullet `:2700`.
- **As rendered:** headline AUC chip **0.93 [0.91–0.94]**; reasoning bullet **"Outcome AUC 0.92 beats …"**.
- **Evidence:** the chip's point estimate is `naive_effect.value = 0.932`, but its CI is
  `leakage_margin.outcome_ci = [0.906, 0.938]` — the CI of `leakage_margin.outcome_auc = 0.922`, which is
  also the 0.92 the bullet prints. So the headline shows 0.93 with a CI computed for 0.92, and the same
  drawer shows both 0.93 and 0.92 for "the AUC." Both are real numbers (naive-fold vs leakage-eval fold),
  but the mix reads as a small inconsistency a careful judge would question.
- **Verdict:** **HONEST but imprecise.**
- **Minimal fix:** pair the CI with its own estimate — show `leakage_margin.outcome_auc` (0.92) with
  `outcome_ci`, or drop the CI from the naive-value chip.

### F5 — HONEST · the comparative "beats scanner-only" bullet (promoted path) — PASS
- **File:line:** `_reasoningBullets` `app/neuroad.html:2696-2701`.
- **As rendered (promoted):** "Outcome AUC 0.92 beats the scanner-only classifier (0.37) — margin 0.55
  [0.52–0.58]."
- **Evidence:** `leakage_margin` = outcome_auc 0.922, scanner_auc 0.374, margin 0.549,
  margin_ci [0.522, 0.575] — all real, correctly compared and rounded. This is the genuine
  double-dissociation result and is the right thing to lead with.
- **Verdict:** **HONEST.**

### F6 — HONEST · "REAL" dataset badges and coverage — PASS (one minor n note)
- **File:line:** `_datasetRows` `app/neuroad.html:2710-2717`; badge render `:2809`.
- **As rendered:** "REAL ADNI · 2,951 subjects · 72 sites · 2 scanners" [REAL]; "plasma p-tau217 · 47%
  measured coverage" [REAL].
- **Evidence:** `cohort.badge="REAL ADNI"`, `n_subjects=2951`, `biomarker_coverage.p_tau217=0.467` — real,
  gated ADNI. `src='real'` is keyed off `co.badge`, correctly. **Minor:** the row shows the full-cohort
  n (2,951) while the hero AUC was computed on the 3T SURVIVOR subset (`naive_effect.n = 1615`); a judge
  could ask why the dataset says 2,951 but the AUC chip says n=1615. Not an overclaim — the Datasets
  section is about the source cohort — but a one-line "(3T subset n=1,615 analyzed)" would preempt it.
- **Verdict:** **HONEST.**

### F7 — HONEST · pipeline-trail lit layers are derived, not asserted — PASS
- **File:line:** `_pipelineTrail` `app/neuroad.html:2659-2676`.
- Layers L3–L6 light only from real case fields (`tx.cross_attention_fusion`, `ranked_targets`,
  `structure/repurposing`, `ranTests`), and L5/L6 are suppressed on killed nodes. No fabricated
  "lit" state. **HONEST.** (Note: L1 "JEPA" / L2 "Probe" are hardcoded `lit:true` regardless of case,
  but that reflects the fixed pipeline topology, not a per-case claim.)

---

## Provenance ledger

| Number (as rendered) | Source field | Tier | Presented at true tier? |
|---|---|---|---|
| AUC 0.93 chip | `adni SURVIVOR.naive_effect.value` 0.932 | 1 real | Yes (see F4 CI note) |
| CI [0.91–0.94] | `leakage_margin.outcome_ci` (of 0.922) | 1 real | Mixed with 0.932 (F4) |
| Confounds 4/5 · 1 n/a | `tests[].result` | 1 real | Yes |
| Anchor 0.99 | `tests[key=biomarker_anchor].effect` 0.992 | 1 real | Yes |
| "Outcome 0.92 beats scanner 0.37, margin 0.55" | `leakage_margin.*` | 1 real | Yes |
| verdict "strong candidate" / 100 | `case.verdict` / `case.score` | 3 frozen replay | Verbatim; renorm-inflated 100 (F3) |
| Driving-analysis fig: tau_hot promoted, AUC 0.70, k=3, sil 0.063, ARI 1.0 | `discovery` (planted-recovery) | **2 synthetic** | **NO — shown as real discovery (F1)** |
| Killed-node AUC 0.93 / 4/5 / 0.99 chips | `SURVIVOR` (not the killed case) | 1 real but WRONG NODE | **NO — attached to killed branch (F2)** |
| "REAL ADNI · 2,951", p-tau217 47% | `cohort.badge/n_subjects/biomarker_coverage` | 1 real | Yes (F6 n note) |

No CAL-table target (`calibration.py`) is surfaced by these tabs. No `FORBIDDEN_WORDS`/bigrams
("proven/validated/breakthrough/…") appear in the rendered strings.

---

## Verdict + single most important fix

The number *plumbing* is honest — these helpers read real `heroCase()` fields and avoid CAL/fabricated
values. The two defects are **provenance/framing**, not fabricated digits: a synthetic planted-recovery
run is dressed as the real "Driving analysis" (F1), and a killed node borrows the survivor's strong
chips and a self-contradicting bullet (F2). Both make a *negative/killed* result read as a *positive*
one — the exact direction of overclaim this project polices.

**Highest-leverage fix:** F1 — point `discoveryFigure` at `data.discovery_real` (or badge `discovery`
as SYNTHETIC planted-cluster recovery and drop the "Driving analysis / unsupervised discovery" framing).
The real cohort found **0 promotable phenotypes**; showing a synthetic run where `tau_hot` *is promoted*
under a "Driving analysis" header is the one finding a hostile judge would call fabrication.

*Epistemic tags: F1 `discovery`-is-synthetic = my-inference (high confidence: planted ground truth,
balanced n=150, ARI 1.0); `discovery_real`-is-the-real-result = asserted-by-data. F2/F3/F4/F6 = verified
against `demo_data.json` + `policy/*.md`. No live pipeline was run; all `case.*`/`discovery*` values are
frozen-replay (`demo_data.json`), not proof of live-engine output.*
