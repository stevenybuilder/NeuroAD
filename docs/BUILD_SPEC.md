# NeuroAD Discovery Engine — Build Spec (contract-first)

Every module codes against `src/neuroad/contract.py` (frozen, v1.0.0). The
contract is the single interface; modules below are **file-disjoint** so they
build in parallel without collisions.

## The one-line product
An Alzheimer's structural-MRI referee: finds a candidate signal in brain MRI,
stress-tests it to throw out artifacts (age/sex → site/scanner leakage →
brain-age → biomarker anchor → replication), anchors survivors to a fluid
biomarker, and — only for what survives — names the likely mechanism and the
next experiment. **Imaging finds it. Proteins confirm it. It tells you what to do next.**

## Track: Builder ("Build Beyond the Bench")
Named user: a computational/translational AD researcher with imaging + partial
metadata who must decide *"is this signal worth a quarter of my time, or is it
scanner noise, aging, or atrophy in disguise?"* Working software they can run
without us in the room.

## The reused head (the whole architecture)
One small head maps a frozen embedding vector → a number. Point it at different
label columns (`LABEL_TARGETS` in the contract) → three of four tools:
- `conversion`/`dx_binary` → the signal
- `site`/`scanner` → the ⭐ leakage test (same code, different label column)
- biomarker regression → molecular anchor
- clustering (unsupervised) → the Detective / discovery entry point

## Modules (owners = parallel build agents)

### M1 — Core referee engine  (`src/neuroad/{probe,gauntlet,scoring,detective}.py`)
- `probe.py`: linear probe (logistic reg) + optional tiny MLP; cross-validated
  AUC; train/test split honoring `subject_id` (no leakage); helper to point the
  head at any `LABEL_TARGETS` column.
- `gauntlet.py`: the five tests. Each returns a `TestEvidence` (result + stats):
  age/sex adjustment (re-fit with covariates), **site/scanner leakage** (same
  head predicts scanner; compare AUC to outcome AUC), **brain-age control**
  (regress out embedding-derived brain-age, measure effect drop), biomarker
  anchor (correlate probe score / embedding with p_tau217/gfap on complete
  subset, report n), replication split (held-out site/cohort).
- `scoring.py`: assemble `ClaimCard` from tests via `contract.robustness_score`,
  `verdict_for`, `is_promoted`.
- `detective.py`: unsupervised phenotype discovery (KMeans + sklearn HDBSCAN;
  PCA 2-D coords for the UI). Every gauntlet test can run per cluster.

### M2 — Data layer  (`src/neuroad/data/`, `data/`, `scripts/download_*.py`)
- `data/registry.yaml`: dataset tiering with access + notation (download-now /
  gated / hardcode-pending). **Filled from research.**
- `scripts/download_open.py`: download the genuinely open tabular datasets.
- `src/neuroad/data/synthetic.py`: schema-matched synthetic cohort generator
  with an **injected conversion signal**, an **injected site/scanner confound**
  (so the ⭐ leakage test has something real to catch), realistic biomarker
  missingness, and a `brain_age` structure. Two presets:
  - `SURVIVOR` (DISEASE_LOAD high, SITE_COUPLE moderate) → partially robust.
  - `KILL` (DISEASE_LOAD low, SITE_COUPLE high) → mostly artifact, collapses.
- `src/neuroad/data/real.py`: adapter that maps a real open tabular dataset
  (e.g. OASIS demographics + structural-derived eTIV/nWBV/ASF) into a contract
  table, treating the structural-derived measures as the weight-free embedding.
- Gated datasets (ADNI/OASIS-3/NACC): **hardcode a small representative stub +
  clear NOTATION** that these are placeholders pending access; wire so a real
  file drop-in replaces the stub with no code change.

### M3 — Claude reasoning layer  (`src/neuroad/claude/`)
Product uses Claude as its reasoning/reviewer engine. Every module: live
Anthropic API when `ANTHROPIC_API_KEY` is set, **deterministic template
fallback** otherwise (demo must run fully offline). Structured tool-use I/O.
- `claim_parser.py`: NL hunch → structured `Claim` (target, groups, covariates).
- `narrator.py`: gauntlet results → plain-language verdict narration.
- `bridge.py`: survivors only → one biomarker-routed mechanism hypothesis + one
  falsifiable experiment + falsification criteria. Biomarker routing:
  amyloid+p-tau→amyloid-cascade; GFAP/weak-amyloid→neuroinflammatory/glial;
  NfL+WMH→vascular/axonal.
- `reviewer.py`: reviewer-agent that critiques the claim card for overclaiming,
  checks every number against the evidence ledger, flags coverage caveats.

### M4 — Visual demo UI  (`app/`)  ⭐ 30% of the score
Self-contained (offline) page: cohort card, a **live gauntlet checklist that
ticks through tests**, the claim card, and a **KILL vs SURVIVOR toggle**. Reads
JSON produced by the engine. Must look like real scientific software and read
well on a 3-minute video.

### M5 — Orchestration, docs, packaging  (`src/neuroad/{cli,pipeline}.py`, root docs)
- `pipeline.py`: end-to-end `run_referee(table, claim) -> ClaimCard`.
- `cli.py`: `neuroad demo`, `neuroad run <table> <claim>`.
- `README.md`, `docs/METHODS.md`, `docs/SUMMARY.md` (100–200 word submission),
  `docs/DEMO_SCRIPT.md`, reproducible `notebooks/referee_walkthrough.ipynb`.

## FINAL DELTAS (research-informed — build these in)

**Reposition (novelty is now published prior art):** cite arXiv:2604.14441 /
2606.09189 / PathoROB; own the *tool + closed loop + biomarker gate + Claude
adversary*. Never say "we discovered leakage." Never say "co-scientist."
Import citations/positioning from `src/neuroad/calibration.py`.

**Claude as ADVERSARY (M3, the Claude-Use score-mover):**
- `courtroom.py`: Prosecution subagent (argue artifact) + Defense subagent
  (argue real biology) — Claude argues *both sides* of the artifact-vs-biology
  question. The verdict itself is deterministic arithmetic (`robustness_score`),
  so the courtroom frames the tension while the verdict meter rules — no separate
  judge. Live API when `ANTHROPIC_API_KEY` set; deterministic template fallback
  otherwise.
- `reviewer.py` argues AGAINST the final verdict (proxy brain-age control,
  p-tau217 missingness, "partially robust ≠ robust").

**Removed: the Judge.** An earlier draft had a Judge subagent "render the verdict."
It was inconsequential — the verdict is fixed arithmetic and the Judge only
restated it — so it was dropped from the product and spec. Effort re-pointed to
the self-supervised discovery + clustering track (the Detective) below.

**Science/trust (M1):**
- Headline metric = subject-disjoint **leakage margin = outcome_AUC − scanner_AUC**.
- **Biomarker anchor is a hard GATE:** a promoted claim MUST show a p-tau217/GFAP
  correlation on the complete subset, else it cannot reach "robust enough".
- **Double dissociation:** residualize the embedding against a scanner-predicting
  direction; survivor still predicts outcome, kill collapses.
- **Confound leaderboard:** rank variance each confound (scanner, age, sex)
  explains in the signal.
- All numbers via `calibration.py` (`CAL`, `target()`), never free-floating.

**Real data (M2) — vendored + verified 2026-07-08, no login:**
- `data/real/oasis_longitudinal.csv` (OASIS-2): 150 subjects / 373 sessions.
  Cols: `Subject ID,MRI ID,Group,Visit,MR Delay,M/F,Hand,Age,EDUC,SES,MMSE,CDR,
  eTIV,nWBV,ASF`. Group ∈ {Nondemented 190, Demented 146, **Converted 37**}.
  → real conversion + diagnosis + longitudinal replication.
- `data/real/oasis_cross-sectional.csv` (OASIS-1): 436 subjects (235 CDR-labeled),
  ages 18–96. → second cohort + brain-age (wide age range).
- The weight-free "embedding" for OASIS = the structural-derived features
  `[nWBV, eTIV, ASF, (Age-derived), MMSE-free]` (do NOT leak MMSE/CDR into the
  probe features — those define the label). Standardize; treat as emb_0..emb_k.
- **Honest star on real data:** OASIS-1 & OASIS-2 are single-scanner → reframe the
  leakage test as **cohort/batch leakage** (predict OASIS-1 vs OASIS-2 membership).
  The *ground-truth scanner-confound* KILL lives in the synthetic harness.
- Biomarker anchor (p-tau217/GFAP): no open cohort has plasma markers → synthetic
  surrogate + "route to ADNI/EPAD" notation.
- Gated stubs: `data/real/_stubs/{adni,oasis3,nacc,epad}_stub.csv` + `STUBS.md`
  notation (clearly marked placeholder pending access; drop-in-ready).

**Demo UI (M4):** viewer over the REAL exported artifacts (reads
`reports/*.json|yaml`). Deterministic staged timeline. Substrate badge visible.
Reviewer(Claude) margin critique. KILL vs SURVIVOR split. See
`docs/RECONCILIATION_AND_ENHANCEMENTS.md` Part 3 and the demo choreography.

## Parallel build agent assignments
- **Agent 1 (core engine, M1):** `probe.py`, `gauntlet.py`, `scoring.py`,
  `detective.py`, `leakage.py` (margin + double-dissociation + leaderboard).
- **Agent 2 (data, M2):** `data/synthetic.py`, `data/real.py`, `data/loaders.py`,
  `data/registry.yaml`, stubs + `STUBS.md`, `scripts/download_open.py`.
- **Agent 3 (Claude layer, M3):** `claude/claim_parser.py`, `claude/courtroom.py`,
  `claude/narrator.py`, `claude/bridge.py`, `claude/reviewer.py`, `claude/_client.py`
  (API+fallback), gauntlet-stage SKILL.md pack under `skills/`.
- **Agent 4 (UI, M4):** `app/index.html` (self-contained) + `app/build_demo_data.py`
  (engine → demo JSON). Must render even before other modules land (mock JSON).
- **Agent 5 (orchestration+docs, M5):** `pipeline.py`, `cli.py`, `README.md`,
  `docs/METHODS.md`, `docs/SUMMARY.md`, `docs/DEMO_SCRIPT.md`,
  `BUILD_WITH_CLAUDE.md`, `notebooks/referee_walkthrough.ipynb`.

## Non-negotiables
- Demo runs offline with zero external access (synthetic harness guarantees it).
- No fabricated science: numbers calibrated to literature ranges (from research).
- Verdict language stays hedged (fragile/partially robust/…); biology speaks
  only about promoted survivors, each claim paired with its evidence.
- Everything open-source (MIT); NeuroJEPA weights used frozen (no derivative).
