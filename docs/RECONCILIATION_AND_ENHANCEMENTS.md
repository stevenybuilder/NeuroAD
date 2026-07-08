# NeuroAD Discovery Engine — Reconciliation, Criteria Audit & Enhancement Plan

**Date:** 2026-07-08 · Grounded in a 5-scout research pass (judging, novelty,
datasets, scientific facts, demo) + the hackathon Participant Guide.

---

## Part 1 — Reconciling the five source docs

| Doc | Role | Verdict |
|---|---|---|
| `neuroad_referee_master.md` | Canonical brief (structural track) | **Governs.** Source of truth. |
| `neuroad_referee_combo.md` | Detective+Referee fusion (functional/fMRI framing) | Earlier evolution; its "Referee deep, Bridge narrow" discipline is **kept**. |
| `ad_fmri_discovery_referee_idea.md` | First idea (fMRI claim-referee) | Earlier evolution; **fMRI demoted to a documented future "functional track."** |
| `neuroad_bridge_strategy.md` | Positioning + competitive map | Kept, **except** its 6-artifact biology suite — superseded by the narrow bridge. |
| `ideation_skill.md` | The method | Used to run the audit below. |

**Two conflicts resolved:**
1. **Bridge breadth.** `bridge_strategy` wants a six-artifact biology suite;
   `master`/`combo` warn that is exactly how the biology step becomes shallow.
   → **Narrow bridge:** one survivor → one mechanism → one falsifiable experiment.
2. **Modality.** `bridge_strategy`/`ad_fmri` lean fMRI + Brain-JEPA; `master`
   locks **structural-only** for the MVP (Neuro-JEPA is a *structural* model).
   → **Structural MVP;** fMRI is a named future track (and a real "pushed past
   the first idea" depth story).

The docs are a clean evolution, not a set of contradictions — that lineage
(fMRI referee → Detective+Referee combo → structural master) *is* the
"Depth & Execution" narrative.

---

## Part 2 — Audit against the actual judging criteria

Weights from the Participant Guide: **Demo 30% · Impact 25% · Claude Use 25% ·
Depth & Execution 20%.** Track: **Builder** ("build the tool a named scientist
is missing — working software they could use without you, built to outlast the
week").

| Criterion | Master brief today | Gap | Fix |
|---|---|---|---|
| **Impact 25%** | Strong — "save months chasing artifacts," fits Builder statement | minor | Name the user (Gladstone/UCSF ML-genomics PI); ship one real KILL finding |
| **Claude Use 25%** | **Weak** — Claude only in the thin biology bridge | **big** | Claude as **adjudicator** (prosecution/defense/judge), reviewer-agent self-audit, gauntlet-as-Skills, one-command CLI, multi-agent build story |
| **Depth 20%** | Good — dual substrate, 5-doc evolution | minor | Verified calibration; double-dissociation control |
| **Demo 30%** | **Weak** — CLI/card, not "cool to watch" | **big** | Self-contained visual **workbench UI** that is a *viewer over the real exported artifacts* |

The two big gaps (Claude Use, Demo) are **50% of the score** and are exactly
where the master brief was thinnest. The enhancements below target them.

### Novelty correction (blunt — this is the important one)
Running the ideation skill's "does this already exist?" test against July-2026
literature: **the leakage insight itself is now published prior art** —
*Batch Effects in Brain Foundation Model Embeddings* (arXiv:2604.14441),
*Pretrained, Frozen, Still Leaking* (arXiv:2606.09189), *PathoROB* (Nat. Commun.).
Pitching "we found that embeddings leak scanner" gets a polite nod and a dock.

**Reposition:** cite the prior art openly; own the parts nobody shipped —
1. the **runnable, agent-orchestrated referee** (one command, a named scientist can run it),
2. the **closed loop**: verdict → biomarker-routed mechanism → ONE falsifiable experiment,
3. the **biomarker anchor as a hard GATE** (no p-tau217/GFAP correlation → does not survive),
4. **Claude as the adjudicator**, not just the coder.
Kill the words "co-scientist" and "discovery platform" (Biomni-AD / PARTHENON's
turf — they won $2M there). Own **referee / auditor / red-team / gauntlet.**

---

## Part 3 — Enhancements we are building (the polish list)

**Claude Use (the score-movers):**
- **Courtroom adjudication.** Two Claude subagents — *Prosecution* (argue the
  signal is an artifact) and *Defense* (argue it is real biology) — then a
  *Judge* agent renders the verdict with reasoning. Every agent makes a
  **consequential** decision (defeats the "Claude is decoration" risk).
- **Reviewer agent that argues against its own verdict** — peer-review critique
  flagging the proxy brain-age control, p-tau217 missingness, "partially robust
  ≠ robust." A referee that referees itself.
- **Gauntlet stages as drop-in Agent Skills** + a one-command `neuroad` CLI /
  `/referee` slash command → "built to outlast the week."
- **`BUILD_WITH_CLAUDE.md`** documenting the multi-agent Claude Code build.

**Demo (30%):**
- **Self-contained visual workbench** (offline HTML): masthead + honest
  *substrate badge*, left-rail **cohort card**, center **docket + embedding
  figure**, right-rail **live gauntlet checklist** that ticks queued→running→
  result, a filling **claim card**, a **KILL vs SURVIVOR split**, and a
  **Reviewer (Claude)** margin critique. Deterministic timeline (identical every
  take). It is a **viewer over the real exported artifacts**, not a mockup.

**Science / trust:**
- **Biomarker anchor = hard gate.** Headline metric in the frontier's currency:
  subject-disjoint **(outcome-AUC − scanner-AUC) leakage margin**.
- **Double dissociation:** scrub the scanner signal from the embedding — the
  SURVIVOR still predicts the outcome; the KILL collapses.
- **Confound leaderboard:** rank how much each confound (scanner, age, sex)
  explains the signal, so the scientist sees which artifact to fix.
- **Calibrated numbers only** (`calibration.py`): diagnosis AUC ~0.89, conversion
  AUC ~0.74, site-leakage AUC ~0.92 (kill) / ~0.64 (survivor), p-tau217 r ~0.43,
  brain-age R² ~0.85 + MAE ~3 yr (softened from the brief's 0.89), effect
  retained ~80% (survivor) / ~25% (kill). Neuro-JEPA spelled hyphenated.

**Data (real-first, honest):**
- **Real, vendored into the repo (curl-verified 2026-07-08, no login):**
  OASIS-2 longitudinal + OASIS-1 cross-sectional CSVs — real structural-derived
  features (eTIV, nWBV, ASF) + real labels (CDR, MMSE, **Converted**). Gives a
  genuine **AD-vs-CN diagnosis + conversion + brain-age + replication** demo on
  real data. *Honest caveat:* both are single-scanner, so the real "star" is
  reframed as **cohort/batch leakage** (OASIS-1 vs OASIS-2 as pseudo-sites).
- **Synthetic harness** carries the *ground-truth* scanner-confound KILL and the
  p-tau217 biomarker anchor (no open cohort has plasma markers) — and is the
  guaranteed offline live path.
- **Gated (ADNI / OASIS-3 / NACC / EPAD):** hardcoded stub + clear NOTATION,
  drop-in-ready (real file replaces stub, zero code change).

---

## Part 4 — What we are explicitly NOT building

- No fMRI/functional track (documented future work — Neuro-JEPA is structural).
- No six-artifact biology suite (keeps the bridge honest and narrow).
- No "co-scientist" / "discovery platform" framing (crowded, funded, off-wedge).
- No hard dependency on Neuro-JEPA weights (weight-free structural features are
  a first-class feeder; the contract makes the encoder swappable).
- No UMAP/HDBSCAN-only dependencies that risk the build (PCA + sklearn suffice).
- No live-API dependency in the demo path (template fallbacks keep it offline).

---

## Part 5 — One-line positioning (final)

> **NeuroAD Discovery Engine** — an Alzheimer's structural-MRI **referee** that falsifies
> embedding-derived findings against scanner leakage, demographics, brain-age
> and replication, **gates** the survivors behind a plasma-biomarker anchor, and —
> for what's left standing — has **Claude adjudicate** the likely biology and the
> one next experiment that would confirm or kill it. Cites the batch-effect prior
> art; ships the tool nobody else did.
