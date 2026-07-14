# NeuroAD — Canonical Facts for the Animations (use these EXACT names/numbers)

The one-liner: **"Imaging finds it. Proteins confirm it. The system says what to test next."**
Positioning: **"An AlphaFold for Alzheimer's neuroimaging."**
Core idea: build ONE probe on a FROZEN brain-MRI foundation model, ask it THREE questions, and
treat the DISAGREEMENT between the answers as the discovery. Everything downstream is a filter
that only lets real biology through. Then: **kill weak hypotheses, surface strong ones, and turn
each survivor into ONE ranked protein target + ONE falsifiable wet-lab experiment.**

## The 7 pipeline stages (for scene1_pipeline — condense to 6–7 cards)

- **STAGE 0 · INPUT** — "Raw signal." Multi-sequence MRI (T1w, T2w, FLAIR) + tabular clinical data
  (cognitive tests, plasma p-tau217, demographics). Cohorts: ADNI · OASIS · EPAD (EPAD multi-site
  enables the scanner-leakage test). Glyph idea: brain / layered scans. State/neutral color: ink/gray.
- **STAGE 1 · FROZEN FOUNDATION** — "The eyes." Neuro-JEPA, a 3D vision transformer pre-trained on
  **1.5M+ brain scans**, kept FROZEN (never retrained). Each scan → one latent structural
  embedding vector. No labels, no fine-tuning → cheap, generalizes. Metric: `1.5M scans · frozen`.
  Glyph: snowflake/lock (frozen) or nested squares. Color: clay-ish neutral (it's the given model).
- **STAGE 2 · ONE PROBE, THREE QUESTIONS** — "The science." A single lightweight attentive-MLP
  probe on the frozen embedding (+ FastSurfer hippocampal/cortical volumes). Same head, swap the
  label column: Disease (AD-vs-CN / conversion), Scanner (site ID), Protein (p-tau217 / GFAP);
  plus unsupervised k-means/HDBSCAN clustering to surface unlabeled subgroups. Color: gold (the
  active, generative step). Glyph: three-way split / prism.
- **STAGE 3 · THE REFEREE** — "Kill weak, surface strong." Every candidate signal must survive a
  falsification gauntlet: age/sex adjustment, site/scanner leakage, brain-age & atrophy control,
  held-out replication split, biomarker anchor. Verdict bands: fragile → partially robust →
  robust enough for follow-up → strong candidate. Only "partially robust" and up pass. Color:
  split green(pass)/gray(killed). Glyph: shield/filter/gate.
- **STAGE 4 · BIOMARKER BRIDGE** — "Turn a signal into a hypothesis." Does the surviving imaging
  signal align with an INDEPENDENT molecular axis? Whichever biomarker it tracks ROUTES the
  mechanism: tracks p-tau217 → amyloid/tau (primary); GFAP → glial inflammation; NfL/WMH →
  vascular/axonal; none → "unanchored, imaging-only." Kill rule: if p-tau217 correlation r < 0.2,
  the amyloid/tau claim dies. Color: green. Glyph: bridge / routing arrows. Metric: `r ≥ 0.2`.
- **STAGE 5 · MULTI-OMICS TARGETING (PI4AD)** — "The ranking." Survivors only. PI4AD logic
  (open-source): multi-omic scoring over GWAS + QTL/PCHi-C + STRING networks, Random-Walk-with-
  Restart propagation, self-organizing-map clustering, PCST pathway-crosstalk → nodal hub genes
  (e.g. Ras hubs HRAS / MAPK1). Output: candidate proteins RANKED 0–10. Calibration: it recovers
  known targets (APP, ESR1). Color: green/clay. Glyph: network graph. Metric: `ranked 0–10`.
- **STAGE 6 · MOLECULAR STRUCTURE (AlphaFold 3 / Boltz)** — "Grounded in 3D." Top candidates get
  3D structure/interaction modeling (Aβ oligomers, tau filaments, APP complexes) to ground
  compound & repurposing decisions. Metric: `pLDDT`. Glyph: folded-protein ribbon / helix.
- **STAGE 7 · OUTPUT: ONE EARNED EXPERIMENT** — "Not a dashboard." For each survivor: ONE
  biomarker-routed mechanism hypothesis → ONE ranked protein target → ONE falsifiable wet-lab
  experiment in a fast human model (iPSC / 3D "Alzheimer's-in-a-dish" organoids, **~6 weeks vs
  decades**). Every sentence paired to its artifact + protein evidence. Color: clay (the payoff).
  Glyph: flask/target. Caveat: "Decision-support candidate — a hypothesis to test, not a
  validated target."

## The flywheel cycle (for scene2_flywheel) — 6 stations around the ring

1. **Hypothesis** (typed / Claude-proposed) → 2. **Falsify** (5-test referee gauntlet) →
3. **Survivor** (verdict: partially-robust+) → 4. **Route** (biomarker bridge names the mechanism)
→ 5. **Rank** (PI4AD → protein target 0–10 + AlphaFold structure) → 6. **Experiment** (one
falsifiable iPSC/organoid test) → feeds a **sharper Hypothesis** (loop closes; each survivor
tightens the next question). Hub label: **"Kill weak · surface strong."**

## Claude orchestration (for scene3_claude)

Claude (Opus) is the harness that ties stages together via tool-calling. TOOLS IN (left):
Neuro-JEPA encoder (embeddings), clinical/biomarker table, the probe + referee statistics,
PI4AD multi-omics, AlphaFold 3 / Boltz structure, literature / RAG search. ARTIFACTS OUT (right):
ranked protein target, falsifiable experiment card, and a full provenance/audit trail (every
claim paired to evidence; real-vs-synthetic + live-vs-offline badges). Design contract that makes
it swappable: one cached-embedding table is the interface every downstream piece reads
(`subject_id, embedding[d], dx, conversion, age, sex, site, scanner, amyloid, p_tau217, gfap,
nfl, apoe`). Build-time caption: "Built in days with Claude Code auto-mode, subagents, skills,
and Claude Design — one cohesive system."

## One-probe-three-questions verdict logic (for scene4_probe) — illustrative AUCs

- Q1 Disease (AD vs CN): high, e.g. `AUC 0.86` → signal present.
- Q2 Scanner (site ID): should be LOW for a real signal, e.g. `AUC 0.54` (near chance = good).
- Q3 Protein (p-tau217): correlated, e.g. `r 0.61` (≥ 0.2 threshold) → molecular anchor holds.
- Verdict A (REAL BIOLOGY, green): predicts disease AND tracks p-tau217 but NOT the scanner.
- Verdict B (ARTIFACT, killed/gray): a competing signal that predicts disease AND the scanner
  (`scanner AUC 0.83`) → refused as a machine artifact, not biology.
All target/AUC numbers here are ILLUSTRATIVE for the animation — label the scene's outputs as
"illustrative" per the honesty rule; they demonstrate the logic, not a specific validated result.
