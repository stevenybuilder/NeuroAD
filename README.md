# NeuroAD Discovery Engine

**An Alzheimer's structural-MRI referee.** It finds a candidate signal in brain
MRI embeddings, stress-tests it to throw out artifacts (age/sex → site/scanner
leakage → brain-age → biomarker anchor → replication), anchors survivors to a
plasma biomarker, and — only for what survives — has **Claude adjudicate** the
likely mechanism and the one next experiment that would confirm or kill it.

> **Imaging finds it. Proteins confirm it. The system tells you what to do next.**

---

## Positioning (read this first)

The insight that frozen foundation-model embeddings leak scanner/site is
**published prior art**, and we cite it openly rather than claim it:

- *Batch Effects in Brain Foundation Model Embeddings* — arXiv:2604.14441 (2026).
  Brain-FM embeddings predict acquisition site/scanner about as well as the
  biological outcome — the same "star" mechanic our leakage test exploits.
- *Pretrained, Frozen, Still Leaking* — arXiv:2606.09189 (2026). Subject-disjoint
  lower bounds on attribute leakage from **frozen** embeddings (leakage margin
  ~0.16–0.37).
- *PathoROB* — Nature Communications (2026). Biological vs non-biological
  variation across medical centers (digital pathology; same genre).
- *REFUTE / AI-Scientist-v2* falsification agents (2024–2026). Automated claim
  falsification is an established sub-genre.

Our contribution is **productization**, quoting `calibration.POSITIONING`:

> The insight that frozen embeddings leak scanner/site is published prior art.
> NeuroAD Discovery Engine's contribution is productization: a runnable, agent-orchestrated
> referee that chains the full adversarial gauntlet, issues a fragile/robust
> verdict a named scientist can run in one command, gates survivors behind a
> plasma-biomarker anchor, and routes them to ONE falsifiable next experiment —
> with Claude as the adjudicator, not just the coder. It is a referee/auditor/
> red-team, NOT a co-scientist or discovery platform.

We do **not** say "we discovered leakage," "co-scientist," or "discovery
platform." We own **referee / auditor / red-team / gauntlet.**

---

## What it is

A translational AD researcher has imaging plus partial metadata and one
recurring question: *"Is this signal worth a quarter of my time, or is it
scanner noise, generic aging, or atrophy in disguise?"* NeuroAD Discovery Engine answers
it in one command, and exports a decision artifact they can defend to a reviewer.

### One probe, three questions

The whole architecture is **one small linear head** pointed at different label
columns of a cached embedding table (`contract.LABEL_TARGETS`):

| Point the head at… | …and it becomes |
|---|---|
| `conversion` / `dx_binary` | the candidate **signal** |
| `site` / `scanner` | the ⭐ **leakage test** (same code, different label) |
| a plasma biomarker (regression) | the **molecular anchor** |
| nothing (unsupervised) | the **Detective** (phenotype discovery) |

The referee runs a five-test **gauntlet** (weights sum to 100; the two ⭐ tests
carry the most weight):

1. **Age / sex adjustment** (15) — survives demographic covariates?
2. ⭐ **Site / scanner leakage** (25) — disease signal, or just the machine?
3. ⭐ **Brain-age control** (25) — more than generic aging/atrophy?
4. **Biomarker anchor** (20) — backed by p-tau217 / GFAP? **Hard gate.**
5. **Replication split** (15) — reproduces on a held-out site/cohort?

A weighted, NA-renormalized **robustness score** maps to a hedged verdict —
*fragile → partially robust → robust enough for follow-up → strong candidate* —
and only promoted survivors reach the biology step. See `docs/METHODS.md` for the
exact statistic behind each test.

---

## Quickstart

```bash
# 1. environment (a ready venv already exists at .venv)
python -m venv .venv && ./.venv/bin/pip install -e .

# 2. run the demo (fully offline: synthetic harness + Claude template fallback)
PYTHONPATH=src ./.venv/bin/python -m neuroad.cli demo
#   ...or, once installed as a script:
neuroad demo

# 3. open the visual workbench
open app/index.html   # ticks the gauntlet through reports/*.json

# 4. run your own claim on a chosen dataset
neuroad run synthetic:KILL "MCI converters have a distinct structural signature"
neuroad run oasis        "AD differs structurally from cognitively normal"
```

Set `ANTHROPIC_API_KEY` to use live Claude for the adjudicator / reviewer /
narrator; without it, every Claude call falls back to a deterministic template so
the demo runs with zero external access.

---

## The honest substrate story

We are explicit about what is real and what is a harness:

- **Real, vendored (no login):** OASIS-2 longitudinal + OASIS-1 cross-sectional
  CSVs — real structural-derived features (eTIV, nWBV, ASF) and real labels
  (CDR, MMSE, *Converted*). Gives a genuine AD-vs-CN diagnosis + conversion +
  brain-age + replication demo. *Honest caveat:* both are single-scanner, so on
  real data the OASIS star test is reframed as **cohort/batch leakage** (OASIS-1
  vs OASIS-2 as pseudo-sites).
- **Real, vendored (no login) — OpenBHB:** a 3,984-subject, 62-site, multi-scanner
  healthy cohort (Apache-2.0 HuggingFace mirror). On these **healthy** subjects
  with no disease at all, the structural embedding predicts the **scanner (field
  strength) at AUC 0.89** — the batch effect the referee gates against,
  demonstrated on *real* multi-scanner data (not synthetic). Healthy-only, so it
  is the leakage/brain-age control cohort, not an AD-signal source. Run it:
  `neuroad scanner-leakage`.
- **Real, frozen Neuro-JEPA embeddings — `openbhb:neurojepa`:** 96 OpenBHB subjects
  across 6 real sites, embedded by the **actual frozen Neuro-JEPA ViT-B MoE** over
  their MNI152 T1w volumes (run on a Colab T4; see `scripts/openbhb_embed.py`). The
  foundation model's **own 768-d representation** predicts scanner field strength at
  **AUC 0.93** (PCA-10, honest) / 0.998 (raw) and brain age at **R² 0.83** on healthy
  brains with no disease — the leakage the referee gates against, measured on the
  encoder itself. Weights are gated/frozen and **never committed**; see
  `docs/HF_ACCESS.md`.
- **The Detective (`neuroad discover`):** unsupervised phenotype discovery
  (reduce-then-cluster, bootstrap-Jaccard stability as the primary quality gate)
  with the gauntlet run **per cluster**. On a planted-phenotype synthetic cohort
  it recovers ground truth (ARI 1.0), then **promotes** the tau-hot phenotype and
  **flags** the age-atrophy and pure-scanner clusters as artifacts.
- **Synthetic harness** carries the *ground-truth* scanner-confound KILL and the
  p-tau217 biomarker anchor (no open cohort ships plasma markers), and is the
  guaranteed offline live path. Two presets: `SURVIVOR` (promoted, anchored to
  p-tau217) and `KILL` (higher naive AUC, collapses to scanner/aging artifact).
- **Gated (ADNI / OASIS-3 / NACC / EPAD):** a clearly-marked stub + notation,
  drop-in-ready — a real file replaces the stub with zero code change
  (`neuroad.data.gated.load_gated`). The contract makes the encoder/feeder
  swappable. See **`docs/DATA_ACCESS.md`** for the exact steps to obtain each
  (OASIS-3's ~1-week DUA is the one worth doing: FreeSurfer + AD labels +
  multi-scanner, i.e. a real scanner-leakage test *with* disease signal).
  Real Neuro-JEPA embeddings are now **shipped** for OpenBHB (`openbhb:neurojepa`,
  above); gated access requires your own HuggingFace grant (`docs/HF_ACCESS.md`).

Neuro-JEPA (hyphenated) weights, if used, are used **frozen** (CC BY-NC-ND):
no fine-tuning, no derivative.

**Prefer real data.** We use real cohorts wherever they exist and reserve synthetic
strictly for signals no open dataset carries (chiefly the plasma-biomarker gate). By
volume the substrate is ≈ **87% real / 13% synthetic**; every synthetic artifact is
badged as a harness, never shown as evidence. Full breakdown, honest assessment, and
the roadmap to replace remaining synthetic beats with real data:
**`docs/DATA_PROVENANCE.md`**.

---

## How the judging criteria map

- **Demo (30%):** a self-contained offline workbench that is a *viewer over the
  real exported artifacts* — a live gauntlet checklist, a filling claim card, and
  a KILL vs SURVIVOR toggle. See `docs/DEMO_SCRIPT.md`.
- **Claude Use (25%):** Claude as **adjudicator** (prosecution / defense / judge),
  a **reviewer agent that argues against its own verdict**, gauntlet stages as
  drop-in Skills, and a contract-first multi-agent build. See
  `BUILD_WITH_CLAUDE.md`.
- **Impact (25%):** a named scientist runs it without us in the room and saves
  months chasing artifacts; one real KILL on vendored data.
- **Depth & Execution (20%):** dual substrate, a double-dissociation control, a
  confound leaderboard, and calibrated-only numbers (`calibration.py`).

---

## Repository layout

```
src/neuroad/
  contract.py       frozen interface (schema, gauntlet, verdict bands)
  calibration.py    literature-pinned numbers + prior-art citations
  probe.py          the reused linear head
  gauntlet.py       the five adversarial tests
  leakage.py        leakage margin + double dissociation + confound leaderboard
  scoring.py        assemble the ClaimCard / verdict
  detective.py      unsupervised phenotype discovery
  data/             synthetic harness + OASIS adapter + loaders
  claude/           claim parser, courtroom, narrator, bridge, reviewer
  pipeline.py       run_referee(df, claim) -> ClaimCard   (this module set)
  cli.py            `neuroad demo` / `neuroad run`
app/                offline visual workbench
docs/               METHODS, SUMMARY, DEMO_SCRIPT
notebooks/          referee_walkthrough.ipynb
```

## License

MIT (see `LICENSE`). Neuro-JEPA weights, if used, remain under their upstream
CC BY-NC-ND license and are used frozen only.
