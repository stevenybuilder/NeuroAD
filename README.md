# NeuroAD Discovery Engine

### An AlphaFold for Alzheimer's neuroimaging.

A single pipeline that takes a raw brain MRI and carries it — through a frozen
neuroimaging foundation model, a rigor gauntlet that kills batch artifacts, and a
multi-omics target layer — all the way to a **ranked list of wet-lab-testable
protein targets.** It bridges the imaging world and the molecular world, and it
refuses to hand you a target it can't defend.

> **Imaging finds it. Confounds try to fake it. The engine proves what's real, then tells you which protein to test.**

---

## Who this is for

- **Translational AD researchers** who have imaging plus partial metadata and one
  recurring question: *"Is this signal worth a quarter of my time, or is it
  scanner noise, generic aging, or atrophy in disguise?"*
- **Neurodegeneration drug-discovery teams** who need imaging-derived findings
  turned into prioritized, structurally-modeled molecular hypotheses they can
  bring to an iPSC / 3D-organoid bench.
- **Biomarker and foundation-model labs** who need an external referee that
  quantifies how much of a foundation-model embedding's apparent signal is
  actually batch/site artifact before anyone builds on it.

The output is a **decision artifact a named scientist can defend to a reviewer**:
a verdict, the confounds it survived, and the one experiment that would confirm
or kill it.

---

## The purpose

**Most Alzheimer's imaging "findings" are the machine, not the disease — and
chasing one costs a lab a quarter.** A foundation-model embedding will separate
patients from controls, but it separates scanners, sites, and field strengths just
as well.

NeuroAD closes the loop no single tool closes today:

**raw MRI → foundation-model embedding → adversarial rigor gauntlet → surviving signal → molecular pathway → ranked, structurally-modeled protein targets → a wet-lab experiment.**

It wires two worlds that normally never touch — clinical **neuroimaging** and
**molecular target discovery** — into one auditable pipeline. That bridge is the
novel core: getting a defensible molecular hypothesis *out of a brain scan* has no
off-the-shelf answer today.

### The AlphaFold aspiration

AlphaFold took a protein **sequence → 3D structure** and became infrastructure —
a reusable, trusted layer the whole field builds on. NeuroAD aspires to be that
layer for Alzheimer's neuroimaging: **brain scan → testable molecular
hypothesis**, reusable, cohort-agnostic, and honest about its own uncertainty.
We do not claim to have arrived there. We have built the full pipeline end to
end — every layer real, wired, and running — held to a rigor standard worthy of
the name.

---

## Technical specification

The engine is a six-layer pipeline. We used the full modern neuro-AI stack — not
for its own sake, but because bridging imaging to molecules genuinely requires
every layer.

```
        [Input] Raw Multi-Sequence MRI  (T1w, T2w, FLAIR)
                        │
                        ▼
  1. NEURO-JEPA FOUNDATION LAYER   — frozen 3D Vision Transformer (ViT-Base-MoE),
     (pretrained on 1.5M+ scans)     V-JEPA-2 latent prediction → 768-d embedding
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
  Tabular clinical feed          2. ATTENTIVE MLP PROBE
  (cognitive tests, plasma          frozen-JEPA head + leave-one-group-out
   p-tau217/GFAP/NfL, demo)         region attribution (interpretable grounding)
        │                               │
        └───────────────┬───────────────┘
                        ▼
  3. MULTIMODAL CROSS-ATTENTION   — from-scratch multi-head scaled-dot-product
     FUSION LAYER (transformer)     attention over imaging × plasma tokens;
     (needs ≥2 modalities: plasma)  per-modality leave-one-out attribution;
                                    N/A ({}) for plasma-less cohorts (OASIS)
                        │
                        ▼
  4. HYPOTHESIS REFINEMENT ENGINE — the rigor gauntlet: "kill weak / surface
     (kill-weak / surface-strong)   strong"; maps structural loss → disease stage
                        │
                        ▼
  5. PI4AD MULTI-OMICS LAYER      — network propagation & pathway analysis over
     (Priority Index for AD)        the protein interactome; ranks candidate genes
                        │
                        ▼
  6. MOLECULAR TARGETING LAYER    — AlphaFold DB structures (live) + open Boltz-2
     (structure-guided)             GPU folding for the complex step
                        │
                        ▼
  [Output] TARGET PRIORITIZATION — ranked proteins + one falsifiable wet-lab
                                    experiment for iPSC / 3D organoids
```

### The stack, layer by layer

| Layer | What we used | What it does |
|---|---|---|
| **1. Foundation** | **NeuroJEPA** — frozen 3D **Vision Transformer** (ViT-Base-MoE, 768-d, V-JEPA-2 latent prediction), used **frozen** under CC BY-NC-ND | Turns a raw 3D brain volume into a 768-dimensional structural embedding. Standing on a frozen encoder is the deliberate, published-best-practice choice (see rigor parity below). *This is transformer #1 in the stack.* |
| **2. Probe** | **Attentive MLP** on the frozen embedding | Yields the AD-vs-CN signal plus **leave-one-group-out attribution** — interpretable grounding of *what* drives the signal (embedding vs plasma). Exact hippocampal/cortical volumes come from FastSurfer. |
| **3. Fusion** | **Multi-head cross-attention** + softmax attention-gate | A from-scratch implementation of the core transformer operation — scaled-dot-product attention, multiple heads, layernorm, per-subject tokenization — over imaging × plasma tokens, producing data-dependent cross-modal weights. *Transformer #2, disclosed honestly: a **fixed, non-trained feature map** (seeded Q/K/V; only a downstream linear head is fitted, under leakage-honest CV).* It needs ≥2 modalities, so it runs for plasma-bearing cohorts (ADNI) and is **N/A** for plasma-less ones (OASIS). |
| **4. Rigor gauntlet** | Custom adversarial referee | Five confound tests (below) that try to falsify every signal before it is reported. |
| **5. Multi-omics** | **PI4AD** (Priority Index for AD) + Open Targets network propagation | Routes a surviving imaging signal through the protein interactome to ranked candidate genes/pathways. |
| **6. Molecular targeting** | **AlphaFold DB** (live precomputed structures) + **Boltz-2** (open, MIT-licensed GPU folding) | Structure-guides the ranked targets. AlphaFold3 de-novo complex folding is account/weight-gated; open Boltz-2 is the license-clean substitute for the complex step. |

### The rigor gauntlet — why the number is trustworthy

Every candidate signal runs a five-test gauntlet (weights sum to 100; the two ⭐
tests carry the most weight). The **same linear head** is simply pointed at
different label columns — point it at diagnosis and it's the signal; point it at
`scanner`/`site` and it's the leakage test:

1. **Age / sex adjustment** (15) — survives demographic covariates?
2. ⭐ **Site / scanner leakage** (25) — disease signal, or just the machine?
3. ⭐ **Brain-age control** (25) — more than generic aging/atrophy?
4. **Biomarker anchor** (20) — backed by p-tau217 / GFAP when available?
5. **Replication split** (15) — reproduces on a held-out site/cohort?

A weighted robustness score maps to a hedged verdict — *fragile → partially robust
→ robust enough for follow-up → strong candidate* — and only promoted survivors
reach the biology step. The machinery is leakage-honest throughout:
site-disjoint cross-validation, PCA fit *inside* each fold, bootstrap 95% CIs, and
a within-site permutation null.

### Performance

| Task | Result | Notes |
|---|---|---|
| **ADNI AD-vs-CN** (frozen 768-d embedding) | **AUC 0.85 [0.81–0.89]**, cross-cohort **0.83** | site-disjoint, post-harmonization; permutation p ≤ 0.001 |
| OASIS clinical AD-vs-CN | AUC 0.81 | independent replication |
| **Leakage collapse (the headline rigor result)** | **AUC 0.9996 → 0.563** | raw features separate cohort almost perfectly; ComBat harmonization strips the batch effect, and the biological signal survives at 0.85 |
| Scanner/field-strength leakage (OpenBHB, healthy) | AUC 0.89 (structural) / 0.93 (NeuroJEPA embedding) | the artifact the referee gates against, measured on real multi-scanner data |
| MCI→AD conversion | AUC 0.71 | the hard prognostic task; honestly rated fragile |
| **Target ranking** (clean, non-circular held-out) | **AUC 0.728, p = 0.003** | recovers independently-known AD-risk genes from out-of-evidence signal |

The point of the leakage collapse is the product: a shiny 0.99 that is almost
entirely scanner artifact, exposed and reduced to a defensible 0.85 **before**
anyone ships it. That we still post a real, reproducible number on data this noisy
is the strongest evidence the pipeline is rigorous.

### Rigor parity with the state of the art

The closest published system is the *Nature Medicine* paper **NeuroVFM**
(*"Health system learning enables generalist neuroimaging models,"*
s41591-026-04497-1), which runs almost exactly our Layer 2 — a frozen encoder +
attentive-MLP probe, AD classifier trained on ADNI, validated on OASIS-1/AIBL.

On the **statistical rigor of that shared experiment**, we match it and in the
control battery arguably exceed it: same frozen-encoder + attentive-probe design,
plus **permutation nulls, negative controls, and ComBat harmonization**. We are
explicit about where we do **not** match it — its pretraining scale (5.24M
volumes, ~1,000 GPU-hours to build the encoder) and its prospective clinical
breadth (156 tasks, blinded experts). We stand on a *frozen* encoder by design and
build the unsolved downstream bridge; matching a health system's pretraining is a
solved, published problem, not our contribution.

### Data scale

- **Scans processed:** **2,951 curated ADNI T1 MRI scans** (1,153 CN / 462 AD /
  1,299 MCI; 2,109 at 3T, 842 at 1.5T), pulled from an 8,372-image IDA query.
- **Subjects across cohorts:** ≈ **7,700 real human subjects** — ADNI + OASIS-1 +
  OASIS-2 + OpenBHB — of which ~1,500 were embedded through the frozen encoder.
- **Raw imaging volume:** ≈ **17 GB** of 3D brain MRI downloaded and processed,
  drawn from ADNI — a multi-site, longitudinal, ~20-year clinical study.

The atomic unit is a **brain volume** — one 3D scan, ≈ 11.5 million voxels,
compressed by the frozen foundation model to a 768-dimensional embedding.

### Compute

- **Frozen-embedding extraction:** NeuroJEPA inference across thousands of
  ADNI/OASIS/OpenBHB T1w volumes on Colab GPUs (T4/A100), with resumable,
  per-subject-checkpointed drivers (`scripts/neurojepa_embed_colab.py`,
  `scripts/run_conversion_embed_colab.py`).
- **Structure folding:** Boltz-2 GPU folding for the Layer-6 molecular step
  (`scripts/boltz_fold_colab.py`).

*Honest note: the foundation encoder is used frozen — we did not pretrain it. The
GPU spend is real, but it is inference/extraction and structure folding, not
foundation-model training.*

---

## Quickstart

```bash
pip install -e .          # a ready venv already exists at .venv

# serve the demo — the ZUI product surface + live Claude API
PYTHONPATH=src ./.venv/bin/python -m app.server     # → http://localhost:8080/

# or run a claim from the CLI on a real cohort
neuroad run oasis "AD differs structurally from cognitively normal"
```

Set `ANTHROPIC_API_KEY` to enable the live Claude roles (router, orchestrator, Ask
Claude). Without it the demo runs fully offline — the referee is deterministic
either way, and hypothesis routing falls back to a keyword classifier.

---

## How Claude is used

The referee's verdict is deterministic by design — Claude never sets a score. Claude
runs in exactly three live roles, each a real, consequential decision:

- **Intent router (Claude Sonnet)** — classifies a free-text hypothesis into the finite
  target enum {conversion, dx_binary, site, scanner} with enum-constrained output, so a
  word like "predicts" no longer misroutes a diagnosis claim to conversion. It only
  chooses which precomputed cell is read; it never produces or alters a number.
- **Orchestrator (Claude Sonnet)** — a tool-runner that sequences the engine's tools
  toward a goal (`/api/orchestrate`); it drives the pipeline but can never override a verdict.
- **Ask Claude (Claude Opus)** — the grounded Q&A rail (`/api/ask`) that answers a
  scientist's follow-ups and drills into the decision tree, grounded in the live case
  plus a curated knowledge base.

Everything else in the referee — the confound gauntlet, scoring, and verdict prose — is
deterministic Python. The build itself was contract-first multi-agent Claude Code; see
`BUILD_WITH_CLAUDE.md`.

---

## Architecture & repository layout

```
src/neuroad/
  contract.py         frozen interface — embedding schema, gauntlet weights, verdict bands
  probe.py            the one reused linear probe (same head → disease or scanner)
  gauntlet.py         the five adversarial confound tests (Layer 4)
  leakage.py          leakage margin + within-field-strength invariance + double dissociation
  scoring.py          assembles the ClaimCard + biomarker promotion gate
  pipeline.py         run_referee(df, claim) → ClaimCard
  claude/             router (Sonnet LLM-judge) + deterministic parse/narrate/bridge/courtroom
  harness/            orchestrator (investigate entry), agent (Claude tool-runner), translation
  integrations/       multi-omics adapters — pi4ad, alphafold, boltz, string
  data/               real cohort feeders + loaders (ADNI / OASIS / OpenBHB)
  cli.py              `neuroad demo` / `neuroad run <dataset> "<claim>"`
app/
  server.py           dependency-free stdlib HTTP backend (live Claude + /api/*)
  investigate_cache.*  precomputed real-engine grid, served by O(1) lookup
  router_cache.json   sha1(hypothesis) → routed target enum
  neuroad.html        the ZUI product surface ('/'); claude_science.html = the /start rail
scripts/              Colab GPU drivers (embed, fold) + cache-warming
docs/                 METHODS, FRAMING, RANKING_METHODOLOGY, handoffs
Dockerfile.backend    the live Cloud Run image (the deployed demo)
```

## Positioning & prior art

The insight that frozen foundation-model embeddings leak scanner/site is
**published prior art**, cited openly (*Batch Effects in Brain Foundation Model
Embeddings*, arXiv:2604.14441; *Pretrained, Frozen, Still Leaking*,
arXiv:2606.09189; *PathoROB*, Nat. Commun. 2026). Our contribution is
**productization**: a runnable, agent-orchestrated referee that chains the full
adversarial gauntlet, issues a fragile/robust verdict, and routes survivors to one
falsifiable next experiment. We own the **referee / auditor / gauntlet** layer.

## License

MIT (see `LICENSE`). NeuroJEPA weights, if used, remain under their upstream
CC BY-NC-ND license and are used **frozen only** — no fine-tuning, no derivative.
