# NeuroAD Discovery Engine — Submission Write-up

> A scientific referee whose **credibility is the product**: it visibly refuses
> confounded imaging claims — including our own — and lets through exactly one real,
> reproducible finding: a foundation model that leaks the scanner.

**One command:** `neuroad demo` (offline) · `neuroad reproduce-finding` (regenerates the
headline leakage number from a checked-in fixture) · `app/index.html` served over http for
the visual workbench.

Every number in this document matches the regenerated `reports/*.json`. Every synthetic
number is badged `SYNTHETIC HARNESS`. Every prior-art citation resolves to a live
arXiv/DOI (`scratch/CITATIONS_VERIFIED.md`).

---

## The thesis in one paragraph

Frozen brain foundation-model embeddings leak acquisition site/scanner — a documented
batch effect (arXiv:2604.14441, Tao et al. 2026). That is *prior art*; we do not claim to
have discovered it. NeuroAD's contribution is **productization**: a runnable,
Claude-adjudicated referee that chains the full adversarial gauntlet
(age/sex → site/scanner leakage → brain-age → biomarker anchor → replication), issues a
fragile/robust verdict with confidence intervals a named scientist can reproduce in one
command, and refuses plausible-looking claims — starting with our own synthetic test case.
The one "wow number" we assert as a *result* is the real OpenBHB / Neuro-JEPA
scanner-leakage finding, and it is reproducible from a clean clone.

---

## Mapping to the rubric

### Impact — 25%

**The build-on unit is the runnable audit artifact, not the leakage fact.** That frozen
brain foundation-model embeddings leak acquisition scanner/site is openly published prior
art (arXiv:2604.14441, Tao et al. 2026) — we cite it, we don't claim it, and a judge should
dock any re-demonstration. What a researcher actually inherits here is the *tooling that
catches it on their own encoder*, three drop-in pieces:

- **A one-command reproduction over a checked-in fixture.** `neuroad reproduce-finding`
  recomputes the headline leakage number from a committed, license-safe PCA-10 feature
  table → **AUC 0.958 (≈0.96), 95% CI [0.907, 0.997], permutation p=0.001** (CI excludes
  chance) — no gated weights, no network. `reports/openbhb_neurojepa_leakage.json`.
- **A swappable contract feeder.** Any cohort that maps into the one-name
  `neuroad.data.loaders.load` contract runs the entire gauntlet unchanged — the referee is
  encoder- and cohort-agnostic by construction.
- **A drop-in gated loader.** `neuroad.data.gated.load_gated` pulls CC-BY-NC-ND weights via
  a token-from-env path and ships only frozen-inference-derived PCA features; the raw 768-d
  table and the weights are **never redistributed** (the raw 768-d AUC 0.998 is explicitly
  flagged p≫n inflated, reported only beside the defensible ~0.96).

Run against that machinery, the finding is just the worked example: on **3,984 healthy
OpenBHB brains** (62 sites, no disease) the structural embedding predicts the **scanner
(field strength)** at **AUC 0.891** (site AUC 0.784), and the frozen Neuro-JEPA 768-d
embedding leaks it at **AUC 0.96 (PCA-10)** on a 96-subject real multi-site healthy subset.
An OASIS-3 imaging lead points the same three pieces at their own frozen embeddings and
gets a fragile/robust verdict — with uncertainty attached, not a claimed cure.

**Honesty guardrails (why this is credible, not hype):**
- We deleted "Proteins confirm it." The plasma-biomarker anchor (p-tau217 / GFAP) is a
  **calibration target drawn to sit inside a literature range, not a measurement** — no
  open cohort pairs MRI with plasma markers. It is badged `SYNTHETIC HARNESS` in the UI,
  the reports, and the downloadable `evidence_ledger.csv`.
- The demo never leads with a beat that fails our own gauntlet. The synthetic SURVIVOR
  (score 80) is demoted to an explicitly-badged demonstration of the *promotion-gate
  mechanic*, not a discovery.

### Claude Use — 25%

**Claude is the adjudicator, not a chatbot.**

- A **three-voice courtroom** — *prosecution* argues artifact (citing the leakage margin),
  *defense* argues biology, *judge* renders the verdict — followed by a **reviewer that
  critiques its own verdict** (proxy brain-age control, small-n, same-probe-family leakage
  bound). This self-critique is the strongest differentiator and it is auto-expanded on
  camera, not hidden behind a chevron.
- **Strict structured tool-use:** a single-tool structured-output contract, a
  server-side-refusal fallback beta + model-retry chain, and a rich **deterministic offline
  fallback on every call** so the pipeline never crashes without a key.
- **Verifiably real, honestly badged:** every report and CLI card prints a
  **live-vs-offline Claude badge** (`claude.live`, `model`, `path`). The shipped default
  path is the **deterministic offline template** (`live=false`, clearly labeled in
  `reports/live_transcript.json` and every report) so the differentiator is provable
  *without* the recording depending on a flaky live call. With `ANTHROPIC_API_KEY` set, the
  same code path runs live and the badge flips — no template swap.

### Depth — 20%

**Statistical rigor upgraded from assertions to tests.**

- **Confidence intervals + permutation nulls on the headline numbers.** Verdicts that the
  demo says out loud — leakage margin, outcome/scanner AUC, replication AUC — now carry
  bootstrap 95% CIs and stratified label-permutation p-values, reframed as
  *"margin CI excludes zero"* / *"CI excludes chance"* rather than point-estimate cutoffs
  (e.g. synthetic KILL: margin −0.08, 95% CI [−0.14, −0.02]; SURVIVOR: margin +0.06, 95%
  CI [−0.04, +0.16], honestly flagged as *not* excluding zero).
- **PCA-in-probe for p≫n cohorts.** The referee reduces dimensionality (PCA/whitening when
  n < k·D) so the shipped leakage number is the defensible ~0.96, not the inflated 0.998 —
  reconciling the builder-track engine with the researcher-track honesty note that
  previously distrusted the very AUCs the engine emitted.
- **Brain-age reported two ways.** Both the standard Franke/Gaser brain-age-**GAP** control
  and the stricter predicted-brain-age control, so a reviewer sees the survivor is not
  silently killed by one over-aggressive residualization.
- **Seed-sweep verdict stability.** 20 seeds on the synthetic SURVIVOR/KILL; the score
  distribution and verdict-flip behavior are emitted into `demo_data.seed_sweep` and shown
  as the needle-fan beat, converting single-seed determinism from a hidden risk into a
  demonstrated robustness claim.
- **Self-describing artifacts.** `evidence_ledger.csv` carries provenance columns (n_used,
  ci_lo, source, `synthetic` flag); the biomarker row reports **r with its Fisher-z CI**,
  not the old UI-transform value that contradicted its own detail text.
- **Honest limitations, documented not hidden:** nested-in-fold residualization,
  Benjamini-Hochberg FDR across tests×clusters, and probability calibration (the module
  holds literature plausibility ranges, not ECE/Brier) are named as future work rather than
  overclaimed.

### Demo — 30%

**A deterministic, byte-identical 3-minute autopilot take — no manual keypresses.**

The beats are driven by a timeline array in `app/index.html`, served over http with
presentation-mode on (`docs/DEMO_SCRIPT.md`):

1. **Cold open (0:00–0:20)** — hold on the badged **SYNTHETIC HARNESS KILL**; naive probe
   AUC counts up 0.50 → 0.87. *"A finding that looks publishable. Watch the referee."*
2. **The star leakage test (0:20–0:55)** — the **pair-bar race**: the outcome bar fills to
   0.87, then the **scanner** bar races past it to 0.95; margin flashes red at −0.08. The
   scatter **morphs** from color-by-outcome to color-by-scanner — centroids slide apart,
   the dominant ellipse rotates: the machine *is* the axis. Verdict **rubber-stamp SLAM**,
   red flood: **REFUSED, 40/100.**
3. **Claude courtroom (0:55–1:25)** — prosecution/defense/judge auto-expand, then the
   reviewer argues *against its own verdict*; live-vs-offline badge visible.
4. **The real finding (1:25–2:00)** — cut to OpenBHB: **3,984 healthy brains, no disease.**
   Frozen Neuro-JEPA embeddings predict scanner at **AUC 0.96 (PCA-10)**; a bootstrap CI
   band snaps in and a permutation-null p appears. A terminal chip types
   `$ neuroad reproduce-finding`. *"Reproducible from a clean clone."*
5. **Survivor mechanic (2:00–2:25)** — explicitly badged `SYNTHETIC HARNESS`: the
   promotion gate and biomarker anchor (p-tau217 r=+0.40), score 80 → PROMOTED. *"This is
   how the gate works — not a claimed result."*
6. **Seed-sweep (2:25–2:45)** — 20 needles fan out and collapse into a tight cluster: the
   KILL never crosses the promotion line (0/20), the SURVIVOR promotes on 19/20 — a
   distribution, not a single-seed accident.
7. **Close (2:45–3:00)** — export tray + `neuroad demo`: open-source, reproducible, a
   referee that kills plausible-looking claims — including its own.

Supporting craft: video-legibility pass (12–13px mono, WCAG-AA contrast), eased count-ups
synced to the 1.1s needle glide, prefers-reduced-motion honored, and a green-CI
open-source signal.

---

## What we deliberately did *not* claim

- No new real AD cohort acquired 5 days out — we reframed the thesis around the real
  **leakage** finding instead of chasing a survivor that clears every star.
- The OASIS AD-vs-CN case is not promoted as a hero; the biomarker anchor is synthetic and
  badged; "Proteins confirm it" is gone.
- Every point estimate near a threshold is reported with its CI so a reviewer never has to
  guess whether a margin is significant — the referee says so itself.

**The pitch:** watch the referee kill good-looking results, including ours, then reproduce
the one real number yourself in one command.
