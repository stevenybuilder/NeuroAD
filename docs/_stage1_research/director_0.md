Judge simulation complete. I read the full build: `README.md`, `docs/BUILD_SPEC.md`, the core engine (`gauntlet.py`, `leakage.py`, `detective.py`, `discovery.py`, `probe.py`, `scoring.py`, `contract.py`), the Claude layer (`courtroom.py`, `reviewer.py`, `claim_parser.py`, `narrator.py`, `bridge.py`), `pipeline.py`, `cli.py`, `app/index.html` (4,368 lines, self-contained), `app/build_demo_data.py`, and the baked `app/demo_data.json`. 60 test functions across 7 files.

---

# NeuroAD Discovery Engine — Judge Scorecard (simulated)

**Headline: 74/100.** A genuinely rigorous *referee* that would place upper-quartile on craft, but the "discovery" the vision promises is not yet real — the only phenotype that promotes in the demo is synthetic planted ground truth (ARI=1.0 on a `tau_hot` cluster you injected), while the one time the Detective touches *real* frozen embeddings it correctly finds nothing but scanner artifacts. That gap is the whole ballgame for the Gladstone award.

---

## Impact — 25% → I'd give ~17/25

**Current strength.** The named user is real and crisp (translational AD researcher: *"is this signal worth a quarter of my time, or is it scanner noise?"*), the tool runs in one command without you in the room, and it exports a decision artifact they could defend to a reviewer. The honest-substrate framing (README lines 104–155) is unusually mature — you cite the leakage prior art (arXiv:2604.14441 / 2606.09189 / PathoROB) instead of claiming it.

**Highest-leverage gap.** It's a tool that mostly says **NO**. A skeptic's filter caps its own impact ceiling — "I stopped you chasing an artifact" is valuable but defensive. Judges reward tools that *produce*, not only *reject*. There is no moment where the researcher walks away with a new candidate they didn't have before.

**What a top-3 build adds.** One real surviving candidate on real data plus the exact confirmatory cohort to run next — so the tool both kills junk *and* hands the researcher a lead.

## Claude Use — 25% → I'd give ~16/25

**Current strength.** Claude-as-adversary is the right creative instinct: `courtroom.py` runs Prosecution/Defense over the same evidence, `reviewer.py` argues *against* the final verdict (p-tau217 missingness, "partially robust ≠ robust"), the contract-first multi-agent build, and gauntlet stages as drop-in Skills. This is well past "Claude wrote my code."

**Highest-leverage gap.** Claude is **decorative around a deterministic core.** The verdict is fixed arithmetic (`robustness_score`); the courtroom "frames tension" but rules nothing, and the Judge was explicitly removed as inconsequential (BUILD_SPEC 104–107). Worse for scoring: the demo is offline-first, so a judge watching the video is almost certainly seeing the **template fallback**, not live Claude — the `_fallback()` path in every module. Nothing here would "surprise Anthropic."

**What a top-3 build adds.** Put Claude on the *critical path* of discovery, not narration: a couple-sentence hypothesis → Claude picks the target column, the contrast, the confounds that matter, and pre-registers the falsification criterion — and that structured output *drives the actual run*. That is the "harness personalized to the researcher" your own vision names, and it makes Claude load-bearing.

## Depth & Execution — 20% → I'd give ~17/20

**Current strength.** This is your best axis and it's not close. The stats are real and honest: subject-disjoint leakage margin with a documented **conservative CV asymmetry** (`leakage.py` 42–72), brain-age control that regresses out *predicted brain age* not the residual gap with a stated reason (`gauntlet.py` 154–169), biomarker anchor gated on the **Fisher-z CI lower bound** not raw r (233–283), bootstrap-Jaccard cluster stability as the primary gate (`detective.py`), double-dissociation via LDA scanner directions, Wilson CIs and Cohen's d in `discovery.py`. 60 tests. Reduce-then-cluster to fight distance degeneration. This is grad-level rigor.

**Highest-leverage gap.** All that machinery mostly runs against a synthetic harness and single-cohort real data. The depth is in the *referee*; the *discovery* half is comparatively shallow — `discover_and_referee` is never even run on your 61 real AD/MCI/CN embeddings (only on synthetic and on healthy OpenBHB).

**What a top-3 build adds.** Point the exact same rigorous machinery at real disease-bearing embeddings and let it produce a defensible real result.

## Demo — 30% → I'd give ~23/30

**Current strength.** The offline workbench is genuinely cool to watch: live gauntlet ticking, KILL-vs-SURVIVOR toggle, PCA scatter with color-by-scanner/outcome, verdict meter, courtroom panel, Detective panel, confound leaderboard — all self-contained, all reading real exported artifacts. This will screen-record beautifully in 3 minutes.

**Highest-leverage gap — and it's a trap.** Your hero moment is **synthetic**. `demo_data.json` promotes cluster 0 `tau_hot` at score 90 / "strong candidate" with p-tau217 d=2.24 and ARI=1.0 — that is planted ground truth. The one real-embedding Detective run (`discovery_real`) finds 2 clusters, *both* flagged scanner/site artifacts, with age_sex / brain_age / biomarker / replication all `not_available`. A judge who reads the panels closely sees the compelling result is fake and the real result is empty. That's a credibility cliff.

**What a top-3 build adds.** Make the **real** run the hero and demote synthetic to an explicitly-labeled positive control ("this is our calibration phantom; here is the real cohort").

## Gladstone "advance the field" bar — currently your weakest fit

A referee that filters artifacts is **methodology/infrastructure**, not a field-advancing finding. To a Gladstone judge, "advance the field" means a *new candidate biomarker or phenotype*, surfaced honestly, with a path to confirmation. Your vision (SSL surfacing a novel imaging pattern) is exactly the winning shape — and it is the single thinnest part of the current build.

---

## Blunt strong-vs-thin summary

- **Already strong (don't spend more time here):** referee rigor (gauntlet + leakage + calibration), the offline self-contained demo, the honest-positioning/prior-art framing, the Detective's SSL + bootstrap-stability machinery. These are done and defensible.
- **Thin (where every remaining hour should go):** (1) a *real* novel discovery — the only promoted phenotype is synthetic; (2) the researcher-hypothesis entry point — `claim_parser.py` is a basic NL→target-column mapper, not the "couple-sentence hypothesis → personalized discovery run" the vision sells; (3) the domain-knowledge harness — `calibration.py` pins literature numbers, but there is no deterministic domain-logic layer that turns AD knowledge into *hypotheses to test*.

---

## The 3 highest-ROI moves for the next 5 days

**1. Run the referee on your 61 real OASIS-1 Neuro-JEPA embeddings — TODAY. (Biggest ROI, near-zero cost.)**
You already have `data/real/oasis1_neurojepa_embeddings.csv` (36 CN / 17 MCI / 8 AD, 768-d, real frozen encoder) via `src/neuroad/data/oasis_jepa.py`, but `app/build_demo_data.py` only runs `discover_and_referee` on synthetic and on *healthy* OpenBHB — it never touches the disease-bearing real cohort. Wire a `discovery_real_oasis` block that runs the identical Detective+gauntlet on these embeddings. Even a cluster that separates AD-from-CN and survives age/sex + the batch control on *real frozen foundation-model embeddings* is a genuine, non-planted result. Then push to `n≈235`: OASIS-1 raw volumes are open (no application), so embed the ~174 additional CDR-labeled subjects on Colab (compute is abundant; the only risk is the preprocessing tax — MNI reg / skull-strip / bias-field via your existing `scripts/neurojepa_embed.py`). This single move converts your weakest axis (Gladstone / real discovery) into a real one and de-risks the demo's credibility cliff.

**2. Make Claude drive discovery from a two-sentence hypothesis — the Claude-Use + Impact multiplier.**
Upgrade `claim_parser.py` from "NL → target column" into a structured *research-plan* generator: hypothesis → {target, subgroup/contrast, which confounds to prioritize, a **pre-registered falsification criterion**}, and have that output actually parameterize the run end-to-end (it currently doesn't). Surface it live in the UI as a "hypothesis in → falsifiable plan out" card. This is the exact "harness personalized to the researcher" from your vision, it puts Claude on the critical path instead of narrating, and it's a 1–2 day build on top of infrastructure you already have.

**3. Re-choreograph the demo so REAL is the hero and synthetic is the labeled control — plus a "novel candidate + next experiment" close.**
Reframe the centerpiece: lead with the real OASIS-1 run from Move 1, badge the synthetic `tau_hot` as an explicit positive-control phantom ("we validate the referee catches a known planted signal, then run it blind on real data"). End on a single card: the best *real* surviving candidate, its one falsifiable next experiment (you already have `bridge.py` for this), and the exact cohort to confirm the plasma anchor (ADNI/EPAD, per your routing). That final card is the Gladstone "advance the field" moment and it turns the demo from "cool but planted" into "trustworthy real lead + honest next step."

**Sequencing:** Move 1 first (it unblocks the honest data for 2 and 3), Move 3 last (it packages 1+2 into the video). If OASIS-1 re-embedding hits the preprocessing tax and stalls, the 61 embeddings you already have are enough to execute all three moves — don't let the n=235 stretch goal block the honest real run you can do this afternoon.