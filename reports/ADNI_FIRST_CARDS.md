# NeuroAD Discovery Engine — First Real ADNI Claim Cards

**Dataset:** real ADNI contract table (`loaders.load("adni")`) — 2,951 subjects, D=323 FreeSurfer `emb_*` features, `is_stub=False`, 37 real `<NA>` dx rows.
**Engine:** `pipeline.run_referee` (5-test gauntlet) + `scoring.apply_honesty_caps`. Run date 2026-07-09.
**Status:** every headline number below was independently re-derived with standalone sklearn code and reproduced to displayed precision. Where a test "passed" for the wrong reason, it is reported as a caught artifact, not a finding.

---

## 1. Headline

Running the real referee on real ADNI is a clean demonstration of the product thesis: **the tool catches its own feeder cheating.** A structural-MRI embedding separates AD from CN with a naive AUC of **0.935** — an eye-catching number a naive pipeline would ship. The gauntlet's STAR scanner-leakage test then shows the same embedding predicts **scanner field strength at AUC 0.989** — i.e. the features encode acquisition hardware *better* than they encode disease (leakage margin **−0.055**, 95% CI entirely below zero). All three cards (AD-vs-CN, MCI→AD conversion, unsupervised discovery) collapse to **score 39 / verdict fragile / promoted False** because the honesty cap (CAP-2) refuses to let any other test rescue a likely batch artifact. The one piece of signal that pulls partly back toward real biology is the **p-tau217 molecular anchor** (r ≈ +0.46 AD-vs-CN, +0.26 conversion, on measured plasma), which passes honestly — but under the current gate a passing anchor is not enough to promote a card the scanner test has failed. **Zero phenotypes promoted, and that is the correct, honest result.**

---

## 2. Card-by-card

### Card A — AD vs CN (`adni_dx_binary`)

**Confirmed naive effect:** AUC **0.9349** (n=1615: 462 AD / 1153 CN, D=323, site-disjoint StratifiedGroupKFold, full-dimensional logistic — n ≥ 2D so no PCA reduction).

**Gauntlet, test by test:**

| Test (weight) | Result | Confirmed numbers |
|---|---|---|
| age_sex (15) | **PASS** | effect retained 0.94 after age/sex adjustment (0.9349 → 0.9089) |
| site_scanner ★ (25) | **FAIL** | scanner_auc **0.9894** vs outcome 0.9349 → margin **−0.0545**, 95% CI [−0.070, −0.041] entirely below zero |
| brain_age ★ (25) | **PASS** | R²=0.236, MAE=4.71 yr, retained_gap 0.902 (n_healthy=1153) |
| biomarker_anchor (20) | **PASS** | p-tau217 Pearson r **+0.459**, n=873, Fisher-z CI_lo +0.405; GFAP r +0.252; provenance = measured |
| replication (15) | **PASS (degenerate — see §3)** | held-out AUC 1.00, CI [1.00,1.00], but **n_test=6** (4 CN / 2 AD) |

**Final verdict:** raw weighted robustness = 15+0+25+20+15 = **75/100** (would band "robust-follow-up"); CAP-2 fires (a ★ test FAILED while verdict ≠ fragile) → **score min(75,39)=39 → FRAGILE → promoted False.** Reproduced exactly against `card.robustness_score=39`.

**KILL / SURVIVOR read:** **KILLED as a disease claim.** The flagship story lives here — a beautiful 0.935 AD signal is a scanner artifact. What *survives* is the molecular anchor: the embedding's AD axis is genuinely correlated with measured p-tau217 (r=0.46), so part of the signal is real AD biology riding on top of the scanner confound. The card is honest about both: the acquisition leakage is fatal for promotion, but the anchor says "don't throw all of it away — there is disease signal entangled in here worth de-confounding."

---

### Card B — MCI → AD conversion (`adni_conv_001`)

**Confirmed naive effect:** AUC **0.7065** (n=1199: 412 converters / 787 non-converters, site-grouped CV, full 323-d logistic).

**Gauntlet, test by test:**

| Test (weight) | Result | Confirmed numbers |
|---|---|---|
| age_sex (15) | **PASS** | retained 0.959 (0.707 → 0.698) |
| site_scanner ★ (25) | **FAIL** | scanner_auc **0.9894** vs outcome 0.707 → margin **−0.2829** (far larger leakage than Card A) |
| brain_age ★ (25) | **PASS** | R²=0.236, MAE=4.71 yr, retained_gap 0.93 (→ 0.692) |
| biomarker_anchor (20) | **PASS** | p-tau217 r **+0.263**, n=490, Fisher-z CI_lo +0.179 (≥ 0.12 floor); measured plasma |
| replication (15) | **FAIL (honest)** | held-out AUC 0.875 point, n_test=6 (2 conv / 4 non), bootstrap CI_lo 0.5 → correctly fails |

**Final verdict:** raw = 15+0+25+20+0 = **60/100** (would band "partially robust"); CAP-2 → **score 39 → FRAGILE → promoted False.** Reproduced exactly.

**KILL / SURVIVOR read:** **KILLED**, and cleanly. The leakage is worse here (margin −0.28), and unlike Card A the replication test **honestly fails** — the tiny held-out site (2 conv / 4 non) is not trivially separable, so the bootstrap CI includes chance and the engine correctly refuses the pass. Survivor: the p-tau217 anchor is again genuinely measured and positive (r=0.26), confirming a real biomarker gradient underneath the confound. This is the "no false pass anywhere" card.

---

### Card C — Unsupervised discovery (`adni_discovery`, kmeans)

**Confirmed structure:** reduce-then-cluster (StandardScaler → PCA-whiten 20d → KMeans, k by silhouette over 2..6) → **k=2, silhouette 0.038** (near-zero structure), cluster sizes **1511 / 1440**.

**What the clusters actually are:** Cramér's V(cluster, scanner) = **0.572** (site 0.448, sex 0.110, age η² 0.057). Scanner here = field strength, binary (3T=2109, 1.5T=842). **cluster0 holds 812/842 = 96% of all 1.5T scans** — the split *is* a field-strength artifact. Embeddings predict field strength at AUC 0.989 (0.978/0.962 within-cluster) versus conversion AUC ~0.70.

**Gauntlet per cluster (both → CAP-2):**
- cluster0: {age_sex pass 15, site_scanner FAIL 0, brain_age pass 25, biomarker_anchor **weakened** 10, replication FAIL 0} = raw 50 → CAP-2 → **39 / fragile**
- cluster1: {biomarker_anchor **pass** 20 instead} = raw 60 → CAP-2 → **39 / fragile**
- Per-cluster biomarker separation: p-tau217 Cohen's d ±0.337, GFAP ±0.349; conversion rates 0.461 (cluster0) vs 0.205 (cluster1), n_mci 648/551. Replication FAILs on genuine 2-class n=6 splits (not a degenerate one-class artifact).

**Final verdict:** **score 39 / fragile / promoted False — 0 promotable phenotypes.**

**KILL / SURVIVOR read:** **KILLED as a discovery.** This is an honest NULL: the only "structure" the unsupervised engine finds is the 3T/1.5T hardware split. The per-cluster biomarker effect sizes (d≈0.34) look like biology but are confounded with scanner/cohort, and the card flags them as such. Nothing survives to promotion — correctly.

---

## 3. What the verifiers caught

1. **Replication false pass on Card A (CONFIRMED artifact, non-verdict-changing).** The AD-vs-CN replication test earns full 15/15 with AUC 1.00 and CI [1.00, 1.00], but this is a **degenerate split, not generalization.** The held-out site has n_test=6 (4 CN / 2 AD); both AD subjects score above all four CN (AD [0.624, 1.0] vs CN [0.0, 0.003, 0.003, 0.048]), so AUC=1.0 is *guaranteed*, and the "tight" bootstrap CI is an artifact of perfectly separating 6 points (100% of valid resamples return exactly 1.0). Honest result should be NA/uninformative. **It does not change the verdict** — CAP-2 floors Card A to fragile regardless (dropping replication → 71 still caps to 39; failing it → 60 still caps to 39; verdict is invariant). The runner already self-labels it "passed (degenerate) / not meaningful." Card B and Card C do **not** have this problem — their small held-out splits are genuinely 2-class and honestly fail.

2. **Score / verdict reconciliation (CONFIRMED correct).** The apparent paradox — Card A passes 4 of 5 tests yet scores 39 — is fully explained and is **not a bug**. `robustness_score` computes the raw weighted 75; then `scoring.apply_honesty_caps` **CAP-2** fires because a ★ test (site_scanner) FAILED while the banded verdict was not already fragile, flooring the score to **min(raw, 39)=39** and the verdict to fragile (39 < 40 partially-robust floor). Promotion is then blocked by `is_promoted(fragile)=False` even though the molecular hard-gate passes. Independently recomputed for all three cards; matches every `card.robustness_score`. The design is defensible: a scanner batch artifact "replicates" in any cohort sharing the confound, so no downstream test may rescue a failed acquisition-leakage check.

3. **Scoring concern worth a human decision (flag only — no engine edit made).** CAP-2 collapses raw scores of 75 (Card A), 60 (Card B), and 50/60 (Card C) all to the *same* 39. That is safe (all should be fragile) but **lossy** — it discards the information that Card A is a far stronger, more anchor-supported underlying signal than the discovery null. Consider whether a failed ★ test should hard-cap to a band ceiling (e.g. fragile) while preserving a within-band ordinal so reviewers can triage which killed cards are worth de-confounding first. This is a product judgment call, not a correctness bug.

4. **Minor provenance caveat (Cards A & B).** The "measured (not synthetic)" biomarker label rests on `df.attrs['synthetic_biomarkers']` being None/unset in the loader. The r-values are internally consistent, but the "measured" tag is only as trustworthy as that loader flag, which was not independently source-verified.

---

## 4. Real cohort caveats

- **Coarse scanner proxy.** The ★ site_scanner test's "scanner" is **field strength only (binary 3T vs 1.5T)**, not scanner model/serial. That the embedding hits AUC 0.989 on a *2-way* label is itself a strong leakage signal, but the confound is almost certainly finer-grained than what the test can currently see. Mitigation (site/model-level harmonization, e.g. ComBat) is the obvious next de-confounding step.
- **Plasma date gap.** p-tau217 / GFAP are real measured ADNI plasma but drawn at a visit that may not coincide with the MRI; the anchor correlation is a cross-time association, not same-day.
- **Missing dx.** 37 subjects have real `<NA>` dx. They are dropped from the AD-vs-CN labeled analysis (n falls to 1615 of 2951 after AD/CN filtering); no imputation was done. Decide whether to filter explicitly upstream vs rely on the head's NA handling.
- **Conversion class balance.** 412 converters / 787 non-converters (~34% positive) — imbalanced but workable; the naive 0.707 and the honest replication failure both account for it via stratified/grouped CV.
- **Methodological asymmetry (disclosed in `leakage.py`).** Outcome AUC uses site-disjoint StratifiedGroupKFold while scanner AUC uses plain StratifiedKFold, making the leakage margin a **conservative** (skeptic-safe) lower bound on the outcome's edge — it does not overstate the failure.

---

*File: `reports/ADNI_FIRST_CARDS.md`. All numbers reproduced independently to displayed precision; the single caught artifact (Card A degenerate n=6 replication pass) is disclosed and shown to be verdict-invariant.*
