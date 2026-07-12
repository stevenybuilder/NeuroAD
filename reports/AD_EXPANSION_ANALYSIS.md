# AD Imaging Expansion — Honest Before/After Verdict

**Task:** AD-vs-CN on the *frozen* NeuroJEPA 768-d embedding, site-disjoint out-of-fold,
via `neuroad.probe.auc_ci_perm` (bootstrap 95% CI + within-site permutation null).
**Source of record:** `reports/ad_expansion_analysis.json`,
`reports/adni_neurojepa_crosscohort.json`,
`data/real/adni_neurojepa_embeddings_expanded.csv`.

> **One-line verdict:** The 5.7x AD expansion bought **precision, not discrimination and
> not de-confounding.** The 95% CI roughly halved (0.081 → 0.039), which is real and
> trustworthy. But the headline AUC gain (+0.034) does **not** survive scanner matching:
> hold field strength constant and the improvement over baseline collapses to ~+0.004.
> More AD did **not** de-confound the embedding (it still reads 3T-vs-1.5T at AUC 0.990)
> and did **not** demonstrate out-of-distribution transfer.

---

## 1. The headline, stated plainly

| | n | AD / CN | sites | AUC | 95% CI | CI width | p_perm |
|---|---|---|---|---|---|---|---|
| **BEFORE** | 590 | 87 / 503 | 61 | **0.857** | [0.815, 0.896] | 0.081 | 0.001 |
| **AFTER** | 997 | 494 / 503 | 70 | **0.891** | [0.871, 0.910] | 0.039 | 0.001 |
| **Δ** | +407 AD (5.7x) | | +9 | **+0.034** | | **−0.042 (halved)** | — |

Taken at face value this reads as "bigger cohort, higher AUC, tighter interval." Two of
those three are misleading. Only the tighter interval holds up. (Reproduced this session:
pooled AUC 0.884 [0.864, 0.904] at seed 0; the 0.891 headline is within split-seed noise.)

---

## 2. What three skeptics found

### 2a. Class-balance — *threat real.* The AUC gain is NOT rebalancing, but IS "which subjects were added."

The AD:CN ratio flipped from ~15% AD to ~50% AD. AUC is prevalence-robust here, so that
flip alone buys nothing: down-sampling the AFTER data back to 87 AD / 503 CN (matched
prevalence *and* n) across 5 random draws gives AUC **0.836–0.879, mean ~0.853** —
straddling the 0.857 baseline. **The +0.034 does not come from having more/balanced data;
it comes from a scanner giveaway in the added subjects.** 40% of the 494 AD are 1.5T scans
while **100% of CN are 3T**, so "is-1.5T" alone predicts AD at AUC 0.70. Scanner-matched
(3T-only, 269 AD / 503 CN) the AUC is **0.861 [0.835, 0.885]** — statistically
indistinguishable from 0.857, just tighter.

### 2b. Scanner-leakage — *threat real.* The confound inflates the headline by ~0.03 but does not hollow it out.

The disease signal is genuine and does **not** collapse toward chance when field strength
is controlled: 3T-only AD-vs-CN holds at **0.861 [0.835, 0.885], p_perm=0.001**
(reproduced this session: 0.860 [0.834, 0.886], 269 AD / 503 CN). But two structural facts
matter. (1) The premise that "CN mix field strengths" is **false** — all 503 CN are 3T, 0
CN at 1.5T (crosstab reproduced). Field strength is therefore *perfectly collinear* with dx
for the 200 1.5T AD: those subjects are trivially separable from every CN by acquisition
alone, and a 1.5T-only control is **impossible** (one class, degenerate). (2) Holding field
strength constant, the expansion improves over the 0.857 baseline by only **~+0.004**
(0.861 vs 0.857), not the advertised +0.034. The pooled→3T drop of ~0.030 AUC is the
leakage tax.

### 2c. CI precision — *threat NOT real.* The tighter interval is trustworthy.

The CI halving is a genuine n-driven precision gain, **not** the Card A degenerate-split
failure mode. The 5 AFTER site-disjoint OOF folds are balanced (~95–105 AD / ~96–104 CN),
site-disjoint (13–15 held-out sites each), with fold AUCs tightly clustered 0.884–0.910 —
no near-separable fold. The bootstrap spans 0.860–0.923 with **zero** resamples at 1.0
(Card A had 100% of resamples = 1.0 on an n=6 perfect-separation holdout). Hanley–McNeil
√(494/87)=2.38x predicts an after-width of 0.034 vs observed 0.039 — within ~15%, exactly
sqrt(n) scaling. **The one caveat:** this validates the CI as a *precision* statement only.
It is a tight interval around a **partly-confounded estimand** — the same embedding predicts
3T-vs-1.5T at AUC 0.990. "Tighter" means "more precisely estimated," not "more confidently
Alzheimer's."

---

## 3. Does the gain survive? — the three stress tests

| Stress test | Result | AUC | vs 0.857 baseline |
|---|---|---|---|
| **Down-sample to original n & prevalence** (87 AD / 503 CN) | gain vanishes | 0.836–0.879 (mean ~0.853) | ~0.00 |
| **Restrict within field strength** (3T-only, 269 AD / 503 CN) | gain nearly vanishes | 0.861 [0.835, 0.885] | **+0.004** |
| **Is the tighter CI real?** (fold audit + bootstrap) | **yes, trustworthy** | folds 0.884–0.910, 0 resamples at 1.0 | precision only |
| 1.5T-only control | **untestable** (0 CN at 1.5T) | undefined | — |
| New-AD-vs-CN (407 vs 503) | new subjects are real signal | 0.898 [0.877, 0.917] | (entangled w/ batch) |

The new AD subjects carry real disease signal (0.898 slice) — but they **arrive entangled
with acquisition batch**, so the pooled 0.891 cannot be cleanly attributed to disease.

---

## 4. The verdict — precision, not de-confounding, not generalization

**What the expansion bought: PRECISION.** The 95% CI tightened from 0.081 to 0.039 (roughly
halved), a genuine, audited √n gain on the 5.7x larger AD sample. Folds are balanced and
site-disjoint; no degenerate separation. This is the one honest win.

**What it did NOT buy — DE-CONFOUNDING.** More AD did not remove the scanner confound; it
*added* one. The embedding still reads field strength at **AUC 0.990**, and because the
added 1.5T AD have no CN counterpart, field strength became a *partial AD proxy* (AUC 0.70).
Scanner-matched discrimination is 0.861 ≈ 0.857 baseline. De-confounding requires ComBat/
harmonization or a scanner-balanced design — **not more subjects from an imbalanced batch.**

**What it did NOT buy — GENERALIZATION.** No external cohort was tested and no OOD transfer
was demonstrated. A tighter CI around an in-distribution, partly-confounded estimand is not
evidence of generalization. Precision and validity are **separate axes**; this expansion
moved precision and left validity where it was.

**Honest headline to report going forward:**
> "The 5.7x AD expansion tightened the AUC estimate (95% CI 0.081 → 0.039) while
> discrimination held steady. Scanner-matched (3T-only) it is **0.861 [0.835, 0.885]** vs a
> 0.857 baseline — a real, precise, but essentially unchanged disease signal. The raw
> pooled 0.891 is inflated ~0.03 by an acquisition confound (40% of added AD are 1.5T,
> 100% of CN are 3T; field strength is itself readable at AUC 0.990)."

Do **not** state "expanding the AD cohort improved AD-vs-CN discrimination from 0.857 to
0.891." Report the 3T-only number as the scanner-matched figure, and note that ComBat plus
an external cohort remain required before any generalization claim. `p_perm=0.001` is the
permutation-resolution floor (0/1000), not a literal p.

---

## 5. How to frame this for a judge

Consistent with `docs/FRAMING.md`: **the tool's value is that it measured and reported the
confound honestly, not that the number went up.** Tell the judge: "We 5.7x'd the AD imaging
cohort and our referee caught that the headline AUC gain was mostly a scanner batch effect —
so we report the scanner-matched 0.861, not the confounded 0.891, and we flag that the same
embedding reads field strength at 0.990." That is the product working as designed: a
gatekeeper that only lets *validated* signal across. Precision is not generalization, and
a tighter interval around a confounded quantity is a more precise wrong answer unless you
say so — which is exactly what this analysis does.
