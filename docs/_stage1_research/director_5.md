# NeuroAD Stage 2 — Compute-Costed Data + Method Plan
### Director of Biostatistics — decision memo (verified against disk state, 2026-07-08)

---

## TL;DR (the decision)

**GO — conditional on one afternoon of embedding work.** A credible *honest novel-candidate* demo is achievable, but **only if we embed the remaining ~174 CDR-labeled OASIS-1 subjects (61 → ~235).** At n=61 the referee's own math forces a KILL or a null anchor (see §3 power), which is honest but weak. At n≈235 we clear the power threshold where a moderate cluster↔severity anchor can survive the Fisher-z lower-bound test — that is the difference between a demo whose hero is a *rejection* and one whose hero is a *surviving candidate*. The GPU cost of the 61→235 jump is **~2–4 compute units of ~1799** (0.2%). GPU is not the bottleneck; **raw-volume download + the OASIS preprocessing path is**, and OASIS-1 gives us a shortcut that removes the FreeSurfer tax entirely.

Verified on disk:
- `data/real/oasis1_neurojepa_embeddings.csv` = **61 rows** → 36 CN (CDR 0) / 17 (CDR 0.5) / 8 (CDR 1.0).
- `data/real/oasis_cross-sectional.csv` = 436 rows, **235 with CDR**: CDR 0=135, 0.5=70, 1=28, 2=2.
- `data/real/openbhb_neurojepa_embeddings.csv` = **96 healthy**, 6 sites, 2 field strengths — scanner-leakage exercise set, ready.
- `scripts/neurojepa_embed.py` is runnable: MONAI transform pipeline (orient RAS → 1mm → crop-foreground → resize 100×120×100 → center-crop 96×108×96 → intensity scale), frozen `NYUMedML/Neuro-JEPA` ViT-MoE, mean-pooled 768-d. **No FreeSurfer, no skull-strip step in-pipeline.**

---

## 1. DATA — exactly what to pull and why

### Tier 0 — already in hand (build on this today, zero risk)
| Dataset | n | Role in demo | Status |
|---|---|---|---|
| OASIS-1 embeddings | 61 | Disease-bearing discovery cohort (Act II) | ✅ on disk |
| OpenBHB embeddings | 96 | Scanner-leakage gauntlet (Act I kill) | ✅ on disk |
| OASIS-1 `cross-sectional.csv` | 235 CDR-labeled | Anchor labels + embedding target list | ✅ on disk |

### Tier 1 — THE KEY LEVER: embed the remaining ~174 labeled OASIS-1 subjects (do this Day 1)

**Why OASIS-1 and not a new cohort:** we already hold the CDR labels for all 235; we only lack their embeddings. Same site, same scanner, same label semantics as the 61 — so adding them is a pure power injection with **no new confound, no new DUA, no cross-cohort harmonization**. It is the single highest-leverage, lowest-risk data action available.

**Why the preprocessing tax is smaller than the grounding note fears:** OASIS-1 ships, per subject, a **T88-registered, skull-stripped, gain-field-corrected volume** (`*_masked_gfc.img`, Analyze 3D, already in Talairach atlas space). That is exactly the input `neurojepa_embed.py`'s transform expects — its pipeline does orientation/spacing/resize/intensity but **no skull-strip and no registration of its own**. So we skip FreeSurfer and skip MNI registration entirely: **download → point manifest at `*_masked_gfc.img` → run the existing script.** The 61 we already have almost certainly came through this path, which is why it worked; the 174 are mechanically identical.

**Obtainable in 5 days?** Yes. OASIS-1 cross-sectional is open (click-through DUA, no committee review, unlike OASIS-3/ADNI/NACC/EPAD which are gated stubs and stay stubs). Discs are ~ a few GB each; we only need the subjects in the 235 list that aren't in the 61. Download to Colab (not local — local `data/raw/` is empty and the Mac is CPU-only Py3.14, can't run torch anyway).

**Manifest to build:** one CSV, columns `subject_id, image_path (→ the masked_gfc .img on the Colab disk), age, sex, cdr, dx, site='OASIS1', scanner='1.5T'` — passed straight through by the script's metadata pass-through. This makes the anchor label (`cdr`) travel with the embedding into the referee contract.

### Tier 2 — assessed and DEFERRED (name them as the "next experiment," don't build on them)
- **IXI (healthy, open):** useful only as an *additional* scanner-leakage / brain-age control set. Marginal over OpenBHB, which we already have. **Skip for the demo; mention as extensible.**
- **MIRIAD (AD/CN, open, longitudinal 46 subj):** genuinely tempting as a *second disease cohort* for the "external replication" beat. But it's a different scanner/protocol → needs harmonization we can't validate in 5 days, and small n. **Do NOT claim replication on it.** Correct honest use: name it (with ADNI/EPAD) as the *stated next-experiment replication cohort*. Embedding it is a stretch-goal only if Tier 1 finishes with a day to spare.
- **OpenBHB (healthy, multi-site):** already embedded — this is the scanner set, keep it exactly there. It has **no disease signal**, so it can never be a discovery cohort; using it as one would be a category error the referee must reject.
- **ADNI / OASIS-3 / NACC / EPAD:** gated stubs, no access. **Blocked. Do not plan around them.** They are the honest "confirmatory cohort to run next," and naming the specific one (ADNI for plasma p-tau217 anchor) is the Gladstone close.

**Data red line for the demo:** the labeled cohort is **single-site OASIS-1 (1.5T)**. Every claim says "confound-robust to the extent the 6-site OpenBHB test exercises it; the labeled cohort is single-site." No cross-scanner generalization claim on the disease cohort.

---

## 2. COMPUTE — concrete Colab plan + unit budget

GPU is abundant and the workload is tiny (frozen inference, ~174 volumes, batch 1). **Use T4** — the cheapest eligible GPU. L4/A100/H100 are strictly wasteful here; the job is latency-trivial, not throughput-bound, and there is nothing to train.

**Approx Colab Pro+ burn rates:** T4 ≈ **1.8 CU/hr**, L4 ≈ 4.8, A100 ≈ 11.8, H100 ≈ 14+. Budget everything on T4.

### Runtime estimate (embedding the 174)
Frozen ViT-MoE, 96×108×96, batch 1, `cudnn.enabled=False` (per script) → ~1–3 s/volume compute + I/O. **174 volumes ≈ 10–20 min GPU wall.** Add cold-start (backbone HF pull + MONAI import) ~5 min, and OASIS disc download/unpack ~20–40 min (network-bound, off-GPU — do it before attaching, or on CPU runtime). **Total GPU-attached time ≤ 1 hr with slack.**

### Unit budget vs ~1799 CU
| Task | GPU | GPU-hrs | CU |
|---|---|---|---|
| Embed 174 OASIS-1 (main run) | T4 | ~0.5 | **~1** |
| One re-run / debug buffer | T4 | ~1.0 | **~2** |
| (Stretch) embed MIRIAD ~46 | T4 | ~0.3 | **~1** |
| **Total planned burn** | | | **~4 CU (0.2% of 1799)** |

**Strategy: hoard units, don't idle.** Do all download/preprocessing on a **CPU runtime or before GPU attach**; attach T4 only for the inference loop; `colab stop` the instant `embeddings.csv` is downloaded. Never leave a GPU session warm. Even a worst-case 10× overrun is <2% of budget — so the constraint is wall-clock and correctness, not units. Reserve headroom (we won't need it) rather than reaching for A100.

### Runnable command sequence (colab CLI, via the `colab-gpu-cli` skill)
```bash
colab auth
colab start --gpu t4 --timeout 3600

# 1. Pull OASIS-1 discs + stage masked_gfc volumes for the 174 not-yet-embedded IDs
colab exec scripts/fetch_oasis1_raw.py        # writes /content/oasis1/<id>/*_masked_gfc.img
# (new small helper: diff the 235-label list against the 61 embedded, download only the gap)

# 2. Build the manifest (id, image_path, age, sex, cdr, dx, site, scanner)
colab exec -c "import build_manifest"         # or upload a prebuilt manifest.csv:
colab upload manifest.csv /content/manifest.csv

# 3. Frozen embedding run (HF_TOKEN in Colab env — never in repo; per gated-weights-compliance skill)
colab exec scripts/neurojepa_embed.py -- \
    --manifest /content/manifest.csv \
    --out /content/oasis1_extra.csv \
    --image-col image_path --id-col subject_id

# 4. Stream results back DURABLY, then kill the GPU
colab download /content/oasis1_extra.csv data/real/oasis1_extra_neurojepa_embeddings.csv
colab stop
```
Then locally concat the 61 + ~174 into `oasis1_neurojepa_embeddings.csv` (dedupe on `participant_id`), re-run `contract.validate_table`. **`HF_TOKEN` from env only; embeddings persisted, weights never written to repo — CC-BY-NC-ND frozen-inference compliance holds** (weights gated, inference-only, no derivative, embeddings are the only artifact and stay unpublished).

---

## 3. METHOD — the honest novel-discovery statistical harness

Two phases, pre-registered in that order: **discovery** (generate the candidate) then **validation** (the 5-test gauntlet tries to kill it). The pre-registration is load-bearing — the `HypothesisSpec` commits the accept/reject thresholds *before* the run, which is what separates discovery from data-dredging.

### 3.1 Discovery (unsupervised — the "Detective")
- **Substrate:** frozen 768-d NeuroJEPA embeddings. First **regress out age + sex + ICV(eTIV) from every embedding dimension** (residualize) *before* clustering — otherwise clusters recover aging, not disease. This is the single most important design choice; do it up front, not as a post-hoc "confound test."
- **Dimensionality:** 768-d on n≈235 is p≫n → cluster in a reduced space (PCA to ~10–20 comps capturing the residual variance, or UMAP for viz only, never for the statistics). Clustering on raw 768-d would manufacture unstable structure.
- **Clusterers (already in `src/neuroad/detective.py`):** KMeans + GMM + HDBSCAN, k swept small (2–4 — n forces this). **Consensus across methods, not a single fit.**
- **Embedding-direction search (Concept B panel):** a supervised linear probe direction for CDR *after* residualizing out chronological age, sex, AND brain-predicted age (the OpenBHB-trained brain-age model). Reports "structure beyond accelerated aging" as a residual r with CI. Supporting panel, not the hero.

### 3.2 Validation — the 5-test gauntlet (already in `src/neuroad/gauntlet.py`; point it at the real 61+174)
Run in this fixed order; each is a deterministic "Trusted" UDF with a pre-registered threshold:

1. **Bootstrap-Jaccard stability** — resample subjects B=1000×, re-cluster, match clusters, report mean Jaccard per cluster. **Kill if < 0.60.** This is the primary FDR guard: unstable clusters (the n=61 failure mode) die here.
2. **Age/sex confound** — cluster membership must retain association with CDR *after* the residualization; verify no residual age/sex separation (AUC of age/sex predicting cluster ≈ chance).
3. **Scanner-leakage double-dissociation** — LDA scanner directions from the 6-site OpenBHB split; **kill if scanner-AUC > threshold** (real OpenBHB scanner-AUC is ~0.90 per `real_scanner_leakage.py` — this is the live Act-I trap on a planted cluster).
4. **Brain-age residual** — association survives regressing out brain-predicted age (not just the gap).
5. **Biomarker/severity anchor** — **Fisher-z CI on cluster-membership ↔ CDR**; **survive only if the CI lower bound excludes 0.** Anchor specificity stated as "severity/AD-like," never "amyloid-specific."

**Verdict:** Prosecution/Defense (Claude, adversarial) argue over the *same numeric evidence*; scored rubric → KILLED / SURVIVOR. Every verdict sentence hyperlinks to its statistic.

### 3.3 Multiple-comparison / false-discovery control (the credibility spine)
- **Pre-register the single primary contrast** in the `HypothesisSpec` (cluster ↔ CDR). Everything else (k values, clusterer variants) is explicitly labeled *exploratory* and its p-values/CIs are **not** used for the survive/kill decision — the anchor CI on the pre-registered contrast is.
- Across the k-sweep and multiple clusters, apply **Benjamini–Hochberg FDR** to the family of anchor tests and report q-values; the demo shows the corrected number, not the cherry-picked min.
- **No p-value theater on small n.** Report **effect sizes + bootstrap CIs** everywhere (grounding red line ⚠). Let the referee *visibly reject* underpowered findings — a shown rejection is the honesty feature, not a failure.
- **Held-out replication proxy:** split-half the disease cohort (or leave-site-out on OpenBHB for the scanner test) so stability is measured out-of-sample, not in-sample.

### 3.4 Power analysis for the achievable n (this is why §1 Tier-1 is mandatory)
Anchor test = Fisher-z CI lower bound must clear 0.
- SE(z) = 1/√(n−3). 95% half-width ≈ 1.96·SE.
- **n=61:** half-width ≈ 0.257 → need **r ≳ 0.25** just to exclude 0. With CDR-1 cell = 8, any subtype found within the impaired group has cluster cells of ~4–12 → bootstrap-Jaccard almost certainly < 0.6 → **dies at test 1.** Honest outcome at n=61: KILL or null anchor.
- **n≈235:** half-width ≈ 0.129 → **r ≳ 0.13** excludes 0; CDR cells 135/70/28 give clusters large enough to clear Jaccard 0.60. Detecting a moderate association (r≈0.3) at 80% power needs n≈85 — **235 clears it comfortably, 61 does not.**

**Conclusion: n=61 is structurally underpowered for a SURVIVOR; n≈235 is the minimum credible cohort. The 61→235 embedding job is the demo's pivot, and it costs ~1 CU.**

### 3.5 Where SL / SSL / RL genuinely help vs. are overkill
- **SSL (NeuroJEPA):** genuinely load-bearing — it is the substrate that makes unsupervised discovery on 768-d structure possible without labels. Frozen, behind the harness. Keep it there.
- **SL (linear/MLP probe):** genuinely helps as the *anchor and leakage instruments* — one reused linear head pointed at CDR (anchor) and at scanner (leakage). Deliberately linear: interpretable, small-n-robust, and a linear probe leaking scanner is a *stronger* indictment than a deep net doing so. MLP is optional and risks overfitting n=235 — **default to linear.**
- **RL: overkill and a red flag here.** No environment, no reward that isn't circular, no time to validate. Using RL would be capability theater that the depth judges see through. **Do not use RL.** The intelligence in the loop is Claude-as-referee + the pre-registered gauntlet, not a policy network.

---

## 4. GO / NO-GO + fallback

**GO** for the honest novel-candidate demo (Concept A: confound-robust, stability-vetted, severity-anchored candidate phenotype), **contingent on the Day-1 embedding of the ~174 OASIS-1 subjects.** All four judging axes are reachable with data we can obtain: real hero (Demo), Claude on the critical path as parser+referee (Claude Use), grad-level stats shown live (Depth), a produced lead + named next experiment (Impact/Gladstone).

**Sequenced execution:**
- **Day 1:** OASIS-1 raw fetch (masked_gfc) → manifest → T4 embed 174 → concat to 235 → validate contract. (~1 CU, ½ day.)
- **Day 2:** Residualize + Detective on real 235; run full gauntlet; read out whether a cluster survives all 5 tests. **This is the honesty checkpoint.**
- **Day 3–4:** wire the surviving (or killed) result into the two-act demo; Act I = planted `tau_hot` phantom killed by scanner test on OpenBHB (labeled positive control), Act II = real 235 cohort; Claude Prosecution/Defense; closing card = candidate + mechanism *hypothesis* + single falsifiable next experiment (replicate in ADNI/EPAD; anchor to plasma p-tau217 vs NfL; named confirmatory n).
- **Day 5:** hardening, red-line audit, offline demo pack.

**Fallback (if no cluster survives all 5 tests on real 235, or if raw-volume fetch stalls):** this is **not a demo failure — it is the thesis.** Ship the *method* as the result: (a) the synthetic `tau_hot` phantom (ARI=1.0) as an explicitly labeled calibration positive control proving the gauntlet detects true structure; (b) the real OpenBHB scanner-AUC≈0.90 kill as the live batch-effect catch; (c) the real OASIS-1 cohort shown with the referee *honestly rejecting* an underpowered/unstable candidate, with the power math on screen and the exact cohort+n that would make it decidable. The wow line survives intact: **"Watch it kill our best-looking result because it was scanner noise — then trust the one it lets through (or trust it to tell you n isn't there yet)."** A visible, quantified rejection with a named fix *is* the Gladstone "advance the field" contribution — a trustworthy referee the field's own reproducibility literature (Prevot/Oxtoby: subtypes survived 3/8 models) proves is missing.

**Non-negotiable red lines enforced in-demo (self-flagged):** candidate ≠ biomarker; symptomatic/MCI range only, not preclinical; not "better than plasma / replaces PET" (imaging owns topography/N-axis, not amyloid detection); single-site labeled cohort stated explicitly; "internally stability-vetted, external replication is the next experiment," never "validated/reproducible"; mechanism always as *hypothesis*; effect-size + bootstrap CI, no p-value theater on n=8 cells.

---

### Relevant paths
- Embedder: `/Users/stevenyang/Documents/claude-life-sciences-hack/neuroad-discovery-engine/scripts/neurojepa_embed.py`
- Scanner-leakage: `/Users/stevenyang/Documents/claude-life-sciences-hack/neuroad-discovery-engine/scripts/real_scanner_leakage.py`
- Gauntlet / Detective: `/Users/stevenyang/Documents/claude-life-sciences-hack/neuroad-discovery-engine/src/neuroad/gauntlet.py`, `.../src/neuroad/detective.py`
- Skills to reuse: `.../skills/brain_age_control/run.py`, `.../skills/biomarker_anchor/run.py`
- Labels: `.../data/real/oasis_cross-sectional.csv` (235 CDR); embeddings: `.../data/real/oasis1_neurojepa_embeddings.csv` (61), `.../data/real/openbhb_neurojepa_embeddings.csv` (96)
- New files needed: `scripts/fetch_oasis1_raw.py` (download masked_gfc for the 174-subject gap), a manifest builder, and the concat step.