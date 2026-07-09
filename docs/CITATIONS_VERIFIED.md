# CITATIONS_VERIFIED — WAVE 1 (QA / citation verification)

Verifier: QA engineer. Date: 2026-07-08.
Method: arXiv export API (`export.arxiv.org/api/query`), direct `arxiv.org/abs/<id>` fetch,
and independent WebSearch cross-checks (Nature, HuggingFace papers, GitHub). A citation is
marked RESOLVED only when at least two independent sources agree on id + title + authors.

## TL;DR for WAVE 2 (backend)

**Every load-bearing citation RESOLVES. Nothing gets DELETED.** All four `PRIOR_ART`
entries and the `Neuro-JEPA` fact are real, resolvable papers with correct years. The
required edits are precision/defensibility upgrades only: add real titles + resolvable
URLs/DOIs, and remove ONE unverifiable inline number. Apply the edits in the
"APPLY THIS" blocks verbatim.

Files WAVE 2 will touch: `src/neuroad/calibration.py` (PRIOR_ART list, FACTS["neurojepa"])
and `README.md` (Positioning section lines ~18-27, and line ~121-122 Neuro-JEPA mention).

---

## Per-citation verdicts

### 1. arXiv:2604.14441 — "Batch Effects In Brain Foundation Model Embeddings"  → RESOLVED
- Resolvable URL: https://arxiv.org/abs/2604.14441
- Exact title: *Batch Effects In Brain Foundation Model Embeddings*
- Authors: Ye Tao, Bradley T. Baker, Yu Wu, Anand D. Sarwate, Sandeep Panta, Sergey Plis,
  Vince D. Calhoun
- Year: 2026 (submitted 2026-04-15; v2 2026-06-08). Category: eess.SP.
- Confirmed by: arXiv export API + WebSearch (arxiv.org abstract/html/pdf all live).
- Claim check: paper evaluates BrainLM and SwiFT (fMRI foundation models) and finds
  "foundation model embeddings encode substantial batch-related variability, often
  dominating diagnosis-related information across heterogeneous datasets." This SUPPORTS
  the calibration claim that brain-FM embeddings predict acquisition site/scanner as well
  as biological outcome.
- CAVEAT (non-blocking): the paper's modality is **fMRI** (BrainLM/SwiFT), not the
  structural-MRI embeddings NeuroAD itself runs on. The "same star mechanic" framing is
  fair, but do not imply it studied structural T1w embeddings. Keep wording generic
  ("brain-FM embeddings"), which the current text already does. No number to remove.
- ACTION: keep. Year and id already correct. No edit strictly required.

### 2. arXiv:2606.09189 — "Pretrained, Frozen, Still Leaking"  → RESOLVED (with 2 required fixes)
- Resolvable URL: https://arxiv.org/abs/2606.09189
- Exact FULL title: *Pretrained, Frozen, Still Leaking: Auditing Cross-Encoder Attribute
  Transfer in EEG Foundation Models*
- Author: Jianwei Tai. Year: 2026 (submitted 2026-06-08). Category: cs.CR (+ cs.AI).
- Confirmed by: arXiv export API + direct arxiv.org/abs/2606.09189v1 fetch (both agree on
  title/author/category/abstract first line). NOTE: general web-search engines had not yet
  indexed it at check time (normal lag for a June-2026 preprint); the paper itself is live
  on arXiv.
- **FIX A (modality):** this paper audits **EEG** foundation models, NOT MRI/imaging
  embeddings. Current calibration text ("attribute leakage from FROZEN foundation-model
  embeddings") is generic enough to stay true, but do not let the pitch imply it is about
  brain-imaging embeddings specifically. It is a same-genre (frozen-embedding attribute
  leakage) citation from a different modality.
- **FIX B (unverifiable number — REMOVE):** the inline claim "leakage margin ~0.16-0.37"
  could NOT be verified against the abstract from any source. Per the "no free-floating
  number" rule in calibration.py's own docstring, DELETE this specific number rather than
  ship an unverifiable figure. Replace with a qualitative description.
- ACTION: update title to the full title, keep the id/year, drop the numeric margin.

### 3. arXiv:2606.14957 — Neuro-JEPA  → RESOLVED (add the real paper title)
- Resolvable URL: https://arxiv.org/abs/2606.14957
- Exact paper title: *Learning Sparse Latent Predictive Foundation Model for Multimodal
  Neuroimaging* (the model is named **Neuro-JEPA** inside the paper).
- Authors: Haoxu Huang, Long Chen, Jingyun Chen, Jinu Hyun, James Ryan Loftus, Kara Melmed,
  Daniel Orringer, Jennifer Frontera, Seena Dehkharghani, Arjun Masurkar, Narges Razavian
  (NYU Langone / NYU Long Island / Massachusetts General Hospital).
- Year: 2026 (submitted 2026-06-12, v2). Confirmed by: WebSearch + HuggingFace papers page
  (huggingface.co/papers/2606.14957) + arXiv export API.
- Claim check: pretrained on 1,551,862 scans / 428,647 studies across T1w/T2w/FLAIR,
  latent-predictive (JEPA) objective + Mixture-of-Experts. This EXACTLY matches
  FACTS["neurojepa"] (~1.55M scans, T1w/T2w/FLAIR, JEPA+MoE). License claim (code MIT,
  weights CC BY-NC-ND) is out of scope for this citation pass — gated-weights owner should
  confirm — but the scientific description is SUPPORTED.
- ACTION: keep id/year; add the real paper title alongside the "Neuro-JEPA" model name so a
  reviewer clicking through lands on a title that matches.

### 4. "PathoROB" — Nature Communications (2026)  → RESOLVED (add real title + DOI)
- Resolvable URL: https://www.nature.com/articles/s41467-026-73923-2
- DOI: 10.1038/s41467-026-73923-2  (Nature Communications, vol. 17, 2026)
- arXiv preprint: https://arxiv.org/abs/2507.17845
- Exact article title: *Towards robust foundation models for digital pathology*
  (**PathoROB** is the robustness benchmark introduced in the paper).
- Confirmed by: nature.com article page + arXiv + RePEc (ideas.repec.org) + Aignostics blog.
- Claim check: introduces PathoROB — robustness benchmark with a "robustness index," 4
  datasets, 28 biological classes across 34 medical centers, 20 evaluated FMs; shows FMs
  susceptible to non-biological technical variation (staining, scanner hardware, lab
  procedure). SUPPORTS the calibration "biological vs non-biological variation across
  medical centers" claim.
- ACTION: keep, but cite with the real article title + DOI (currently only "PathoROB /
  Nature Communications 2026", which is not resolvable as written).

### 5. REFUTE  → RESOLVED (add arXiv id)
- Resolvable URL: https://arxiv.org/abs/2502.19414  (GitHub: falsifiers/REFUTE)
- Exact title: *Can Language Models Falsify? Evaluating Algorithmic Reasoning with
  Counterexample Creation*. Year: 2025 (submitted 2025-02-26).
- Confirmed by: arXiv + Semantic Scholar + OpenReview (forum M7cl4Ldw61) + project site
  falsifiers.github.io.
- Claim check: REFUTE is a benchmark for automated counterexample creation / falsification
  of incorrect solutions. SUPPORTS the "automated scientific-claim falsification is an
  established sub-genre" framing. (Note: it is code/algorithmic falsification, not
  biomedical claim falsification — keep the framing as "established sub-genre," which is
  accurate; do not imply it is domain-specific to AD or imaging.)
- ACTION: add the arXiv id 2502.19414 so the reference resolves.

### 6. AI-Scientist-v2  → RESOLVED (add arXiv id)
- Resolvable URL: https://arxiv.org/abs/2504.08066  (code: github.com/SakanaAI/AI-Scientist-v2)
- Exact title: *The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via
  Agentic Tree Search*. Authors: Sakana AI et al. Year: 2025 (submitted 2025-04-10).
- Confirmed by: arXiv + HuggingFace papers + Sakana AI (pub.sakana.ai) + ResearchGate.
- Claim check: end-to-end agentic scientific-discovery system (agentic tree search) that
  produced an AI-generated workshop-accepted paper. SUPPORTS the "automated scientific
  discovery / falsification sub-genre" framing.
- ACTION: add arXiv id 2504.08066.

---

## APPLY THIS — exact edits for WAVE 2 (backend)

### Edit 1 — `src/neuroad/calibration.py`, replace the whole `PRIOR_ART` list (lines 14-27)

Replace with:

```python
PRIOR_ART = [
    ("Batch Effects in Brain Foundation Model Embeddings",
     "arXiv:2604.14441 (2026)",
     "Tao et al. show brain-FM embeddings (BrainLM, SwiFT) encode substantial batch/"
     "site variability that often dominates diagnosis-related signal — the same 'star' "
     "mechanic. We cite it, we don't claim it. https://arxiv.org/abs/2604.14441"),
    ("Pretrained, Frozen, Still Leaking: Auditing Cross-Encoder Attribute Transfer in "
     "EEG Foundation Models",
     "arXiv:2606.09189 (2026)",
     "Tai audits attribute leakage from FROZEN foundation-model embeddings with "
     "subject-disjoint lower bounds. Same-genre evidence (EEG modality) that frozen "
     "embeddings leak protected attributes. https://arxiv.org/abs/2606.09189"),
    ("Towards Robust Foundation Models for Digital Pathology (PathoROB)",
     "Nature Communications 2026, doi:10.1038/s41467-026-73923-2",
     "Digital-pathology robustness benchmark (PathoROB): biological vs non-biological "
     "variation across 34 medical centers. Same genre, different modality. "
     "https://www.nature.com/articles/s41467-026-73923-2"),
    ("REFUTE (Can Language Models Falsify?) / The AI Scientist-v2",
     "arXiv:2502.19414 (2025) / arXiv:2504.08066 (2025)",
     "Automated scientific-claim falsification is an established sub-genre; our "
     "novelty is the closed AD-specific loop, not falsification per se."),
]
```

Rationale: full titles added; unverifiable "leakage margin ~0.16-0.37" number REMOVED
(could not be verified against the abstract — violates the file's own "no free-floating
number" rule); PathoROB and REFUTE/AI-Scientist-v2 now carry resolvable ids/DOI.

### Edit 2 — `src/neuroad/calibration.py`, `FACTS["neurojepa"]` (lines 42-46)

Change the first sentence so the cited title matches what a reviewer sees on arXiv. Replace:

```
        "Neuro-JEPA (hyphenated; NYU/NYUMedML, arXiv:2606.14957): self-supervised "
```
with:
```
        "Neuro-JEPA — 'Learning Sparse Latent Predictive Foundation Model for Multimodal "
        "Neuroimaging' (Huang et al., NYU Langone/MGH, arXiv:2606.14957): self-supervised "
```
(Leave the rest of the entry — 1.55M scans, T1w/T2w/FLAIR, JEPA+MoE, license, SUPPORTED —
unchanged; it is accurate.)

### Edit 3 — `README.md`, Positioning bullets (lines 18-27)

Replace the four bullets with:

```markdown
- *Batch Effects in Brain Foundation Model Embeddings* — arXiv:2604.14441 (Tao et al., 2026;
  https://arxiv.org/abs/2604.14441). Brain-FM embeddings (BrainLM, SwiFT) encode batch/site
  variability that often dominates the biological outcome — the same "star" mechanic our
  leakage test exploits.
- *Pretrained, Frozen, Still Leaking: Auditing Cross-Encoder Attribute Transfer in EEG
  Foundation Models* — arXiv:2606.09189 (Tai, 2026; https://arxiv.org/abs/2606.09189).
  Subject-disjoint lower bounds on attribute leakage from **frozen** embeddings (EEG
  modality; same-genre evidence).
- *Towards Robust Foundation Models for Digital Pathology* (PathoROB) — Nature Communications
  2026, doi:10.1038/s41467-026-73923-2 (https://www.nature.com/articles/s41467-026-73923-2).
  Biological vs non-biological variation across 34 medical centers (digital pathology).
- *REFUTE — Can Language Models Falsify?* (arXiv:2502.19414, 2025) and *The AI Scientist-v2*
  (arXiv:2504.08066, 2025). Automated claim falsification is an established sub-genre.
```

Rationale: dropped the unverifiable "leakage margin ~0.16-0.37" figure; added resolvable
ids/DOI for all four; corrected the "Still Leaking" title (EEG modality) so the pitch
cannot be attacked as misattributing modality.

### Edit 4 — `README.md`, Neuro-JEPA mention (line ~121)

No hard change required; the id 2606.14957 is correct. OPTIONAL for polish: append the real
paper title so a clicking reviewer sees a matching title, e.g. after
"the **actual frozen Neuro-JEPA ViT-B MoE**" add a parenthetical
"(paper: *Learning Sparse Latent Predictive Foundation Model for Multimodal Neuroimaging*,
arXiv:2606.14957)". Verify the "ViT-B" architecture descriptor against the paper before
shipping — the abstract describes JEPA + Mixture-of-Experts; "ViT-B" was not confirmed in
this pass and should be checked by whoever owns the embeddings claim.

---

## Open items for other owners (NOT this wave's edits)

- **Neuro-JEPA "ViT-B" descriptor** (README line ~122): not verified in this citation pass;
  the paper describes JEPA + MoE over T1w/T2w/FLAIR but I did not confirm a "ViT-B" backbone.
  Embeddings/model owner should confirm or soften.
- **Neuro-JEPA license** (code MIT / weights CC BY-NC-ND): out of scope here; gated-weights
  owner should confirm against the arXiv paper / repo before the license line ships.
- **"leakage margin ~0.16-0.37"**: removed as unverifiable. If a real number is wanted, pull
  it directly from arXiv:2606.09189's tables and cite the table — do not reintroduce the old
  figure.
