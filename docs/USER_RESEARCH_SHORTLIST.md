# User Research Shortlist — AD / Neuroimaging Researchers

Date prepared: 2026-07-09

Purpose: rapid expert interviews for NeuroAD Discovery Engine before hackathon submission. This list prioritizes early-career researchers whose recent work maps to the core thesis: researcher-driven discovery from brain-MRI/foundation-model embeddings, followed by a falsification gauntlet for scanner/site leakage, age/sex, brain-age, biomarker anchoring, and replication.

## Ranking Rubric

Rank is based on:

1. Direct relevance to NeuroAD's thesis: discovery plus referee, not just classification.
2. Fit with the gauntlet: scanner/site leakage, batch effects, harmonization, brain-age, biomarker anchors, subtype reproducibility, or foundation-model MRI embeddings.
3. Likelihood of actionable feedback in a 15-20 minute interview.
4. Early-career fit: PhD students, postdocs, or first/co-first authors preferred over senior PIs.
5. Recency: 2024-2026 papers or public research activity.

Contact note: use only public professional contact routes. Where a direct email was not verified from public sources during the source pass, the table says "not publicly verified" and gives the safest route.

## Top 15

| Rank | Person | Affiliation | Email / Contact Route | Recent Relevant Work | Why Ranked Here |
|---:|---|---|---|---|---|
| 1 | Ye Tao | Rutgers ECE; collaborators include TReNDS / Calhoun group | Direct email not publicly verified in source pass; contact via Rutgers profile or corresponding authors on paper | *Batch Effects in Brain Foundation Model Embeddings*, arXiv, submitted Apr 15 2026; v2 Jun 8 2026 | Exact thesis match. This paper shows brain foundation-model embeddings can encode batch/site variability that dominates diagnosis-related signal. Best person to critique the STAR scanner/site leakage test. |
| 2 | Haoxu Huang | NYU Center for Data Science / NYU Langone | Direct email not publicly verified in source pass; contact via NYU / NYU Langone profile or Neuro-JEPA author page | *Learning Sparse Latent Predictive Foundation Model for Multimodal Neuroimaging* (Neuro-JEPA), arXiv, submitted Jun 12 2026 | Directly relevant because NeuroAD uses frozen Neuro-JEPA-style structural MRI embeddings as the substrate. Can critique whether the way we probe frozen embeddings is scientifically fair. |
| 3 | Moona Mazher | UCL Hawkes Institute | Direct email not publicly verified in source pass; contact via UCL profile / Hawkes Institute | *Towards Generalisable Foundation Models for 3D Brain MRI*, arXiv, Oct 27 2025 | BrainFound / 3D brain-MRI foundation model work. Strong fit for feedback on frozen embeddings, label-scarce evaluation, multi-dataset transfer, and whether NeuroAD is meaningfully different from another model paper. |
| 4 | Emma Prevot | UCL / Oxford Statistics | Direct email not publicly verified in source pass; contact via UCL/Oxford profile or paper page | *How reproducible are data-driven subtypes of Alzheimer's disease atrophy?*, arXiv, Nov 29 2024 | Best fit for subtype reproducibility. NeuroAD's "candidate phenotype must survive stability + confound gauntlet" maps directly to this work. |
| 5 | Divyanshu Tak | MGB AIM / Harvard | Direct email not publicly verified in source pass; contact via MGB AIM / BrainIAC lab route | *BrainIAC: A generalizable foundation model for analysis of human brain MRI*, Nature Neuroscience, Feb 5 2026 | Built/evaluated a clinical brain-MRI foundation model with downstream neurodegenerative tasks. Can challenge the product's claim that the novelty is the referee layer, not another foundation model. |
| 6 | Xiaotong Wei | University of Electronic Science and Technology of China | Direct email not publicly verified in source pass; contact via corresponding author route | *Mapping heterogeneous brain structural subtypes in AD/MCI using normative models*, Translational Psychiatry, Mar 2 2026 | Very close to the AD/MCI subtype and normative-deviation version of the product. Good interview for whether our "stability-vetted candidate phenotype" framing is credible. |
| 7 | Bethany Little | Newcastle CNNP Lab | Direct email not publicly verified in source pass; contact via CNNP / Yujiang Wang lab route | *Brain MoNoCle: A Mouse Brain Normative Calculator*, arXiv v4, Apr 23 2025 | Normative modeling tooling perspective. Useful for feedback on calibration, scanner adjustment, and what an analyst expects from a practical neuroimaging workbench. |
| 8 | Nida Alyas | Newcastle CNNP Lab | Direct email not publicly verified in source pass; contact via CNNP lab route | *Normative Modelling in Neuroimaging: Practical Guide*, arXiv, Sep 8 2025 | Strong methodological fit for scanner-matched controls, calibration size, and whether NeuroAD's confound screens are enough for an honest "candidate" claim. |
| 9 | Fangqi Cheng | University of Glasgow | Direct email not publicly verified for first author; corresponding route via Xiaochen Yang / Glasgow | *Self-Supervised Cross-Encoder for Neurodegenerative Disease Diagnosis*, arXiv, Sep 9 2025 | ADNI/OASIS longitudinal MRI and self-supervised learning. Good critique target for model evaluation, interpretability, and avoiding "yet another AD classifier." |
| 10 | Cheng Wang | CUHK | Direct email not publicly verified in source pass; contact via CUHK author page / NeuroSTORM paper | *NeuroSTORM: self-supervised foundation model for task-free fMRI analysis*, arXiv, Jun 11 2025; v2 Mar 24 2026 | Less AD-specific, but relevant to foundation-model neuroimaging, reproducibility, and what it means to use embeddings as a scientific substrate. |
| 11 | Cameron Shand | Francis Crick Institute / UCL CMIC | Direct email not publicly verified in source pass; contact via Crick/UCL profile or paper page | Coauthor, *How reproducible are data-driven subtypes of Alzheimer's disease atrophy?*, arXiv, Nov 29 2024 | Strong technical reviewer for SuStaIn, subtype reproducibility, cohort-split methodology, and whether our bootstrap-Jaccard / replication framing is convincing. |
| 12 | Akhil Kondepudi | University of Michigan MLINS | Direct email not publicly verified in source pass; contact via MLINS / NeuroVFM site | *Health system learning achieves generalist neuroimaging models*, arXiv, Nov 23 2025 | Health-system-scale neuroimaging foundation model perspective. Useful for clinical translation critique and "what would make this operationally useful?" |
| 13 | Biniam A. Garomsa | MGB AIM / Harvard | Direct email not publicly verified in source pass; contact via MGB AIM / BrainIAC lab route | Coauthor, *BrainIAC*, Nature Neuroscience, Feb 5 2026 | Good alternate BrainIAC contact if Divyanshu Tak is unreachable. Same foundation-model clinical MRI context. |
| 14 | Jianwei Tai | Anhui University | Direct email not publicly verified in source pass; contact via arXiv author route / university profile | *Pretrained, Frozen, Still Leaking: Auditing Cross-Encoder Attribute Transfer in EEG Foundation Models*, arXiv, Jun 8 2026 | Different modality, but directly relevant to frozen-embedding attribute leakage. Useful for the general "foundation embeddings leak nuisance variables" framing. |
| 15 | Xiongri Shen | Shenzhen University / collaborators | Direct email not publicly verified in source pass; GitHub route: `SXR3015` | *Brain-Atlas-Guided Generative Counterfactual Attention Network for Early Detection of Alzheimer's Disease*, arXiv, Jun 2026 | ADNI/SCD/MCI explainability and early-detection modeling. Useful for probing whether the workbench's explanations and claim cards would be trusted by ML-neuroimaging researchers. |

## Fastest Outreach Order

If time is tight, start with:

1. Ye Tao
2. Moona Mazher
3. Emma Prevot
4. Cameron Shand
5. Fangqi Cheng
6. Xiaotong Wei
7. Bethany Little
8. Nida Alyas

These are the strongest matches for the product's scientific credibility rather than generic AI/product polish.

## Suggested Interview Ask

Subject: 15-min feedback request: AD MRI discovery/referee tool

Short note:

> I am building a hackathon MVP called NeuroAD Discovery Engine: a researcher-driven tool that takes a brain-MRI embedding signal or candidate phenotype and runs a falsification gauntlet for scanner/site leakage, age/sex, brain-age, biomarker anchoring, and replication. The goal is not to claim a new biomarker, but to decide whether a candidate imaging phenotype is worth follow-up or likely an artifact. Given your work on [specific paper], I would value a 15-minute critique of whether the evidence card and failure modes would be useful to an AD/neuroimaging researcher.

Best questions:

1. When was the last time you worried an imaging/ML signal was scanner, site, age, or atrophy rather than biology?
2. What do you currently do to test that?
3. Would this evidence card change whether you spend time on a signal?
4. What evidence would make you trust or reject the verdict?
5. Would you run this on ADNI/OASIS/your own cohort if setup took under 30 minutes?

## Source Links

- *Batch Effects in Brain Foundation Model Embeddings* — https://arxiv.org/abs/2604.14441
- *Learning Sparse Latent Predictive Foundation Model for Multimodal Neuroimaging* / Neuro-JEPA — https://arxiv.org/abs/2606.14957
- *Towards Generalisable Foundation Models for 3D Brain MRI* — https://arxiv.org/abs/2510.23415
- *How reproducible are data-driven subtypes of Alzheimer's disease atrophy?* — https://arxiv.org/abs/2412.00160
- *BrainIAC: A generalizable foundation model for analysis of human brain MRI* — https://www.nature.com/articles/s41593-026-02202-6
- *Brain MoNoCle* — https://arxiv.org/abs/2406.01107
- *Normative Modelling in Neuroimaging: Practical Guide* — https://arxiv.org/abs/2509.07237
- *Self-Supervised Cross-Encoder for Neurodegenerative Disease Diagnosis* — https://arxiv.org/abs/2509.07623
- *NeuroSTORM* — https://arxiv.org/abs/2506.11167
- *Health system learning achieves generalist neuroimaging models* / NeuroVFM — https://arxiv.org/abs/2511.18640
- *Pretrained, Frozen, Still Leaking* — https://arxiv.org/abs/2606.09189
- *Brain-Atlas-Guided Generative Counterfactual Attention Network for Early Detection of Alzheimer's Disease* — https://arxiv.org/abs/2606.01237
