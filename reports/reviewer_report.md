# Reviewer (Claude) — peer-review critique

## ADNI-A — strong candidate [REAL ADNI]

- A ‘strong candidate’ verdict is a licence to follow up, not to conclude — the referee promotes hypotheses, it does not confirm them.
- The brain-age control is a proxy, not a gold standard: it is embedding-derived (this cohort's fit: R2=0.48, MAE=4.0yr), so residual generic-aging signal can survive the adjustment and masquerade as disease-specific.
- The biomarker anchor likely rests on a partial complete-case subset; with realistic p-tau217 missingness the correlation is estimated on far fewer subjects than the headline cohort.
- Confirm the effective sample size per split — subject-disjoint cross-validation on a modest cohort leaves each fold thin.
- The leakage test audits the embeddings with the same linear-probe family it uses for the outcome, so it bounds — but cannot fully rule out — shared confounding; this is the mechanic quantified in ‘Pretrained, Frozen, Still Leaking: Auditing Cross-Encoder Attribute Transfer in EEG Foundation Models’ (arXiv:2606.09189 (2026)), which we cite rather than claim.
