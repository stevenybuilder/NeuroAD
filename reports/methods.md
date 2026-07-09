# NeuroAD Discovery Engine — Methods (auto-generated)

One small linear head is pointed at different label columns of a frozen structural-embedding table. Pointed at `conversion`/`dx_binary` it is the signal; pointed at `site`/`scanner` it is the leakage test. The gauntlet chains five adversarial challenges; the headline metric is the subject-disjoint leakage margin (outcome AUC - scanner AUC). Survivors are gated behind a plasma-biomarker anchor before any biology is proposed. All demo numbers are calibrated in `src/neuroad/calibration.py`.

**Permutation-null limitation.** `probe.auc_ci_perm` computes the OOF scores once and holds them fixed under the label permutation (the probe is never refit, and the bootstrap resamples the frozen OOF `(y, proba)` pair). Model-selection variance is therefore under-propagated and the reported permutation `p` is a LOWER BOUND on the true p-value (anticonservative) — a deliberate speed tradeoff, disclosed here.
