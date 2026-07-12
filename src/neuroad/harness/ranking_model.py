"""
ranking_model — a LEARNED, calibrated replacement for the hand-weighted composite.

The shipped composite (``harness/ranking.py``) fuses the discovery signals with
HAND-SET weights (PI4AD 0.30 / OT-heldout 0.35 / STRING 0.20 / pLDDT 0.15) and min-max
normalization. Those weights are a judgement call and min-max is outlier-sensitive. This
module turns the fusion into a fitted model: rank-normalized features, an L2-regularized
logistic regression whose coefficients ARE the data-derived weights, with leave-one-out
cross-validated out-of-fold AUC, a bootstrap CI, a permutation p-value, and a calibration
(Brier) score — plus the LINCS efficacy feature the hand composite never had.

HONEST FRAMING (low-n): the clean training label is a tiny gold set (~15 GWAS genes), so
this is a SENSITIVITY ANALYSIS — "what weights does the data imply, and how uncertain are
they?" — not a definitive re-weighting. Every number rides with its uncertainty.

Read-only side artifact: NOT imported by ``translation.py`` / the referee / the demo.
"""
from __future__ import annotations

import logging
from typing import Optional

_log = logging.getLogger("neuroad.harness.ranking_model")

#: Model features, in a fixed order. ``struct_plddt`` and ``lincs_efficacy`` have
#: narrower coverage than the rest; coverage is reported so imputed features are visible.
FEATURES: tuple[str, ...] = (
    "pi4ad_priority", "ot_assoc_heldout", "net_centrality", "struct_plddt",
    "lincs_efficacy",
)


def rank_normalize(values: dict) -> dict:
    """Rank-based normalization to [0,1] — outlier-robust (unlike min-max).

    Ties share the average rank; ``None`` stays ``None`` (imputed later). A single
    present value maps to 0.5."""
    present = {g: v for g, v in values.items() if v is not None}
    if not present:
        return {g: None for g in values}
    order = sorted(present, key=lambda g: present[g])
    n = len(order)
    ranks: dict = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and present[order[j + 1]] == present[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        norm = (avg / (n - 1)) if n > 1 else 0.5
        for k in range(i, j + 1):
            ranks[order[k]] = norm
        i = j + 1
    return {g: ranks.get(g) for g in values}


def build_design(genes: list[str], raw_features: dict) -> tuple:
    """Rank-normalize each feature over ``genes`` and impute missing to 0.5.

    ``raw_features`` maps each name in FEATURES to a ``{gene: value_or_None}`` dict.
    Returns ``(X, coverage)`` where X is a list of per-gene feature-vectors (aligned to
    FEATURES) and coverage is the present-fraction per feature (so a mostly-imputed
    feature is never mistaken for a measured one)."""
    normed = {f: rank_normalize(raw_features.get(f, {})) for f in FEATURES}
    coverage = {f: (sum(1 for g in genes if raw_features.get(f, {}).get(g) is not None)
                    / max(1, len(genes))) for f in FEATURES}
    X = [[(normed[f].get(g) if normed[f].get(g) is not None else 0.5)
          for f in FEATURES] for g in genes]
    return X, coverage


def fit_learned_ranker(genes: list[str], raw_features: dict, gold: frozenset,
                       *, n_boot: int = 2000, n_perm: int = 1000, seed: int = 0,
                       C: float = 1.0) -> Optional[dict]:
    """Fit + LOO-cross-validate the learned ranker. Returns a report dict or None.

    - **Learned weights**: coefficients of an L2 logistic fit on the FULL design
      (features are already on a common [0,1] rank scale, so coefficients are directly
      comparable — the data-derived analogue of the hand-set weights).
    - **Honest performance**: leave-one-gene-out CV out-of-fold predictions -> OOF AUC,
      bootstrap CI, label-shuffle permutation p, and Brier calibration score.
    - **Scores**: per-gene fitted probability (for ranking) from the full-fit model.
    Never raises — returns None on degeneracy/failure."""
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, brier_score_loss

        X, coverage = build_design(genes, raw_features)
        X = np.asarray(X, dtype=float)
        y = np.array([1 if g.upper() in {h.upper() for h in gold} else 0
                      for g in genes])
        if y.sum() < 3 or y.sum() == y.size or X.shape[0] != y.size:
            return None

        def _mk():
            # liblinear defaults to L2; passing penalty= is deprecated in sklearn 1.9.
            return LogisticRegression(C=C, solver="liblinear", max_iter=1000)

        # Full fit -> learned weights + in-universe scores.
        full = _mk().fit(X, y)
        coef = {f: float(c) for f, c in zip(FEATURES, full.coef_[0])}
        scores = full.predict_proba(X)[:, 1]

        # Leave-one-out out-of-fold predictions (honest generalization on tiny n).
        oof = np.zeros(y.size)
        for i in range(y.size):
            mask = np.ones(y.size, dtype=bool)
            mask[i] = False
            if y[mask].sum() == 0 or y[mask].sum() == mask.sum():
                oof[i] = y[mask].mean()
                continue
            oof[i] = _mk().fit(X[mask], y[mask]).predict_proba(X[i:i + 1])[:, 1][0]

        oof_auc = float(roc_auc_score(y, oof))
        brier = float(brier_score_loss(y, oof))

        # Bootstrap CI on the OOF AUC.
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(max(1, int(n_boot))):
            idx = rng.integers(0, y.size, y.size)
            if y[idx].sum() == 0 or y[idx].sum() == y[idx].size:
                continue
            boots.append(roc_auc_score(y[idx], oof[idx]))
        ci = ([float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
              if len(boots) >= 2 else None)

        # Permutation p-value on the OOF AUC (shuffle labels).
        ge = 0
        for _ in range(max(1, int(n_perm))):
            yp = rng.permutation(y)
            if yp.sum() == 0 or yp.sum() == yp.size:
                continue
            if roc_auc_score(yp, oof) >= oof_auc:
                ge += 1
        perm_p = (ge + 1) / (int(n_perm) + 1)

        gene_scores = sorted(({"gene": g, "learned_score": round(float(s), 4),
                               "is_gold": bool(y[i])}
                              for i, (g, s) in enumerate(zip(genes, scores))),
                             key=lambda r: r["learned_score"], reverse=True)
        return {
            "n_genes": int(y.size), "n_gold_in_universe": int(y.sum()),
            "features": list(FEATURES), "coverage": coverage,
            "learned_weights": coef,
            "oof_auc": oof_auc, "oof_auc_ci": ci, "permutation_p": perm_p,
            "brier": brier, "C": C,
            "gene_scores": gene_scores,
        }
    except Exception as exc:  # noqa: BLE001
        _log.debug("fit_learned_ranker failed: %r", exc)
        return None
