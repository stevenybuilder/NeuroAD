"""Offline / deterministic tests for the learned-ranker module.

No network: they verify rank-normalization (outlier-robustness + tie handling),
missing-value imputation with coverage tracking, and that fit_learned_ranker returns
a well-formed, deterministic report on a controlled separable fixture (and degrades to
None on degenerate labels rather than raising).
"""
from __future__ import annotations

from neuroad.harness import ranking_model as M


def test_rank_normalize_is_outlier_robust_and_handles_ties_and_none():
    # An extreme outlier must NOT compress the rest (unlike min-max).
    vals = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 1000.0, "e": None}
    n = M.rank_normalize(vals)
    assert n["a"] == 0.0 and n["d"] == 1.0            # ends of the rank scale
    assert n["b"] == 1 / 3 and n["c"] == 2 / 3         # evenly spaced by rank
    assert n["e"] is None                               # missing stays missing
    # ties share the average rank
    t = M.rank_normalize({"x": 5.0, "y": 5.0, "z": 9.0})
    assert t["x"] == t["y"]
    # all-missing -> all None, no raise
    assert M.rank_normalize({"p": None, "q": None}) == {"p": None, "q": None}


def test_build_design_imputes_and_reports_coverage():
    genes = ["g1", "g2", "g3", "g4"]
    raw = {f: {} for f in M.FEATURES}
    raw["ot_assoc_heldout"] = {"g1": 0.1, "g2": 0.9, "g3": 0.5, "g4": 0.7}
    raw["lincs_efficacy"] = {"g1": 1.0, "g2": None, "g3": None, "g4": None}  # 25% cov
    X, cov = M.build_design(genes, raw)
    assert len(X) == 4 and len(X[0]) == len(M.FEATURES)
    assert cov["ot_assoc_heldout"] == 1.0
    assert cov["lincs_efficacy"] == 0.25
    # a fully-missing feature is imputed to 0.5 everywhere
    idx = M.FEATURES.index("pi4ad_priority")
    assert all(row[idx] == 0.5 for row in X)


def test_fit_learned_ranker_on_separable_fixture_is_sane_and_deterministic():
    # 6 gold genes with high ot_heldout + a clean background -> the model should learn
    # a positive ot weight and achieve OOF AUC > 0.5 on a clean separation.
    genes = [f"gold{i}" for i in range(6)] + [f"bg{i}" for i in range(24)]
    gold = frozenset(f"GOLD{i}" for i in range(6))
    raw = {f: {} for f in M.FEATURES}
    raw["ot_assoc_heldout"] = {g: (0.9 if g.startswith("gold") else 0.1) for g in genes}
    raw["pi4ad_priority"] = {g: (0.8 if g.startswith("gold") else 0.2) for g in genes}
    fit = M.fit_learned_ranker(genes, raw, gold, n_boot=300, n_perm=300, seed=0)
    assert fit is not None
    assert fit["n_gold_in_universe"] == 6
    assert set(fit["learned_weights"]) == set(M.FEATURES)
    assert fit["oof_auc"] is not None and fit["oof_auc"] > 0.5
    assert fit["learned_weights"]["ot_assoc_heldout"] > 0     # clean signal is positive
    assert 0.0 <= fit["brier"] <= 1.0
    # deterministic given the seed
    fit2 = M.fit_learned_ranker(genes, raw, gold, n_boot=300, n_perm=300, seed=0)
    assert fit2["oof_auc"] == fit["oof_auc"]
    assert fit2["learned_weights"] == fit["learned_weights"]


def test_fit_returns_none_on_degenerate_labels():
    genes = [f"g{i}" for i in range(10)]
    raw = {f: {g: 0.5 for g in genes} for f in M.FEATURES}
    # too few positives -> None (not an exception)
    assert M.fit_learned_ranker(genes, raw, frozenset({"G0"}), n_perm=10) is None
    # all-positive -> None
    assert M.fit_learned_ranker(genes, raw, frozenset(g.upper() for g in genes),
                                n_perm=10) is None
