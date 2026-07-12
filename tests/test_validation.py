"""Tests for the Output target-prioritization outcome-validation layer.

All offline / deterministic (no network, no creds). They verify:
  * the metrics core is correct on toy fixtures (incl. a real AUC>0.5 & p<0.05
    significance smoke on a controlled perfect ranking);
  * the offline reports are deterministic, provenance-stamped, and NEVER raise;
  * the two honesty guards are wired in: the CIRCULAR overall-score signal is
    real (OT drugs, AUC>0.5, p<0.05) and flagged optimistic, while the honest
    held-out AUC is reported alongside it;
  * every gold gene carries a citation.

Deliberately NO assertion that the non-circular OFFLINE held-out AUC clears 0.5:
the bundled snapshots are curation-biased with a tiny background, so the honest
offline non-circular signal is at/below chance BY CONSTRUCTION. Forcing a passing
assertion there would fabricate a result — the rigorous read requires the live
run (prefer_offline=False), which tests do not touch.
"""
from __future__ import annotations

from neuroad.harness import validation as V


# --------------------------------------------------------------------------
# Metrics core — toy fixtures with controlled ground truth
# --------------------------------------------------------------------------

def test_precision_and_recall_at_k_on_toy_ranking():
    ranked = ["A", "B", "C", "D", "E"]      # best-first
    gold = frozenset({"A", "C", "X"})       # X is out-of-universe
    assert V.precision_at_k(ranked, gold, 2) == 0.5    # A gold, B not
    assert V.precision_at_k(ranked, gold, 1) == 1.0    # A gold
    # recall denominator = recoverable gold in universe = {A, C} = 2
    assert V.recall_at_k(ranked, gold, 1) == 0.5       # found A
    assert V.recall_at_k(ranked, gold, 3) == 1.0       # found A and C
    # degenerate guards return None, never raise
    assert V.precision_at_k(ranked, gold, 0) is None
    assert V.precision_at_k([], gold, 3) is None
    assert V.recall_at_k(ranked, frozenset({"X"}), 3) is None


def test_roc_auc_perfect_and_reversed():
    ranked = ["A", "B", "C", "D"]
    scores = [4.0, 3.0, 2.0, 1.0]
    gold = frozenset({"A", "B"})            # top-scored are gold -> AUC 1.0
    assert V.roc_auc(ranked, scores, gold) == 1.0
    rev = frozenset({"C", "D"})             # bottom-scored are gold -> AUC 0.0
    assert V.roc_auc(ranked, scores, rev) == 0.0
    # single-class labels -> undefined, None not exception
    assert V.roc_auc(ranked, scores, frozenset()) is None
    assert V.roc_auc(ranked, scores, frozenset({"A", "B", "C", "D"})) is None


def test_permutation_pvalue_significant_on_perfect_ranking():
    # A clean, controlled separation: real AUC>0.5 with a significant p-value.
    ranked = [f"g{i}" for i in range(20)]
    scores = [float(20 - i) for i in range(20)]     # descending
    gold = frozenset(ranked[:5])                     # the 5 top-scored are gold
    auc = V.roc_auc(ranked, scores, gold)
    p = V.permutation_pvalue(ranked, scores, gold, n_perm=2000, seed=0)
    assert auc is not None and auc > 0.5
    assert p is not None and p < 0.05
    # deterministic given the seed
    p2 = V.permutation_pvalue(ranked, scores, gold, n_perm=2000, seed=0)
    assert p == p2
    # p-value is add-one smoothed, so never exactly 0 and always in (0, 1]
    assert 0.0 < p <= 1.0


# --------------------------------------------------------------------------
# Gold sets — every gene cited
# --------------------------------------------------------------------------

def test_every_gold_gene_carries_a_citation():
    for gs in V.GOLD_SETS.values():
        assert gs.genes, f"{gs.name} is empty"
        for gene in gs.genes:
            assert gene.gene and gene.gene.isupper() is not None
            assert gene.citation and len(gene.citation) > 10, \
                f"{gs.name}:{gene.gene} lacks a citation"
    # spot-check provenance content
    assert any("35379992" in g.citation for g in V.GWAS_GOLD.genes)
    assert any("FDA" in g.citation for g in V.DRUG_GOLD.genes)


# --------------------------------------------------------------------------
# Offline reports — deterministic, provenance-stamped, never raise
# --------------------------------------------------------------------------

def test_offline_validators_never_raise_and_are_stamped():
    reports = [
        V.validate_pi4ad_gwas(prefer_offline=True, n_perm=200),
        V.validate_opentargets_gwas(prefer_offline=True, n_perm=200),
        V.validate_opentargets_drugs(prefer_offline=True, n_perm=200),
    ]
    for r in reports:
        d = r.to_dict()
        assert d["source"] == "offline_snapshot"
        assert d["background_size"] > 0
        assert d["n_gold"] > 0                      # gold genes are in the universe
        assert d["gold_citations"]                  # citations rode along
        assert "offline_snapshot" in d["caveat"]    # curation-bias caveat present
        assert set(d["precision_at_k"]) == {5, 10, 20}
        assert d["roc_auc"] is None or 0.0 <= d["roc_auc"] <= 1.0
        assert d["permutation_p"] is None or 0.0 < d["permutation_p"] <= 1.0


def test_offline_reports_are_deterministic():
    a = V.validate_opentargets_drugs(prefer_offline=True, n_perm=300, seed=0).to_dict()
    b = V.validate_opentargets_drugs(prefer_offline=True, n_perm=300, seed=0).to_dict()
    assert a == b


def test_run_default_validation_is_serializable_and_stamped():
    out = V.run_default_validation(prefer_offline=True, n_perm=100)
    assert isinstance(out, list) and len(out) == 3
    for d in out:
        assert isinstance(d, dict)
        assert d["source"] in ("offline_snapshot", "live")
        assert "background_size" in d and "roc_auc" in d and "permutation_p" in d


# --------------------------------------------------------------------------
# Honesty Guard 1 (circularity) — the real signal is the circular one, flagged
# --------------------------------------------------------------------------

def test_circular_drug_signal_is_real_but_flagged_optimistic():
    # Naive overall score vs the drug gold set: this is CIRCULAR (the clinical
    # datatype defines the gold set) — it is genuinely significant AND must be
    # labeled optimistic so it is never read as an unbiased validation.
    naive = V.validate_opentargets_drugs(
        prefer_offline=True, held_out=False, n_perm=2000, seed=0).to_dict()
    assert naive["optimistic"] is True
    assert naive["evidence_mode"] == "overall"
    assert naive["roc_auc"] is not None and naive["roc_auc"] > 0.5
    assert naive["permutation_p"] is not None and naive["permutation_p"] < 0.05
    assert "CIRCULAR" in naive["caveat"]


def test_held_out_mode_reports_both_and_is_honest():
    # Held-out drug validation: the honest (non-clinical) AUC is the primary
    # roc_auc; the circular overall AUC rides along as naive_roc_auc for
    # comparison. Both are present; we do NOT assert the held-out number clears
    # 0.5 (offline curation bias makes that false by construction).
    held = V.validate_opentargets_drugs(
        prefer_offline=True, held_out=True, n_perm=200, seed=0).to_dict()
    assert held["evidence_mode"] == "non_clinical"
    assert held["optimistic"] is False
    assert held["roc_auc"] is not None
    assert held["naive_roc_auc"] is not None
    # the honest held-out AUC drops below the circular naive one
    assert held["roc_auc"] < held["naive_roc_auc"]
    assert "Held-out" in held["caveat"]

    held_g = V.validate_opentargets_gwas(
        prefer_offline=True, held_out=True, n_perm=200, seed=0).to_dict()
    assert held_g["evidence_mode"] == "non_genetic"
    assert held_g["naive_roc_auc"] is not None


# --------------------------------------------------------------------------
# Honesty Guard 2 (curation bias) — offline background is tiny and stamped
# --------------------------------------------------------------------------

def test_offline_background_is_small_and_flagged():
    ot = V.validate_opentargets_gwas(prefer_offline=True, n_perm=50).to_dict()
    pi = V.validate_pi4ad_gwas(prefer_offline=True, n_perm=50).to_dict()
    # bundled snapshots are curated to a small canonical background
    assert ot["background_size"] <= 100
    assert pi["background_size"] <= 1000
    for d in (ot, pi):
        assert "enriched-by-construction" in d["caveat"]
        assert "prefer_offline=False" in d["caveat"]     # points to rigorous run


# --------------------------------------------------------------------------
# Rigor hardening — bootstrap CI, BH-FDR, degree-matched null, decoy control
# --------------------------------------------------------------------------

def test_bootstrap_auc_ci_brackets_and_is_deterministic():
    ranked = [f"g{i}" for i in range(20)]
    scores = [float(20 - i) for i in range(20)]     # descending
    gold = frozenset(ranked[:5])                     # top-scored are gold -> AUC 1.0
    ci = V.bootstrap_auc_ci(ranked, scores, gold, n_boot=500, seed=0)
    assert ci is not None
    lo, hi = ci
    assert 0.0 <= lo <= hi <= 1.0
    assert hi >= 0.9                                  # a near-perfect ranking
    # deterministic given the seed
    assert V.bootstrap_auc_ci(ranked, scores, gold, n_boot=500, seed=0) == ci
    # degenerate (single-class) -> None, never raises
    assert V.bootstrap_auc_ci(ranked, scores, frozenset(), n_boot=100) is None


def test_benjamini_hochberg_values_order_and_none_passthrough():
    # Known BH result for [0.01, 0.02, 0.03, 0.04] with m=4.
    q = V.benjamini_hochberg([0.01, 0.02, 0.03, 0.04])
    assert q == [0.04, 0.04, 0.04, 0.04]
    # None entries pass through and are excluded from m (m=2 here).
    q2 = V.benjamini_hochberg([0.01, None, 0.5])
    assert q2[1] is None
    assert q2[0] is not None and 0.0 <= q2[0] <= 1.0
    # q-values are monotone non-decreasing in p rank and clipped to [0,1].
    q3 = V.benjamini_hochberg([0.001, 0.6, 0.9])
    assert all(x is None or 0.0 <= x <= 1.0 for x in q3)
    assert q3[0] <= q3[1] <= q3[2]


def test_degree_matched_null_returns_control_dict_or_none():
    # Toy: gold genes are the highest-scored AND highest-degree; a degree-matched
    # null should therefore sit well below the observed (which is ~1.0).
    genes = [f"g{i}" for i in range(20)]
    scores = [float(20 - i) for i in range(20)]
    gold = frozenset(genes[:4])
    degrees = {g: float(len(genes) - i) for i, g in enumerate(genes)}
    res = V.degree_matched_null_auc(genes, scores, gold, degrees,
                                    n_draws=300, seed=0)
    assert res is not None
    for k in ("observed_auc", "null_mean", "null_ci", "empirical_p"):
        assert k in res
    assert 0.0 <= res["empirical_p"] <= 1.0
    # degenerate gold -> None, never raises
    assert V.degree_matched_null_auc(genes, scores, frozenset(), degrees) is None


def test_decoy_gold_is_registered_and_housekeeping():
    assert V.DECOY_GOLD.name in V.GOLD_SETS
    syms = V.DECOY_GOLD.symbols
    assert {"ACTB", "GAPDH", "B2M"} <= syms          # canonical housekeeping genes
    # decoys are NOT AD-risk genes (disjoint from the GWAS gold set)
    assert not (syms & V.GWAS_GOLD.symbols)


def test_validate_with_ci_populates_ci_and_todict_keys():
    ranked = [f"g{i}" for i in range(20)]
    scores = [float(20 - i) for i in range(20)]
    uni = V.RankingUniverse(genes=ranked, scores=scores, source="live")
    gold = V.GoldSet(name="toy", description="toy",
                     genes=tuple(V.GoldGene(g, "cited toy gold gene") for g in ranked[:5]))
    r = V.validate(uni, gold, ranking_source="toy", evidence_mode="toy",
                   n_perm=200, with_ci=True, n_boot=300).to_dict()
    assert r["roc_auc_ci"] is not None and len(r["roc_auc_ci"]) == 2
    # new keys are always present in the serialization (default None otherwise)
    for k in ("roc_auc_ci", "q_value", "null_auc_mean"):
        assert k in r
    # with_ci defaults off -> no CI, unchanged for existing callers
    r0 = V.validate(uni, gold, ranking_source="toy", evidence_mode="toy",
                    n_perm=50).to_dict()
    assert r0["roc_auc_ci"] is None
