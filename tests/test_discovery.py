"""
Tests for the Detective / discovery upgrade (agent B).

Covers:
  * the planted-phenotype cohort is contract-valid and carries ground truth,
  * reduce-then-cluster recovers the 3 planted phenotypes (ARI > 0.5) and
    returns bootstrap stability + projection trustworthiness,
  * discover_and_referee flags the scanner phenotype as an artifact and the
    tau-hot phenotype as a promotable phenotype.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from neuroad import contract, detective, discovery
from neuroad.data import synthetic


# --------------------------------------------------------------------------- #
# planted cohort
# --------------------------------------------------------------------------- #
def test_phenotype_cohort_is_contract_valid():
    df = synthetic.generate_phenotype_cohort(seed=0)
    contract.validate_table(df)                       # raises on any violation
    assert "phenotype" in df.columns
    assert set(df["phenotype"].dropna().unique()) == set(synthetic.PHENOTYPE_LEVELS)
    # every planted subgroup carries CN/MCI/AD so its sub-cohort is refereeable
    for pheno in synthetic.PHENOTYPE_LEVELS:
        sub = df[df["phenotype"] == pheno]
        assert set(sub["dx"].dropna().unique()) == set(contract.DX_LEVELS)


def test_phenotype_cohort_is_deterministic():
    a = synthetic.generate_phenotype_cohort(seed=3)
    b = synthetic.generate_phenotype_cohort(seed=3)
    np.testing.assert_allclose(contract.embedding_matrix(a),
                               contract.embedding_matrix(b), rtol=0, atol=1e-12)
    assert a["phenotype"].tolist() == b["phenotype"].tolist()


# --------------------------------------------------------------------------- #
# discover(): recovery + stability + trustworthiness
# --------------------------------------------------------------------------- #
def test_discover_recovers_planted_phenotypes():
    df = synthetic.generate_phenotype_cohort(seed=0)
    res = detective.discover(df)
    ari = adjusted_rand_score(df["phenotype"].astype(str), res["labels"])
    assert res["k"] >= 3, f"expected >=3 clusters, got {res['k']}"
    assert ari > 0.5, f"ARI {ari:.3f} too low — phenotypes not recovered"
    # stability is the primary quality gate and must be reported per cluster
    assert res["stability"] and len(res["stability"]) == res["k"]
    assert all(v is not None for v in res["stability"].values())
    assert res["trustworthiness"] is not None


def test_gmm_method_also_recovers():
    df = synthetic.generate_phenotype_cohort(seed=0)
    res = detective.discover(df, method="gmm")
    ari = adjusted_rand_score(df["phenotype"].astype(str), res["labels"])
    assert res["method"] == "gmm"
    assert ari > 0.5


def test_bootstrap_stability_shape():
    df = synthetic.generate_phenotype_cohort(seed=0)
    from neuroad.detective import _reduce, _n_reduced_dims
    from sklearn.preprocessing import StandardScaler
    X = contract.embedding_matrix(df)
    Xs = StandardScaler().fit_transform(X)
    X_red = _reduce(Xs, _n_reduced_dims(len(X), X.shape[1]))
    stab = detective.bootstrap_stability(X_red, k=3, B=20)
    assert set(stab.keys()) == {0, 1, 2}
    assert all(0.0 <= v <= 1.0 for v in stab.values())


# --------------------------------------------------------------------------- #
# discover_and_referee(): artifact vs promotable adjudication
# --------------------------------------------------------------------------- #
def test_discover_and_referee_adjudicates():
    df = synthetic.generate_phenotype_cohort(seed=0)
    out = discovery.discover_and_referee(df, B=20)

    assert out["ari"] > 0.5
    assert out["ami"] is not None
    assert len(out["clusters"]) >= 3

    by_pheno = {}
    for c in out["clusters"]:
        by_pheno.setdefault(c["dominant_phenotype"], c)

    # the scanner phenotype must be caught as an acquisition artifact
    scanner = by_pheno["scanner_artifact"]
    assert scanner["artifact"]["flag"] is True
    assert "artifact" in scanner["status"]
    assert not scanner["gauntlet"]["promoted"]

    # the tau-hot phenotype must survive its own gauntlet and be promotable
    tau = by_pheno["tau_hot"]
    assert tau["gauntlet"]["promoted"] is True
    assert tau["status"] == "promotable phenotype"
    assert not tau["artifact"]["flag"]

    # every refereed cluster carries a stability number
    assert all(c["stability"] is not None for c in out["clusters"])
