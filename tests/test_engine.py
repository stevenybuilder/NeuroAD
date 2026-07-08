"""
Unit tests for the core referee engine (M1).

These build SURVIVOR-like and KILL-like *contract-valid* DataFrames INLINE
(no dependency on the data agent's synthetic generator). The cohorts inject a
disease direction, an acquisition (scanner) direction, and an age direction into
the embedding so every gauntlet test has real structure to bite on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neuroad import contract, detective, leakage, probe, scoring
from neuroad.contract import Claim, TestResult as Result
from neuroad.gauntlet import run_gauntlet

D = 16
_DIS = slice(0, 5)      # disease direction
_SCAN = slice(5, 9)     # scanner/acquisition direction
_AGE = slice(9, 12)     # age direction
_NOISE = slice(12, 16)  # pure noise


def build_cohort(preset: str, seed: int = 0, n: int = 270) -> pd.DataFrame:
    """A contract-valid cohort with tunable disease vs scanner coupling."""
    rng = np.random.default_rng(seed)
    if preset == "SURVIVOR":
        w_d, w_s, anchor = 1.6, 0.8, True
    elif preset == "KILL":
        w_d, w_s, anchor = 0.28, 2.4, False
    else:
        raise ValueError(preset)

    sites = np.array(["siteA", "siteB", "siteC"])
    site = rng.choice(sites, size=n)
    scanner_of = {"siteA": "Prisma3T", "siteB": "Trio3T", "siteC": "GE15T"}
    scanner = np.array([scanner_of[s] for s in site])
    scan_code = np.array([np.where(sites == s)[0][0] for s in site], dtype=float)

    # Cohort composition: MCI (carry conversion), CN (brain-age), AD.
    dx = rng.choice(["MCI", "CN", "AD"], size=n, p=[0.6, 0.25, 0.15])
    d_latent = rng.normal(size=n)
    d_latent[dx == "CN"] -= 0.8
    d_latent[dx == "AD"] += 0.9

    age = 60 + 12 * (d_latent * 0.3) + rng.normal(0, 6, size=n)

    # Build the embedding from orthogonal directions.
    X = rng.normal(0, 1.0, size=(n, D))
    dis_dir = rng.normal(size=(5,))
    scan_dir = rng.normal(size=(4,))
    age_dir = rng.normal(size=(3,))
    X[:, _DIS] += w_d * np.outer(d_latent, dis_dir)
    X[:, _SCAN] += w_s * np.outer((scan_code - scan_code.mean()), scan_dir)
    X[:, _AGE] += 0.9 * np.outer((age - age.mean()) / age.std(), age_dir)

    # conversion label for MCI only, from the disease latent.
    conv = np.full(n, np.nan)
    mci = dx == "MCI"
    thr = np.median(d_latent[mci])
    conv[mci] = (d_latent[mci] > thr).astype(float)

    # plasma p-tau217: anchored to disease latent for survivor; noise for kill.
    ptau = np.full(n, np.nan)
    present = rng.random(n) < 0.6
    if anchor:
        ptau[present] = 0.9 + 0.6 * d_latent[present] + rng.normal(0, 0.5, present.sum())
    else:
        ptau[present] = 0.9 + rng.normal(0, 1.0, present.sum())

    df = pd.DataFrame({
        "subject_id": [f"S{i:04d}" for i in range(n)],
        "dx": pd.Categorical(dx, categories=contract.DX_LEVELS),
        "conversion": pd.array(conv, dtype="Int8"),
        "age": age,
        "sex": pd.Categorical(rng.choice(["M", "F"], size=n), categories=contract.SEX_LEVELS),
        "site": pd.Categorical(site),
        "scanner": pd.Categorical(scanner),
        "amyloid": pd.array(rng.integers(0, 2, n), dtype="Int8"),
        "p_tau217": ptau,
        "gfap": np.where(present, 100 + 40 * np.nan_to_num(d_latent) + rng.normal(0, 30, n), np.nan),
        "nfl": rng.normal(20, 5, n),
        "apoe4": pd.array(rng.integers(0, 3, n), dtype="Int8"),
    })
    for i in range(D):
        df[f"emb_{i}"] = X[:, i]
    contract.validate_table(df)
    return df


@pytest.fixture(scope="module")
def survivor():
    return build_cohort("SURVIVOR", seed=1)


@pytest.fixture(scope="module")
def kill():
    return build_cohort("KILL", seed=2)


# ---------------------------------------------------------------------------
def test_point_head_shapes(survivor):
    X, y, g = probe.point_head(survivor, "conversion")
    assert X.shape[0] == len(y) == len(g)
    assert set(np.unique(y)) <= {0, 1}
    # dx_binary drops MCI.
    Xb, yb, _ = probe.point_head(survivor, "dx_binary")
    assert set(np.unique(yb)) == {0, 1}
    assert Xb.shape[1] == D


def test_probe_learns_signal(survivor):
    X, y, g = probe.point_head(survivor, "conversion")
    auc = probe.cross_val_auc(X, y, groups=g)
    assert auc > 0.6, f"survivor conversion AUC too low: {auc}"


def test_run_gauntlet_returns_five(survivor):
    tests = run_gauntlet(survivor, Claim("c1", "MCI converters vs stable", "conversion"))
    assert len(tests) == 5
    keys = {t.key for t in tests}
    assert keys == {"age_sex", "site_scanner", "brain_age",
                    "biomarker_anchor", "replication"}


def test_leakage_margin_sign_differs(survivor, kill):
    ms = leakage.leakage_margin(survivor, "conversion")
    mk = leakage.leakage_margin(kill, "conversion")
    assert ms["margin"] > 0, f"survivor margin should be positive: {ms}"
    assert mk["margin"] < ms["margin"], f"kill margin should be worse: {mk} vs {ms}"
    assert mk["margin"] <= 0.05, f"kill should not clearly beat the scanner: {mk}"


def test_double_dissociation(survivor, kill):
    ds = leakage.double_dissociation(survivor, "conversion")
    dk = leakage.double_dissociation(kill, "conversion")
    assert ds["retained"] >= dk["retained"]


def test_confound_leaderboard(kill):
    board = leakage.confound_leaderboard(kill, "conversion")
    assert board and all("confound" in r and "variance_explained" in r for r in board)
    # sorted descending
    ve = [r["variance_explained"] for r in board]
    assert ve == sorted(ve, reverse=True)


def test_survivor_promotes(survivor):
    claim = Claim("c1", "MCI converters vs stable", "conversion")
    tests = run_gauntlet(survivor, claim)
    naive = {"metric": "AUC", "value": probe.cross_val_auc(*probe.point_head(survivor, "conversion")[:2])}
    card = scoring.build_claim_card(claim, naive, tests)
    assert card.score >= 40, f"survivor score too low: {card.score} ({card.verdict})"
    assert card.promoted, f"survivor should promote: {card.verdict}"
    # biomarker anchor should hold on the survivor.
    anchor = next(t for t in tests if t.key == "biomarker_anchor")
    assert anchor.result in (Result.PASSED, Result.WEAKENED)


def test_kill_does_not_promote(kill):
    claim = Claim("c2", "MCI converters vs stable", "conversion")
    tests = run_gauntlet(kill, claim)
    naive = {"metric": "AUC", "value": 0.58}
    card = scoring.build_claim_card(claim, naive, tests)
    assert not card.promoted, f"kill should NOT promote: {card.verdict} score={card.score}"
    site = next(t for t in tests if t.key == "site_scanner")
    assert site.result in (Result.FAILED, Result.WEAKENED)


def test_evidence_ledger_shape(survivor):
    claim = Claim("c1", "x", "conversion")
    tests = run_gauntlet(survivor, claim)
    card = scoring.build_claim_card(claim, {"metric": "AUC", "value": 0.74}, tests)
    assert len(card.evidence_ledger) == 5
    for row in card.evidence_ledger:
        assert {"test", "metric", "value", "n_used", "n_missing"} <= set(row)


def test_detective_discovers(survivor):
    out = detective.discover(survivor)
    assert set(["labels", "coords_2d", "k"]) <= set(out)
    assert out["coords_2d"].shape == (len(survivor), 2)
    assert len(out["labels"]) == len(survivor)
