"""
Unit tests for the M3 Claude reasoning layer.

These run with NO ANTHROPIC_API_KEY (the demo default), so they exercise the
deterministic offline path. They assert shape + non-empty deterministic text,
that the bridge routes gfap-dominant vs ptau-dominant to different mechanisms,
and that the courtroom returns all three parts. Fixtures are built inline from
contract types so this module is importable and testable without the other
agents' code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from neuroad.contract import (
    Claim,
    TestEvidence,
    TestResult,
    ClaimCard,
    robustness_score,
    verdict_for,
    is_promoted,
)
from neuroad.claude import _client, claim_parser, courtroom, narrator, bridge, reviewer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _survivor_tests() -> list[TestEvidence]:
    return [
        TestEvidence("age_sex", TestResult.PASSED, "survives age/sex",
                     {"naive_auc": 0.74, "adjusted_auc": 0.71, "retained": 0.80}),
        TestEvidence("site_scanner", TestResult.PASSED, "outcome exceeds scanner",
                     {"outcome_auc": 0.74, "scanner_auc": 0.64, "margin": 0.10}),
        TestEvidence("brain_age", TestResult.WEAKENED, "some aging overlap",
                     {"brain_age_r2": 0.85, "brain_age_mae_yr": 3.0,
                      "naive_auc": 0.74, "residualized_auc": 0.68, "retained": 0.80}),
        TestEvidence("biomarker_anchor", TestResult.PASSED, "p-tau217 anchor",
                     {"ptau217_r": 0.43, "gfap_r": 0.35, "n": 48}),
        TestEvidence("replication", TestResult.WEAKENED, "held-out shrinks",
                     {"in_cohort_auc": 0.74, "held_out_auc": 0.69, "retained": 0.85,
                      "split": "cohort"}),
    ]


def _kill_tests() -> list[TestEvidence]:
    return [
        TestEvidence("age_sex", TestResult.WEAKENED, "part demographics",
                     {"naive_auc": 0.72, "adjusted_auc": 0.60, "retained": 0.40}),
        TestEvidence("site_scanner", TestResult.FAILED, "scanner >= outcome",
                     {"outcome_auc": 0.72, "scanner_auc": 0.92, "margin": -0.20}),
        TestEvidence("brain_age", TestResult.FAILED, "just aging",
                     {"naive_auc": 0.72, "residualized_auc": 0.58, "retained": 0.25}),
        TestEvidence("biomarker_anchor", TestResult.NA, "no plasma marker", {}),
        TestEvidence("replication", TestResult.FAILED, "collapses",
                     {"in_cohort_auc": 0.72, "held_out_auc": 0.55, "retained": 0.30}),
    ]


def _card(claim: Claim, tests: list[TestEvidence], n: int = 180) -> ClaimCard:
    score = robustness_score({t.key: t.result for t in tests})
    verdict = verdict_for(score)
    return ClaimCard(
        claim=claim,
        naive_effect={"metric": "AUC", "value": 0.74, "n": n},
        tests=tests,
        score=score,
        verdict=verdict,
        promoted=is_promoted(verdict),
    )


def _biomarker_df(*, gfap_high: bool, n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Contract-shaped mini table with one biomarker dominating in converters."""
    rng = np.random.default_rng(seed)
    conv = np.array([1] * (n // 2) + [0] * (n - n // 2))
    disease = conv == 1

    def marker(base, hi_disease):
        x = rng.normal(base, 0.3, n)
        x[disease] += hi_disease
        return x

    if gfap_high:
        gfap = marker(3.0, 1.5)      # GFAP dominates in converters
        ptau = marker(0.4, 0.05)     # p-tau217 nearly flat
    else:
        gfap = marker(3.0, 0.05)     # GFAP nearly flat
        ptau = marker(0.4, 1.5)      # p-tau217 dominates in converters

    return pd.DataFrame({
        "subject_id": [f"s{i}" for i in range(n)],
        "conversion": conv,
        "dx": ["MCI"] * n,
        "p_tau217": ptau,
        "gfap": gfap,
        "nfl": marker(15.0, 0.05),
        "amyloid": (rng.random(n) < 0.5).astype(int),
    })


# ---------------------------------------------------------------------------
# _client
# ---------------------------------------------------------------------------


def test_client_offline_by_default():
    assert _client.USING_LIVE_API is False
    text = _client.complete("sys", "The signal separates converters. Details follow.")
    assert isinstance(text, str) and text.strip()

    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "array"}},
        "required": ["a", "b"],
    }
    data = _client.complete("sys", "A prompt here.", schema=schema)
    assert isinstance(data, dict)
    assert set(data) == {"a", "b"}
    assert isinstance(data["b"], list)


# ---------------------------------------------------------------------------
# claim_parser
# ---------------------------------------------------------------------------


def test_parse_conversion_claim():
    c = claim_parser.parse_claim(
        "I think MCI patients who will convert to AD look different structurally"
    )
    assert isinstance(c, Claim)
    assert c.target == "conversion"
    assert c.group_a and c.group_b
    assert c.covariates  # non-empty
    assert c.claim_id.startswith("claim-")


def test_parse_diagnosis_claim():
    c = claim_parser.parse_claim("These embeddings separate AD vs CN diagnosis")
    assert c.target == "dx_binary"
    assert c.group_a == "AD" and c.group_b == "CN"


def test_parse_uses_table_when_ambiguous():
    df = pd.DataFrame({"conversion": pd.array([1, 0], dtype="Int8"),
                       "dx": ["MCI", "MCI"]})
    c = claim_parser.parse_claim("some structural pattern of interest", df=df)
    assert c.target in ("conversion", "dx_binary")


# ---------------------------------------------------------------------------
# courtroom
# ---------------------------------------------------------------------------


def test_courtroom_returns_all_three_parts():
    claim = claim_parser.parse_claim("MCI to AD conversion signal")
    result = courtroom.adjudicate(claim, _survivor_tests())
    assert set(result) == {"prosecution", "defense", "judge_reasoning"}
    for part in result.values():
        assert isinstance(part, str) and len(part.strip()) > 20


def test_courtroom_kill_prosecution_cites_leakage():
    claim = claim_parser.parse_claim("MCI to AD conversion signal")
    result = courtroom.adjudicate(claim, _kill_tests())
    assert "leakage" in result["prosecution"].lower()
    # Judge reasoning reflects the low (fragile) verdict.
    assert "0/100" in result["judge_reasoning"] or "fragile" in result["judge_reasoning"].lower()


# ---------------------------------------------------------------------------
# narrator
# ---------------------------------------------------------------------------


def test_narrate_survivor_names_assumption_and_cites():
    claim = claim_parser.parse_claim("MCI to AD conversion signal")
    card = _card(claim, _survivor_tests())
    text = narrator.narrate(card)
    assert isinstance(text, str) and text.strip()
    assert "stops being true" in text.lower()
    assert "arxiv" in text.lower()  # cites the batch-effect prior art


def test_narrate_kill_is_not_promoted():
    claim = claim_parser.parse_claim("MCI to AD conversion signal")
    card = _card(claim, _kill_tests())
    text = narrator.narrate(card)
    assert card.promoted is False
    assert isinstance(text, str) and text.strip()


# ---------------------------------------------------------------------------
# bridge
# ---------------------------------------------------------------------------


def test_bridge_routes_gfap_vs_ptau_to_different_mechanisms():
    claim = claim_parser.parse_claim("MCI to AD conversion signal")
    card = _card(claim, _survivor_tests())
    assert card.promoted is True

    glial = bridge.propose_biology(card, _biomarker_df(gfap_high=True))
    amyloid = bridge.propose_biology(card, _biomarker_df(gfap_high=False))

    for out in (glial, amyloid):
        assert set(out) == {"hypothesis", "next_experiment", "falsification"}
        assert out["hypothesis"].strip()
        assert out["next_experiment"] and out["falsification"]

    assert glial["hypothesis"] != amyloid["hypothesis"]
    assert "glial" in glial["hypothesis"].lower() or "neuroinflammatory" in glial["hypothesis"].lower()
    assert "amyloid" in amyloid["hypothesis"].lower()


def test_bridge_gates_non_promoted():
    claim = claim_parser.parse_claim("MCI to AD conversion signal")
    card = _card(claim, _kill_tests())
    out = bridge.propose_biology(card, _biomarker_df(gfap_high=True))
    assert out["next_experiment"] == []
    assert out["falsification"] == []
    assert "not promoted" in out["hypothesis"].lower()


# ---------------------------------------------------------------------------
# reviewer
# ---------------------------------------------------------------------------


def test_reviewer_argues_against_verdict():
    claim = claim_parser.parse_claim("MCI to AD conversion signal")
    card = _card(claim, _survivor_tests(), n=150)
    out = reviewer.review(card)
    assert set(out) == {"critique", "revised_caveats"}
    assert out["critique"] and out["revised_caveats"]
    joined = " ".join(out["critique"]).lower()
    # Flags the proxy brain-age control and small N.
    assert "proxy" in joined
    assert "n is small" in joined or "sample size" in joined
    # Cites prior art rather than claiming the insight.
    assert "arxiv" in " ".join(out["revised_caveats"]).lower()
