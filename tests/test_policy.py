"""
Tests for the L3 policy LOADER (neuroad.harness.policy).

Two guarantees are exercised:
  (a) values LOAD from the real policy/ docs (table / thresholds / brief), and
  (b) with policy/ MISSING or MALFORMED the loader falls back to today's
      hardcoded constants BYTE-IDENTICALLY — so the frozen demo never changes.

Because the policy docs are transcriptions of the live constants, the
loaded-from-policy values and the fallback values must AGREE. Both directions
are asserted here.
"""
from __future__ import annotations

import pytest

from neuroad.calibration import CAL
from neuroad.contract import (
    GAUNTLET,
    PROMOTION_FLOOR,
    RESULT_CREDIT,
    VERDICT_BANDS,
    Verdict,
)
from neuroad.harness import policy


@pytest.fixture
def empty_policy_dir(tmp_path, monkeypatch):
    """Point the loader at an EMPTY directory (every file missing)."""
    monkeypatch.setattr(policy, "POLICY_DIR", tmp_path)
    policy.reload()
    yield tmp_path
    policy.reload()


# ---------------------------------------------------------------------------
# (a) values load from policy/
# ---------------------------------------------------------------------------
def test_policy_dir_exists():
    assert policy.POLICY_DIR.is_dir(), "policy/ docs must be present in the repo"


def test_table_loads_from_policy():
    cp = policy.table("confound_priors")
    assert cp["retained_bands"]["survivor_retained"] == 0.70
    assert cp["retained_bands"]["kill_retained"] == 0.40

    br = policy.table("biomarker_routing")
    assert "amyloid_cascade" in br["mechanisms"]
    assert br["mechanisms"]["amyloid_cascade"]["label"] == "amyloid-cascade (tau-driven)"
    assert br["routing"]["marker_to_mechanism"]["gfap"] == "glial"

    vr = policy.table("verdict_rubric")
    keys = {row["key"] for row in vr["verdict_bands"]}
    assert {"STRONG", "ROBUST_FOLLOWUP", "PARTIALLY_ROBUST", "FRAGILE"} <= keys


def test_thresholds_load_from_policy():
    assert policy.thresholds("retained") == {
        "survivor_retained": 0.70, "kill_retained": 0.40}

    anchor = policy.thresholds("anchor")
    assert anchor["ci_pass"] == 0.12
    assert anchor["ci_weak"] == 0.0
    assert anchor["min_n"] == 20

    verdict = policy.thresholds("verdict")
    assert verdict["strong"] == 85.0
    assert verdict["robust_followup"] == 70.0
    assert verdict["partially_robust"] == 40.0
    assert verdict["fragile"] == 0.0
    assert verdict["promotion_floor"] == 40.0


def test_brief_returns_markdown_body():
    body = policy.brief("verdict_rubric")
    assert isinstance(body, str) and body.strip()
    # front matter must be stripped: the body should not start with the fence.
    assert not body.lstrip().startswith("---")
    assert "Verdict" in body

    atn = policy.brief("atn_framework")
    assert "anchor" in atn.lower()


# ---------------------------------------------------------------------------
# (b) missing-file -> identical fallback
# ---------------------------------------------------------------------------
def test_missing_dir_thresholds_identical(empty_policy_dir):
    # Capture the fallback (policy/ now empty) ...
    fb_retained = policy.thresholds("retained")
    fb_anchor = policy.thresholds("anchor")
    fb_verdict = policy.thresholds("verdict")
    fb_credit = policy.thresholds("result_credit")
    fb_weights = policy.thresholds("gauntlet_weights")
    fb_repl = policy.thresholds("replication")

    # ... they must equal the live constants exactly.
    assert fb_retained == {"survivor_retained": float(CAL["survivor_retained"][0]),
                           "kill_retained": float(CAL["kill_retained"][1])}
    assert fb_anchor == {"ci_pass": 0.12, "ci_weak": 0.0, "min_n": 20.0}
    assert fb_repl == {"pass": 0.65, "weak": 0.58}
    assert fb_credit == {k.value: float(v) for k, v in RESULT_CREDIT.items()}
    assert fb_weights == {d.key: float(d.weight) for d in GAUNTLET}

    expected_verdict = {v.name.lower(): float(lo) for lo, v in VERDICT_BANDS}
    expected_verdict["promotion_floor"] = 40.0
    assert fb_verdict == expected_verdict


def test_loaded_equals_fallback_on_real_docs():
    """With the real docs present, loaded thresholds still equal the fallback —
    proving the policy files are faithful transcriptions of the constants."""
    for g in policy.THRESHOLD_GROUPS:
        assert policy.thresholds(g) == policy._fallback_thresholds(g)


def test_missing_dir_table_falls_back(empty_policy_dir):
    cp = policy.table("confound_priors")
    assert cp["retained_bands"]["survivor_retained"] == float(CAL["survivor_retained"][0])
    assert cp["retained_bands"]["kill_retained"] == float(CAL["kill_retained"][1])

    br = policy.table("biomarker_routing")
    # bridge._MECHANISMS transcription survives with policy/ absent.
    assert br["mechanisms"]["amyloid_cascade"]["label"] == "amyloid-cascade (tau-driven)"
    assert br["mechanisms"]["glial"]["cohort"]
    assert br["calibration_anchors"]["ptau217_r_target"] == float(CAL["ptau217_r"][2])

    vr = policy.table("verdict_rubric")
    assert vr["promotion_floor_min_score"] == 40
    assert vr["score"]["dimension_weights"] == {d.key: int(d.weight) for d in GAUNTLET}


def test_missing_dir_brief_falls_back(empty_policy_dir):
    for name in ("verdict_rubric", "atn_framework", "novelty_rubric",
                 "confound_priors", "biomarker_routing"):
        body = policy.brief(name)
        assert isinstance(body, str) and body.strip()
    # load-bearing thresholds survive into the fallback prose.
    assert "0.12" in policy.brief("atn_framework")
    assert "40" in policy.brief("verdict_rubric")


def test_malformed_file_falls_back(tmp_path, monkeypatch):
    # Write a syntactically broken YAML where confound_priors should be.
    (tmp_path / "confound_priors.yaml").write_text(
        "retained_bands: {: this is : not : valid", encoding="utf-8")
    monkeypatch.setattr(policy, "POLICY_DIR", tmp_path)
    policy.reload()
    try:
        assert policy.thresholds("retained") == {
            "survivor_retained": float(CAL["survivor_retained"][0]),
            "kill_retained": float(CAL["kill_retained"][1])}
        assert policy.table("confound_priors")["retained_bands"]["survivor_retained"] \
            == float(CAL["survivor_retained"][0])
    finally:
        policy.reload()


def test_fallback_promotion_floor_matches_contract():
    """The fallback promotion floor equals PROMOTION_FLOOR's band lower bound."""
    floor_score = next(lo for lo, v in VERDICT_BANDS if v == PROMOTION_FLOOR)
    assert policy.thresholds("verdict")["promotion_floor"] == float(floor_score)
    assert PROMOTION_FLOOR == Verdict.PARTIALLY_ROBUST


def test_unknown_name_raises():
    with pytest.raises(KeyError):
        policy.table("does_not_exist")
    with pytest.raises(KeyError):
        policy.brief("does_not_exist")
