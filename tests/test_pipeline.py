"""Import-guarded smoke tests for the orchestration layer.

These must NOT fail the build just because a sibling module (probe/gauntlet/
scoring/data) has not landed yet — if a dependency is missing we skip. What we
CAN always assert: `cli` and `pipeline` import cleanly against the contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neuroad import contract


# ---------------------------------------------------------------------------
# Always-on: the M5 modules must import against only the frozen contract.
# ---------------------------------------------------------------------------

def test_pipeline_imports():
    from neuroad import pipeline  # noqa: F401
    assert hasattr(pipeline, "run_referee")


def test_cli_imports_and_parser():
    from neuroad import cli
    parser = cli.build_parser()
    args = parser.parse_args(["run", "synthetic:KILL", "a hunch"])
    assert args.dataset == "synthetic:KILL"
    assert args.claim == "a hunch"


def test_cli_demo_subcommand_parses():
    from neuroad import cli
    args = cli.build_parser().parse_args(["demo"])
    assert args.command == "demo"


# ---------------------------------------------------------------------------
# Fixture: a tiny contract-valid table with a real, separable signal.
# ---------------------------------------------------------------------------

def _tiny_cohort(n: int = 80, d: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    # Embedding carries the label in the first two dims (so the probe works).
    X = rng.normal(size=(n, d))
    X[:, 0] += 1.5 * y
    X[:, 1] += 1.0 * y
    frame = contract.make_embedding_frame(X)
    frame["subject_id"] = [f"s{i:03d}" for i in range(n)]
    frame["dx"] = pd.Categorical(
        np.where(y == 1, "MCI", "CN"), categories=contract.DX_LEVELS)
    frame["conversion"] = pd.array(y, dtype="Int8")
    frame["age"] = rng.normal(72, 6, size=n)
    frame["sex"] = pd.Categorical(
        rng.choice(["M", "F"], size=n), categories=contract.SEX_LEVELS)
    frame["site"] = pd.Categorical(rng.choice(["S1", "S2"], size=n))
    frame["scanner"] = pd.Categorical(rng.choice(["A", "B"], size=n))
    frame["amyloid"] = pd.array(rng.integers(0, 2, size=n), dtype="Int8")
    frame["p_tau217"] = rng.normal(1.0, 0.3, size=n)
    frame["gfap"] = rng.normal(120, 30, size=n)
    frame["nfl"] = rng.normal(20, 5, size=n)
    frame["apoe4"] = pd.array(rng.integers(0, 3, size=n), dtype="Int8")
    contract.validate_table(frame)
    return frame


# ---------------------------------------------------------------------------
# End-to-end: only runs if the core engine is present.
# ---------------------------------------------------------------------------

def test_run_referee_end_to_end():
    pytest.importorskip("neuroad.probe")
    pytest.importorskip("neuroad.gauntlet")
    pytest.importorskip("neuroad.scoring")
    from neuroad import pipeline

    df = _tiny_cohort()
    claim = contract.Claim(
        claim_id="t1",
        claim_text="MCI converters show a distinct structural signature.",
        target="conversion",
    )
    card = pipeline.run_referee(df, claim)

    assert isinstance(card, contract.ClaimCard)
    assert isinstance(card.verdict, contract.Verdict)
    assert 0 <= card.score <= 100
    assert card.naive_effect.get("metric") == "AUC"
    # narration is attached as a side artifact for the UI/exporter.
    assert isinstance(getattr(card, "narration", ""), str)

    # GAP 1: the two STAR trust features are computed in the referee path and
    # surface in the exported dict (not demo-only).
    d = card.to_dict()
    assert "double_dissociation" in d and "confound_leaderboard" in d
    assert isinstance(d["double_dissociation"], dict)
    assert isinstance(d["confound_leaderboard"], list)
    # Non-empty: the tiny cohort has two scanners/sites, so both compute.
    assert {"auc_before", "auc_after", "retained", "confound"} <= set(
        d["double_dissociation"])
    assert card.double_dissociation is d["double_dissociation"] \
        or card.double_dissociation == d["double_dissociation"]


def test_write_reports_emits_reviewer_and_biology_and_star(tmp_path, monkeypatch):
    """GAP 1 + GAP 2: the CLI report payload carries the STAR trust features and
    the structured reviewer / biology blocks — not just narration/adjudication."""
    pytest.importorskip("neuroad.probe")
    pytest.importorskip("neuroad.gauntlet")
    pytest.importorskip("neuroad.scoring")
    import json

    from neuroad import cli, pipeline

    df = _tiny_cohort(seed=2)
    card = pipeline.run_referee(df, "a hunch about converters")
    # Reviewer always runs; assert the side artifact is present so the export has
    # something structured to write.
    assert isinstance(getattr(card, "reviewer", None), dict)

    monkeypatch.setattr(cli, "_REPORTS", tmp_path)
    written = cli._write_reports("synthetic:TEST", card)
    jp = next(p for p in written if p.suffix == ".json")
    payload = json.loads(jp.read_text())

    # GAP 2: structured reviewer block written (mirrors adjudication).
    assert "reviewer" in payload
    assert "critique" in payload["reviewer"]
    # GAP 1: STAR trust features carried into the written report.
    assert "double_dissociation" in payload
    assert "confound_leaderboard" in payload


def test_run_referee_accepts_raw_string_claim():
    pytest.importorskip("neuroad.probe")
    pytest.importorskip("neuroad.gauntlet")
    pytest.importorskip("neuroad.scoring")
    from neuroad import pipeline

    df = _tiny_cohort(seed=1)
    card = pipeline.run_referee(df, "a hunch about converters")
    assert isinstance(card, contract.ClaimCard)
