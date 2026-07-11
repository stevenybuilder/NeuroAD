"""Tests for the Claude ORCHESTRATOR harness (harness/agent.py).

All offline / deterministic — the live tool-runner path needs ANTHROPIC_API_KEY
and is not exercised here. These cover the tool wrappers and the scripted
orchestration, including the load-bearing invariant: the molecular chain never
runs on a killed (non-promoted) hypothesis.
"""
from __future__ import annotations

import json

from neuroad.harness import agent


# --- tools return well-formed JSON -----------------------------------------

def test_list_datasets_tool():
    d = json.loads(agent.list_datasets())
    assert "synthetic:SURVIVOR" in d["datasets"]


def test_referee_tool_returns_verdict_and_mechanism():
    d = json.loads(agent.referee_hypothesis(
        "Does the embedding predict AD vs CN diagnosis?", "synthetic:SURVIVOR"))
    assert d["verdict"] and d["robustness_score"] is not None
    assert "Neuro-JEPA" not in d["substrate"]           # honest substrate
    assert "promoted" in d


def test_referee_tool_unknown_dataset_is_error_json():
    d = json.loads(agent.referee_hypothesis("x", "not_a_dataset"))
    assert "error" in d


def test_prioritize_targets_tool_is_pi4ad_ranked():
    d = json.loads(agent.prioritize_targets("amyloid_cascade"))
    scored = [r for r in d["ranked_targets"] if r.get("priority_score") is not None]
    assert scored and scored[0]["priority_score"] is not None


def test_protein_structure_tool_provenance_labeled():
    d = json.loads(agent.protein_structure("APP"))
    # live or offline_snapshot — never unlabeled.
    assert d.get("source") in ("live", "offline_snapshot") or "error" in d


# --- scripted orchestration ------------------------------------------------

def test_scripted_orchestration_promoted_runs_full_chain():
    out = agent.orchestrate(
        "Investigate AD vs CN on synthetic:SURVIVOR and propose an experiment",
        api=False)
    assert out["path"] == "scripted_offline"
    tools = [c["tool"] for c in out["tool_calls"]]
    assert tools[:2] == ["describe_cohort", "referee_hypothesis"]
    ref = out["tool_calls"][1]["result"]
    if ref.get("promoted"):
        # promoted -> the molecule chain must have been sequenced
        assert "prioritize_targets" in tools
        assert "protein_structure" in tools
        assert "repurposing_candidates" in tools


def test_scripted_orchestration_kill_does_not_run_molecular_chain():
    out = agent.orchestrate(
        "Does the embedding predict conversion on synthetic:KILL?", api=False)
    ref = out["tool_calls"][1]["result"]
    assert ref["promoted"] is False
    tools = [c["tool"] for c in out["tool_calls"]]
    # THE INVARIANT: a killed imaging signal never reaches target discovery.
    assert "prioritize_targets" not in tools
    assert "protein_structure" not in tools


def test_orchestrate_api_false_is_offline():
    # Forcing api=False must never touch the live path even if a key were set.
    out = agent.orchestrate("AD vs CN on synthetic:SURVIVOR", api=False)
    assert out["model"] == "none (deterministic script)"


def test_guess_dataset():
    assert agent._guess_dataset("run on adni:3t please") == "adni:3t"
    assert agent._guess_dataset("the synthetic KILL cohort") == "synthetic:KILL"
    assert agent._guess_dataset("no dataset named here") == "adni:3t"  # default
