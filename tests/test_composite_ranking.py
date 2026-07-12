"""Focused, OFFLINE tests for the opt-in COMPOSITE target ranking.

The referee's ``harness.translation._rank_targets`` gains an opt-in
``method='composite'`` that fuses four signals (PI4AD priority + Open Targets
non-genetic held-out + STRING-RWR centrality + AlphaFold pLDDT) via the shared
``harness.ranking`` helper. These tests assert, all offline / deterministic
(prefer_offline=True, no network, no creds):

  * the DEFAULT single-signal PI4AD path is unchanged (back-compat);
  * the composite path returns ranked candidates with the expected keys, keeps
    per-signal provenance, is honestly sorted, and NEVER crashes.
"""
from __future__ import annotations

from neuroad.harness import ranking, translation


def test_default_rank_targets_is_single_signal_pi4ad():
    """DEFAULT behavior unchanged: PI4AD rows keep priority_score/source."""
    rows = translation._rank_targets("amyloid_cascade", prefer_offline=True)
    assert isinstance(rows, list) and rows
    top = rows[0]
    # Single-signal contract keys the referee already depends on.
    assert "gene" in top and "priority_score" in top and "source" in top
    # No composite-only key leaks into the default path.
    assert "composite_score" not in top


def test_composite_path_returns_ranked_candidates_with_expected_keys():
    rows = translation._rank_targets(
        "glial", prefer_offline=True, method="composite")
    assert isinstance(rows, list) and rows, "composite must return candidates"
    expected = {
        "gene", "composite_score", "n_signals", "signals_present",
        "pi4ad_priority", "pi4ad_rank", "ot_assoc_heldout", "net_centrality",
        "net_degree", "struct_plddt", "source",
    }
    for r in rows:
        assert expected <= set(r), f"missing keys: {expected - set(r)}"
        # Per-signal provenance / self-describing source stamp.
        assert r["source"] == "composite_multi_signal"
        # n_signals honestly matches the count of present signals listed.
        assert r["n_signals"] == len(r["signals_present"])


def test_composite_only_names_this_mechanisms_genes():
    mech = "glial"
    rows = translation._rank_targets(mech, prefer_offline=True, method="composite")
    genes = {r["gene"] for r in rows}
    assert genes == set(translation.MECHANISM_GENES[mech])


def test_composite_is_honestly_sorted_scored_first_desc():
    rows = translation._rank_targets(
        "amyloid_cascade", prefer_offline=True, method="composite")
    scored = [r["composite_score"] for r in rows
              if r["composite_score"] is not None]
    # Ranked descending; None composites (no signal) sink to the bottom.
    assert scored == sorted(scored, reverse=True)
    seen_none = False
    for r in rows:
        if r["composite_score"] is None:
            seen_none = True
        elif seen_none:
            raise AssertionError("scored gene ranked below an unscored one")


def test_composite_never_raises_on_unknown_mechanism():
    # Unknown mechanism must not crash; helper degrades gracefully.
    rows = translation._rank_targets(
        "not_a_real_mechanism", prefer_offline=True, method="composite")
    assert isinstance(rows, list)


def test_shared_helper_matches_expected_weight_signals():
    # The referee and the CLI script share ONE weight set (the four task signals).
    assert set(ranking.WEIGHTS) == {
        "pi4ad_priority", "ot_assoc_heldout", "net_centrality",
        "struct_confidence",
    }
    assert abs(sum(ranking.WEIGHTS.values()) - 1.0) < 1e-9


def test_composite_targets_helper_is_offline_safe():
    rows = ranking.composite_targets("vascular", prefer_offline=True)
    assert isinstance(rows, list)
    # Every row carries a source stamp (no unstamped/fabricated rows).
    assert all(r.get("source") == "composite_multi_signal" for r in rows)
