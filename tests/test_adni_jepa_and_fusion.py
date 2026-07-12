"""Tests for the newly-wired pipeline components:

  1. The ADNI Neuro-JEPA feeder (``adni:neurojepa``) — the 590-subject real-data
     cohort made consumable through the loaders dispatch.
  2. The L3 multimodal fusion transformer, now actually CALLED from
     ``translation.translate`` (it was dead code before) and surfaced as
     ``biomarker_fusion`` molecular-pathology corroboration.

The feeder test skips when the git-ignored embedding cache is absent (CI); the
fusion-wiring test is fully synthetic and always runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neuroad import contract
from neuroad.data import adni_jepa, loaders
from neuroad.harness import translation as T


# --- 1. ADNI Neuro-JEPA feeder -------------------------------------------

_HAS_ADNI = adni_jepa.EMBEDDINGS_CSV.exists()
_skip_no_adni = pytest.mark.skipif(
    not _HAS_ADNI, reason="ADNI Neuro-JEPA embedding cache absent (git-ignored)")


@_skip_no_adni
def test_adni_neurojepa_loads_contract_valid():
    df = loaders.load("adni:neurojepa")
    contract.validate_table(df)                       # raises if invalid
    assert len(df) > 100                              # the real cohort is ~590
    assert {"AD", "CN"} <= set(df["dx"].dropna().unique())
    # real plasma is carried on the cohort (the anchor has data)
    assert df["p_tau217"].notna().mean() > 0.5
    # multi-site => site-leakage test is informative here
    assert df["site"].nunique() > 1


@_skip_no_adni
def test_adni_neurojepa_alias_matches():
    a = loaders.load("adni:neurojepa")
    b = loaders.load("adni:jepa")
    assert a.shape == b.shape


# --- 2. Multimodal fusion transformer is now wired into translate --------

def _synthetic_contract(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = 16
    X = rng.normal(size=(n, d))
    cols = {f"{contract.EMBED_PREFIX}{i}": X[:, i] for i in range(d)}
    df = pd.DataFrame(cols)
    df["subject_id"] = [f"S{i:04d}" for i in range(n)]
    df["dx"] = pd.Categorical(rng.choice(["AD", "CN"], n),
                              categories=contract.DX_LEVELS)
    df["conversion"] = pd.array([pd.NA] * n, dtype="Int8")
    df["age"] = rng.normal(72, 6, n)
    df["sex"] = pd.Categorical(rng.choice(["M", "F"], n),
                               categories=contract.SEX_LEVELS)
    df["site"] = pd.Categorical([f"s{i % 4}" for i in range(n)])
    df["scanner"] = pd.Categorical(rng.choice(["1.5T", "3T"], n))
    df["amyloid"] = pd.array(rng.integers(0, 2, n), dtype="Int8")
    df["apoe4"] = pd.array(rng.integers(0, 3, n), dtype="Int8")
    df["p_tau217"] = rng.normal(1.0, 0.4, n)
    df["gfap"] = rng.normal(150, 40, n)
    df["nfl"] = rng.normal(30, 8, n)
    contract.validate_table(df)
    return df


def test_translate_surfaces_biomarker_fusion():
    """The transformer runs over the cohort and lands in the lead (was dead code)."""
    df = _synthetic_contract()
    out = T.translate("amyloid_cascade", df, prefer_offline=True)
    bf = out.get("biomarker_fusion")
    assert bf, "biomarker_fusion should be populated when df is provided"
    assert bf["n_subjects"] == len(df)
    assert 0.0 <= bf["mean_abeta_prob"] <= 1.0
    assert 0.0 <= bf["tau_positive_rate"] <= 1.0
    assert bf["model"] in ("jasodanand2025", "surrogate_logistic")
    assert out["provenance"].get("biomarker_fusion") == bf["source"]


def test_translate_without_df_omits_fusion():
    """No cohort => no fabricated fusion summary (empty, not invented)."""
    out = T.translate("amyloid_cascade", None, prefer_offline=True)
    assert out.get("biomarker_fusion") in ({}, None)


def test_biomarker_fusion_helper_empty_on_no_features():
    """A df with no fusion features returns {} rather than raising."""
    df = pd.DataFrame({"subject_id": ["a", "b"]})
    assert T._biomarker_fusion(df, prefer_offline=True) == {}
