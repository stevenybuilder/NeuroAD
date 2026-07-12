"""Tests for the ADNI MCI-conversion Neuro-JEPA feeder (``adni:conversion``).

The 334-subject prognostic cohort (pMCI vs sMCI) made consumable through the
loaders dispatch — the conversion-arm counterpart to ``adni:neurojepa``. The
converter label is DXSUM-derived and joined from the gated contract, NOT the
labels manifest (which is ``group='MCI'`` for all), so the test also locks the
verified 58/276 split to catch any silent re-derivation drift.

Skips when the git-ignored embedding cache is absent (CI / fresh clone).
"""
from __future__ import annotations

import pytest

from neuroad import contract
from neuroad.data import adni_conversion_jepa, loaders

_HAS_CSV = adni_conversion_jepa.EMBEDDINGS_CSV.exists()
_skip_no_csv = pytest.mark.skipif(
    not _HAS_CSV,
    reason="ADNI conversion Neuro-JEPA embedding cache absent (git-ignored)")


@_skip_no_csv
def test_adni_conversion_loads_contract_valid():
    df = loaders.load("adni:conversion")
    contract.validate_table(df)                       # raises if invalid
    assert len(df) > 300                              # the real cohort is 334
    # These are all baseline-MCI subjects (prognostic cohort, not cross-sectional dx)
    assert set(df["dx"].dropna().unique()) == {"MCI"}
    # 768-d frozen Neuro-JEPA foundation space (NOT the 323-d FreeSurfer contract)
    assert len(contract.embedding_columns(df)) == 768


@_skip_no_csv
def test_conversion_label_is_dxsum_derived_split():
    df = loaders.load("adni:conversion")
    conv = df["conversion"].dropna()
    # every subject in this cohort resolves to a converter/stable label
    assert conv.notna().mean() > 0.99
    assert set(conv.unique()) <= {0, 1}
    # lock the DXSUM-derived split (build_adni_contract._conversion): 58 pMCI / 276 sMCI
    assert int((conv == 1).sum()) == 58
    assert int((conv == 0).sum()) == 276


@_skip_no_csv
def test_conversion_carries_real_plasma_and_multisite():
    df = loaders.load("adni:conversion")
    # real plasma panel present on every row (the imaging-vs-plasma anchor has data)
    assert df["p_tau217"].notna().mean() > 0.9
    assert df["gfap"].notna().mean() > 0.9
    assert df["nfl"].notna().mean() > 0.9
    # multi-site => site-leakage / leave-one-site-out split is informative here
    assert df["site"].nunique() > 1
    # apoe4 joined from the gated contract (never fabricated where absent)
    assert 0.5 < df["apoe4"].notna().mean() <= 1.0


@_skip_no_csv
def test_adni_conversion_alias_matches():
    a = loaders.load("adni:conversion")
    b = loaders.load("adni:mci")
    assert a.shape == b.shape
    assert a["conversion"].tolist() == b["conversion"].tolist()


def test_conversion_substrate_is_neurojepa():
    # honest substrate label: these emb_* ARE the frozen foundation model, not
    # morphometry — must not be mislabeled as FreeSurfer features.
    assert loaders.honest_substrate("adni:conversion") == \
        "frozen Neuro-JEPA structural embeddings"
    assert loaders.honest_substrate("adni:mci") == \
        "frozen Neuro-JEPA structural embeddings"
