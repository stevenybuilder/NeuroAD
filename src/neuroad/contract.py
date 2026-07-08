"""
NeuroAD Discovery Engine — THE CONTRACT (frozen interface).

This module is the single source of truth that every downstream piece reads.
Once the schema exists, downstream is data-source-independent: swapping real
embeddings in later changes one thing — the table, not the code.

Design stance (from the master brief):
  "Imaging finds it. Proteins confirm it. The system tells you what to do next."
  Build against the embedding-table CONTRACT, not against any specific encoder.

Three interchangeable feeders satisfy this contract:
  1. real frozen Neuro-JEPA structural embeddings (gated weights),
  2. a substitute open structural encoder,
  3. weight-free structural features (eTIV, nWBV, ASF, hippocampal volume,
     cortical thickness, WMH burden) — the OASIS-style real-data feeder.

Everything the Referee, the Detective (clustering), the Bridge (biology), and
the demo UI consume is defined here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. The cached-embedding table schema
# ---------------------------------------------------------------------------
# One row per subject. Embedding is stored as columns emb_0 .. emb_{D-1}.
# The metadata/label/biomarker columns below are the fixed contract.

EMBED_PREFIX = "emb_"

#: Fixed non-embedding columns. NaN is allowed where marked (partial coverage).
METADATA_COLUMNS: dict[str, str] = {
    "subject_id": "string",     # unique subject id
    "dx": "category",           # diagnosis: CN | MCI | AD
    "conversion": "Int8",       # MCI->AD within horizon: 1/0/<NA> (NA if not MCI or unknown)
    "age": "float64",           # years
    "sex": "category",          # M | F
    "site": "category",         # acquisition site / study
    "scanner": "category",      # scanner model / field strength label
    "amyloid": "Int8",          # amyloid positivity: 1/0/<NA>
    "p_tau217": "float64",      # plasma p-tau217 (pg/mL), <NA> allowed
    "gfap": "float64",          # plasma GFAP (pg/mL), <NA> allowed
    "nfl": "float64",           # plasma NfL (pg/mL), <NA> allowed
    "apoe4": "Int8",            # APOE e4 allele count 0/1/2, <NA> allowed
}

DX_LEVELS = ["CN", "MCI", "AD"]
SEX_LEVELS = ["M", "F"]

#: Columns that carry molecular/protein evidence (biomarker anchor + routing).
BIOMARKER_COLUMNS = ["amyloid", "p_tau217", "gfap", "nfl", "apoe4"]

#: Label columns the ONE reused head can be pointed at (contract of the probe).
#: Pointing the same head at these different targets is the whole product.
LABEL_TARGETS = {
    "conversion": "MCI->AD conversion (or AD-vs-CN diagnosis on open data)",
    "dx_binary": "AD vs CN diagnosis (derived from dx)",
    "site": "acquisition site  (STAR leakage test — same head, label=site)",
    "scanner": "scanner model   (STAR leakage test — same head, label=scanner)",
}


def embedding_columns(df: pd.DataFrame) -> list[str]:
    """Return the ordered list of embedding columns present in a table."""
    cols = [c for c in df.columns if c.startswith(EMBED_PREFIX)]
    return sorted(cols, key=lambda c: int(c[len(EMBED_PREFIX):]))


def embedding_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract the (n_subjects, D) embedding matrix from a contract table."""
    return df[embedding_columns(df)].to_numpy(dtype=float)


def make_embedding_frame(X: np.ndarray) -> pd.DataFrame:
    """Wrap an (n, D) array into emb_0..emb_{D-1} columns."""
    X = np.asarray(X, dtype=float)
    cols = [f"{EMBED_PREFIX}{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols)


# ---------------------------------------------------------------------------
# 2. The Referee gauntlet — tests, verdict bands, robustness score
# ---------------------------------------------------------------------------


class TestResult(str, Enum):
    __test__ = False           # not a pytest test class despite the "Test" prefix
    PASSED = "passed"          # signal survives the challenge intact
    WEAKENED = "weakened"      # survives but effect materially shrinks
    MIXED = "mixed"            # ambiguous / borderline
    FAILED = "failed"          # collapses under the challenge
    NA = "not_available"       # data insufficient to run this test


#: Structural-track gauntlet. Each dimension answers one way a finding can be
#: an artifact. Weights sum to 100. The two ⭐ tests carry the most weight —
#: they are the ones a scanner or generic aging can most easily fake.
@dataclass(frozen=True)
class GauntletDimension:
    key: str
    label: str
    question: str
    weight: int
    star: bool = False


GAUNTLET: list[GauntletDimension] = [
    GauntletDimension(
        "age_sex", "Age / sex adjustment",
        "Does the signal survive demographic covariates?", 15),
    GauntletDimension(
        "site_scanner", "Site / scanner leakage",
        "Is it disease signal, or just which machine acquired the scan?",
        25, star=True),
    GauntletDimension(
        "brain_age", "Brain-age control",
        "More than generic aging/atrophy?", 25, star=True),
    GauntletDimension(
        "biomarker_anchor", "Biomarker anchor",
        "Backed by molecular pathology (p-tau217 / GFAP)?", 20),
    GauntletDimension(
        "replication", "Replication split",
        "Reproduces on a held-out site / cohort?", 15),
]

GAUNTLET_BY_KEY = {d.key: d for d in GAUNTLET}
assert sum(d.weight for d in GAUNTLET) == 100, "gauntlet weights must sum to 100"

#: Fraction of a dimension's weight earned per TestResult.
RESULT_CREDIT: dict[TestResult, float] = {
    TestResult.PASSED: 1.0,
    TestResult.WEAKENED: 0.5,
    TestResult.MIXED: 0.5,
    TestResult.FAILED: 0.0,
    TestResult.NA: 0.0,
}


class Verdict(str, Enum):
    FRAGILE = "fragile"
    PARTIALLY_ROBUST = "partially robust"
    ROBUST_FOLLOWUP = "robust enough for follow-up"
    STRONG = "strong candidate"


#: (inclusive lower bound, Verdict). Score in [0, 100].
VERDICT_BANDS: list[tuple[int, Verdict]] = [
    (85, Verdict.STRONG),
    (70, Verdict.ROBUST_FOLLOWUP),
    (40, Verdict.PARTIALLY_ROBUST),
    (0, Verdict.FRAGILE),
]

#: Only signals at or above this verdict are allowed to reach the biology step.
PROMOTION_FLOOR = Verdict.PARTIALLY_ROBUST


def robustness_score(results: dict[str, TestResult]) -> int:
    """Weighted, renormalized over the tests that actually ran (NA excluded).

    Returns an integer 0..100. A dimension that could not be run (NA) is
    dropped from BOTH numerator and denominator so the score reflects only
    evidence we actually have — with an explicit completeness caveat surfaced
    elsewhere.
    """
    earned = 0.0
    possible = 0.0
    for dim in GAUNTLET:
        res = results.get(dim.key, TestResult.NA)
        if res == TestResult.NA:
            continue
        possible += dim.weight
        earned += dim.weight * RESULT_CREDIT[res]
    if possible == 0:
        return 0
    return int(round(100 * earned / possible))


def verdict_for(score: int) -> Verdict:
    for lo, v in VERDICT_BANDS:
        if score >= lo:
            return v
    return Verdict.FRAGILE


def is_promoted(verdict: Verdict) -> bool:
    order = [Verdict.FRAGILE, Verdict.PARTIALLY_ROBUST,
             Verdict.ROBUST_FOLLOWUP, Verdict.STRONG]
    return order.index(verdict) >= order.index(PROMOTION_FLOOR)


# ---------------------------------------------------------------------------
# 3. Structured claim + claim-card schema (the exported artifact)
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    """A structured, testable claim (output of the Claude claim parser)."""
    claim_id: str
    claim_text: str
    target: str                       # one of LABEL_TARGETS (usually conversion/dx_binary)
    group_a: str = ""                 # e.g. "MCI converters" / "AD"
    group_b: str = ""                 # e.g. "MCI non-converters" / "CN"
    substrate: str = "frozen Neuro-JEPA structural embeddings"
    head: str = "linear probe"
    covariates: list[str] = field(default_factory=lambda: ["age", "sex"])


@dataclass
class TestEvidence:
    """One gauntlet test's result plus the numbers that justify it."""
    __test__ = False           # not a pytest test class despite the "Test" prefix
    key: str
    result: TestResult
    detail: str = ""                  # human-readable one-liner
    stats: dict = field(default_factory=dict)  # e.g. {"scanner_auc": 0.75}


@dataclass
class ClaimCard:
    """The exported decision artifact. Serializes to the demo/report YAML."""
    claim: Claim
    naive_effect: dict                     # {"metric": "AUC", "value": 0.74, ...}
    tests: list[TestEvidence]
    score: int
    verdict: Verdict
    promoted: bool
    biology_hypothesis: str = ""
    next_experiment: list[str] = field(default_factory=list)
    falsification: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    evidence_ledger: list[dict] = field(default_factory=list)
    #: STAR trust features computed in the referee path (leakage.py). Empty until
    #: run_referee wires them in; both are pure pandas/numpy (no API).
    double_dissociation: dict = field(default_factory=dict)
    confound_leaderboard: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim.claim_id,
            "claim_text": self.claim.claim_text,
            "substrate": self.claim.substrate,
            "head": self.claim.head,
            "population": {"group_a": self.claim.group_a,
                           "group_b": self.claim.group_b},
            "naive_effect": self.naive_effect,
            "robustness": {t.key: t.result.value for t in self.tests},
            "robustness_detail": {t.key: {"detail": t.detail, "stats": t.stats}
                                  for t in self.tests},
            "robustness_score": self.score,
            "verdict": self.verdict.value,
            "promoted": self.promoted,
            "biology_hypothesis": self.biology_hypothesis,
            "next_experiment": self.next_experiment,
            "falsification": self.falsification,
            "caveats": self.caveats,
            "evidence_ledger": self.evidence_ledger,
            "double_dissociation": self.double_dissociation,
            "confound_leaderboard": self.confound_leaderboard,
        }


# ---------------------------------------------------------------------------
# 4. Table validation (fail fast, loudly, with a helpful message)
# ---------------------------------------------------------------------------


class ContractError(ValueError):
    pass


def validate_table(df: pd.DataFrame, *, require_embeddings: bool = True) -> None:
    """Raise ContractError if `df` violates the contract. Coverage gaps (NaNs)
    are allowed by design and reported by cohort_summary(), not rejected here."""
    missing = [c for c in METADATA_COLUMNS if c not in df.columns]
    if missing:
        raise ContractError(f"table missing required columns: {missing}")
    if require_embeddings and not embedding_columns(df):
        raise ContractError(
            "no embedding columns (emb_0..emb_D) found — provide a feeder")
    if df["subject_id"].duplicated().any():
        raise ContractError("subject_id must be unique (one row per subject)")
    bad_dx = set(df["dx"].dropna().unique()) - set(DX_LEVELS)
    if bad_dx:
        raise ContractError(f"dx has values outside {DX_LEVELS}: {bad_dx}")
    bad_sex = set(df["sex"].dropna().unique()) - set(SEX_LEVELS)
    if bad_sex:
        raise ContractError(f"sex has values outside {SEX_LEVELS}: {bad_sex}")


def cohort_summary(df: pd.DataFrame) -> dict:
    """The cohort card falls out of the table for free (Step 3 of the plan)."""
    D = len(embedding_columns(df))
    n = len(df)

    def cov(col: str) -> float:
        if col not in df.columns:
            return 0.0
        return float(df[col].notna().mean())

    return {
        "n_subjects": int(n),
        "embedding_dim": D,
        "dx_counts": {k: int(v) for k, v in
                      df["dx"].value_counts(dropna=False).items()},
        "n_sites": int(df["site"].nunique(dropna=True)),
        "n_scanners": int(df["scanner"].nunique(dropna=True)),
        "age_mean": round(float(df["age"].mean()), 1) if n else None,
        "pct_female": round(100 * float((df["sex"] == "F").mean()), 1) if n else None,
        "label_coverage": {"conversion": cov("conversion")},
        "biomarker_coverage": {b: round(cov(b), 3) for b in BIOMARKER_COLUMNS},
    }


# Contract version — bump if the schema changes so cached tables can be checked.
CONTRACT_VERSION = "1.0.0"
