"""
Discovery + Referee — adjudicate every unsupervised phenotype.

`discover_and_referee(df)` runs the Detective (`detective.discover`), then for
each recovered cluster:

  1. RE-RUNS the existing five-test gauntlet on the sub-cohort and scores a
     per-cluster ClaimCard (same machinery the supervised referee uses), so a
     putative subtype must survive its OWN leakage / brain-age / anchor
     challenges to be promotable.
  2. CHARACTERIZES it:
       * conversion rate among its MCI, with a Wilson 95% CI,
       * plasma p-tau217 & GFAP standardized mean difference (Cohen's d + 95% CI)
         vs the rest of the cohort,
       * an ARTIFACT FLAG — how well cluster membership is explained by
         acquisition scanner/site (Cramer's V), age (eta^2) or sex (Cramer's V).
         A cluster whose membership is best explained by scanner/age/sex — rather
         than by biology or a surviving gauntlet — is flagged "likely artifact,
         not a phenotype."

When the table carries a ground-truth `phenotype` column, discovered-vs-planted
agreement is scored with ARI + AMI.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sstats

from . import contract, detective, gauntlet, scoring
from .contract import Claim
from .leakage import _eta_squared

#: A confound association (Cramer's V or eta^2) at/above this dominates biology.
_ARTIFACT_ASSOC = 0.25


# ---------------------------------------------------------------------------
# Small statistics (self-contained, no new deps)
# ---------------------------------------------------------------------------
def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score 95% CI for a binomial proportion. Returns (p, lo, hi)."""
    if n == 0:
        return (None, None, None)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(float(p), 3), round(float(center - half), 3),
            round(float(center + half), 3))


def _cohens_d(x_in: np.ndarray, x_out: np.ndarray) -> dict:
    """Standardized mean difference (in-cluster minus rest) with a 95% CI."""
    a = x_in[np.isfinite(x_in)]
    b = x_out[np.isfinite(x_out)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return {"d": None, "ci": (None, None), "n_in": n1, "n_out": n2}
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / (n1 + n2 - 2))
    if pooled == 0:
        return {"d": 0.0, "ci": (0.0, 0.0), "n_in": n1, "n_out": n2}
    d = (a.mean() - b.mean()) / pooled
    se = np.sqrt((n1 + n2) / (n1 * n2) + d * d / (2 * (n1 + n2)))
    return {"d": round(float(d), 3),
            "ci": (round(float(d - 1.96 * se), 3), round(float(d + 1.96 * se), 3)),
            "n_in": n1, "n_out": n2}


def _cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    """Bias-uncorrected Cramer's V between two categorical vectors."""
    a = pd.Series(a).astype("string").fillna("__na__")
    b = pd.Series(b).astype("string").fillna("__na__")
    table = pd.crosstab(a, b)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = sstats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    denom = min(table.shape[0] - 1, table.shape[1] - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-cluster referee
# ---------------------------------------------------------------------------
def _naive_effect(sub: pd.DataFrame) -> dict:
    from . import probe
    X, y, groups = probe.point_head(sub, "conversion")
    if len(np.unique(y)) < 2:
        return {"metric": "AUC", "value": 0.5, "target": "conversion", "n": int(len(y))}
    auc = probe.cross_val_auc(X, y, groups=groups)
    return {"metric": "AUC", "value": round(float(auc), 3),
            "target": "conversion", "n": int(len(y))}


def _referee_cluster(sub: pd.DataFrame, cluster_id: int):
    claim = Claim(
        claim_id=f"cluster-{cluster_id}",
        claim_text=f"Discovered cluster {cluster_id} is a distinct MCI->AD "
                   f"conversion phenotype.",
        target="conversion",
        group_a="cluster converters", group_b="cluster non-converters",
    )
    naive = _naive_effect(sub)
    tests = gauntlet.run_gauntlet(sub, claim)
    card = scoring.build_claim_card(claim, naive, tests)
    return card, tests, naive


# ---------------------------------------------------------------------------
# Characterization + artifact flag
# ---------------------------------------------------------------------------
def _characterize(df: pd.DataFrame, member: np.ndarray) -> dict:
    inside = df.loc[member]
    outside = df.loc[~member]

    # Conversion rate among the cluster's MCI (Wilson 95% CI).
    mci = inside[inside["dx"].astype("string") == "MCI"]
    conv = pd.to_numeric(mci["conversion"], errors="coerce").dropna()
    p, lo, hi = _wilson_ci(int(conv.sum()), int(conv.size))
    conversion = {"rate": p, "ci": (lo, hi), "n_mci": int(conv.size)}

    # Biomarker SMD (Cohen's d + CI) vs the rest.
    biomarker = {
        "p_tau217": _cohens_d(inside["p_tau217"].to_numpy(float),
                              outside["p_tau217"].to_numpy(float)),
        "gfap": _cohens_d(inside["gfap"].to_numpy(float),
                          outside["gfap"].to_numpy(float)),
    }
    return conversion, biomarker


def _artifact_associations(df: pd.DataFrame, member: np.ndarray) -> dict:
    memb = member.astype(int)
    scanner_v = _cramers_v(memb, df["scanner"].to_numpy())
    site_v = _cramers_v(memb, df["site"].to_numpy())
    sex_v = _cramers_v(memb, df["sex"].to_numpy())
    age = df["age"].to_numpy(float)
    ok = np.isfinite(age)
    age_eta2 = _eta_squared(age[ok], memb[ok]) if ok.sum() > 2 else 0.0
    return {
        "scanner_site": round(float(max(scanner_v, site_v)), 3),
        "scanner_cramers_v": round(float(scanner_v), 3),
        "site_cramers_v": round(float(site_v), 3),
        "age_eta2": round(float(age_eta2), 3),
        "sex_cramers_v": round(float(sex_v), 3),
    }


def _classify(card, assoc: dict) -> tuple:
    """Return (status, artifact_flag, driver). A cluster whose membership is best
    explained by acquisition/age/sex — and does not survive the gauntlet — is an
    artifact, not a phenotype."""
    confounds = {
        "scanner/site": assoc["scanner_site"],
        "age/atrophy": assoc["age_eta2"],
        "sex": assoc["sex_cramers_v"],
    }
    driver, driver_val = max(confounds.items(), key=lambda kv: kv[1])
    artifact_flag = (driver_val >= _ARTIFACT_ASSOC) and not card.promoted

    if card.promoted and not artifact_flag:
        status = "promotable phenotype"
    elif artifact_flag:
        status = f"likely artifact ({driver})"
    else:
        status = "inconclusive"
    return status, artifact_flag, driver


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def discover_and_referee(df: pd.DataFrame, method: str = "kmeans",
                         B: int = 50) -> dict:
    """Discover phenotypes, then referee + characterize each one.

    Returns {'discovery', 'clusters', 'note', 'ari', 'ami'} where `clusters` is a
    per-cluster list of {cluster, n, stability, gauntlet, characterization,
    artifact, status, dominant_phenotype?}.
    """
    contract.validate_table(df)
    disc = detective.discover(df, method=method, B=B)
    labels = np.asarray(disc["labels"])
    stability = disc.get("stability", {}) or {}
    has_truth = "phenotype" in df.columns

    clusters: list[dict] = []
    for c in sorted(int(x) for x in np.unique(labels) if x >= 0):
        member = labels == c
        sub = df.loc[member].reset_index(drop=True)
        card, tests, naive = _referee_cluster(sub, c)
        conversion, biomarker = _characterize(df, member)
        assoc = _artifact_associations(df, member)
        status, artifact_flag, driver = _classify(card, assoc)

        summary = {
            "cluster": c,
            "n": int(member.sum()),
            "stability": stability.get(c),
            "unstable": (stability.get(c) is not None
                         and stability.get(c) < detective.STABILITY_FLOOR),
            "naive_effect": naive,
            "gauntlet": {
                "score": card.score,
                "verdict": card.verdict.value,
                "promoted": bool(card.promoted),
                "tests": {t.key: t.result.value for t in tests},
            },
            "characterization": {"conversion": conversion, "biomarker": biomarker},
            "artifact": {**assoc, "flag": artifact_flag, "driver": driver},
            "status": status,
        }
        if has_truth:
            modes = df.loc[member, "phenotype"].astype("string").mode()
            summary["dominant_phenotype"] = (str(modes.iloc[0])
                                             if len(modes) else None)
        clusters.append(summary)

    ari = ami = None
    if has_truth:
        from sklearn.metrics import (adjusted_mutual_info_score,
                                     adjusted_rand_score)
        truth = df["phenotype"].astype("string").to_numpy()
        ari = round(float(adjusted_rand_score(truth, labels)), 3)
        ami = round(float(adjusted_mutual_info_score(truth, labels)), 3)

    n_promotable = sum(1 for c in clusters if c["status"] == "promotable phenotype")
    n_artifact = sum(1 for c in clusters if c["artifact"]["flag"])
    note = (f"Discovered {disc['k']} cluster(s) via {disc['method']} "
            f"(silhouette={disc['silhouette']}). "
            f"{n_promotable} promotable phenotype(s); {n_artifact} flagged as "
            f"likely acquisition/age/sex artifact(s).")
    if ari is not None:
        note += f" Recovery vs planted ground truth: ARI={ari}, AMI={ami}."

    return {"discovery": disc, "clusters": clusters, "note": note,
            "ari": ari, "ami": ami}
