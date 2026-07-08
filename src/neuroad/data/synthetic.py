"""
Synthetic contract-cohort generator (the guaranteed-offline live path).

Two presets, both contract-valid, both deterministic given ``seed``:

  SURVIVOR  strong disease load, modest site coupling
            -> conversion AUC ~0.74, site/scanner AUC ~0.64, signal survives.
  KILL      weak disease load, strong site coupling, disease is age-driven
            -> site/scanner AUC ~0.92 >= outcome AUC, collapses under brain-age.

Injected structure (so the referee's gauntlet has something *real* to catch):
  * a latent disease axis in the embedding driving dx / conversion,
  * a SITE/SCANNER confound direction in the embedding (the leakage ⭐),
  * a brain-age axis (chronological age is partly recoverable from embeddings),
  * plasma p-tau217 / GFAP correlated with the disease axis at the calibrated r,
  * realistic p-tau217 missingness (~45%, per calibration.PTAU217_MISSINGNESS).

Every headline behaviour is pinned to ``calibration.CAL`` — nothing free-floating.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from neuroad import contract
from neuroad import calibration as cal

# ---------------------------------------------------------------------------
# Preset knobs. These are the *design* levers; the emergent AUCs they produce
# are what calibration.CAL pins. Tuned so agent-1's linear probe + gauntlet
# reproduce the calibrated verdict bands (see tests/test_data.py).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Preset:
    name: str
    n_subjects: int
    embed_dim: int
    disease_load: float       # magnitude of the disease axis in the embedding
    site_couple: float        # magnitude of the site/scanner axis in the embedding
    age_load: float           # magnitude of the brain-age axis in the embedding
    disease_age_mix: float    # 0 = disease independent of age, 1 = disease is age
    disease_site_mix: float    # 0 = disease independent of site, >0 = outcome leaks the site
    disease_age_align: float  # 0 = disease axis ⟂ age axis, 1 = same direction
    conv_slope: float         # logistic slope: conversion vs within-MCI disease
    dx_noise: float           # jitter in dx banding (band overlap; widens MCI)
    noise: float              # isotropic embedding noise SD
    frac_cn: float            # dx mix
    frac_mci: float
    # remainder -> AD
    bio_disease_r: float = 0.55   # latent corr of plasma markers with the DISEASE axis
    #  SURVIVOR: high -> the anchor holds (proteins confirm real biology).
    #  KILL:     ~0   -> proteins do NOT track the artifact, so the anchor FAILS
    #                   the hard gate (a scanner/age artifact has no molecular support).


PRESETS: dict[str, Preset] = {
    # Strong, age-independent disease axis on its own direction; the confound is
    # present but modest -> outcome clearly exceeds the confound and survives the
    # brain-age control.
    "SURVIVOR": Preset(
        name="SURVIVOR", n_subjects=360, embed_dim=48,
        disease_load=3.10, site_couple=1.02, age_load=3.0,
        disease_age_mix=0.12, disease_site_mix=0.0, disease_age_align=0.34, conv_slope=2.8,
        dx_noise=0.85, noise=1.0, frac_cn=0.38, frac_mci=0.36,
        bio_disease_r=0.85,
    ),
    # Dominant confound; what disease signal exists is age-driven AND loaded onto
    # the age axis -> the naive effect looks real but the brain-age control
    # removes it (generic atrophy in disguise), and site leakage >= outcome.
    # Proteins do NOT track the artifact (bio_disease_r=0) -> anchor gate FAILS.
    # Age-driven disease on the age axis + a dominant, outcome-independent
    # scanner component in the embedding. Group-aware CV keeps the naive effect
    # honestly positive (the age signal generalizes across sites), so the demo
    # still opens on a tempting naive AUC; the gauntlet then shows the signal is
    # generic aging (age/sex + brain-age collapse) AND the embedding is
    # scanner-contaminated (leakage margin < 0) — two independent reasons to
    # refuse it — with no molecular anchor (bio_disease_r=0).
    "KILL": Preset(
        name="KILL", n_subjects=360, embed_dim=48,
        disease_load=1.55, site_couple=2.55, age_load=3.0,
        disease_age_mix=0.90, disease_site_mix=0.0, disease_age_align=0.95, conv_slope=2.6,
        dx_noise=0.75, noise=1.0, frac_cn=0.40, frac_mci=0.38,
        bio_disease_r=0.0,
    ),
}

SCANNER_BY_SITE = {
    "SITE_A": "GE_Signa_1.5T",
    "SITE_B": "Siemens_Trio_3T",
}
_SITE_NAMES = list(SCANNER_BY_SITE.keys())


def _unit(rng: np.random.Generator, d: int) -> np.ndarray:
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x - x.mean()


def generate_cohort(preset: str, seed: int = 0) -> pd.DataFrame:
    """Generate a contract-valid synthetic cohort for ``preset``.

    Parameters
    ----------
    preset : {'SURVIVOR', 'KILL'}
    seed   : int   deterministic RNG seed.

    Returns
    -------
    pd.DataFrame  passing ``contract.validate_table``.
    """
    key = preset.upper()
    if key not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {list(PRESETS)}")
    p = PRESETS[key]
    rng = np.random.default_rng(seed)
    n, d = p.n_subjects, p.embed_dim

    # --- latent axes (fixed given seed) ---------------------------------
    u_site = _unit(rng, d)
    u_age = _unit(rng, d)
    u_disease_raw = _unit(rng, d)
    # In KILL the disease axis is (nearly) the age axis, so removing brain-age
    # removes the disease signal too; in SURVIVOR it is its own direction.
    a = p.disease_age_align
    u_disease = a * u_age + (1.0 - a) * u_disease_raw
    u_disease = u_disease / np.linalg.norm(u_disease)

    # --- chronological age (wide-ish, AD-cohort skewed) -----------------
    age = np.clip(rng.normal(74.0, 8.0, n), 55.0, 96.0)
    age_z = _standardize(age)

    # --- site / scanner membership (defined before disease so the KILL's
    #     outcome can leak the site — cases drawn disproportionately from one
    #     scanner, the classic real-world batch confound). -------------------
    site_ind = rng.integers(0, 2, n)               # 0/1 membership
    site_centered = site_ind - 0.5
    site_z = _standardize(site_centered.astype(float))

    # --- latent disease score -------------------------------------------
    # SURVIVOR: disease is largely its own biology (site/age independent).
    # KILL: disease is mostly SITE + age -> the naive signal is which scanner
    # acquired the scan plus generic atrophy; controlling either guts it, and a
    # scanner probe predicts it better than the outcome probe (leakage).
    own_mix = max(1.0 - p.disease_age_mix - p.disease_site_mix, 0.0)
    disease_own = rng.standard_normal(n)
    disease = (p.disease_age_mix * age_z
               + p.disease_site_mix * site_z
               + own_mix * disease_own)
    disease_z = _standardize(disease)

    # --- diagnosis from the disease score -------------------------------
    # Jittered ranking so the CN/MCI/AD bands overlap in true disease (realistic
    # diagnostic uncertainty) — this also widens the within-MCI disease spread,
    # which is what a prognostic conversion probe actually has to work with.
    dx_rank = disease_z + p.dx_noise * rng.standard_normal(n)
    order = np.argsort(dx_rank)
    dx = np.empty(n, dtype=object)
    n_cn = int(round(p.frac_cn * n))
    n_mci = int(round(p.frac_mci * n))
    dx[order[:n_cn]] = "CN"
    dx[order[n_cn:n_cn + n_mci]] = "MCI"
    dx[order[n_cn + n_mci:]] = "AD"

    # --- conversion label (only defined for MCI) ------------------------
    # Among MCI, higher disease -> higher conversion probability. Logistic
    # link keeps it probabilistic (not a clean threshold the probe memorizes).
    is_mci = dx == "MCI"
    conversion = np.array([pd.NA] * n, dtype=object)
    mci_idx = np.where(is_mci)[0]
    if mci_idx.size:
        mci_d = _standardize(disease_z[mci_idx])
        p_conv = 1.0 / (1.0 + np.exp(-p.conv_slope * mci_d))
        conv = (rng.random(mci_idx.size) < p_conv).astype(int)
        for i, ci in zip(mci_idx, conv):
            conversion[i] = int(ci)

    # --- site / scanner names (membership drawn above) ------------------
    site = np.array([_SITE_NAMES[s] for s in site_ind], dtype=object)
    scanner = np.array([SCANNER_BY_SITE[s] for s in site], dtype=object)

    # --- embedding = disease axis + site axis + age axis + noise --------
    emb = (
        p.disease_load * np.outer(disease_z, u_disease)
        + p.site_couple * np.outer(site_centered, u_site)
        + p.age_load * np.outer(age_z, u_age)
        + p.noise * rng.standard_normal((n, d))
    )

    # --- plasma biomarkers correlated with the disease axis -------------
    # Injected against the DISEASE axis at the preset's bio_disease_r. The probe
    # score is a noisy estimate of that axis, so the *observed* anchor r (probe
    # score vs marker) lands lower than bio_disease_r — SURVIVOR ~0.4 (PASSED),
    # KILL ~0 (FAILED, no molecular support for the artifact).
    br = p.bio_disease_r
    p_tau217 = _biomarker(rng, disease_z, r=br,
                          mean=0.30, sd=0.18, floor=0.02)
    gfap = _biomarker(rng, disease_z, r=0.80 * br,
                      mean=110.0, sd=45.0, floor=5.0)
    nfl = _biomarker(rng, disease_z, r=0.60 * br,
                     mean=18.0, sd=8.0, floor=2.0)

    # amyloid positivity: prevalence rises with disease
    p_amy = 1.0 / (1.0 + np.exp(-(0.9 * disease_z - 0.2)))
    amyloid = (rng.random(n) < p_amy).astype(int)

    # APOE e4 allele count (0/1/2), weakly disease-linked
    p_e4 = np.clip(0.25 + 0.10 * disease_z, 0.02, 0.85)
    apoe4 = rng.binomial(2, p_e4)

    sex = rng.choice(["M", "F"], size=n)

    # --- assemble the contract table ------------------------------------
    frame = contract.make_embedding_frame(emb)
    frame.insert(0, "subject_id", [f"{key[:3]}_{seed:02d}_{i:04d}" for i in range(n)])
    frame["dx"] = pd.Categorical(dx, categories=contract.DX_LEVELS)
    frame["conversion"] = pd.array(conversion, dtype="Int8")
    frame["age"] = age.astype("float64")
    frame["sex"] = pd.Categorical(sex, categories=contract.SEX_LEVELS)
    frame["site"] = pd.Categorical(site, categories=_SITE_NAMES)
    frame["scanner"] = pd.Categorical(scanner, categories=list(SCANNER_BY_SITE.values()))
    frame["amyloid"] = pd.array(amyloid, dtype="Int8")

    # realistic p-tau217 missingness (surface a completeness caveat downstream)
    miss = rng.random(n) < cal.PTAU217_MISSINGNESS
    p_tau217 = p_tau217.astype("float64")
    p_tau217[miss] = np.nan
    frame["p_tau217"] = p_tau217
    frame["gfap"] = gfap.astype("float64")
    frame["nfl"] = nfl.astype("float64")
    frame["apoe4"] = pd.array(apoe4.astype("int64"), dtype="Int8")

    contract.validate_table(frame)
    return frame


def _biomarker(rng: np.random.Generator, disease_z: np.ndarray, *,
               r: float, mean: float, sd: float, floor: float) -> np.ndarray:
    """A plasma marker with population Pearson ~= r against the disease axis,
    shifted/scaled into a plausible pg/mL range and floored at a detection LOD."""
    noise = rng.standard_normal(disease_z.size)
    latent = r * disease_z + np.sqrt(max(1.0 - r * r, 0.0)) * noise
    latent = _standardize(latent)
    return np.maximum(floor, mean + sd * latent)


# ===========================================================================
# Planted-phenotype cohort (the Detective's ground-truth benchmark)
# ===========================================================================
# A single contract table carrying a `phenotype` ground-truth column that
# plants THREE deliberately distinct subgroups, each encoding its disease signal
# on a *different* latent axis so the unsupervised Detective can recover them and
# the per-cluster gauntlet can adjudicate them differently:
#
#   1. tau_hot          — disease on its OWN biology axis (⟂ age, ⟂ scanner),
#                         plasma p-tau217/GFAP strongly elevated + tracking it.
#                         SURVIVES the gauntlet -> a promotable phenotype.
#   2. age_atrophy      — disease gradient IS chronological age, encoded on the
#                         brain-age axis; no molecular correlate. COLLAPSES under
#                         the brain-age control -> flagged age/atrophy artifact.
#   3. scanner_artifact — separation lives ENTIRELY on the site/scanner axis;
#                         membership maps 1:1 to two dedicated scanners, no
#                         biomarker. Fails site/scanner leakage -> flagged
#                         acquisition artifact, not a phenotype.
#
# Ground truth is the `phenotype` column; recovery is scored by ARI/AMI in
# discovery.py. Deterministic given ``seed``. Reuses the same latent-axis and
# biomarker machinery as generate_cohort so nothing is free-floating.

PHENOTYPE_LEVELS = ["tau_hot", "age_atrophy", "scanner_artifact"]

#: The age/atrophy + tau_hot phenotypes share the two "real" scanners; the
#: scanner-artifact phenotype gets two DEDICATED scanners so its cluster
#: membership is (by construction) fully explained by acquisition hardware.
_PHENO_SCANNER_BY_SITE = {
    "SITE_A": "GE_Signa_1.5T",
    "SITE_B": "Siemens_Trio_3T",
    "SITE_C": "Philips_Achieva_3T",
    "SITE_D": "Philips_Ingenia_3T",
}
_PHENO_SITE_NAMES = list(_PHENO_SCANNER_BY_SITE.keys())


def generate_phenotype_cohort(seed: int = 0, n_per: int = 150) -> pd.DataFrame:
    """Generate a contract-valid cohort with a planted ``phenotype`` ground truth.

    Parameters
    ----------
    seed  : int   deterministic RNG seed.
    n_per : int   subjects per planted phenotype (total = 3 * n_per).

    Returns
    -------
    pd.DataFrame passing ``contract.validate_table`` with an extra categorical
    ``phenotype`` column in {tau_hot, age_atrophy, scanner_artifact}.
    """
    rng = np.random.default_rng(seed)
    d = 48

    # Orthonormal latent axes: disease (own biology), brain-age, site/scanner.
    Q, _ = np.linalg.qr(rng.standard_normal((d, 4)))
    u_disease, u_age, u_site = Q[:, 0], Q[:, 1], Q[:, 2]

    SEP = 10.0           # phenotype centroid separation (dominates within spread)
    signal_load = 2.6    # within-phenotype disease signal magnitude
    age_load = 1.6       # shared brain-age structure magnitude
    site_couple = 1.0    # mild A/B acquisition confound (C/D handled by the axis)
    noise_sd = 1.0
    conv_slope = 2.7
    frac_cn, frac_mci = 0.35, 0.40

    pheno_idx: list = []
    d_local: list = []
    age_parts: list = []
    dx_parts: list = []
    conv_parts: list = []
    site_parts: list = []
    ptau_parts: list = []
    gfap_parts: list = []
    nfl_parts: list = []

    for pi, pname in enumerate(PHENOTYPE_LEVELS):
        n = n_per
        if pname == "tau_hot":
            dl = rng.standard_normal(n)                     # own biology axis
            ag = np.clip(rng.normal(71.0, 6.5, n), 55.0, 96.0)
            st = rng.choice(["SITE_A", "SITE_B"], size=n)   # site-independent
            dlz = _standardize(dl)
            ptau = _biomarker(rng, dlz, r=0.85, mean=0.48, sd=0.16, floor=0.02)
            gfap = _biomarker(rng, dlz, r=0.64, mean=150.0, sd=45.0, floor=5.0)
            nfl = _biomarker(rng, dlz, r=0.47, mean=22.0, sd=8.0, floor=2.0)
        elif pname == "age_atrophy":
            base = rng.standard_normal(n)
            ag = np.clip(85.0 + 2.6 * base, 60.0, 96.0)     # OLD; disease == age
            dl = base
            st = rng.choice(["SITE_A", "SITE_B"], size=n)
            dlz = _standardize(dl)
            ptau = _biomarker(rng, dlz, r=0.0, mean=0.20, sd=0.10, floor=0.02)
            gfap = _biomarker(rng, dlz, r=0.0, mean=95.0, sd=35.0, floor=5.0)
            nfl = _biomarker(rng, dlz, r=0.0, mean=16.0, sd=7.0, floor=2.0)
        else:  # scanner_artifact
            half = n // 2
            sgn = np.array([1.0] * half + [-1.0] * (n - half))
            rng.shuffle(sgn)
            dl = sgn + 0.15 * rng.standard_normal(n)        # gradient == scanner
            ag = np.clip(rng.normal(71.0, 6.5, n), 55.0, 96.0)
            st = np.where(sgn > 0, "SITE_C", "SITE_D")       # dedicated scanners
            dlz = _standardize(dl)
            ptau = _biomarker(rng, dlz, r=0.0, mean=0.20, sd=0.10, floor=0.02)
            gfap = _biomarker(rng, dlz, r=0.0, mean=95.0, sd=35.0, floor=5.0)
            nfl = _biomarker(rng, dlz, r=0.0, mean=16.0, sd=7.0, floor=2.0)

        # dx bands within the phenotype by (jittered) disease rank.
        rank = dlz + 0.7 * rng.standard_normal(n)
        order = np.argsort(rank)
        dxp = np.empty(n, dtype=object)
        n_cn = int(round(frac_cn * n))
        n_mci = int(round(frac_mci * n))
        dxp[order[:n_cn]] = "CN"
        dxp[order[n_cn:n_cn + n_mci]] = "MCI"
        dxp[order[n_cn + n_mci:]] = "AD"

        # conversion among MCI (logistic in the within-phenotype disease score).
        convp = np.array([pd.NA] * n, dtype=object)
        mci = np.where(dxp == "MCI")[0]
        if mci.size:
            md = _standardize(dlz[mci])
            p_conv = 1.0 / (1.0 + np.exp(-conv_slope * md))
            conv = (rng.random(mci.size) < p_conv).astype(int)
            for i, c in zip(mci, conv):
                convp[i] = int(c)

        pheno_idx.append(np.full(n, pi))
        d_local.append(dlz)
        age_parts.append(ag)
        dx_parts.append(dxp)
        conv_parts.append(convp)
        site_parts.append(np.asarray(st, dtype=object))
        ptau_parts.append(ptau)
        gfap_parts.append(gfap)
        nfl_parts.append(nfl)

    pheno_idx = np.concatenate(pheno_idx)
    d_local = np.concatenate(d_local)
    age = np.concatenate(age_parts)
    dx = np.concatenate(dx_parts)
    conversion = np.concatenate(conv_parts)
    site = np.concatenate(site_parts)
    p_tau217 = np.concatenate(ptau_parts)
    gfap = np.concatenate(gfap_parts)
    nfl = np.concatenate(nfl_parts)
    N = len(pheno_idx)

    age_z = _standardize(age)

    # Mild A/B acquisition confound; the C/D separation is carried by the axis.
    site_c = np.zeros(N)
    site_c[site == "SITE_A"] = 0.5
    site_c[site == "SITE_B"] = -0.5

    # Each phenotype is offset SEP along its characteristic axis and carries its
    # within-phenotype disease signal on the SAME axis.
    axis_by_pheno = [u_disease, u_age, u_site]
    centroid = np.zeros((N, d))
    signal = np.zeros((N, d))
    for pi in range(len(PHENOTYPE_LEVELS)):
        m = pheno_idx == pi
        centroid[m] = axis_by_pheno[pi]
        signal[m] = axis_by_pheno[pi]

    emb = (
        SEP * centroid
        + signal_load * d_local[:, None] * signal
        + age_load * age_z[:, None] * u_age[None, :]
        + site_couple * site_c[:, None] * u_site[None, :]
        + noise_sd * rng.standard_normal((N, d))
    )

    # amyloid / apoe4 / sex (weakly disease-linked where relevant).
    p_amy = 1.0 / (1.0 + np.exp(-(0.9 * d_local - 0.2)))
    amyloid = (rng.random(N) < p_amy).astype(int)
    p_e4 = np.clip(0.25 + 0.10 * d_local, 0.02, 0.85)
    apoe4 = rng.binomial(2, p_e4)
    sex = rng.choice(["M", "F"], size=N)

    scanner = np.array([_PHENO_SCANNER_BY_SITE[s] for s in site], dtype=object)
    phenotype = np.array([PHENOTYPE_LEVELS[i] for i in pheno_idx], dtype=object)

    frame = contract.make_embedding_frame(emb)
    frame.insert(0, "subject_id", [f"PHE_{seed:02d}_{i:04d}" for i in range(N)])
    frame["dx"] = pd.Categorical(dx, categories=contract.DX_LEVELS)
    frame["conversion"] = pd.array(conversion, dtype="Int8")
    frame["age"] = age.astype("float64")
    frame["sex"] = pd.Categorical(sex, categories=contract.SEX_LEVELS)
    frame["site"] = pd.Categorical(site, categories=_PHENO_SITE_NAMES)
    frame["scanner"] = pd.Categorical(
        scanner, categories=list(_PHENO_SCANNER_BY_SITE.values()))
    frame["amyloid"] = pd.array(amyloid, dtype="Int8")

    miss = rng.random(N) < cal.PTAU217_MISSINGNESS
    p_tau217 = p_tau217.astype("float64")
    p_tau217[miss] = np.nan
    frame["p_tau217"] = p_tau217
    frame["gfap"] = gfap.astype("float64")
    frame["nfl"] = nfl.astype("float64")
    frame["apoe4"] = pd.array(apoe4.astype("int64"), dtype="Int8")

    # Ground-truth phenotype (NOT a contract metadata column — extra is allowed).
    frame["phenotype"] = pd.Categorical(phenotype, categories=PHENOTYPE_LEVELS)

    contract.validate_table(frame)
    return frame
