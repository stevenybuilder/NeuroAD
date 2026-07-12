"""
validation — outcome-validation for the Output target-prioritization layer.

A READ-ONLY side artifact that asks a single, honest question of the ranking
adapters the engine already ships: *when we rank AD proteins by priority /
association, do the genes we independently KNOW to matter (GWAS-validated risk
genes; targets of FDA-approved AD drugs) actually float to the top?* It scores
``pi4ad.rank_ad_targets`` and ``opentargets.disease_targets`` against small,
in-code, per-gene-CITED GOLD SETS, and reports precision@k / recall@k / ROC-AUC
with a label-shuffle permutation p-value.

It touches NOTHING in the referee path — not ``gauntlet.py``, ``scoring.py``, or
``contract.py``. It never raises offline. Every report is provenance-stamped
(``source=offline_snapshot|live``, ``background_size``) so a curation-biased tiny
offline background is never mistaken for a rigorous full-universe evaluation.

TWO HONESTY GUARDS, both wired in and both LOUD in the output:

  1. Circularity. Open Targets' overall ``association_score`` is BUILT FROM the
     ``genetic_association`` and ``clinical`` (known-drug) datatypes — the very
     evidence that DEFINES a GWAS/drug gold set. Scoring the overall score
     against those gold sets is near-circular, so this module reports it as the
     *optimistic* number and ALSO computes a leave-evidence-out AUC from the
     per-datatype breakdown (predict the GWAS gold set from NON-genetic evidence;
     predict the drug gold set from NON-clinical evidence). The held-out number
     is the honest one.

  2. Curation bias. The bundled OFFLINE snapshots deliberately include canonical
     AD genes (13/15 GWAS gold genes sit in the 50-gene OT snapshot; 9/15 in the
     74-gene PI4AD snapshot), and the background is tiny. Offline precision@k /
     AUC are therefore enriched-by-construction and NOT an unbiased estimate;
     every offline report carries that caveat, and the rigorous full-background
     run requires ``prefer_offline=False`` (freely fetchable, keyless: the
     ~14.7k-gene PI4AD HTTP table and paged Open Targets GraphQL).

Honest empirical note (offline snapshots, seed=0, this repo): the only
statistically significant offline signal is the CIRCULAR one — OT drug-targets
via the overall score, AUC≈0.85, p<0.001. Every non-circular offline case is at
or below chance by construction (PI4AD-vs-GWAS AUC≈0.13; OT-vs-GWAS held-out
AUC≈0.54, p≈0.33; OT-vs-drugs held-out AUC≈0.43). That is the expected
consequence of Guards 1+2, not a bug — a rigorous verdict needs the live run.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

_log = logging.getLogger("neuroad.harness.validation")

# ---------------------------------------------------------------------------
# Datatype partitions for the leave-evidence-out (Honesty Guard 1) mode.
# Keys match the Open Targets per-datatype breakdown in the snapshot/API.
# ---------------------------------------------------------------------------
#: Evidence that DEFINES a GWAS gold set — excluded when predicting it.
GENETIC_DATATYPES: frozenset[str] = frozenset(
    {"genetic_association", "genetic_literature"})
#: Evidence that DEFINES an approved-drug gold set — excluded when predicting it.
CLINICAL_DATATYPES: frozenset[str] = frozenset({"clinical", "known_drug"})


# ---------------------------------------------------------------------------
# GOLD SETS — frozen, in-code, every gene carrying a public citation.
# Mirrors the honesty discipline of translation.MECHANISM_GENES: narrow, cited,
# no invented labels. Genes are the intersection of "well-established AD truth"
# with "resolvable to a symbol", not a fabricated exhaustive list.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldGene:
    """One gold-standard gene and the public source that validates it."""
    gene: str
    citation: str


@dataclass(frozen=True)
class GoldSet:
    """A named, cited set of ground-truth AD genes for outcome validation."""
    name: str
    description: str
    genes: tuple[GoldGene, ...]

    @property
    def symbols(self) -> frozenset[str]:
        return frozenset(g.gene.upper() for g in self.genes)

    def citations(self) -> dict[str, str]:
        return {g.gene: g.citation for g in self.genes}


_BELLENGUEZ = (
    "Bellenguez C, et al. New insights into the genetic etiology of Alzheimer's "
    "disease and related dementias. Nat Genet. 2022;54(4):412-436. PMID:35379992"
)

#: GWAS-validated AD risk genes — genome-wide-significant loci in Bellenguez 2022.
GWAS_GOLD = GoldSet(
    name="gwas_ad_bellenguez_2022",
    description=(
        "Genome-wide-significant Alzheimer's risk genes from the Bellenguez et "
        "al. 2022 GWAS meta-analysis (75 loci). Narrow, canonical subset."),
    genes=tuple(GoldGene(g, f"{_BELLENGUEZ} — genome-wide-significant locus {g}")
                for g in (
        "SORL1", "TREM2", "ABCA7", "CLU", "BIN1", "PICALM", "CD33",
        "MS4A6A", "CR1", "EPHA1", "PLCG2", "ADAM10", "ACE", "APOE", "APH1B",
    )),
)

#: Molecular targets of FDA-approved Alzheimer's drugs.
DRUG_GOLD = GoldSet(
    name="fda_approved_ad_drug_targets",
    description=(
        "Molecular targets of FDA-approved Alzheimer's therapeutics: "
        "cholinesterase inhibitors, the NMDA-receptor antagonist memantine, and "
        "the anti-amyloid-beta antibodies."),
    genes=(
        GoldGene("ACHE", "FDA labels: donepezil (Aricept), rivastigmine (Exelon), "
                          "galantamine (Razadyne) — acetylcholinesterase inhibitors"),
        GoldGene("BCHE", "FDA label: rivastigmine (Exelon) — also inhibits "
                         "butyrylcholinesterase"),
        GoldGene("APP", "FDA labels: lecanemab (Leqembi), aducanumab (Aduhelm), "
                        "donanemab (Kisunla) — anti-amyloid-beta (APP/Abeta)"),
        GoldGene("GRIN1", "FDA label: memantine (Namenda) — NMDA-receptor "
                          "(GRIN subunit) antagonist"),
        GoldGene("GRIN2A", "FDA label: memantine (Namenda) — NMDA-receptor "
                           "(GRIN subunit) antagonist"),
        GoldGene("GRIN2B", "FDA label: memantine (Namenda) — NMDA-receptor "
                           "(GRIN subunit) antagonist"),
        GoldGene("GRIN2C", "FDA label: memantine (Namenda) — NMDA-receptor "
                           "(GRIN subunit) antagonist"),
        GoldGene("GRIN2D", "FDA label: memantine (Namenda) — NMDA-receptor "
                           "(GRIN subunit) antagonist"),
        GoldGene("GRIN3A", "FDA label: memantine (Namenda) — NMDA-receptor "
                           "(GRIN subunit) antagonist"),
    ),
)

# ---------------------------------------------------------------------------
# TEMPORAL gold sets — the pre/post split for prospective novel-target validation.
# KNOWN_2019 = AD-risk genes established by Kunkle 2019 (and Lambert 2013 before it).
# NOVEL_2022 = genome-wide-significant AD genes NEW in Bellenguez 2022 that were NOT
# genome-wide significant in Kunkle 2019 — the held-out "future discoveries". The
# engine is scored on ranking NOVEL_2022 from evidence orthogonal to / predating the
# 2022 GWAS (STRING network seeded on KNOWN_2019; non-genetic OT). See
# docs/TEMPORAL_VALIDATION_SPEC.md. Both are NARROW, CITED, high-confidence subsets
# (same discipline as GWAS_GOLD) — verify/expand against the source supplementary
# tables before over-interpreting a null.
# ---------------------------------------------------------------------------
_KUNKLE = (
    "Kunkle BW, et al. Genetic meta-analysis of diagnosed Alzheimer's disease "
    "identifies new risk loci. Nat Genet. 2019;51(3):414-430. PMID:30820047")
_BELLENGUEZ_NEW = (
    _BELLENGUEZ + " — locus NOT genome-wide significant in Kunkle 2019 (new in 2022)")

#: Established (pre-2020) AD-risk genes — the network seeds / "already known" set.
KNOWN_2019 = GoldSet(
    name="known_ad_kunkle_2019",
    description=("AD-risk genes at genome-wide significance by Kunkle et al. 2019 "
                 "(building on Lambert 2013) — the pre-Bellenguez established set."),
    genes=tuple(GoldGene(g, f"{_KUNKLE} — established locus {g}") for g in (
        "APOE", "BIN1", "CR1", "CLU", "PICALM", "MS4A6A", "ABCA7", "EPHA1", "CD33",
        "CD2AP", "SORL1", "TREM2", "INPP5D", "MEF2C", "FERMT2", "CASS4", "PTK2B",
        "SLC24A4", "CELF1", "ZCWPW1", "ACE", "ADAM10", "IQCK", "WWOX",
    )),
)

#: The held-out "future discovery" set — the nearest protein-coding gene at EVERY
#: locus new in Bellenguez 2022 (Table 2: "new loci at the time of analysis with a
#: genome-wide significant signal"). This is the COMPLETE, source-verified new-loci
#: set (41 mappable single-gene loci of the 42; locus 27 = the IGH gene cluster,
#: which has no single mappable symbol, is omitted). None are in KNOWN_2019. Verified
#: 2026-07 against the Nature Genetics full text, Table 2 (open-access PDF). This
#: replaces an earlier 20-gene curated subset that had erroneously included TSPOAP1
#: (a KNOWN locus in the paper's Table 1), CCDC6, and TMEM163 (not Table-2 new-loci
#: genes) — see git history.
_BELLENGUEZ_NEW_LOCI = (
    # (gene, Bellenguez-2022 Table-2 locus number)
    ("SORT1", 1), ("ADAM17", 2), ("PRKD3", 3), ("NCK2", 4), ("WDR12", 5),
    ("MME", 6), ("IDUA", 7), ("RHOH", 8), ("ANKH", 9), ("COX7C", 10),
    ("TNIP1", 11), ("RASGEF1C", 12), ("HS3ST5", 13), ("UMAD1", 14), ("ICA1", 15),
    ("TMEM106B", 16), ("JAZF1", 17), ("SEC61G", 18), ("CTSB", 19), ("SHARPIN", 20),
    ("ABCA1", 21), ("ANK3", 22), ("TSPAN14", 23), ("BLNK", 24), ("PLEKHA1", 25),
    ("TPCN1", 26), ("SNX1", 28), ("CTSH", 29), ("DOC2A", 30), ("MAF", 31),
    ("FOXF1", 32), ("PRDM7", 33), ("WDR81", 34), ("MYO15A", 35), ("GRN", 36),
    ("KLF16", 37), ("SIGLEC11", 38), ("LILRB2", 39), ("RBCK1", 40),
    ("SLC2A4RG", 41), ("APP", 42),
)
NOVEL_2022 = GoldSet(
    name="novel_ad_bellenguez_2022",
    description=("Nearest protein-coding gene at every locus reaching genome-wide "
                 "significance for the FIRST time in Bellenguez et al. 2022 "
                 "(Table 2, new loci; not significant in Kunkle 2019) — the complete, "
                 "source-verified prospective novel-target test set."),
    genes=tuple(
        GoldGene(g, f"{_BELLENGUEZ_NEW} — Table 2 new-locus {loci} (nearest gene {g})")
        for g, loci in _BELLENGUEZ_NEW_LOCI),
)

#: NEGATIVE-CONTROL gold set — ubiquitously-expressed housekeeping genes with no
#: established AD-risk role. A specificity check: an HONEST ranker must score these
#: at CHANCE (AUC ~0.5). If a decoy set scores high, the "signal" is an artifact
#: (background bias / curation leakage), not AD biology. Housekeeping genes per
#: Eisenberg & Levanon 2013 (Trends Genet 29:569) + canonical references.
DECOY_GOLD = GoldSet(
    name="housekeeping_decoy_negative_control",
    description=("Ubiquitously-expressed housekeeping genes with no established "
                "Alzheimer's-risk role — a negative control that an honest ranker "
                "must score at chance (AUC~0.5)."),
    genes=tuple(GoldGene(g, "Housekeeping/negative-control gene (Eisenberg & Levanon "
                            "2013, Trends Genet 29:569-574) — no established AD-risk "
                            f"role [{g}]") for g in (
        "ACTB", "GAPDH", "B2M", "RPL13A", "TUBB", "PGK1", "HPRT1", "TBP",
        "PPIA", "YWHAZ", "SDHA", "UBC", "RPLP0", "GUSB", "TFRC",
    )),
)

GOLD_SETS: dict[str, GoldSet] = {
    GWAS_GOLD.name: GWAS_GOLD,
    DRUG_GOLD.name: DRUG_GOLD,
    KNOWN_2019.name: KNOWN_2019,
    NOVEL_2022.name: NOVEL_2022,
    DECOY_GOLD.name: DECOY_GOLD,
}


# ---------------------------------------------------------------------------
# Metrics core — pure numpy/sklearn, exception-safe, deterministic given a seed.
# ---------------------------------------------------------------------------


def precision_at_k(ranked_genes: Sequence[str],
                   gold_symbols: frozenset[str], k: int) -> Optional[float]:
    """Fraction of the top-``k`` ranked genes that are in the gold set.

    ``ranked_genes`` must be ordered best-first. Returns None if ``k<=0`` or the
    ranking is empty (an honest "undefined", never a fabricated number)."""
    try:
        if k <= 0 or not ranked_genes:
            return None
        gold = {g.upper() for g in gold_symbols}
        top = [g.upper() for g in ranked_genes[:k]]
        if not top:
            return None
        return sum(1 for g in top if g in gold) / len(top)
    except Exception as exc:  # noqa: BLE001
        _log.debug("precision_at_k failed: %r", exc)
        return None


def recall_at_k(ranked_genes: Sequence[str],
                gold_symbols: frozenset[str], k: int) -> Optional[float]:
    """Fraction of the RECOVERABLE gold genes found in the top-``k``.

    Denominator is the number of gold genes present anywhere in the ranked
    universe (recall against an out-of-universe gene is not this metric's job).
    Returns None if no gold gene is in the universe."""
    try:
        if k <= 0 or not ranked_genes:
            return None
        gold = {g.upper() for g in gold_symbols}
        universe = {g.upper() for g in ranked_genes}
        recoverable = gold & universe
        if not recoverable:
            return None
        top = {g.upper() for g in ranked_genes[:k]}
        return len(top & recoverable) / len(recoverable)
    except Exception as exc:  # noqa: BLE001
        _log.debug("recall_at_k failed: %r", exc)
        return None


def roc_auc(ranked_genes: Sequence[str], scores: Sequence[float],
            gold_symbols: frozenset[str]) -> Optional[float]:
    """ROC-AUC of the gold membership label vs the ranking score.

    Uses ``sklearn.metrics.roc_auc_score`` over the whole universe. Returns None
    when the labels are single-class (all/none gold) or on any failure."""
    try:
        import numpy as np
        from sklearn.metrics import roc_auc_score
        gold = {g.upper() for g in gold_symbols}
        y = np.array([1 if g.upper() in gold else 0 for g in ranked_genes])
        s = np.asarray(scores, dtype=float)
        if y.size == 0 or s.size != y.size:
            return None
        if y.sum() == 0 or y.sum() == y.size:
            return None
        return float(roc_auc_score(y, s))
    except Exception as exc:  # noqa: BLE001
        _log.debug("roc_auc failed: %r", exc)
        return None


def permutation_pvalue(ranked_genes: Sequence[str], scores: Sequence[float],
                       gold_symbols: frozenset[str], *, n_perm: int = 1000,
                       seed: int = 0) -> Optional[float]:
    """Label-shuffle permutation p-value for the observed ROC-AUC.

    Shuffles the gold labels ``n_perm`` times and returns the fraction of
    shuffles whose AUC is >= the observed AUC (add-one smoothed, so the p-value
    is never exactly 0). Deterministic given ``seed``. Returns None if the AUC is
    undefined (single-class) or on any failure."""
    try:
        import numpy as np
        from sklearn.metrics import roc_auc_score
        gold = {g.upper() for g in gold_symbols}
        y = np.array([1 if g.upper() in gold else 0 for g in ranked_genes])
        s = np.asarray(scores, dtype=float)
        if y.size == 0 or s.size != y.size:
            return None
        if y.sum() == 0 or y.sum() == y.size:
            return None
        observed = roc_auc_score(y, s)
        rng = np.random.default_rng(seed)
        ge = 0
        for _ in range(max(1, int(n_perm))):
            yp = rng.permutation(y)
            if yp.sum() == 0 or yp.sum() == yp.size:
                continue
            if roc_auc_score(yp, s) >= observed:
                ge += 1
        return (ge + 1) / (int(n_perm) + 1)
    except Exception as exc:  # noqa: BLE001
        _log.debug("permutation_pvalue failed: %r", exc)
        return None


def bootstrap_auc_ci(ranked_genes: Sequence[str], scores: Sequence[float],
                     gold_symbols: frozenset[str], *, n_boot: int = 2000,
                     seed: int = 0, alpha: float = 0.05
                     ) -> Optional[tuple[float, float]]:
    """Percentile bootstrap confidence interval for the ROC-AUC.

    Case-resamples the (score, label) universe ``n_boot`` times and returns the
    ``(lo, hi)`` percentile CI at level ``1-alpha`` (default 95%). This is the
    honest companion to a point AUC on a tiny gold set — a 0.73 with n_gold=15 has
    a wide CI, and reporting it prevents over-reading. Deterministic given
    ``seed``. Returns None if the AUC is undefined or on any failure."""
    try:
        import numpy as np
        from sklearn.metrics import roc_auc_score
        gold = {g.upper() for g in gold_symbols}
        y = np.array([1 if g.upper() in gold else 0 for g in ranked_genes])
        s = np.asarray(scores, dtype=float)
        if y.size == 0 or s.size != y.size or y.sum() == 0 or y.sum() == y.size:
            return None
        rng = np.random.default_rng(seed)
        n = y.size
        aucs = []
        for _ in range(max(1, int(n_boot))):
            idx = rng.integers(0, n, n)
            yb, sb = y[idx], s[idx]
            if yb.sum() == 0 or yb.sum() == yb.size:
                continue  # degenerate resample — skip
            aucs.append(roc_auc_score(yb, sb))
        if len(aucs) < 2:
            return None
        lo = float(np.percentile(aucs, 100 * (alpha / 2)))
        hi = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
        return (lo, hi)
    except Exception as exc:  # noqa: BLE001
        _log.debug("bootstrap_auc_ci failed: %r", exc)
        return None


def benjamini_hochberg(pvalues: Sequence[Optional[float]]) -> list[Optional[float]]:
    """Benjamini-Hochberg FDR q-values, preserving input order and Nones.

    Multiple-testing correction across the honest-test battery: the engine reports
    several AUC permutation p-values, so raw p-values overstate significance.
    ``None`` entries (undefined tests) pass through as ``None`` and are excluded
    from the correction's ``m``. Never raises."""
    try:
        import numpy as np
        idx = [i for i, p in enumerate(pvalues) if p is not None]
        if not idx:
            return [None] * len(pvalues)
        p = np.array([float(pvalues[i]) for i in idx])
        m = p.size
        order = np.argsort(p)
        ranked = p[order]
        q = ranked * m / (np.arange(m) + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]  # enforce monotonicity
        q = np.clip(q, 0.0, 1.0)
        out_by_pos = np.empty(m)
        out_by_pos[order] = q
        result: list[Optional[float]] = [None] * len(pvalues)
        for j, i in enumerate(idx):
            result[i] = float(out_by_pos[j])
        return result
    except Exception as exc:  # noqa: BLE001
        _log.debug("benjamini_hochberg failed: %r", exc)
        return [None] * len(pvalues)


def degree_matched_null_auc(ranked_genes: Sequence[str], scores: Sequence[float],
                            gold_symbols: frozenset[str], degrees: dict[str, float],
                            *, n_draws: int = 1000, seed: int = 0
                            ) -> Optional[dict]:
    """Specificity control for a network-centrality ranking.

    Hub genes score high on any centrality metric by construction, so a raw AUC
    against a gold set can be inflated by the gold genes simply being high-degree.
    This draws ``n_draws`` random "gold sets" MATCHED to the real gold set's STRING
    degree distribution (sampling within degree strata), recomputes the AUC for
    each, and reports the null mean/CI and an empirical p-value (fraction of
    degree-matched nulls whose AUC >= observed). If the observed AUC sits well
    above this degree-matched null, the ranking carries signal beyond "hubs win".
    Returns None if inputs are degenerate. Never raises."""
    try:
        import numpy as np
        gold = {g.upper() for g in gold_symbols}
        genes = [g.upper() for g in ranked_genes]
        universe = [g for g in genes]
        gold_in = [g for g in universe if g in gold]
        k = len(gold_in)
        if k < 2 or len(universe) <= k:
            return None
        observed = roc_auc(ranked_genes, scores, gold_symbols)
        if observed is None:
            return None
        deg = {g.upper(): float(degrees.get(g, degrees.get(g.upper(), 0.0)))
               for g in universe}
        # Degree strata via quantile bins so matched draws share the degree profile.
        vals = np.array([deg[g] for g in universe])
        nbins = max(1, min(10, len(set(vals.tolist()))))
        edges = np.quantile(vals, np.linspace(0, 1, nbins + 1))
        edges[-1] += 1e-9
        bin_of = {g: int(np.clip(np.searchsorted(edges, deg[g], side="right") - 1,
                                 0, nbins - 1)) for g in universe}
        strata: dict[int, list[str]] = {}
        for g in universe:
            strata.setdefault(bin_of[g], []).append(g)
        gold_bins = [bin_of[g] for g in gold_in]
        rng = np.random.default_rng(seed)
        null_aucs = []
        score_by_gene = {g.upper(): float(s) for g, s in zip(ranked_genes, scores)}
        for _ in range(max(1, int(n_draws))):
            drawn: set[str] = set()
            ok = True
            for b in gold_bins:
                pool = [g for g in strata.get(b, []) if g not in drawn]
                if not pool:
                    ok = False
                    break
                drawn.add(pool[int(rng.integers(0, len(pool)))])
            if not ok or len(drawn) < 2:
                continue
            y = np.array([1 if g in drawn else 0 for g in universe])
            s = np.array([score_by_gene[g] for g in universe])
            from sklearn.metrics import roc_auc_score
            null_aucs.append(roc_auc_score(y, s))
        if len(null_aucs) < 2:
            return None
        null_aucs = np.array(null_aucs)
        ge = int((null_aucs >= observed).sum())
        return {
            "observed_auc": float(observed),
            "null_mean": float(null_aucs.mean()),
            "null_ci": [float(np.percentile(null_aucs, 2.5)),
                        float(np.percentile(null_aucs, 97.5))],
            "n_draws_effective": int(null_aucs.size),
            "empirical_p": (ge + 1) / (null_aucs.size + 1),
        }
    except Exception as exc:  # noqa: BLE001
        _log.debug("degree_matched_null_auc failed: %r", exc)
        return None


# ---------------------------------------------------------------------------
# Ranking adapters — call the shipped clients, extract (gene, score) pairs.
# ---------------------------------------------------------------------------


@dataclass
class RankingUniverse:
    """A ranked gene universe with parallel scores and a provenance stamp."""
    genes: list[str]
    scores: list[float]
    source: str            # "offline_snapshot" | "live"

    @property
    def background_size(self) -> int:
        return len(self.genes)


def pi4ad_universe(*, prefer_offline: bool = True,
                   top_n: int = 100_000) -> RankingUniverse:
    """PI4AD whole-disease ranking as a scored universe (priority_score, desc).

    ``prefer_offline=True`` (default) is the deterministic bundled snapshot;
    ``prefer_offline=False`` best-effort fetches the full ~14.7k-gene HTTP table.
    Never raises — degrades to an empty universe on total failure."""
    try:
        from ..integrations.pi4ad import rank_ad_targets
        recs = rank_ad_targets(top_n=top_n, prefer_offline=prefer_offline)
    except Exception as exc:  # noqa: BLE001
        _log.debug("PI4AD universe fetch failed: %r", exc)
        return RankingUniverse(genes=[], scores=[], source="offline_snapshot")
    recs = sorted(recs, key=lambda r: r.priority_score, reverse=True)
    source = recs[0].source if recs else "offline_snapshot"
    return RankingUniverse(genes=[r.gene for r in recs],
                           scores=[float(r.priority_score) for r in recs],
                           source=source)


def _ot_score(assoc, evidence: str) -> float:
    """Score one Open Targets association under an evidence-partition mode.

    ``overall`` = the aggregate association_score (OPTIMISTIC — includes the
    circular datatypes). ``non_genetic`` / ``non_clinical`` = mean of the
    per-datatype scores with the gold-defining datatypes removed (Honesty
    Guard 1). Missing/empty breakdown -> 0.0."""
    if evidence == "overall":
        return float(assoc.association_score)
    excluded = GENETIC_DATATYPES if evidence == "non_genetic" else CLINICAL_DATATYPES
    kept = [float(v) for k, v in (assoc.datatype_scores or {}).items()
            if k not in excluded]
    if not kept:
        return 0.0
    return sum(kept) / len(kept)


def opentargets_universe(*, prefer_offline: bool = True, evidence: str = "overall",
                         top_n: int = 100_000) -> RankingUniverse:
    """Open Targets AD disease->targets ranking as a scored universe.

    ``evidence`` selects the scoring: ``overall`` (naive/optimistic),
    ``non_genetic`` (held-out for the GWAS gold set), or ``non_clinical``
    (held-out for the drug gold set). ``prefer_offline=True`` is the deterministic
    bundled snapshot; ``False`` best-effort pages the live GraphQL API. Never
    raises — degrades to an empty universe on total failure."""
    try:
        from ..integrations.opentargets import OpenTargetsClient
        client = OpenTargetsClient(prefer_offline=prefer_offline)
        targets = client.disease_targets(top_n=top_n)
    except Exception as exc:  # noqa: BLE001
        _log.debug("Open Targets universe fetch failed: %r", exc)
        return RankingUniverse(genes=[], scores=[], source="offline_snapshot")
    scored = [(t.gene, _ot_score(t, evidence), t.source) for t in targets
              if t.gene]
    scored.sort(key=lambda x: x[1], reverse=True)
    source = scored[0][2] if scored else "offline_snapshot"
    return RankingUniverse(genes=[g for g, _, _ in scored],
                           scores=[s for _, s, _ in scored],
                           source=source)


# ---------------------------------------------------------------------------
# The report + the validate() core
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """A serializable, provenance-honest outcome-validation record.

    Read-only side artifact. ``optimistic=True`` flags a naive/CIRCULAR score
    (the ranking evidence overlaps the gold-set definition); for held-out reports
    ``naive_roc_auc`` carries the optimistic number alongside for comparison.
    ``source`` + ``background_size`` stamp exactly what was evaluated so a tiny,
    curation-biased offline background is never read as a rigorous verdict."""
    ranking_source: str                    # "pi4ad" | "opentargets"
    evidence_mode: str                     # "overall" | "non_genetic" | ...
    gold_set_name: str
    gold_citations: dict[str, str]
    n_gold: int                            # gold genes present in the universe
    background_size: int
    precision_at_k: dict[int, Optional[float]]
    recall_at_k: dict[int, Optional[float]]
    roc_auc: Optional[float]
    permutation_p: Optional[float]
    source: str                            # "offline_snapshot" | "live"
    optimistic: bool = False
    naive_roc_auc: Optional[float] = None
    caveat: str = ""
    #: Rigor-hardening additive fields (default None so existing serialization and
    #: every current caller are unchanged). Populated when ``validate(with_ci=True)``
    #: or by the discovery-rigor script.
    roc_auc_ci: Optional[tuple[float, float]] = None  # 95% bootstrap CI
    q_value: Optional[float] = None                   # BH-FDR across the battery
    null_auc_mean: Optional[float] = None             # degree-matched null mean

    def to_dict(self) -> dict:
        return {
            "ranking_source": self.ranking_source,
            "evidence_mode": self.evidence_mode,
            "gold_set_name": self.gold_set_name,
            "gold_citations": dict(self.gold_citations),
            "n_gold": self.n_gold,
            "background_size": self.background_size,
            "precision_at_k": {int(k): v for k, v in self.precision_at_k.items()},
            "recall_at_k": {int(k): v for k, v in self.recall_at_k.items()},
            "roc_auc": self.roc_auc,
            "roc_auc_ci": list(self.roc_auc_ci) if self.roc_auc_ci else None,
            "permutation_p": self.permutation_p,
            "q_value": self.q_value,
            "null_auc_mean": self.null_auc_mean,
            "source": self.source,
            "optimistic": self.optimistic,
            "naive_roc_auc": self.naive_roc_auc,
            "caveat": self.caveat,
        }


_OFFLINE_CAVEAT = (
    "source=offline_snapshot: the bundled OT/PI4AD snapshots were curated to "
    "include canonical AD genes and the background is small (N={bg}), so "
    "precision@k and AUC here are enriched-by-construction and NOT an unbiased "
    "estimate. Run with prefer_offline=False for a rigorous full-universe "
    "(~14.7k-gene PI4AD HTTP table / paged Open Targets GraphQL) evaluation."
)
_CIRCULAR_CAVEAT = (
    " OPTIMISTIC/CIRCULAR: the overall association_score is built from the "
    "gold-defining datatypes; the held-out-evidence AUC is the honest number."
)


def validate(universe: RankingUniverse, gold: GoldSet, *, ranking_source: str,
             evidence_mode: str, k_values: Sequence[int] = (5, 10, 20),
             n_perm: int = 1000, seed: int = 0, optimistic: bool = False,
             naive_roc_auc: Optional[float] = None,
             extra_caveat: str = "", with_ci: bool = False,
             n_boot: int = 2000) -> ValidationReport:
    """Score a ranked universe against a gold set into a ValidationReport.

    Pure and exception-safe: every metric independently degrades to None rather
    than raising, so the report is always returned and always provenance-stamped.
    ``with_ci=True`` also computes the 95% bootstrap AUC CI (default off, so every
    existing caller's cost and output are unchanged).
    """
    symbols = gold.symbols
    uni_upper = {g.upper() for g in universe.genes}
    n_gold = len(symbols & uni_upper)

    prec = {int(k): precision_at_k(universe.genes, symbols, int(k))
            for k in k_values}
    rec = {int(k): recall_at_k(universe.genes, symbols, int(k))
           for k in k_values}
    auc = roc_auc(universe.genes, universe.scores, symbols)
    pval = permutation_pvalue(universe.genes, universe.scores, symbols,
                              n_perm=n_perm, seed=seed)
    auc_ci = bootstrap_auc_ci(universe.genes, universe.scores, symbols,
                              n_boot=n_boot, seed=seed) if with_ci else None

    caveat = _OFFLINE_CAVEAT.format(bg=universe.background_size) \
        if universe.source == "offline_snapshot" else \
        f"source=live: full-universe evaluation (N={universe.background_size})."
    if optimistic:
        caveat += _CIRCULAR_CAVEAT
    if extra_caveat:
        caveat += " " + extra_caveat

    return ValidationReport(
        ranking_source=ranking_source,
        evidence_mode=evidence_mode,
        gold_set_name=gold.name,
        gold_citations=gold.citations(),
        n_gold=n_gold,
        background_size=universe.background_size,
        precision_at_k=prec,
        recall_at_k=rec,
        roc_auc=auc,
        permutation_p=pval,
        source=universe.source,
        optimistic=optimistic,
        naive_roc_auc=naive_roc_auc,
        caveat=caveat,
        roc_auc_ci=auc_ci,
    )


# ---------------------------------------------------------------------------
# Top-level validators — each never raises offline, each provenance-stamped.
# ---------------------------------------------------------------------------


def validate_pi4ad_gwas(*, prefer_offline: bool = True, n_perm: int = 1000,
                        seed: int = 0) -> ValidationReport:
    """PI4AD priority ranking vs the GWAS gold set.

    NON-circular by construction: PI4AD priority is a network-propagation score,
    not built from GWAS labels. (Offline the canonical GWAS genes were appended
    at deep ranks in the snapshot, so the honest offline AUC is LOW — a rigorous
    read needs the live table.)"""
    uni = pi4ad_universe(prefer_offline=prefer_offline)
    return validate(uni, GWAS_GOLD, ranking_source="pi4ad",
                    evidence_mode="priority", n_perm=n_perm, seed=seed)


def validate_opentargets_gwas(*, prefer_offline: bool = True, held_out: bool = True,
                              n_perm: int = 1000, seed: int = 0) -> ValidationReport:
    """Open Targets ranking vs the GWAS gold set, held-out-genetics by default.

    ``held_out=True`` predicts the GWAS gold set from NON-genetic evidence only
    (Honesty Guard 1) and also carries the naive AUC for comparison;
    ``held_out=False`` reports only the naive/optimistic overall score."""
    naive_uni = opentargets_universe(prefer_offline=prefer_offline,
                                     evidence="overall")
    naive_auc = roc_auc(naive_uni.genes, naive_uni.scores, GWAS_GOLD.symbols)
    if not held_out:
        return validate(naive_uni, GWAS_GOLD, ranking_source="opentargets",
                        evidence_mode="overall", n_perm=n_perm, seed=seed,
                        optimistic=True)
    held_uni = opentargets_universe(prefer_offline=prefer_offline,
                                    evidence="non_genetic")
    return validate(held_uni, GWAS_GOLD, ranking_source="opentargets",
                    evidence_mode="non_genetic", n_perm=n_perm, seed=seed,
                    naive_roc_auc=naive_auc,
                    extra_caveat="Held-out: GWAS gold predicted from non-genetic "
                                 "datatypes only; naive_roc_auc is the circular "
                                 "overall-score number.")


def validate_opentargets_drugs(*, prefer_offline: bool = True, held_out: bool = True,
                               n_perm: int = 1000, seed: int = 0) -> ValidationReport:
    """Open Targets ranking vs the FDA-approved-drug-target gold set.

    ``held_out=True`` predicts the drug gold set from NON-clinical evidence only
    (Honesty Guard 1) and carries the naive AUC for comparison; ``held_out=False``
    reports only the naive/optimistic overall score (which is CIRCULAR — the
    clinical/known-drug datatype defines the gold set)."""
    naive_uni = opentargets_universe(prefer_offline=prefer_offline,
                                     evidence="overall")
    naive_auc = roc_auc(naive_uni.genes, naive_uni.scores, DRUG_GOLD.symbols)
    if not held_out:
        return validate(naive_uni, DRUG_GOLD, ranking_source="opentargets",
                        evidence_mode="overall", n_perm=n_perm, seed=seed,
                        optimistic=True)
    held_uni = opentargets_universe(prefer_offline=prefer_offline,
                                    evidence="non_clinical")
    return validate(held_uni, DRUG_GOLD, ranking_source="opentargets",
                    evidence_mode="non_clinical", n_perm=n_perm, seed=seed,
                    naive_roc_auc=naive_auc,
                    extra_caveat="Held-out: drug gold predicted from non-clinical "
                                 "datatypes only; naive_roc_auc is the circular "
                                 "overall-score number.")


def run_default_validation(*, prefer_offline: bool = True, n_perm: int = 1000,
                           seed: int = 0) -> list[dict]:
    """Run the full offline-safe validation suite, newest honesty first.

    Returns serializable report dicts: PI4AD-vs-GWAS (non-circular), Open Targets
    -vs-GWAS (held-out non-genetic), and Open Targets-vs-drugs (held-out
    non-clinical). Never raises."""
    reports = [
        validate_pi4ad_gwas(prefer_offline=prefer_offline, n_perm=n_perm, seed=seed),
        validate_opentargets_gwas(prefer_offline=prefer_offline, held_out=True,
                                  n_perm=n_perm, seed=seed),
        validate_opentargets_drugs(prefer_offline=prefer_offline, held_out=True,
                                   n_perm=n_perm, seed=seed),
    ]
    return [r.to_dict() for r in reports]


__all__ = [
    "GoldGene", "GoldSet", "GWAS_GOLD", "DRUG_GOLD", "KNOWN_2019", "NOVEL_2022", "GOLD_SETS",
    "GENETIC_DATATYPES", "CLINICAL_DATATYPES",
    "precision_at_k", "recall_at_k", "roc_auc", "permutation_pvalue",
    "RankingUniverse", "pi4ad_universe", "opentargets_universe",
    "ValidationReport", "validate",
    "validate_pi4ad_gwas", "validate_opentargets_gwas",
    "validate_opentargets_drugs", "run_default_validation",
]
