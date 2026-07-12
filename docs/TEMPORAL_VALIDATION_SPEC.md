# Temporal Validation Spec — proving *novel* target discovery

_Status: DRAFT v1 (2026-07-11). Reuses `harness/validation.py`. ~Zero GPU; a few
network calls. The one honest way to show the discovery engine finds targets it
did NOT already know._

## 1. The question this answers

The current validation asks: "does the engine rank the ~15 *known* GWAS AD genes
highly?" That can only ever be retrospective. Temporal validation asks the
**prospective** question: **"using only evidence that existed BEFORE a cutoff, does
the engine rank the AD-risk genes that were discovered AFTER the cutoff above
background?"** If yes, the engine *anticipated* real discoveries — genuine de-novo
target-finding, not curve-fitting to known biology.

## 2. The design (a natural experiment already in the literature)

AD genetics has clean, dated milestones we can use as a pre/post split:

| Era | GWAS | ~Genome-wide-significant loci |
|---|---|---|
| Pre-cutoff (known) | Kunkle et al. 2019 (Nat Genet) | ~25 loci |
| Post-cutoff (novel) | **Bellenguez et al. 2022** (Nat Genet) | **75 loci (+~42 new)** |

- **`KNOWN_2019`** = genes at genome-wide significance in Kunkle 2019.
- **`NOVEL_2022`** = Bellenguez-2022 genome-wide-significant genes **NOT** in
  `KNOWN_2019` (the ~42 newly-discovered loci). *This is the held-out "future"
  discovery set.*
- The engine gets **only pre-2022, non-genetic evidence** and must rank `NOVEL_2022`
  above the genomic background.

## 3. Why it's honest (the anti-circularity discipline)

The engine must NOT use the very GWAS we're predicting. So the ranking evidence is
restricted to signals that are **orthogonal to and predate** the 2022 GWAS:

- **STRING-RWR network centrality** — protein-protein interactions (largely
  GWAS-independent); seed from `KNOWN_2019` only, then ask: do the `NOVEL_2022`
  genes surface as network hubs? (We already saw this qualitatively — CD2AP,
  INPP5D, MS4A4A lit up from GWAS seeds.)
- **Open Targets NON-genetic, publication-date-filtered evidence** — expression,
  pathway, animal-model, literature datatypes, filtered to evidence published
  **before 2021** (Open Targets evidence carries publication years). Excludes the
  `genetic_association` datatype entirely (Honesty Guard 1, already implemented).
- **PI4AD priority is EXCLUDED** from the primary test (it ingests genetics → would
  leak the answer); report it only as a labeled "optimistic/circular" comparator.

If `NOVEL_2022` genes rank highly on network + pre-2021 non-genetic evidence alone,
the engine predicted them before the GWAS confirmed them.

## 4. Metrics (reuse the existing harness)

Feed each ranking into `validation.validate(universe, gold=NOVEL_2022, ...)`:
- **ROC-AUC** of `NOVEL_2022` membership vs the ranking score, over the full gene
  universe, with the label-shuffle **permutation p-value** already implemented.
- **precision@k / recall@k** (k = 20, 42) — of the top-k engine picks, how many are
  real future discoveries.
- **Enrichment ratio + hypergeometric p** — are `NOVEL_2022` genes over-represented
  in the top decile of the ranking?
- Report the network-only and non-genetic-only numbers as the honest primary; the
  PI4AD/overall-OT numbers as the circular ceiling.

## 5. Implementation (small, mostly additive)

1. Add to `harness/validation.py`: `KNOWN_2019` and `NOVEL_2022` `GoldSet`s
   (cited, in-code, same discipline as `GWAS_GOLD`).
2. Add a `date_before` filter to `opentargets_universe` (drop evidence whose
   `publicationYear >= cutoff`; the GraphQL evidence query exposes it).
3. New `scripts/run_temporal_validation.py`: builds the pre-2021 non-genetic OT
   universe + the `KNOWN_2019`-seeded STRING-RWR universe, scores both against
   `NOVEL_2022`, writes `reports/temporal_validation.json` (+ .md).
4. Tests: offline unit test that `NOVEL_2022 = Bellenguez \ Kunkle` set-difference
   is correct and the harness runs.

## 6. Expected outcomes + honest failure modes

- **Success:** network/non-genetic AUC materially > 0.5 with p < 0.05 →
  "the engine prospectively ranks future-discovered AD genes." Strong, publishable.
- **Null:** AUC ≈ 0.5 → the engine's novel-target signal is not better than chance
  once you remove genetic circularity. Also honest, and important to know.
- **Caveat to disclose:** Open Targets is a *current* snapshot; the publication-year
  filter approximates but does not perfectly reconstruct the 2021 evidence state
  (a gene's OT *record* exists now even if we drop its recent evidence). Stamp this
  limitation; it makes the test slightly optimistic, not invalid.

## 7. Cost

Negligible — a few Open Targets GraphQL calls + STRING fetches, pure-CPU scoring.
No GPU, no paid compute. Runs in minutes.

## 8. Why this is the right "ambitious but bounded" next step

It converts the discovery half's story from *"rigorously-filtered hypothesis
engine"* to *"demonstrated ability to anticipate real AD-target discoveries,"*
using data we can fetch today, with the anti-circularity discipline the validation
harness already enforces — and it generalizes the engine from the 15 known genes to
the full genome without combinatorial blow-up.
