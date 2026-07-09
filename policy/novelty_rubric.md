---
policy: novelty_rubric
layer: L3
kind: brief
consumers: [code, claude]
version: 1.0.0
schema_version: 1
last_verified: "2026-07-08"
mirrors:
  - src/neuroad/contract.py         # ClaimCard.novelty_class, ClaimCard.honesty_rung
  - src/neuroad/gauntlet.py         # the tests each rung is earned by
  - src/neuroad/leakage.py          # double_dissociation (confound-survivor rung)
# Machine-readable stamps. `novelty_class` and `honesty_rung` are the two fields
# the loader writes onto ClaimCard. Values below are the CLOSED vocabularies.
novelty_class:                      # ClaimCard.novelty_class  (contract.py: "known"|"adjacent"|"novel")
  values: [known, adjacent, novel]
  definitions:
    known: >-
      the contrast/mechanism is already published prior art (the engine is
      re-measuring, not discovering — e.g. embedding scanner leakage itself).
    adjacent: >-
      a known mechanism applied to a new sub-cohort, marker, or contrast; an
      incremental extension of established biology.
    novel: >-
      a pattern with no clean published precedent for THIS contrast; the highest
      novelty the engine will claim, and only ever paired with the honesty rung
      that says how far it has actually been defended.
honesty_rung:                       # ClaimCard.honesty_rung  (the calibrated-honesty stamp)
  ordered: true                     # rung N is only valid if all rungs < N are satisfied
  rungs:
    - id: 1
      key: raw_pattern
      label: "raw pattern"
      earned_by: "a naive effect exists (naive_effect AUC above chance) — no controls yet"
    - id: 2
      key: stable_cluster
      label: "stable cluster"
      earned_by: "the pattern reproduces as a coherent sub-cohort / cluster (detective), not a single-split fluke"
    - id: 3
      key: confound_survivor
      label: "confound-survivor"
      earned_by: "survives the STAR confound tests — site/scanner leakage (margin CI > 0) AND brain-age GAP AND age/sex adjustment"
    - id: 4
      key: severity_anchored
      label: "severity-anchored candidate"
      earned_by: "clears the biomarker-anchor HARD GATE (plasma p-tau217/GFAP CI lower bound >= 0.12) and is promoted (score >= 40)"
    - id: 5
      key: externally_replicated
      label: "externally-replicated"
      earned_by: "reproduces on a GENUINELY INDEPENDENT external cohort (a second, differently-acquired dataset) at held-out AUC >= 0.65. The gauntlet's internal held-out SPLIT of the same cohort does NOT count — that earns rung 4 (ready for replication). Rung 5 requires an explicit external-cohort run; a single-dataset investigation can never honestly reach it."
---

# Novelty Rubric — candidate taxonomy + the 5-rung honesty ladder

Two orthogonal stamps travel on every `ClaimCard`:

- **`novelty_class`** — *how new is the idea?* (`known` / `adjacent` / `novel`)
- **`honesty_rung`** — *how far has it actually been defended?* (rungs 1–5)

They are independent on purpose: a `novel` idea sitting at rung 1 (`raw
pattern`) is an interesting hunch, **not** a finding. The rung is what keeps the
novelty claim honest.

## Candidate taxonomy (`novelty_class`)

| Class | When to stamp it |
|-------|------------------|
| **known** | The contrast/mechanism is published prior art — the engine is re-measuring a known effect (e.g. that frozen embeddings leak scanner/site; we cite it, we do not claim it). |
| **adjacent** | A known mechanism extended to a new sub-cohort, marker, or contrast. Incremental. |
| **novel** | No clean published precedent for *this* contrast. The ceiling of what the engine claims — always paired with an honesty rung. |

## The 5-rung honesty ladder (`honesty_rung`)

The ladder is **cumulative and ordered**: a card may be stamped at rung *N* only
if every lower rung is already satisfied. Each rung corresponds to a concrete,
already-computed piece of evidence in the pipeline — no rung is awarded on vibes.

| Rung | Label | Earned by (evidence) |
|------|-------|----------------------|
| **1** | raw pattern | A naive effect exists — `naive_effect` AUC above chance. No controls applied. |
| **2** | stable cluster | The pattern reproduces as a coherent sub-cohort / cluster (Detective), not a single-split fluke. |
| **3** | confound-survivor | Survives the STAR confounds: site/scanner leakage (margin CI excludes zero) **and** brain-age GAP control **and** age/sex adjustment (each PASSED/WEAKENED, none FAILED). |
| **4** | severity-anchored candidate | Clears the biomarker-anchor **HARD GATE** — plasma p-tau217/GFAP CI lower bound `>= 0.12` — and is **promoted** (score `>= 40`). |
| **5** | externally-replicated | Reproduces on a **genuinely independent external cohort** (a second, differently-acquired dataset) at held-out AUC `>= 0.65`. The internal held-out **split** of the same cohort does **not** count — that caps at rung 4. A single-dataset investigation can never honestly reach rung 5. |

### How the rung is set

Walk the ladder upward and stop at the highest rung whose evidence is present;
that rung's `key` is written to `ClaimCard.honesty_rung`. A card that fails a
STAR confound never rises above rung 2 no matter how novel it looks; a card that
lacks biomarker coverage (anchor NA) is capped at rung 3.

### Honesty coupling with the verdict

- Rung ≥ 4 is required before the language may call something a "candidate."
- The `novel` class + a low rung must be spoken as *"a novel pattern, defended
  only to rung {N} ({label})"* — never as a discovery.
- The rung is the single most honest sentence the engine can offer a skeptic:
  it names exactly how far the evidence has been pushed and where it stopped.
