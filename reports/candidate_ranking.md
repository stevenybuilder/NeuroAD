# Narrowed candidate-protein ranking (composite, multi-signal)

_Generated 2026-07-12T04:25:28.261016+00:00; prefer_offline=False._

Composite = weighted, min-max-normalized fusion of up to five LIVE signals (weights: PI4AD 0.30, OpenTargets-heldout 0.35, STRING-RWR 0.20, AlphaFold-pLDDT 0.15, Boltz-2 complex 0.15). The Boltz-2 signal is OPTIONAL and PRESENT only when a real committed GPU snapshot exists; otherwise it is absent and the composite is the prior 4-signal fusion. Decision-support only — see `target_prioritization_validation.md` for the honesty caveats.

## Overall shortlist (all mechanisms pooled)

| Rank | Gene | Composite | PI4AD (rank) | OT-heldout | STRING deg | pLDDT | Boltz |
|---|---|---|---|---|---|---|---|
| 1 | APP | 0.7827 | 8.597 (r18) | 0.7645 | 66 | 67.4 | 0.5874 |
| 2 | MAPK1 | 0.7746 | 7.966 (r64) | 0.6807 | 9 | 90.4 | None |
| 3 | ESR1 | 0.6888 | 7.992 (r61) | 0.7511 | 10 | 66.4 | None |
| 4 | TREM2 | 0.5614 | 8.135 (r49) | 0.6411 | 32 | 76.8 | None |
| 5 | BIN1 | 0.5213 | 6.754 (r287) | 0.9805 | 43 | 66.7 | None |
| 6 | HRAS | 0.5126 | 8.19 (r45) | 0.3028 | 5 | 91.9 | None |
| 7 | BACE1 | 0.4629 | 5.688 (r781) | 0.6975 | 35 | 87.5 | 0.5874 |
| 8 | APOE | 0.4369 | 7.151 (r185) | 0.4233 | 66 | 75.5 | 0.508 |
| 9 | MAPT | 0.4077 | 7.304 (r151) | 0.5956 | 41 | 49.2 | 0.3729 |
| 10 | CLU | 0.3677 | 6.738 (r292) | 0.506 | 51 | 77.3 | None |
| 11 | PSEN1 | 0.3526 | 6.211 (r492) | 0.588 | 44 | 72.1 | None |

## amyloid_cascade — top candidates

**Lead: APP** (composite 0.7827, 5/5 signals). Runners-up: ESR1(0.6888), BACE1(0.4629), APOE(0.4369).

## glial — top candidates

**Lead: MAPK1** (composite 0.7746, 4/5 signals). Runners-up: TREM2(0.5614), HRAS(0.5126), APOE(0.4369).

## vascular — top candidates

**Lead: APP** (composite 0.7827, 5/5 signals). Runners-up: BIN1(0.5213), APOE(0.4369), CLU(0.3677).

