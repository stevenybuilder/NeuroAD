# Temporal (prospective) validation — anticipating 2022-novel AD genes

_Generated 2026-07-12T03:01:10.815726+00:00; prefer_offline=False, add_nodes=1000, ot_top_n=2000, n_perm=2000._

Seeds: 24 Kunkle-2019 genes. Test set: 41 Bellenguez-2022-NEW genes.

## Verdict

No CLEAN (non-circular) prospective signal clears significance with the complete, source-verified 41-gene NOVEL_2022 set and the widened universe — network 13/41 novel in-universe, AUC=0.587 (p=0.142); OT non-genetic 27/41 novel in-universe, AUC=0.506 (p=0.463). This is now a PROPERLY-POWERED honest null (the network test trends above chance but does not reach p<0.05), not a coverage artifact: the earlier small-sample OT estimate (AUC~0.60 at n=4) regressed to chance once 27 of 41 novel genes were scored. The circular comparators (which leak the 2022 genetics) only reach ~0.63–0.66, confirming most retrospective 'signal' is genetic circularity. Honest framing: the engine is a validated hypothesis engine for KNOWN biology, not a demonstrated novel-target anticipator.

| Test (orthogonal evidence) | universe | novel-in-universe | AUC | perm p |
|---|---|---|---|---|
| STRING-RWR (seeded on KNOWN_2019) | 1000 | 13 | 0.5868599485620762 | 0.14192903548225888 |
| Open Targets non-genetic | 2000 | 27 | 0.5064481612885059 | 0.46326836581709147 |

**Circular ceilings** (leak 2022 genetics — optimistic, NOT clean): OT overall AUC=0.6615043832479209 (p=0.0009995002498750624); PI4AD full-table AUC=0.6308930035528834 (p=0.0014992503748125937, novel-in-universe 40/41 — full coverage but residually circular).

> Honest caveats: Open Targets is a CURRENT snapshot, so the non-genetic universe only approximates the pre-2022 evidence state (slightly optimistic). The network test is cleaner (PPI is GWAS-independent). NOVEL_2022 is now the COMPLETE, source-verified new-loci set (41 genes = the nearest gene at every Bellenguez-2022 Table-2 new locus, IGH cluster excluded), so a residual null reflects the evidence, not gold-set undercounting.
