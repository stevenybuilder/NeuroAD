# Submission summary — NeuroAD Discovery Engine

**Track:** Builder ("Build Beyond the Bench").

For a translational Alzheimer's researcher with brain MRI and partial metadata,
one question recurs: *is this signal worth a quarter of my time, or is it scanner
noise, generic aging, or atrophy in disguise?* NeuroAD answers it in one command.
It points a linear head at a frozen NeuroJEPA embedding table — at the
outcome to find signal, at the scanner label to expose leakage, at a plasma
biomarker to anchor it — and runs a five-test gauntlet (age/sex,
star site/scanner leakage, star brain-age, biomarker anchor, replication) with
fold-honest CV and permutation nulls. It issues a
hedged verdict and refuses to promote biology until a finding
clears leakage plus an independent anchor: a real *measured* ADNI p-tau217
correlation, or leakage-clean held-out replication where cohorts lack plasma.

Every verdict is deterministic Python — we deliberately kept Claude *out* of the
referee so it can never fabricate a promote or kill. Instead Claude is the
**orchestrator**: a live tool-runner that sequences the engine and halts the
molecular chain when a finding is killed, plus a live, source-grounded
"Ask Claude" rail. It boots on real OASIS and ADNI data, cites the batch-effect
prior art openly, and ships the runnable referee nobody else did.
