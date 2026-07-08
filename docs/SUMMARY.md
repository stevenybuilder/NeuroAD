# Submission summary — NeuroAD Discovery Engine

**Track:** Builder ("Build Beyond the Bench").

For a translational Alzheimer's researcher who has brain MRI and partial
metadata, one question recurs: *is this signal worth a quarter of my time, or is
it scanner noise, generic aging, or atrophy in disguise?* NeuroAD Discovery Engine answers
it in one command. It points a single linear head at a frozen embedding table —
at the outcome to find the signal, at the scanner label to expose leakage, at a
plasma biomarker to anchor it — and runs a five-test adversarial gauntlet
(age/sex, star site/scanner leakage, star brain-age, biomarker gate,
replication). It issues a hedged fragile-to-robust verdict, gates every survivor
behind a p-tau217/GFAP correlation, and closes the loop: for what survives,
**Claude adjudicates** the likely mechanism and the single falsifiable next
experiment that would confirm or kill it. Claude is the adjudicator —
Prosecution, Defense, and Judge subagents plus a reviewer that argues against the
verdict — not just the coder. The demo runs fully offline on a synthetic
KILL/SURVIVOR harness and on real vendored OASIS data; it cites the batch-effect
prior art openly and ships the runnable referee nobody else did.
