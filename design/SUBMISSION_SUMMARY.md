# NeuroAD Discovery Engine — Submission Summary

<!-- Written summary for the CV platform. Word count stated at the bottom; every
     number matches app/demo_data.json / reports/*.json and scratch/CITATIONS_VERIFIED.md. -->

NeuroAD Discovery Engine is a scientific referee for Alzheimer's brain-MRI machine
learning. It stress-tests a candidate signal through an adversarial gauntlet —
age/sex, scanner/site leakage, brain-age, a plasma-biomarker anchor, replication —
and refuses plausible-looking claims, including its own.

The real finding anchors it: on 3,984 healthy OpenBHB brains (62 sites, no disease),
the structural embedding predicts which scanner took the scan at AUC 0.89, and the frozen
Neuro-JEPA foundation-model embedding leaks scanner field strength at AUC 0.96 (PCA-10;
raw 768-d 0.998, flagged p-greater-than-n). `neuroad reproduce-finding` prints that same
number — AUC 0.958, 95% CI [0.91, 1.00], permutation p=0.001 — from a checked-in fixture.
That is the batch effect the referee exists to catch: an OASIS-3 imaging lead drops their
own frozen-embedding table into the same PCA-10 contract feeder and audits it for scanner
leakage in one command.

Claude is the adjudicator, not a chatbot: a three-voice courtroom
(prosecution/defense/judge) that then critiques its own verdict through strict
structured tool-use, badged live-vs-offline on camera. The demo opens by refusing our
own synthetic classifier (naive AUC 0.87, scanner 0.95, margin -0.08, REFUSED 40/100) —
a refusal that holds across 20 seeds — and stamps every synthetic number SYNTHETIC
HARNESS. Rigor is the product.

<!-- WORD COUNT: 200 words (body only, excluding the heading and HTML comments;
     whitespace-token count via Python str.split on the comment-stripped body). -->
