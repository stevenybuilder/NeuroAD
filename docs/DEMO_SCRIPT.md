# Demo script — 3 minutes, beat by beat

The video is a screen recording of the offline workbench (`app/index.html`), a
viewer over the real exported artifacts in `reports/*`. The timeline is
deterministic — every take is identical. All numbers below are what the engine
actually emits (see `app/demo_data.json`); nothing is hand-waved.

## 0:00–0:22 — Cold open on the KILL
Open on the workbench with the **KILL** cohort loaded. Claim card:
*"MCI converters show a distinct structural-MRI signature."* The naive probe AUC
lands at **0.82** — higher than most publishable imaging findings.
> VO: "Here's a finding that looks great — conversion AUC 0.82 off frozen brain-
> MRI embeddings. A lab is about to spend a quarter on it. Watch the referee take
> ten seconds."

## 0:22–0:55 — The star test fires
The gauntlet ticks: age/sex **fails** (the effect is gone after demographics),
then the ⭐ **site/scanner leakage** row runs. The *same head* pointed at the
scanner label scores **0.95** — better than it predicts the disease. The
leakage-margin gauge (outcome − scanner) swings to **−0.13** and turns red.
> VO: "Same probe, different label — the scanner. It reads the machine better
> than the disease. Negative leakage margin. This isn't biology, it's hardware."

On-screen, cite the prior art (arXiv:2604.14441 / 2606.09189): *"we didn't
discover embedding leakage — we built the tool that catches it."*

## 0:55–1:18 — Verdict: FRAGILE, refused
Brain-age **fails** (effect → chance) and the biomarker anchor **fails**
(p-tau217 correlation includes zero). Score settles at **15/100**, verdict
stamps **FRAGILE — not promoted**; the biology panel stays locked.
> VO: "Fragile. The molecular gate never opens. A quarter saved in ten seconds."

## 1:18–1:52 — Contrast: flip to SURVIVOR
Hit the **KILL / SURVIVOR** toggle — same claim, a cohort with real disease load,
and a **lower** naive AUC of **0.71**. The tempting one was the fake; the plainer
one is real. Age/sex and brain-age both **hold** (retains 86% / 91%). The ⭐ star
leakage row comes back **amber, not green**: margin only **+0.04** (outcome 0.71
vs scanner 0.68) — positive but thin.
> VO: "The weaker-looking finding is the real one. It survives demographics and
> brain-age — but the leakage margin is thin. On its own, still not enough."

## 1:52–2:16 — The biomarker rescue (the gate opens)
The ⭐ **biomarker anchor** row runs: probe score vs plasma **p-tau217 r = +0.40,
95% CI lower bound +0.20** on the complete subset — confidently non-zero. The
hard gate opens; the score climbs to **88/100**, verdict **STRONG CANDIDATE —
PROMOTED**.
> VO: "What rescues it is molecular. The probe tracks plasma p-tau217, and the
> confidence interval clears zero. A scanner can't fake a protein. Only now is it
> allowed to talk biology."

## 2:16–2:40 — Payoff: Claude adjudicates
The courtroom panel animates: **Prosecution** argues artifact (cites the +0.04
margin), **Defense** argues biology (cites the anchor + survived tests), the
**Judge** renders a hedged verdict. The bridge routes it to an **amyloid-cascade
(tau-driven)** mechanism via p-tau217 and names one **falsifiable next
experiment** (ADNI-3/EPAD, kill criterion attached). The **Reviewer (Claude)**
then argues *against* the verdict — proxy brain-age control, p-tau217
missingness, "strong candidate ≠ proven."
> VO: "Claude adjudicates — prosecution, defense, judge — then argues against its
> own verdict and hands back one experiment that would confirm or kill it."

## 2:40–3:00 — Real data + wow line
Flip the substrate badge to **REAL OASIS**. The AD-vs-CN signal looks strong
(AUC **0.82**) but **collapses under brain-age control** (retains ~1%) and has
**no plasma anchor** — so the referee refuses to promote it and routes it to
ADNI/EPAD for the anchor it can't run. Click **Export**: the claim card, evidence
ledger and reviewer report write to `reports/`.
> VO: "On real, vendored OASIS data it's just as skeptical — a strong-looking
> signal that's mostly brain age, refused pending a protein it can't measure.
> Offline, one command. **Imaging finds it. Proteins confirm it. It tells you
> what to do next.**"
