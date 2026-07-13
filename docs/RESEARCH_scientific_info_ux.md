# Research — Presenting Dense Science Cleanly (applied to NeuroAD)

*Deep-research output, 2026-07-13. Feed to the `ux-design-reviewer` agent (handoff P5).*

## Executive summary
The tension "show a lot of rigorous science" while staying "clean and minimalistic" is solved in principle:
**layer the same rigor across depth** — a calm macro reading grasped in one glance, with every number, interval,
and caveat retrievable on demand at the next layer. Shneiderman ("overview first, zoom and filter, details-on-demand"),
Tufte ("micro/macro readings"), Apple ("defer to content"), NN/g ("progressive disclosure") are four names for the
same discipline. The ZUI is architecturally correct for this. The harder half — where NeuroAD is most exposed — is
**honesty under compression**: when you shrink a rigorous result to fit a clean canvas, the failure mode is not
ugliness, it's *misleading confidence* (an AUROC that doesn't say what task it measured; a "SURVIVOR score = 100"
read as certainty rather than a renormalization artifact; the same protein showing two numbers in graph vs drawer).
Minimalism that hides a caveat is an overclaim.

## Prioritized principles

### Tier 1 — Honesty under compression (do first)
1. **Every number carries the task it measures — the label is part of the number.** A diagnosis-AUROC and a
   conversion/prognosis-AUROC are different quantities; reporting one as the other is the classic misleading move
   (TRIPOD exists to force this). → *NeuroAD:* never render bare "AUROC 0.86"; render "Diagnosis AUROC 0.86 (AD vs
   CN)" / "Conversion AUROC 0.86 (MCI→AD, 24mo)" — the task/contrast/horizon travels with the metric on the node
   label, drawer, and captions **identically**. (This is exactly the 0.64-vs-0.92 bug.)
2. **Show what a number is measured *on* — provenance is not optional** (Model Cards, Datasheets). → inline
   provenance chip: cohort, n, measured-vs-derived. The ADNI SURVIVOR score=100 renormalization must be visible
   AT the number ("score 100 — renormalized, brain-age unavailable"), not in a memory note.
3. **Uncertainty is a first-class visual; prefer honest formats over familiar ones.** Quantile dot plots beat error
   bars for *accuracy* even though clinicians rate error bars "most understandable" (familiarity ≠ accuracy). → CI as
   the primary visual at macro layer; numeric CI text in the drawer; the interval width visible before any digit.
4. **"Not validated" is a real, dignified state — design it.** Non-significant ≠ true null; underpowered-inconclusive
   ≠ ruled-out. → killed/gray nodes must distinguish *inconclusive (underpowered, n=…)* from *confidently excluded*.
5. **One entity, one set of numbers, everywhere — single source of truth.** A target's AUROC/rank/CI must be
   byte-identical on the node, Protein-data tab, Summary tab, and captions. The frozen `demo_data.json`/`translate`
   seam IS the semantic layer — forbid any surface from recomputing/re-rounding. Panel drift is the #1 credibility bug.

### Tier 2 — Managing density (core ZUI discipline)
6. **Overview → zoom/filter → details-on-demand; honor the ZUI literally.** Define what each zoom tier *shows and
   hides*. Far: shape/color + one headline metric. Mid: ranked targets + CI bars. Near/drawer: full numbers,
   provenance, structure. Never render drawer-density on the far canvas "to look impressive."
7. **One focal point per view — everything else defers** (Apple HIG). Exactly one hero (active node or selected
   target); dim/desaturate the rest (focus+context). The drawer supports, never co-stars.
8. **Semantic-only color; never spend color on decoration.** green=survivor, gray=killed, clay=target are the ONLY
   meanings. Confidence rides intensity/opacity, not a new hue. Emphasis = proximity/size/dimming, not a 4th accent.
9. **Hierarchy by proximity and common region, not borders.** Whitespace is the primary grouping tool; a border is a
   last resort.
10. **Tame the graph: lazy reveal, clustering, edge discipline** (avoid hairballs/starbursts/snowstorms). Expand
    targets on focus, cap branch depth, collapse killed confounds into a "3 confounds excluded" chip; uniform edge
    weight unless it encodes something real.

### Tier 3 — Numeric craft
11. **One number + its context beats many.** Prefer one headline per node (target's AUROC-with-CI) over a stat dump;
    word-sized sparkline/CI beside it; full table in the drawer.
12. **Progressive disclosure of *complexity*, not just data.** Default view answers "which survived, what to chase?"
    in plain language + one metric; a "show statistics" affordance reveals CIs/p-values/gauntlet/cohort tables.
13. **Consistency of grammar across surfaces — one bar primitive, one type scale, one motion rule.** Same visual
    means the same thing everywhere; one zoom motion curve.

## Anti-patterns (specific)
- The bare metric ("AUROC 0.86" with no task/cohort/interval). - Certainty laundering via rounding/renormalization
  (SURVIVOR-100 without the caveat at the number). - Panel drift (per-surface recompute). - Gray = "unsure"
  masquerading as gray = "ruled out." - Error bars as the honesty story. - Impressive-clutter on the far canvas. -
  Color inflation (new accent for "emphasis"). - Decorative uncertainty theater. - Two co-equal focal points. -
  Hover-only truth (honesty-critical qualifiers must ride with the number at every layer, not hide in a tooltip).

## Sources
Shneiderman visual-information-seeking mantra + semantic zoom; Tufte (data-ink, micro/macro, sparklines); NN/g
(progressive disclosure, managing visual complexity); Apple HIG (clarity/deference/depth); Gestalt (proximity/common
region/focal point); Cambridge Intelligence (graph UX — hairballs/starbursts); Nguyen et al. PMC10623599 (quantile
dot plots vs error bars); Mitchell et al. Model Cards (arXiv 1810.03993); Gebru et al. Datasheets; AUROC
diagnosis-vs-prognosis case-mix (arXiv 2409.01444); TRIPOD / Lancet Digital Health predictive-AI evaluation;
"absence of evidence is not evidence of absence" (PMC3178960); BI semantic-layer single-source-of-truth.

**Bottom line for NeuroAD:** the ZUI already gives the right *mechanism*. The differentiated win — and current risk —
is **honesty under compression**: wire task-label + provenance chip + CI-as-primary + single-source-of-truth
(principles 1–5) into the one seam that feeds every surface, so no matter how minimal the far view gets, the number
never means more than it earned.
