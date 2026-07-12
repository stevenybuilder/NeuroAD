# Dana Cho design principles — decluttering the NeuroAD ZUI

> Distilled operating checklist derived from Dana Cho's public work (Harvard GSD "Humanizing
> Technology"; RGD "Healthy Design"; danascho.medium.com; Stanford Center for Digital Health).
> Her writing is reflective essays/talks, not a published listicle — this is an honest
> distillation into actionable rules, not a checklist she published verbatim. Through-line:
> **humanize complex technology through simplicity — which matters most in complex,
> high-stakes domains like healthcare.**

## The five principles (and the move each licenses)
1. **Simplicity in complex domains** → **Surface what IS there; cut what's absent.** Zeroed
   rows, N/A stacks, "0%" bars are noise — remove, don't faithfully render.
2. **Reduce cognitive load ("how does it make you feel?")** → **Fewer things competing at
   once.** Dim decoration/chrome; fold multi-line stats to one line; collapse verbose legends.
3. **Lead with the essential; tuck the rest** → **Progressive disclosure.** Primary narrative
   always visible; secondary/technical detail one click away (never deleted).
4. **Trust & information integrity** → **One thing must never recede: provenance/trust.** The
   REAL/SYNTHETIC badge, live/offline Claude state, verdict, and honesty caveat stay prominent
   even as everything else quiets.
5. **Humanize the technical** → **Plain language in the chrome;** numbers precise, framing human.

## Operating checklist (apply in order)
1. **Screenshot first.** Count co-equal elements fighting for attention. ~8+ equal-weight
   panels IS the problem, not any single panel.
2. **Delete noise before styling.** Remove zero/N-A/empty rows and duplicate numbers first —
   biggest calm-per-effort win.
3. **Establish ONE focal point.** Decide the single thing the eye lands on (the verdict / the
   candidate / the active node). Everything else reads quieter.
4. **Fold & merge.** Two-line stats → one line. Collapse dev-facing chrome into tooltips.
5. **Progressive disclosure for secondary panels.** Default-collapse the most technical
   (reviewer notes, confound leaderboards); default-open the payoff / showcase.
6. **Protect the trust signals.** Never collapse or dim provenance, the headline verdict, or
   the honesty caveat.
7. **Verify in the browser, not the diff.** Reload, screenshot, read the console for *page*
   errors; inspect live DOM state rather than trusting pixels.

## Applied to the NeuroAD ZUI (concrete)
- **One focal point:** semantic zoom already frames one node — enforce it; never show two
  competing story blocks at once.
- **Delete noise:** don't render null/`0` protein scores as "0" — show "unranked" or omit;
  drop N/A gauntlet rows from a story block rather than showing empty bars.
- **Progressive disclosure:** candidate story block leads with the ranked target + verdict;
  tuck the full confound-check rows, reviewer/courtroom text, and provenance detail behind
  an expander that opens on deeper zoom / click — nothing deleted.
- **Protect trust (the product's whole thesis):** the Claude live/offline badge, the
  SYNTHETIC-vs-REAL provenance, and the pinned candidate caveat NEVER collapse or dim.
- **Humanize:** keep the plain-language gate questions ("Disease signal, or just which
  machine acquired the scan?") over jargon; mono only for the numbers.

## Pitfalls
- Progressive disclosure ≠ deletion — reduce prominence, never capability.
- Don't collapse the showcase: the Claude/candidate panels are what's being judged — keep visible.
- Watch encoding: ensure `<meta charset="utf-8">` so `·` / `—` / `★` don't mojibake.
- One pass at a time, screenshotted, so subjective changes are reviewable.
