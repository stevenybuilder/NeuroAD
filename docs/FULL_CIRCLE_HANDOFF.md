# Full-circle recommendation — HANDOFF (neuroad.html, active-collision safe)

**Why this doc:** two sessions were editing `app/neuroad.html` at the same time (one adding the
AUROC/CI header below; another mid-refactor of the ranked-targets structure viewer). To avoid
corrupting work, ONE session should own `neuroad.html`. This doc lets that owner finish the
full-circle rec and re-apply the AUROC/CI header cleanly. See also `docs/FULL_CIRCLE_REC_SPEC.md`.

## Status (2026-07-12 ~22:13)
- ✅ **AUROC/CI header** built and rendering (verified: neuroad.html serves 200; header shows
  `AUROC 0.92 [0.91–0.94] · p=0.001` + `anchored to plasma p-tau217 · r=+0.49, n=876 (measured)`).
  It is **UNCOMMITTED** in the working tree, tangled with the other session's refactor. It may be
  bundled into their commit OR overwritten. If overwritten, re-apply with the recipe below.
- ⚠️ **Flow-restructure NOT done.** User's clarified spec: the ranked-targets card is a GATE STEP
  in the right-rail spot, shown BEFORE the standard right-rail card; clicking a protein (top or any)
  opens the per-protein card.

## The AUROC/CI header — exact re-apply recipe
Target: the `if(k==='proteins'){ ... }` branch of `storyShell`/story rendering (search
`'Ranked protein targets'`). All numbers READ FROM THE CASE — never hardcode.

**Edit A** — right after `const tx=this.heroTx();` `if(tx){`, before `const targets=...`, insert:
```js
// Finding-level measured significance (AUROC + CI), read from the case — the ranking DERIVES
// from this imaging finding. Per-protein AUROC does NOT exist (those are PI4AD priority scores),
// so significance shows ONCE here, not per row.
const c=this.heroCase()||{}; const lm=c.leakage_margin||{};
const anc=((c.tests||[]).find(t=>t.key==='biomarker_anchor')||{}).stats||{};
const aucTxt = (lm.outcome_auc!=null)
  ? ('AUROC '+lm.outcome_auc.toFixed(2)
     + (Array.isArray(lm.outcome_ci)?(' ['+lm.outcome_ci[0].toFixed(2)+'–'+lm.outcome_ci[1].toFixed(2)+']'):'')
     + (lm.outcome_p_perm!=null?(' · p='+(lm.outcome_p_perm<=0.001?'0.001':lm.outcome_p_perm.toFixed(3))):''))
  : null;
const ancTxt = (anc.ptau217_r!=null)
  ? ('anchored to plasma p-tau217 · r=+'+anc.ptau217_r.toFixed(2)+', n='+anc.ptau217_n+' (measured)')
  : null;
const sigHeader = aucTxt ? el('div',{key:'sig',style:{padding:'10px 12px',background:C.paper2,border:'1px solid '+C.edge,borderRadius:'10px',marginBottom:'12px'}},[
  el('div',{key:'t',style:{fontSize:'10.5px',letterSpacing:'.05em',textTransform:'uppercase',color:C.mute,marginBottom:'5px'}},'Measured on the imaging finding'),
  el('div',{key:'a',style:{fontFamily:'ui-monospace,SFMono-Regular,Menlo,Consolas,monospace',fontSize:'13px',color:C.ink}}, aucTxt),
  ancTxt ? el('div',{key:'n',style:{marginTop:'4px',fontSize:'11.5px',color:C.ink2}}, ancTxt) : null
]) : null;
```

**Edit B** — as the FIRST child of the `storyShell(node,'Ranked protein targets',[ ... ])` array
(before the `key:'g'` "Ranked by PI4AD priority score…" div), insert:
```js
sigHeader,
```
React renders `null` children harmlessly, so OASIS cases (no p-tau217 anchor / no leakage_margin)
degrade cleanly.

## The flow-restructure (what's still to build)
User spec (confirmed): investigation done → **Ranked protein targets card** in the right-rail spot
(the gate) → user clicks a protein → **standard right-rail card** (the drawer) for that protein.

Code map (search by marker; line numbers drift under active edits):
- **Standard right-rail card = the drawer**: rendered when `state.pinned` is set (search
  `'data-drawer'`). Tabs = `drawerTabs` (Summary/Artifacts/Reasoning/Brain data; defs near
  `defs.push(['summary','Summary'])`). Opened by pinning a node (`onNodeClick` → sets
  `pinned:id, drawerTab:'summary'`).
- **Ranked-targets card = the `'proteins'` storyShell** (search `'Ranked protein targets'`),
  rows via `rankRow(rank,name,desc,score,opts)`.
- **Per-protein structure already exists**: `StructViewer` + per-gene `app/structures/<GENE>.cif`
  (APP/ESR1/MAPT/APOE/PSEN1/BACE1) + AlphaFold pLDDT per gene — from the protein-tab session.

**Open design question for the implementer:** "individualized card for each protein" — does clicking
a protein open (a) the existing finding-summary drawer (image #7; AUC/confounds/pipeline are
FINDING-level, identical per protein), or (b) a genuinely per-protein card (that protein's
AlphaFold structure + PI4AD rank + druggability + evidence)? Option (b) is the honest, more useful
one and the per-protein assets exist. Recommend (b): each `rankRow` gets an `onClick` that sets a
`selectedTarget` and renders a per-protein detail card (structure via StructViewer for that gene +
its priority/rank/evidence), with the finding-level AUROC/CI shown as shared context.

## HONESTY RULES (do not violate)
- AUROC/CI is FINDING-LEVEL, shown once in the header. NEVER a per-protein AUROC (candidates carry
  PI4AD `priority_score`, not AUROC).
- Every number traces to the case (`leakage_margin`, `tests[].stats`, `translation.ranked_targets`).
  Do not hardcode. The GWAS ranking-validation `AUC 0.728` is NOT in the frontend case data (only in
  `reports/target_prioritization_validation.json`); to show it, plumb it into `demo_data.json` first
  (demo_data session's territory) — do not hardcode it in the UI.

## Coordination
One owner for `neuroad.html` at a time. Recommended: the session already doing the structure refactor
finishes the flow-restructure (option b above) AND applies Edits A+B for the header; then commit once.
