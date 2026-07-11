# Session Status — backend build-out (branch `feat/molecule-translation-loop`)

What was built, what's verified, and what's next. All work is committed on
`feat/molecule-translation-loop` (not pushed). Full test suite: **245 passed, 2
skipped** at the last full run.

---

## What was done (in order)

| Commit | What | Verified |
|---|---|---|
| `0a331fd` | **Imaging→molecule translation loop** wired: promoted survivor → mechanism → PI4AD ranked genes → AlphaFold structure → repurposing → falsifiable organoid experiment. Plus honesty fixes (out-of-scope target refusal; honest per-feeder substrate labels). | End-to-end on real ADNI-3T (APP, pLDDT 67.4, Nilotinib/Bexarotene) |
| `0d0eb57` | **Interactive live backend** (`app/server.py`, stdlib HTTP): `POST /api/investigate`, `POST /api/orchestrate`; runs referee + translation live; `Dockerfile.backend` for Cloud Run. | 5 HTTP smoke tests |
| `823f1f1` | **Switched live model off Fable 5 → Sonnet 5** ($10/$50 → $3/$15); Fable's research-bio classifiers risked false-positive refusals. Haiku documented as the cheap switch. | Suite green |
| `a21a71b` | **Claude ORCHESTRATOR harness** (`harness/agent.py`, tool-runner): the engine's capabilities are tools Claude sequences; the tools decide. Scripted deterministic fallback offline. | 9 tests; kill-gate invariant |
| `56a126d` | **Ripped Claude out of the referee** — pipeline is now 100% deterministic; `import anthropic` exists in exactly one file (`agent.py`). `model_badge()` now honest (`referee: deterministic`). | Suite green; no report claims live-Claude referee |
| `bab4ebc` | **Active-learning wet-lab feedback loop** (`harness/discovery_loop.py`): Beta belief per target seeded from a composite prior; acquisition function (ucb/uncertainty/greedy) selects the next experiment; results update the posterior; state persists. 4 orchestrator tools. | 6 tests; simulated rounds |
| `e55f4ed` | **Open Targets integration** — real live AD target-disease association scores + known drugs, fused into the loop's priors (APP: pi4ad 0.86 · structure 0.67 · opentargets 0.81) and exposed as the `target_evidence` orchestrator tool. | 20 adapter tests; live path |
| `82aeb64` | **Plasma biomarker ensemble** (`data/plasma_ensemble.py`): triangulate ADNI's 3 plasma assays → 1,593 p-tau217 (vs 1,366), 917 with two independent assays, + Aβ42/40 and %p-tau217. | 4 tests |

**The 4 external tools** (AlphaFold, PI4AD, multimodal transformer, GNN/LLM) were
integrated earlier as `src/neuroad/integrations/` adapters (offline-first,
provenance-labeled). Open Targets was added this session.

**Architecture now:** deterministic referee (kill/promote, gauntlet, scoring,
molecule loop) + Claude ONLY as the orchestrator that drives tools. Matches plan
§4.2.

---

## Credentials / access state

- `HF_TOKEN` — stored in `.env` (git-ignored), **verified working** (user
  `stevenyml`, Neuro-JEPA gated access granted). Unblocks embedding generation.
- Open Targets, AlphaFold EBI, PI4AD portal — open, no login, live.
- `ANTHROPIC_API_KEY` — not set; the orchestrator runs its scripted fallback
  offline. Live tool-runner loop is written to the SDK API but **unverified
  without a key** (~$0.10–0.40/run on Sonnet, pennies on Haiku).
- Colab CLI — installed and authed; no active runtime.

---

## Next steps (priority order)

1. **[BIGGEST POWER LEVER] Expand NeuroJEPA embeddings on Colab.** Current disease
   signal is n=61 (OASIS, 8 AD) / n=96 (OpenBHB leakage) — the thinnest evidence.
   Now unblocked (token verified). Highest value: **OASIS-1 61 → up to 436**
   (has CDR labels). Manifest exists: `data/real/_manifests/oasis1_gap_manifest.csv`;
   run `scripts/neurojepa_embed.py` on `colab start --gpu t4` (weights fetch
   ephemerally via `HF_TOKEN`, never committed; embeddings stay local per
   CC-BY-NC-ND). This is also what makes real cross-cohort validation possible
   (ADNI FreeSurfer ≠ OASIS morphometry — a shared embedding space is required).

2. **Wire the plasma ensemble into the live biomarker anchor.** `data/plasma_ensemble.py`
   is built + tested but standalone; integrate it into the ADNI contract's anchor
   (triangulated p-tau217 + new Aβ42/40 / %p-tau217 markers) so promotion/mechanism
   routing use it.

3. **Cross-cohort + multiplicity controls.** Leave-one-cohort-out validation +
   Benjamini-Hochberg FDR — the honest fix for single-cohort/multiplicity, and
   what makes the active-learning search trustworthy rather than a confound-finder.
   (Cross-cohort transfer depends on step 1's shared embeddings.)

4. **Longitudinal ADNI.** `ADNIMERGE2` (in the download folder) has PACC/MMSE for
   4,746 subjects (~3 visits) — enables trajectory/slope outcomes, more power than
   binary contrasts.

5. **Verify the live orchestrator + one live adjudication.** With `ANTHROPIC_API_KEY`,
   run one real Sonnet orchestration to confirm the tool-runner loop end-to-end and
   report actual token cost.

6. **Interactive UI + deploy.** Wire `app/index.html` to `POST /api/investigate` /
   `POST /api/orchestrate`; deploy via `Dockerfile.backend` (note `.gcloudignore`
   currently strips `src/`/`data/` for the static deploy).

---

## Known caveats (honest)

- Plasma ensemble built but not yet wired into the anchor (step 2).
- Live Claude tool-runner loop unverified without a key (step 5).
- The engine's discovery *power* is still early: single-cohort (ADNI) dependence,
  thin embeddings (step 1), molecule side is a real-evidence scaffold not validated
  against outcomes.
- Nothing pushed; 8 commits on `feat/molecule-translation-loop`.
