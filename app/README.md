# NeuroAD Discovery Engine — Visual Workbench (M4)

A self-contained, offline scientific **workbench** that renders as a *viewer over
the real exported artifacts* of the referee engine. It is the demo surface:
masthead + substrate badge, cohort card, docket, embedding scatter, the live
gauntlet checklist, a filling claim card, a Claude courtroom + reviewer margin
note, a verdict meter, a confound leaderboard, a KILL-vs-SURVIVOR split, and an
export tray.

## Files
- **`index.html`** — fully self-contained (all CSS/JS inline, zero external
  calls, works offline from `file://`). Loads `demo_data.json` if present, else
  falls back to an **embedded copy of the same JSON** so it renders standalone
  even before the engine has ever run.
- **`build_demo_data.py`** — engine → `app/demo_data.json` + `reports/*`.
- **`demo_data.json`** — the deterministic payload the page renders.

## Run it
Just open the file — no server, no build step:
```
open app/index.html          # macOS
# or drag app/index.html into any browser
```

## Regenerate the data
```
cd <repo root>
PYTHONPATH=src ./.venv/bin/python app/build_demo_data.py            # calibrated fallback (default)
PYTHONPATH=src ./.venv/bin/python app/build_demo_data.py --engine   # overlay a live referee run
```
`build_demo_data.py` writes `app/demo_data.json` and six artifacts into
`reports/` (`cohort_card.json`, `claim.yaml`, `evidence_ledger.csv`,
`methods.md`, `referee_run.ipynb`, `reviewer_report.md`).

After regenerating, re-embed the fallback copy into the page (keeps the offline
standalone identical to the emitted JSON):
```
./.venv/bin/python /path/to/scratchpad/gen_index.py   # or re-run the M4 generator
```
The page also `fetch()`es `demo_data.json` at load when served over `http(s)://`,
so a fresh engine run shows up without re-embedding when hosted.

### Why the default is the calibrated fallback (not the live engine)
The demo must be **deterministic and identical every take**, and it must tell
the clean SURVIVOR-vs-KILL story regardless of how tuned the parallel-built
engine currently is. So `build_demo_data.py` defaults to the calibrated fallback
(every number pinned to `src/neuroad/calibration.py`) and only overlays a live
`pipeline.run_referee()` run when you pass `--engine` (or `NEUROAD_ENGINE=1`).
All engine imports are guarded: if M1/M2/M3 are missing or a case fails, it
degrades to the fallback for that case and prints a clear message.

## What the demo shows (choreography)
- **Substrate badge** (top-right, press `S`): `SYNTHETIC HARNESS` ⇄ `REAL OASIS`.
  Synthetic carries the ground-truth scanner-confound KILL and the p-tau217
  anchor; OASIS is real, vendored, single-scanner (so the ★ leakage test is an
  honest **cohort/batch** reframe) with the biomarker gate marked N/A.
- **Docket** — two cases: **A · SURVIVOR** and **B · KILL**. Status chip goes
  `UNADJUDICATED → RUNNING → ADJUDICATED`.
- **Embedding scatter** (PCA 2-D) — toggle **color: outcome / color: scanner**
  (press `C`). On the KILL, coloring by scanner makes the leakage *visible*:
  the dominant axis of variance is the acquisition scanner, not the disease.
- **The Gauntlet** — five rows tick `queued → running → result`, each with an
  effect-size bar (0.5 = chance … 1.0). The two ★ tests (site/scanner leakage,
  brain-age) have a thick border.
- **Verdict meter** animates `fragile → partially robust → robust`; the headline
  is the subject-disjoint **leakage margin = outcome AUC − scanner AUC**
  (+0.10 survivor, −0.20 kill).
- **Claim card** fills live; biology is **gated** — only a promoted survivor gets
  a mechanism hypothesis + one falsifiable next experiment.
- **Courtroom (Claude)** — prosecution / defense / judge, each consequential.
- **Reviewer (Claude)** margin note argues *against* the verdict ("partially
  robust ≠ robust", proxy brain-age control, p-tau217 missingness).
- **Export tray** — click any chip to preview the real artifact.

## Controls
| Key | Action |
|---|---|
| `Space` | run / replay the gauntlet |
| `←` / `→` | switch case (SURVIVOR / KILL) |
| `C` | toggle scatter color (outcome / scanner) |
| `S` | toggle substrate (synthetic / OASIS) |
| `V` | open / close the KILL-vs-SURVIVOR split |
| `Esc` | close overlays |

## Guarantees
- Offline, zero external requests (strict enough to run from `file://`).
- Deterministic: the scatter uses a fixed-seed PRNG; the timeline uses fixed
  durations — identical on every take.
- No fabricated numbers: every headline value traces to
  `src/neuroad/calibration.py`. Neuro-JEPA is spelled hyphenated. Verdict
  language stays hedged; the tool is a referee / auditor / red-team, and it
  cites the batch-effect prior art rather than claiming the leakage insight.
