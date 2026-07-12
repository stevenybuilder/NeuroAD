# Dataset-connect step for the Claude Science entry (frontend demo) — implementation handoff

**Audience:** an engineer/session implementing this without the original conversation context.
**Scope:** a NEW UX step in the existing Claude Science entry page. Frontend only, scripted,
self-contained. Do NOT touch the backend pipeline or the existing NeuroAD product.

---

## 1. Context — what NeuroAD is and why this step matters

**NeuroAD** is an Alzheimer's imaging *discovery* engine (not a diagnostic tool). A researcher
states a hypothesis; NeuroAD runs it through a confound "referee" on real imaging cohorts and, if
the signal survives, returns ranked druggable protein targets + suggested wet-lab experiments. It
closes the gap between **neuroimaging** (produces a *phenotype*) and the **wet lab** (needs a
*molecular target*).

**The entry point** is a recreation of Anthropic's "Claude Science" surface (already built at
`app/claude_science.html`, served at `/start`). Today the flow is:

1. `empty` — Claude Science greeting + composer. User types `/`.
2. Slash menu → user selects the **NeuroAD** connector.
3. `connected` — "NeuroAD connected", Claude asks for a hypothesis.
4. User types a hypothesis → `result` — a tool card with **"Open in NeuroAD ↗"** that opens the
   real NeuroAD product in a **new browser tab** (`window.open('/?h='+enc,'_blank')`).

**The problem this step solves.** Right now the tool jumps straight from "connect NeuroAD" to
"type a hypothesis" — it never shows *which data* the analysis runs on. In the real product vision,
a researcher's value comes from NeuroAD working against **the datasets they already work with**.
This step makes that legible and personal: after connecting NeuroAD, it **auto-scans the datasets
the researcher has historically used**, surfaces them, and **connects the ones relevant to their
hypothesis**. It demonstrates the "connector-scoped cohort" vision and makes the demo feel like it
knows the user's world.

---

## 2. How to frame the data + product + the researcher's starting point

- **Starting point:** a translational/neuroimaging AD researcher is in Claude Science. They have
  worked with several neuroimaging datasets over time (their own downloads, lab cohorts). They
  connect the NeuroAD skill/connector.
- **The framing to convey in the UI:** *"NeuroAD sees the neuroimaging datasets in your workspace,
  and connects the ones relevant to your hypothesis."* Hypothesis + the researcher's historical
  datasets → scoped analysis → targets.
- **Honesty (important):** in the demo, the "datasets the researcher has historically used" are
  **our real reference cohorts** (ADNI, OASIS, OpenBHB) standing in for the user's workspace. The
  auto-scan is **scripted**, not a real filesystem/connector scan. Badge/caption it so a judge is
  never misled (e.g., a small "demo workspace" note). Do NOT imply we scanned the user's real machine.

---

## 3. The new step — behavior spec

Insert a dataset-connect step **after** the NeuroAD connector is selected and **before** the
hypothesis is submitted. Suggested new phase: `'datasets'` (between `'connected'` and the hypothesis
entry — see §5 for exact hook).

### Beat A — auto-scan (brief, ~1–1.5s scripted)
- Right after NeuroAD connects, show a Claude line + a scanning affordance:
  *"Scanning your workspace for neuroimaging datasets…"* with a small spinner (reuse the existing
  `.spin` class).
- Then resolve to the discovered-datasets list (Beat B).

### Beat B — discovered datasets (the researcher's "historical" data)
- Render a compact list/cards of the discovered datasets, each with: name, a 1-line descriptor,
  and key metadata chips (n subjects · modality · biomarkers · sites). Use the demo datasets in §4.
- Each dataset has a **toggle/checkbox**, defaulting to ON (all connected). The researcher can
  deselect. A primary button **"Use these datasets →"** (or auto-advance) moves to hypothesis entry.
- Keep it minimal and light (match the existing design tokens/classes — no crowding).

### Beat C — hypothesis entry (existing `connected` prompt)
- Claude asks for the hypothesis (existing copy is fine, optionally: "…and I'll use the relevant
  datasets you connected").

### Beat D — relevance matching (on hypothesis submit)
- When the user submits the hypothesis, **match it to the relevant connected datasets** and show
  which ones NeuroAD will use and WHY. Simple keyword→capability matching is enough (scripted):
  - mentions `plasma` / `p-tau` / `p-tau217` / `biomarker` → datasets with plasma (ADNI)
  - mentions `atrophy` / `hippocamp` / `structural` / `MRI` / `volume` → structural datasets (ADNI, OASIS)
  - mentions `conversion` / `MCI` / `progress` → datasets with conversion labels (ADNI, OASIS-2)
  - mentions `scanner` / `healthy` / `brain age` / `control` → OpenBHB
  - default (no keyword hit) → ADNI (primary AD substrate)
- Render this inside the existing tool card: header `NeuroAD · investigate`, a line like
  *"Matched to 2 of your datasets: ADNI, OASIS-1"* (list the matched ones + a one-word reason), then
  the existing **"Open in NeuroAD ↗"** button.

### Beat E — hand-off (existing, extended)
- **Keep the new-tab hand-off** (`window.open(..., '_blank')`). Extend the URL to also pass the
  matched datasets, e.g. `/?h=<hypothesis>&ds=adni,oasis1`. (The product already reads `?h=` to
  prefill the hypothesis; `&ds=` is optional/forward-compatible — the product can ignore it for now.)

---

## 4. Demo datasets (hardcode these — our real cohorts as the "workspace")

These are the real reference cohorts the engine actually uses; present them as the researcher's
historical datasets. Numbers are card-friendly approximations of the real cohorts.

| Dataset | Descriptor | Metadata chips |
|---|---|---|
| **ADNI** | Alzheimer's Disease Neuroimaging Initiative | ~1,600 subjects · T1w MRI · **plasma p-tau217** · AD/MCI/CN + conversion · multi-site |
| **OASIS-1** | Open Access Series of Imaging Studies (cross-sectional) | 210 subjects · T1w MRI · CDR labels · single-scanner (1.5T) |
| **OASIS-2** | OASIS longitudinal | 150 subjects · longitudinal T1w MRI · conversion |
| **OpenBHB** | Healthy multi-scanner controls | 3,984 subjects · T1w MRI · 62 sites · brain-age / scanner control |

Optionally show a couple as "connected" and one (e.g. OpenBHB) as a control cohort. Icons can be
simple monograms (reuse the `.ic` style). Keep descriptions short — no crowding.

---

## 5. Implementation notes

- **File:** `app/claude_science.html` only. It is a self-contained page (inline CSS+JS, no external
  deps — keep it that way). Its state machine is a plain object:
  `const state = { phase:'empty'|'connected'|'result', connected, rawInput, hypothesis, showSlash, ... }`
  with a `render()` that rebuilds `#root` from `state`.
- **Hook point:** `selectNeuroAD()` currently sets `state.phase='connected'`. Insert the new
  `'datasets'` phase here: on NeuroAD select → `phase='datasets'` (Beats A–B) → on "Use these
  datasets" → `phase='connected'` (Beat C). Store selected datasets on `state.datasets` (array).
  On hypothesis submit (existing handler that sets `phase='result'`), compute matched datasets from
  `state.hypothesis` + `state.datasets` and render them in the tool card (Beat D). Extend the
  `window.open` URL with `&ds=` (Beat E).
- **Reuse existing components/classes:** `topbar()`, `composer()`, `aiMsg()`, `userMsg()`, the
  `.toolcard/.thd/.spin/.res/.openbtn`, `.slash/.srow/.ic/.badge/.feat` classes, and the `:root`
  Anthropic light tokens (`--paper #F5F4ED`, `--ink #141413`, `--clay #C96442`, etc.). New dataset
  cards should reuse `.srow`-style rows or a small card; only add CSS if strictly necessary.
- **Keep it scripted + deterministic** (screen-recorded live): fixed ~1–1.5s scan delay, no real
  I/O, no network. Auto-focus behavior and Enter-to-submit already exist — preserve them.
- **Do NOT** modify the served NeuroAD product (`app/neuroad.html`) beyond what already exists (it
  already reads `?h=`), the backend, or `app/server.py` routes. This is purely the entry page.
- **Forward hook (optional, not required for the demo):** the backend already exposes
  `GET /api/datasets -> {datasets:[...]}` (backed by `neuroad.data.loaders.AVAILABLE`), and
  `POST /api/investigate` already accepts a `{"hypothesis", "dataset"}` body. So the "connect
  datasets" step maps onto real plumbing: a future non-demo version could populate the discovered
  list from `/api/datasets` and pass the matched dataset through to `/api/investigate`. For THIS
  demo, keep the list scripted/self-contained (§4) — just know the real seam exists.

---

## 6. Honesty guardrails (non-negotiable)

- The auto-scan is a **scripted demo** of discovering the researcher's datasets — label it as such
  (small "demo workspace" caption). Never imply we scanned the user's actual filesystem/accounts.
- The datasets shown are **our real reference cohorts** (ADNI/OASIS/OpenBHB) used as stand-ins for
  the researcher's workspace — real data, presented as the demo's connected datasets.
- Relevance matching is **keyword-based** in the demo; it's a plausible stand-in for real
  hypothesis→dataset matching, not a claim of semantic understanding.

---

## 7. Acceptance criteria

1. From `/start`: `/` → select NeuroAD → **auto-scan** appears → **discovered datasets** list
   (ADNI, OASIS-1, OASIS-2, OpenBHB) with metadata chips + toggles → "Use these datasets →".
2. Hypothesis entry works as before; on submit, the tool card shows **which datasets were matched**
   to the hypothesis and why (e.g. "p-tau217 …" → ADNI [plasma], OASIS-1 [structural]).
3. **"Open in NeuroAD ↗" still opens a real new tab**; URL carries `?h=<hypothesis>` (and optional
   `&ds=<matched>`). Claude Science stays in tab 1.
4. Page remains **self-contained** (no external requests) and visually consistent with the current
   design (light theme, existing tokens/classes).
5. The existing product at `/` and the backend are **unchanged**.

---

## 8. Reference — real cohort facts (for accuracy if numbers are questioned)

Real, already-ingested cohorts the engine uses (`data/real/`, loaded via `neuroad.data.loaders`):
ADNI (structural FreeSurfer + real 3-assay plasma p-tau217; ~590 embedded / ~1,600 dx-labeled /
1,199 conversion-labeled / 1,377 with plasma), OASIS-1 (210) + OASIS-2 (150), OpenBHB (3,984 healthy,
62 sites; 96 with frozen Neuro-JEPA embeddings). See `docs/FRAMING.md` §8 for the full
dataset-provenance table and `docs/DATA_EXPANSION_SPEC.md` for how new cohorts (e.g. AIBL) plug in —
the same "connect a dataset" mechanic this step visualizes.
