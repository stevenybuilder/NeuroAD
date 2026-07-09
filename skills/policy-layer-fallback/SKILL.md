---
name: policy-layer-fallback
description: >-
  Externalize behavior into a governed "policy" layer read by BOTH deterministic
  code AND an LLM's prompt from ONE source of truth, behind a hardcoded constant
  fallback so the system is byte-identical whether the policy files are present
  or absent. This is the pattern that let 10 parallel agents modify shared, high-
  blast-radius files without drift or breakage in the NeuroAD Stage-2 build
  (74→120 green tests, investigate() runs byte-identically offline). Use when
  moving constants → config or code → policy docs while keeping an offline-
  deterministic guarantee. Reference implementation: src/neuroad/harness/policy.py.
when-to-use: >-
  Invoke when you want thresholds/rubrics/prose to be human-editable and LLM-
  readable, but you cannot afford a missing/edited/poisoned config file to change
  behavior silently or crash an offline demo. Also use before parallelizing agents
  over shared logic — the constant fallback is what makes concurrent edits safe.
  Pairs with skills/director-agent-pipeline (plan) as the safe execution pattern.
---

# Policy-Layer + Constant-Fallback Pattern

One file, two consumers (code + LLM), zero behavior change when it's gone.
"Governed semantic layer / dual-consumption from one file" — the single biggest
architectural decision in the NeuroAD build, and its process safety net.

**Why it matters:** it decouples a risky refactor (externalizing logic) from the
demo guarantee (offline, deterministic, unchanged). That decoupling is what let
many agents touch shared files in parallel.

## The invariant to preserve

> With the policy directory present OR renamed away, the emitted output is
> **byte-identical**. Prove it with `diff`, not just a unit assertion.

## Procedure

### Step 1 — Keep the live constant as the single source of truth
The real value stays a Python constant in the code (`THRESHOLDS`, `TABLE`,
`BRIEF`, …). The policy doc is an *overlay*, never the origin. Never delete the
constant when you add the policy file.

### Step 2 — Write one loader accessor per externalized thing
Each accessor (e.g. `thresholds()`, `table()`, `brief()`) contains a **hardcoded
transcription of the live constant** as its fallback, and:
- **Degrades key-by-key.** Overlay only well-formed keys the code already knows;
  an unknown or malformed key falls through to the constant. Never accept a whole
  foreign dict wholesale.
- **Validates every value.** A `_num`-style guard rejects `bool`/`None`/non-numeric
  so a poisoned doc value cannot pass. (`isinstance(x, bool)` is True for `0/1` —
  reject it explicitly.)
- **Never caches a parse failure.** A bad/absent file this run must not poison the
  next; only cache a successful parse.
- **Never raises to the caller.** Missing dir, unreadable file, bad YAML → silently
  fall back to the constant.

```
def thresholds() -> dict:
    base = dict(_FALLBACK_THRESHOLDS)          # hardcoded copy of the live constant
    doc = _load_doc("thresholds.yaml")         # returns {} on ANY failure, uncached
    for k, v in doc.items():
        if k in base and _num(v) is not None:  # known key + valid value only
            base[k] = _num(v)
    return base
```

### Step 3 — Point BOTH consumers at the accessor
- Deterministic code calls `thresholds()` / `table()`.
- The LLM system prompt is assembled from `brief()` / the same doc text.
Same source, two readers. Editing the doc updates both; deleting it changes
neither's behavior.

### Step 4 — Prove byte-identical, both directions
```
neuroad investigate "..." synthetic:SURVIVOR      # with policy/ present
mv policy policy.bak && neuroad investigate "..." synthetic:SURVIVOR && mv policy.bak policy
diff reports/investigate_synthetic_survivor.json reports/investigate_..._nopolicy.json   # must be clean
```
Then unit-assert the same: `test_loaded_equals_fallback_on_real_docs`,
`test_missing_dir_*`. The empirical `diff` is the real proof; the unit test is the
regression guard.

### Step 5 — Adversarially test the loader's inputs
Not just "does the happy path load." Feed it: missing dir, empty file, malformed
YAML, an unknown key, a poisoned value (`true`, `null`, a string where a number
belongs), a partial overlay. Each must fall back cleanly and never change a
downstream decision. This is the "test the code that enforces your guarantee"
discipline — its *inputs*, not just its outputs.

## Copy-usable checklist
- [ ] Live constant remains the source of truth; policy doc is an overlay.
- [ ] One accessor per thing, each with a hardcoded fallback copy.
- [ ] Overlay is key-by-key, known-keys-only, every value validated.
- [ ] `bool`/`None`/non-numeric rejected explicitly.
- [ ] Parse failures never cached, never raised.
- [ ] Both code and the LLM prompt read from the accessor.
- [ ] `diff` proves byte-identical with policy present vs absent.
- [ ] Loader tested against missing/empty/malformed/poisoned/partial inputs.

## Anti-patterns (we hit or dodged these)
- **Accepting a whole foreign dict** instead of key-by-key overlay → one bad doc
  silently reshapes behavior.
- **Caching a failed parse** → a transient bad file poisons the rest of the run.
- **Two vocabularies for one concept**, where a second code path emits a different
  label set and shadows the policy ladder — any caller that omits the explicit
  value silently gets the wrong vocabulary. Keep ONE governed vocabulary; delete
  or redirect the shadow path.
- **Docstrings that oversell** the guard (a curated 5-item denylist described as
  "refuses any forbidden overclaim"). State the real scope.
