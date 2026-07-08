# Neuro-JEPA weights: access, secrets, and license compliance

The referee is **encoder-agnostic** and runs fully on open data
(`openbhb`, `oasis`) and clearly-labeled `synthetic` cohorts with **no weights and
no GPU**. The frozen Neuro-JEPA embeddings are an *optional* upgrade of the
substrate. This doc is how we use those gated weights without ever leaking a token
or redistributing a model — the rules below are non-negotiable in this repo.

## TL;DR rules

1. **Never commit the weights.** `model.safetensors` and any `*.pt/*.pth/*.ckpt`
   are `.gitignore`d. They live only in the HuggingFace cache
   (`~/.cache/huggingface/hub`) or on an ephemeral GPU runtime — never in git.
2. **Never commit a token.** No token in code, notebooks, or committed files. The
   token is read from the `HF_TOKEN` environment variable only.
3. **Never commit the embedding table.** `*embeddings*.csv` / `*.npy` are
   `.gitignore`d. A large derived-embedding dump could be argued a derivative of
   the CC-BY-NC-ND weights, so it stays local. Result *numbers* (AUCs) in
   `reports/` are fine to publish.
4. **Frozen inference only — never fine-tune.** Training any layer on the weights
   would create a derivative work, which the license forbids (see below).

## The license: CC-BY-NC-ND 4.0 (gated)

Neuro-JEPA (`NYUMedML/Neuro-JEPA`) is released **Creative Commons
Attribution-NonCommercial-NoDerivatives 4.0**, access-gated:

| Clause | What it means for us | What we do |
|---|---|---|
| **BY** (Attribution) | Cite the model | Cited in `README.md` + `reports/*` provenance |
| **NC** (NonCommercial) | Research/education only | Hackathon research use ✔ |
| **ND** (NoDerivatives) | No adapted/retrained model may be shared | **Frozen inference only** — we never fine-tune |
| **Gated** | Each user requests their own access | We authenticate per-user via `HF_TOKEN`; token never shared/committed |

Frozen inference that produces embeddings for an analysis is **use**, not a
distributed adaptation — that is why the `openbhb:neurojepa` path is defensible.
Fine-tuning, or shipping the weights / a bulk embedding table, would not be.

## Getting access (each user, once)

1. Request access on the model card: <https://huggingface.co/NYUMedML/Neuro-JEPA>
   (an institutional email is required; approval is manual).
2. Create a **fine-grained, read-only** token (one per machine/use) at
   <https://huggingface.co/settings/tokens>. Per HuggingFace guidance, scoped
   read tokens limit blast radius if leaked.
3. Put it in your environment, never in the repo:
   ```bash
   export HF_TOKEN=hf_xxx        # or: huggingface-cli login  (stores in ~/.cache, not the repo)
   ```
   `.env` is git-ignored; copy `.env.example` to `.env` for local use.

## How the code uses it (token from env, weights ephemeral)

```python
import os
from neurojepa.utils.init_utils import load_backbone_from_hf
backbone = load_backbone_from_hf("NYUMedML/Neuro-JEPA", token=os.environ["HF_TOKEN"])
```

On Colab the weights are fetched **to the GPU runtime** from HuggingFace and
discarded when the runtime is released — they are never uploaded from your laptop
and never written to the repo (see `scripts/openbhb_embed.py`).

## If you clone this repo with no access

Everything works without the weights. `load("openbhb:neurojepa")` raises a clear
error telling you to run `scripts/openbhb_embed.py` with your own `HF_TOKEN`; every
other feeder (`openbhb`, `oasis`, `synthetic:*`) runs offline on open/synthetic data.

## References

- HuggingFace — User access tokens: <https://huggingface.co/docs/hub/security-tokens>
- HuggingFace — Gated models: <https://huggingface.co/docs/hub/models-gated>
