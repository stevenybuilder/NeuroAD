# Models

Model weights are git-ignored. Check 1.2 (skull-strip) needs the SynthStrip weight here:

```
backend/models/synthstrip.1.pt
```

Fetch it (run in the auth'd `hf` env):

```bash
micromamba run -n hf python -c "from huggingface_hub import hf_hub_download as d; \
    d('jil202/synthstrip','synthstrip.1.pt',local_dir='backend/models')"
```

If the weight is absent, check 1.2 degrades gracefully: it still runs the weak-stripper
verification but skips the SynthStrip reference comparison.
