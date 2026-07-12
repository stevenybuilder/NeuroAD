# Data

The contents of this directory are **git-ignored** (tens of GB of imaging). This file
documents the layout the code expects and how to repopulate it on a fresh clone.

```
data/
├── ixi/                       # raw T1, three scanners - the Phase-1 workhorse
│   ├── IXI002-Guys-0828-T1.nii.gz
│   ├── IXI012-HH-1211-T1.nii.gz
│   ├── IXI035-IOP-0873-T1.nii.gz
│   └── _tars/IXI-T1.tar        # cached source tarball (optional to keep)
├── brats/
│   └── cases/                  # one dir per case: <case>-{t1n,t1c,t2w,t2f,seg}.nii.gz
├── fixtures/                   # generated induced-failure scans (+ .json sidecars)
└── adni/                       # tabular only, unused this release
```

## Repopulate

**IXI** (raw T1, Guy's / Hammersmith / IOP) - the official host 403s direct downloads, so we
pull the original tar from a HuggingFace mirror (run in the auth'd `hf` env):

```bash
micromamba run -n hf python scripts/fetch_ixi.py 4    # 4 subjects per scanner
```

**BraTS-GLI** - download `ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip` from
[Synapse](https://www.synapse.org/) (gated registration) and extract a handful of case dirs
into `data/brats/cases/` (each case dir holds the four modality NIfTIs + `<case>-seg.nii.gz`).

**Fixtures** - generated automatically from an IXI scan on backend startup
(`backend/sfg/fixtures.py`); no manual step.

**ADNI** - not used this release (tabular biomarker data, no volumes).
