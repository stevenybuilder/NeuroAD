#!/usr/bin/env python
"""Fetch a balanced IXI T1 subset covering all three scanners.

The official brain-development.org host 403s direct downloads, so we pull the
original IXI-T1.tar from a HuggingFace mirror (auth via the `hf` env) and extract
N subjects per scanner (Guy's / Hammersmith / IOP). Raw T1 with intact skulls and
real affines - the Phase-1 workhorse.

Run in the `hf` env:  micromamba run -n hf python scripts/fetch_ixi.py [per_scanner]
"""

import sys
import tarfile
from pathlib import Path

REPO = "Santhosh1884/IXI-Datasets"
TAR_NAME = "IXI-T1.tar"
SCANNERS = ["Guys", "HH", "IOP"]
IXI_DIR = Path(__file__).resolve().parents[1] / "data" / "ixi"
TAR_DIR = IXI_DIR / "_tars"


def ensure_tar() -> Path:
    tar = TAR_DIR / TAR_NAME
    if tar.exists():
        return tar
    from huggingface_hub import hf_hub_download

    print(f"downloading {TAR_NAME} from {REPO} ...")
    hf_hub_download(REPO, TAR_NAME, repo_type="dataset", local_dir=str(TAR_DIR))
    return tar


def main() -> int:
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    IXI_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = ensure_tar()

    with tarfile.open(tar_path) as tf:
        names = tf.getnames()
        picked: list[str] = []
        for site in SCANNERS:
            site_members = sorted(n for n in names if f"-{site}-" in n and n.endswith(".nii.gz"))
            picked.extend(site_members[:per])
        for name in picked:
            out = IXI_DIR / Path(name).name
            if out.exists():
                continue
            member = tf.getmember(name)
            with tf.extractfile(member) as src, out.open("wb") as dst:
                dst.write(src.read())
            print("extracted", out.name)

    print(f"\nIXI subset ready: {len(picked)} subjects ({per} per scanner) in {IXI_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
