#!/usr/bin/env python3
"""Stream-extract a tiny IXI T1 subset (N per scanner) from the public HF mirror
WITHOUT storing the 4.8 GB tar. Reads the tar sequentially over HTTP and stops as
soon as it has enough subjects, so it only pulls a few hundred MB.
"""
import sys, tarfile, urllib.request
from pathlib import Path

URL = "https://huggingface.co/datasets/Santhosh1884/IXI-Datasets/resolve/main/IXI-T1.tar"
OUT = Path("mri_visualizations/data/ixi")
SCANNERS = ["Guys", "HH", "IOP"]
PER = int(sys.argv[1]) if len(sys.argv) > 1 else 2
BYTE_CAP = 2_500_000_000  # hard stop so it can never blow disk/time

OUT.mkdir(parents=True, exist_ok=True)


class Counting:
    def __init__(self, fh): self.fh = fh; self.n = 0
    def read(self, size):
        b = self.fh.read(size); self.n += len(b)
        if self.n > BYTE_CAP:
            raise RuntimeError(f"byte cap hit ({self.n} bytes) before subset complete")
        return b


def main() -> int:
    got = {s: 0 for s in SCANNERS}
    req = urllib.request.Request(URL, headers={"User-Agent": "sfg-fetch"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        counting = Counting(resp)
        with tarfile.open(fileobj=counting, mode="r|") as tf:
            for m in tf:
                if not (m.isfile() and m.name.endswith(".nii.gz")):
                    continue
                base = Path(m.name).name
                site = next((s for s in SCANNERS if f"-{s}-" in base), None)
                if site is None or got[site] >= PER:
                    continue
                out = OUT / base
                if not out.exists():
                    with tf.extractfile(m) as src:
                        out.write_bytes(src.read())
                    print("extracted", base, flush=True)
                got[site] += 1
                if all(got[s] >= PER for s in SCANNERS):
                    break
    print("done:", got, "| streamed ~%.0f MB" % (counting.n / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
