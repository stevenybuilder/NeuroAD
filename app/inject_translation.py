#!/usr/bin/env python3
"""
inject_translation.py — surface the molecular translation + normalized tree onto
the ALREADY-COMMITTED app/demo_data.json WITHOUT re-running the full engine.

Why not just rebuild? A full engine re-run is NOT byte-stable (real OASIS
Neuro-JEPA evidence + biomarker-stat fields drift with the current code /
environment), which would blow away frozen numbers the real app/index.html
depends on. This injector instead READS the committed payload and only ADDS:

  * ``translation`` on each promoted survivor (offline PI4AD -> AlphaFold ->
    repurposing -> wet-lab, via harness.translation.translate, routed by the
    case's pre-registered mechanism). KILL / refused cases get NONE.
  * ``translation`` + ``mechanism`` on each promoted Detective cluster.
  * a normalized ``tree`` object per case (+ cluster) so the UI consumes ONE
    shape (build_demo_data._derive_tree).
  * meta.schema -> 1.1.0 + meta.translation_note + molecular_sources_unverified.

Every existing field VALUE is left untouched (keys are only appended), so
`git diff` shows additions only. Re-running build_demo_data.py in full will
produce the same additions (the helpers are shared) — this is just the
byte-safe path for the frozen demo.

    ./.venv/bin/python app/inject_translation.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"

# Import the shared helpers from build_demo_data (it puts src/ on sys.path).
_spec = importlib.util.spec_from_file_location("bdd", APP / "build_demo_data.py")
bdd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bdd)


def main() -> int:
    path = APP / "demo_data.json"
    data = json.loads(path.read_text())

    # 1. Promoted survivors -> real molecular translation (offline, deterministic).
    for sub_key, sub in (data.get("substrates", {}) or {}).items():
        for kind, case in (sub.get("cases", {}) or {}).items():
            if case.get("promoted") and not case.get("translation"):
                mech = ((case.get("investigate", {}) or {}).get("routed_mechanism")
                        or "amyloid_cascade")
                case["translation"] = bdd._static_translation(mech)
                print(f"[inject] {sub_key}/{kind}: translation attached "
                      f"(mechanism={mech}, top={case['translation'].get('top_target')})")
            elif not case.get("promoted"):
                print(f"[inject] {sub_key}/{kind}: KILL/refused -> NO translation")

    # 2. Promoted Detective clusters -> mechanism routing + translation.
    for dk in ("discovery", "discovery_real"):
        disc = data.get(dk)
        if not isinstance(disc, dict):
            continue
        for cluster in disc.get("clusters", []) or []:
            mech = bdd._cluster_mechanism(cluster)
            cluster["mechanism"] = mech
            if cluster.get("promoted") and not cluster.get("translation"):
                cluster["translation"] = bdd._static_translation(mech)
                print(f"[inject] {dk}/cluster {cluster.get('cluster')}: translation "
                      f"attached (mechanism={mech})")

    # 3. Normalized tree per case (+ cluster) + meta bump (idempotent).
    bdd._attach_trees(data)

    path.write_text(json.dumps(data, indent=2))
    print(f"[inject] wrote {path} (schema={data['meta']['schema']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
