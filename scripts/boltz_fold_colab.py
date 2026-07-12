#!/usr/bin/env python3
"""
Self-contained Colab GPU job: fold AD target complexes / target-ligand pairs with
Boltz-2 (open, MIT, AlphaFold3-class) and write a SMALL results JSON of confidence
+ binding-affinity SCALARS only.

Runs ENTIRELY on the ephemeral GPU runtime so a websocket drop / preempted runtime
never leaves a half-done mess (mirrors scripts/neurojepa_embed_colab.py):

  1. pip-installs boltz on top of Colab's CUDA torch,
  2. keylessly fetches each AD target's sequence from UniProt (accessions mirror
     integrations/alphafold.AD_PROTEIN_MAP), writes a Boltz-2 input YAML per job,
  3. runs REAL Boltz-2 GPU inference (`boltz predict`) for each protein-protein
     complex and each target+ligand pair,
  4. parses ONLY the scalar confidence (iptm/ptm/pae/confidence_score) and, for
     target+ligand jobs, the predicted binding-affinity scalars
     (affinity_pred_value -> binding_affinity, affinity_probability_binary ->
     binding_probability) from Boltz's output JSONs,
  5. writes `--out` incrementally in the EXACT schema of
     src/neuroad/integrations/data/boltz_snapshot.json AND streams a durable
     gzip+base64 copy to stdout between ===JSONGZ_START===/===JSONGZ_END===.

HONESTY (paramount): only the SMALL derived scalars leave the runtime. Predicted
COORDINATES (the .cif/.pdb structures) are NEVER downloaded or committed — the
committed snapshot holds confidence + affinity numbers only. Do NOT fabricate any
value: a job that fails to run is simply omitted from the output (never written
with invented numbers). Copy the resulting JSON to
src/neuroad/integrations/data/boltz_snapshot.json to light up the BoltzClient's
precomputed path in the referee.

Usage (see docs — mirrors the Neuro-JEPA runbook):
    colab start --gpu t4                                    # note the session id
    colab exec --session <id> --timeout 20m scripts/boltz_fold_colab.py -- \
        --complexes APP:MAPT,APP:BACE1 --limit 1            # smoke test
    colab exec --session <id> --timeout 120m scripts/boltz_fold_colab.py -- \
        --complexes APP:MAPT,APP:BACE1,APOE:CLU \
        --ligands 'BACE1:verubecestat:O=C(N)...SMILES...'
    # rebuild the JSON locally from the durable blob in the exec log, or:
    colab download --session <id> boltz_targeting.json \
        src/neuroad/integrations/data/boltz_snapshot.json
    colab stop --session <id>

AD target accessions (UniProt) mirror integrations/alphafold.AD_PROTEIN_MAP.
"""
import argparse
import base64
import gzip
import json
import os
import string
import subprocess
import sys
import tempfile
from pathlib import Path

# UniProt accessions for the AD targets (mirror of alphafold.AD_PROTEIN_MAP),
# extended with APP's top STRING partners (SORL1/APBB1) and the canonical targets
# of the repurposing compounds (ABL1 for Nilotinib, RXRA for Bexarotene) so the
# hero-path complexes and target+ligand affinity jobs can fetch their sequences.
AD_PROTEIN_MAP = {
    "APP": "P05067", "MAPT": "P10636", "TAU": "P10636", "APOE": "P02649",
    "PSEN1": "P49768", "PSEN2": "P49810", "BACE1": "P56817", "TREM2": "Q9NZC2",
    "HRAS": "P01112", "MAPK1": "P28482", "ESR1": "P03372", "CLU": "P10909",
    "BIN1": "O00499",
    "SORL1": "Q92673", "APBB1": "O00213", "ABL1": "P00519", "RXRA": "P19793",
}


# Hero fold-set defaults. `colab exec` runs this LOCAL script inside a Jupyter
# kernel and does NOT forward `-- <argv>`, so the reproducible hero targets are
# baked here as the argparse defaults (override locally with real CLI args when
# running outside colab exec). SMILES are canonical PubChem strings.
_NILOTINIB_SMILES = ("CC1=C(C=C(C=C1)C(=O)NC2=CC(=CC(=C2)C(F)(F)F)N3C=C(N=C3)C)"
                     "NC4=NC=CC(=N4)C5=CN=CC=C5")
_BEXAROTENE_SMILES = "CC1=CC2=C(C=C1C(=C)C3=CC=C(C=C3)C(=O)O)C(CCC2(C)C)(C)C"
HERO_COMPLEXES = "APP:APOE,APP:BACE1,APP:MAPT"
HERO_LIGANDS = (f"ABL1:Nilotinib:{_NILOTINIB_SMILES},"
                f"RXRA:Bexarotene:{_BEXAROTENE_SMILES}")


def sh(cmd, check=True):
    print(f"    $ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def install_deps():
    """Install boltz on top of Colab's working CUDA torch (frozen inference)."""
    print("[deps] installing boltz (keeping Colab CUDA torch) ...", flush=True)
    sh("pip install -q boltz")
    print("[deps] done.", flush=True)


def fetch_sequence(gene):
    """Keyless UniProt FASTA -> bare amino-acid sequence. None on failure."""
    import requests
    acc = AD_PROTEIN_MAP.get(gene.upper())
    if acc is None:
        print(f"[seq] {gene}: not an AD target accession -- skipping", flush=True)
        return None
    try:
        resp = requests.get(
            f"https://rest.uniprot.org/uniprotkb/{acc}.fasta", timeout=30)
        if resp.status_code != 200:
            print(f"[seq] {gene} ({acc}): HTTP {resp.status_code}", flush=True)
            return None
        seq = "".join(ln for ln in resp.text.splitlines() if not ln.startswith(">"))
        return seq or None
    except Exception as exc:  # noqa: BLE001
        print(f"[seq] {gene} fetch failed: {exc!r}", flush=True)
        return None


def build_yaml(entities, affinity=False):
    """Render a Boltz-2 input YAML. entities: list of ('protein'|'ligand', str)."""
    lines = ["version: 1", "sequences:"]
    letters = iter(string.ascii_uppercase)
    ligand_chain = ""
    for kind, seq in entities:
        cid = next(letters)
        if kind == "protein":
            lines += ["  - protein:", f"      id: {cid}", f"      sequence: {seq}"]
        else:
            ligand_chain = cid
            lines += ["  - ligand:", f"      id: {cid}", f"      smiles: '{seq}'"]
    if affinity and ligand_chain:
        lines += ["properties:", "  - affinity:", f"      binder: {ligand_chain}"]
    return "\n".join(lines) + "\n"


def parse_outputs(out_dir):
    """(confidence_dict, affinity_dict|None) from a Boltz output tree, else None."""
    out = Path(out_dir)
    conf_files = sorted(out.rglob("confidence_*.json"))
    aff_files = sorted(out.rglob("affinity_*.json"))
    conf = None
    if conf_files:
        try:
            conf = json.loads(conf_files[0].read_text())
        except Exception:
            conf = None
    aff = None
    if aff_files:
        try:
            aff = json.loads(aff_files[0].read_text())
        except Exception:
            aff = None
    if conf is None and aff is None:
        return None
    return conf, aff


def run_boltz(entities, affinity=False, workroot="boltz_work"):
    """Run `boltz predict` for one job; return (conf, aff) or None on any failure.

    Captures stdout+stderr so that a failing job surfaces Boltz's actual error
    (last lines) in the log — otherwise a returncode-1 is undiagnosable remotely.
    """
    base = Path(tempfile.mkdtemp(prefix="job_", dir=workroot))
    yaml_path = base / "input.yaml"
    yaml_path.write_text(build_yaml(entities, affinity=affinity))
    out_dir = base / "out"
    # --no_kernels: use the pure-PyTorch triangular-multiplication path instead of
    # the optional NVIDIA cuequivariance_torch accelerated kernel, which is not
    # installed on Colab's Boltz build (ModuleNotFoundError: cuequivariance_torch).
    # Slightly slower, fully correct — the folds fail without it.
    cmd = (f"boltz predict '{yaml_path}' --out_dir '{out_dir}' "
           f"--use_msa_server --no_kernels")
    # STREAM boltz's output (do NOT capture): a long fold that runs silently starves
    # the `colab exec` websocket of activity and it drops (frame header: EOF) before
    # the job finishes. Streaming its progress keeps the socket alive; any error
    # traceback still lands in the log. (--no_kernels avoids the missing
    # cuequivariance_torch accelerated kernel.)
    try:
        rc = sh(cmd, check=False).returncode
    except Exception as exc:  # noqa: BLE001
        print(f"[boltz] run raised: {exc!r}", flush=True)
        return None
    if rc != 0:
        print(f"[boltz] returncode {rc} -- omitting this job (no fabricated values)",
              flush=True)
        return None
    return parse_outputs(out_dir)


def emit_durable(path):
    """gzip+base64 the result JSON to stdout so a runtime drop can't lose it."""
    blob = base64.b64encode(gzip.compress(open(path, "rb").read())).decode()
    print("===JSONGZ_START===", flush=True)
    print(blob, flush=True)
    print("===JSONGZ_END===", flush=True)


def _complex_key(a, b):
    return "|".join(sorted((a.upper(), b.upper())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complexes", default=HERO_COMPLEXES,
                    help="comma list of protein pairs 'A:B' (AD gene symbols), "
                         "e.g. 'APP:MAPT,APP:BACE1,APOE:CLU'")
    ap.add_argument("--ligands", default=HERO_LIGANDS,
                    help="comma list of 'GENE:ligand_id:SMILES' target+ligand jobs")
    ap.add_argument("--out", default="boltz_targeting.json")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N successfully-folded jobs (0 = all; smoke test)")
    ap.add_argument("--skip-install", action="store_true")
    # parse_known_args (not parse_args): under `colab exec` the script runs inside a
    # Jupyter kernel that injects its own `-f <kernel.json>` into argv — ignore it.
    args, _unknown = ap.parse_known_args()

    if not args.skip_install:
        install_deps()

    os.makedirs("boltz_work", exist_ok=True)

    # Assemble the output in the EXACT schema of data/boltz_snapshot.json.
    result = {
        "_comment": ("PRECOMPUTED Boltz-2 results from a REAL Colab GPU run "
                     "(scripts/boltz_fold_colab.py). Confidence + affinity scalars "
                     "only; coordinates never committed."),
        "_provenance": {
            "model": "Boltz-2", "license": "MIT",
            "predictor_class": "AlphaFold3-class open structure + affinity predictor",
            "producer": "scripts/boltz_fold_colab.py (Colab GPU)",
            "captured": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        },
        "complexes": {},
        "affinities": {},
    }
    n_done = 0

    # --- protein-protein complexes ---
    complexes = [p for p in (x.strip() for x in args.complexes.split(",")) if p]
    for spec in complexes:
        if ":" not in spec:
            continue
        a, b = (s.strip().upper() for s in spec.split(":", 1))
        print(f"\n[job] complex {a}:{b}", flush=True)
        sa, sb = fetch_sequence(a), fetch_sequence(b)
        if not sa or not sb:
            print(f"[job] {a}:{b} missing sequence -- omitting", flush=True)
            continue
        out = run_boltz([("protein", sa), ("protein", sb)])
        if out is None:
            continue
        conf, _ = out
        if not conf:
            continue
        result["complexes"][_complex_key(a, b)] = {
            "gene_a": a, "gene_b": b,
            "iptm": conf.get("iptm"), "ptm": conf.get("ptm"),
            "pae": conf.get("pae"),
            "confidence_score": conf.get("confidence_score"),
        }
        n_done += 1
        json.dump(result, open(args.out, "w"), indent=2)
        print(f"[job] {a}:{b} -> iptm={conf.get('iptm')} ptm={conf.get('ptm')}",
              flush=True)
        emit_durable(args.out)  # incremental: survive a mid-run websocket drop
        if args.limit and n_done >= args.limit:
            break

    # --- target + ligand affinity jobs ---
    if not (args.limit and n_done >= args.limit):
        ligands = [p for p in (x.strip() for x in args.ligands.split(",")) if p]
        for spec in ligands:
            parts = spec.split(":", 2)
            if len(parts) != 3:
                continue
            gene, lid, smiles = (parts[0].strip().upper(), parts[1].strip(),
                                 parts[2].strip())
            print(f"\n[job] affinity {gene} + {lid}", flush=True)
            sg = fetch_sequence(gene)
            if not sg or not smiles:
                continue
            out = run_boltz([("protein", sg), ("ligand", smiles)], affinity=True)
            if out is None:
                continue
            conf, aff = out
            result["affinities"][f"{gene}::{lid}"] = {
                "gene_a": gene, "ligand_id": lid, "ligand_smiles": smiles,
                "iptm": (conf or {}).get("iptm"),
                "ptm": (conf or {}).get("ptm"),
                "confidence_score": (conf or {}).get("confidence_score"),
                "binding_affinity": (aff or {}).get("affinity_pred_value"),
                "binding_probability": (aff or {}).get("affinity_probability_binary"),
            }
            n_done += 1
            json.dump(result, open(args.out, "w"), indent=2)
            print(f"[job] {gene}+{lid} -> affinity="
                  f"{(aff or {}).get('affinity_pred_value')}", flush=True)
            emit_durable(args.out)  # incremental: survive a mid-run websocket drop
            if args.limit and n_done >= args.limit:
                break

    json.dump(result, open(args.out, "w"), indent=2)
    print(f"\n[done] wrote {args.out}: {len(result['complexes'])} complexes, "
          f"{len(result['affinities'])} affinities", flush=True)
    if n_done == 0:
        print("[warn] no jobs succeeded -- nothing to commit (honest empty result).",
              flush=True)
    emit_durable(args.out)


if __name__ == "__main__":
    main()
