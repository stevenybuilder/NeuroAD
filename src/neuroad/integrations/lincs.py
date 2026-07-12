"""
lincs — LINCS L1000 connectivity as a PERTURBATIONAL efficacy-proxy signal.

Every other discovery-half signal (PI4AD, Open Targets, STRING) is an ASSOCIATION
signal: it says a gene is *linked* to AD. None of them test *efficacy* — whether
perturbing the gene would move the disease state. This adapter adds a mechanistically
orthogonal axis: query the LINCS L1000 catalogue (keyless SigCom LINCS API) for genetic
perturbations whose transcriptional signature REVERSES an Alzheimer brain signature
(the perturbation pushes the AD-up genes down and the AD-down genes up). A gene whose
knockout reverses the AD signature is an efficacy-relevant candidate — you would want to
inhibit it.

HONEST CAVEATS (loud, by design — mirrored into every report):
  * This is an efficacy *proxy*, NOT efficacy. The L1000 signatures are measured in
    (mostly cancer) cell lines — HT29, MCF7, ES2, AGS, … — not neurons or microglia,
    so "reverses an AD transcriptomic signature in a cancer line" is a weak surrogate
    for neuronal/glial AD efficacy. Treat a hit as a hypothesis, not a claim.
  * The AD signature itself is a curated consensus (up = neuroinflammation / reactive
    glia / complement; down = synaptic / neuronal), an approximation of the real,
    region- and cell-type-specific AD transcriptome.

Offline-first, exactly like ``opentargets.py``: default (``prefer_offline=False``) hits
the live SigCom LINCS REST API and, on ANY failure, degrades to the bundled snapshot —
never raising, always provenance-stamped. The committed snapshot keeps the demo path
deterministic and network-free.

Live API (keyless): https://maayanlab.cloud/sigcom-lincs (Ma'ayan Lab / SigCom LINCS,
Evangelista et al., Nucleic Acids Res 2022;50:W697). Databases queried: ``l1000_xpr``
(CRISPR knockout) and ``l1000_shRNA`` (shRNA knockdown) — both loss-of-function, so a
reverser => "silencing this gene reverses AD" => an inhibition target.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

_log = logging.getLogger("neuroad.integrations.lincs")

_META_BASE = "https://maayanlab.cloud/sigcom-lincs/metadata-api"
_DATA_BASE = "https://maayanlab.cloud/sigcom-lincs/data-api/api/v1"
_HTTP_TIMEOUT = 45
#: Loss-of-function genetic-perturbation databases (a reverser => inhibition target).
_LOF_DATABASES = ("l1000_xpr", "l1000_shRNA")

_SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "data",
                              "lincs_ad_reversal_snapshot.json")

_BELLENGUEZ = "Bellenguez 2022, Nat Genet 54:412"  # for cross-ref only

# ---------------------------------------------------------------------------
# Curated consensus Alzheimer brain transcriptomic signature (up / down).
# Cited to canonical AD transcriptomics. Deliberately compact and biased toward
# broadly-measured genes so it resolves well into the L1000 landmark space.
# ---------------------------------------------------------------------------
#: UP in AD brain — reactive gliosis, microglial activation, complement, MHC.
AD_SIGNATURE_UP: tuple[str, ...] = (
    "GFAP", "TYROBP", "C1QA", "C1QB", "C1QC", "C3", "CD68", "AIF1", "ITGAX",
    "CTSS", "CTSB", "CTSD", "LAPTM5", "FCER1G", "SERPINA3", "VIM", "CHI3L1",
    "HLA-DRA", "B2M", "CD74", "TREM2", "CST7", "SPP1", "GPNMB",
)
#: DOWN in AD brain — synaptic / neuronal / vesicle-cycling loss.
AD_SIGNATURE_DOWN: tuple[str, ...] = (
    "SNAP25", "SYT1", "SYN1", "SYP", "NEFL", "NEFM", "NRGN", "GAP43", "CAMK2A",
    "VSNL1", "RAB3A", "GABRA1", "GRIN1", "DLG4", "SYN2", "STMN2", "NEFH",
    "RAB3C", "ATP2B2", "CALB1", "PVALB", "ENO2", "MAP2", "TUBB3",
)
_SIGNATURE_CITATION = (
    "Curated consensus AD brain transcriptomic signature (up = reactive glia / "
    "microglial activation / complement, down = synaptic / neuronal), after "
    "Zhang et al. 2013 (Cell 153:707; TYROBP causal network), Mathys et al. 2019 "
    "(Nature 570:332; single-cell AD), and Mostafavi et al. 2018 (Nat Neurosci "
    "21:811). APPROXIMATION — not a single-study DE table."
)


@dataclass
class GeneEfficacyProxy:
    """One gene's LINCS reversal-efficacy proxy, fully auditable."""
    gene: str
    reversal_score: float          # aggregated best reverser strength (>=0)
    n_signatures: int              # supporting LoF reverser signatures
    best_database: str             # l1000_xpr | l1000_shRNA
    best_cell_line: str
    source: str                    # "live" | "offline_snapshot"

    def to_dict(self) -> dict:
        return {
            "gene": self.gene, "reversal_score": round(self.reversal_score, 4),
            "n_signatures": self.n_signatures, "best_database": self.best_database,
            "best_cell_line": self.best_cell_line, "source": self.source,
        }


@dataclass
class LincsClient:
    """Offline-first SigCom LINCS adapter (keyless). Never raises on the network."""
    prefer_offline: bool = False
    timeout: int = _HTTP_TIMEOUT
    meta_base: str = _META_BASE
    data_base: str = _DATA_BASE
    _snapshot: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._snapshot = _load_snapshot()

    # -- low-level REST ----------------------------------------------------
    def _post(self, url: str, body: dict) -> Optional[object]:
        try:
            import requests
            resp = requests.post(url, json=body, timeout=self.timeout)
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception:  # noqa: BLE001 — any failure degrades to snapshot
            return None

    def resolve_entities(self, symbols: list[str]) -> dict[str, str]:
        """Gene symbols -> SigCom entity UUIDs (only the ones that resolve)."""
        if not symbols:
            return {}
        body = {"filter": {"where": {"meta.symbol": {"inq": list(symbols)}},
                           "fields": ["id", "meta.symbol"]}}
        data = self._post(self.meta_base + "/entities/find", body)
        if not isinstance(data, list):
            return {}
        out = {}
        for e in data:
            sym = ((e.get("meta") or {}).get("symbol") if isinstance(e, dict) else None)
            if sym and e.get("id"):
                out[sym] = e["id"]
        return out

    def _reverser_signatures(self, up_uuids: list[str], down_uuids: list[str],
                             database: str, limit: int) -> list[dict]:
        """Top reverser signatures for the up/down query in one LoF database."""
        if not up_uuids or not down_uuids:
            return []
        body = {"up_entities": up_uuids, "down_entities": down_uuids,
                "limit": int(limit), "database": database}
        data = self._post(self.data_base + "/enrich/ranktwosided", body)
        if not isinstance(data, dict):
            return []
        results = data.get("results") or []
        revs = [r for r in results if isinstance(r, dict)
                and r.get("type") == "reversers" and r.get("uuid")]
        # most-negative z-sum = strongest reverser
        revs.sort(key=lambda r: r.get("z-sum", 0.0))
        return revs

    def _signature_perts(self, uuids: list[str]) -> dict[str, dict]:
        """Signature UUID -> {pert_name, cell_line, pert_type} metadata."""
        if not uuids:
            return {}
        body = {"filter": {"where": {"id": {"inq": uuids}},
                           "fields": ["id", "meta.pert_name", "meta.cell_line",
                                      "meta.pert_type"]}}
        data = self._post(self.meta_base + "/signatures/find", body)
        if not isinstance(data, list):
            return {}
        out = {}
        for s in data:
            if not isinstance(s, dict) or not s.get("id"):
                continue
            m = s.get("meta") or {}
            out[s["id"]] = {"pert_name": m.get("pert_name"),
                            "cell_line": m.get("cell_line"),
                            "pert_type": m.get("pert_type")}
        return out

    # -- public: the per-gene efficacy proxy -------------------------------
    def ad_reversal_efficacy(self, *, limit: int = 500
                             ) -> dict[str, GeneEfficacyProxy]:
        """Build the per-gene AD-reversal efficacy proxy.

        Queries each LoF database for the top ``limit`` reversers of the curated AD
        signature, maps each reverser signature to its perturbed gene, and scores
        each gene by its BEST (strongest) reverser strength |z-sum| across all its
        supporting signatures. Returns ``{}`` on total live failure unless a
        snapshot is available (offline path)."""
        if self.prefer_offline:
            return self._from_snapshot()
        up = self.resolve_entities(list(AD_SIGNATURE_UP))
        down = self.resolve_entities(list(AD_SIGNATURE_DOWN))
        if not up or not down:
            return self._from_snapshot()
        agg: dict[str, dict] = {}
        for db in _LOF_DATABASES:
            revs = self._reverser_signatures(list(up.values()), list(down.values()),
                                             db, limit)
            metas = self._signature_perts([r["uuid"] for r in revs])
            for r in revs:
                m = metas.get(r["uuid"]) or {}
                gene = (m.get("pert_name") or "").strip().upper()
                if not gene:
                    continue
                strength = abs(float(r.get("z-sum", 0.0)))
                cur = agg.get(gene)
                if cur is None or strength > cur["reversal_score"]:
                    agg[gene] = {"reversal_score": strength, "best_database": db,
                                 "best_cell_line": m.get("cell_line") or "",
                                 "n_signatures": (cur["n_signatures"] + 1) if cur else 1}
                elif cur:
                    cur["n_signatures"] += 1
        if not agg:
            return self._from_snapshot()
        return {g: GeneEfficacyProxy(gene=g, source="live", **v)
                for g, v in agg.items()}

    def reversal_universe(self, *, limit: int = 500) -> dict[str, float]:
        """Signed connectivity universe for HONEST validation of the efficacy axis.

        Unlike ``ad_reversal_efficacy`` (reversers only, for use as a ranking
        signal), this returns EVERY perturbed gene that appears in the top-``limit``
        results — reversers scored ``+|z-sum|`` (KO reverses AD) and mimickers scored
        ``-|z-sum|`` (KO worsens AD) — giving a real background so an AUC against a
        gold set means something. Live only (returns ``{}`` on failure); validation
        callers stamp provenance from the client's live/offline state."""
        if self.prefer_offline:
            snap = self._snapshot.get("validation_universe")
            return {str(g).upper(): float(s) for g, s in snap.items()} if snap else {}
        up = self.resolve_entities(list(AD_SIGNATURE_UP))
        down = self.resolve_entities(list(AD_SIGNATURE_DOWN))
        if not up or not down:
            return {}
        best: dict[str, float] = {}
        for db in _LOF_DATABASES:
            body = {"up_entities": list(up.values()),
                    "down_entities": list(down.values()),
                    "limit": int(limit), "database": db}
            data = self._post(self.data_base + "/enrich/ranktwosided", body)
            results = (data or {}).get("results") if isinstance(data, dict) else None
            if not results:
                continue
            metas = self._signature_perts([r["uuid"] for r in results
                                           if isinstance(r, dict) and r.get("uuid")])
            for r in results:
                if not isinstance(r, dict):
                    continue
                m = metas.get(r.get("uuid")) or {}
                gene = (m.get("pert_name") or "").strip().upper()
                if not gene:
                    continue
                mag = abs(float(r.get("z-sum", 0.0)))
                signed = mag if r.get("type") == "reversers" else -mag
                # keep the most extreme (largest |signed|) score per gene
                if gene not in best or abs(signed) > abs(best[gene]):
                    best[gene] = signed
        return best

    def _from_snapshot(self) -> dict[str, GeneEfficacyProxy]:
        rows = self._snapshot.get("genes", [])
        out = {}
        for r in rows:
            g = str(r.get("gene", "")).upper()
            if not g:
                continue
            out[g] = GeneEfficacyProxy(
                gene=g, reversal_score=float(r.get("reversal_score", 0.0)),
                n_signatures=int(r.get("n_signatures", 0)),
                best_database=str(r.get("best_database", "")),
                best_cell_line=str(r.get("best_cell_line", "")),
                source="offline_snapshot")
        return out


def _load_snapshot() -> dict:
    try:
        with open(_SNAPSHOT_PATH) as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return {"genes": []}


def efficacy_proxy_map(*, prefer_offline: bool = False, limit: int = 500
                       ) -> dict[str, float]:
    """Convenience: {GENE: reversal_score} for use as a ranking signal."""
    client = LincsClient(prefer_offline=prefer_offline)
    return {g: p.reversal_score
            for g, p in client.ad_reversal_efficacy(limit=limit).items()}
