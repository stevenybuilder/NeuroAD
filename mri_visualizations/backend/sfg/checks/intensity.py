"""Check 1.3 - Intensity normalization consistency (deterministic root of Problem A).

MRI intensities have no fixed units, so raw T1 from different scanners sit on
different scales. A model trained across sites can then learn the *scanner* from
raw intensity instead of biology - the silent scanner confound. This check
measures the cross-scanner spread of robust intensity statistics, shows the raw
per-scanner distributions diverging, and shows that a simple percentile
normalization collapses them - quantifying the confound and its deterministic fix.

Cohort-level: it compares scans across scanners, so it implements run_cohort.
"""

from __future__ import annotations

import numpy as np

from ..flags import Flag
from ..imaging import foreground_mask, load
from ..registry import Scan
from ..resources import ResourceStore
from .base import register

_BINS = 60
_COLORS = {"Guy's": "#4c8dff", "Hammersmith": "#f0a13a", "IOP": "#46c98b"}


class IntensityConsistencyCheck:
    check_id = "1.3.intensity_consistency"
    description = "Measures cross-scanner intensity divergence (scanner confound) and its normalization fix."

    def run_cohort(self, scans: list[Scan], store: ResourceStore) -> list[Flag]:
        ixi = [s for s in scans if s.source in ("ixi", "adni")]
        if len({s.site for s in ixi}) < 2:
            return []  # need multiple scanners to talk about a scanner confound

        raw_by_site: dict[str, list[np.ndarray]] = {}
        norm_by_site: dict[str, list[np.ndarray]] = {}
        medians_by_site: dict[str, list[float]] = {}
        raw_edges = np.linspace(0, 1, _BINS + 1)
        norm_edges = np.linspace(0, 2, _BINS + 1)

        for s in ixi:
            data, _aff, _img = load(s)
            fg = data[foreground_mask(data)]
            if fg.size == 0:
                continue
            p99 = np.percentile(fg, 99)
            med = float(np.median(fg))
            raw_h = np.histogram(fg / max(p99, 1e-6), bins=raw_edges, density=True)[0]
            norm_h = np.histogram(fg / max(med, 1e-6), bins=norm_edges, density=True)[0]
            raw_by_site.setdefault(s.site, []).append(raw_h)
            norm_by_site.setdefault(s.site, []).append(norm_h)
            medians_by_site.setdefault(s.site, []).append(med)

        sites = sorted(raw_by_site)
        raw_series = [self._series(site, raw_by_site[site]) for site in sites]
        norm_series = [self._series(site, norm_by_site[site]) for site in sites]

        # Confound magnitude: coefficient of variation of the per-scanner median.
        site_medians = {k: float(np.mean(v)) for k, v in medians_by_site.items()}
        cov_raw = self._cov(list(site_medians.values()))

        flag = Flag(
            check_id=self.check_id,
            scan_id=ixi[0].scan_id,
            severity="warn" if cov_raw > 0.10 else "info",
            explanation=(
                f"Raw T1 intensities differ systematically across {len(sites)} scanners "
                f"(per-scanner median CoV {cov_raw*100:.0f}%). Un-normalized, this is a scanner "
                "signature a model can exploit instead of biology (Problem A). Percentile "
                "normalization rescales every scan to a common intensity range, collapsing the "
                "per-scanner offset - apply it before any cross-site analysis."
            ),
            location=None,
            extra={
                "scanner_medians": {k: round(v, 1) for k, v in site_medians.items()},
                "median_cov_raw_pct": round(cov_raw * 100, 1),
                "histograms": [
                    {"title": "Raw intensity by scanner (÷ per-scan p99)",
                     "edges": raw_edges.round(3).tolist(), "series": raw_series},
                    {"title": "After percentile normalization (÷ per-scan median)",
                     "edges": norm_edges.round(3).tolist(), "series": norm_series},
                ],
            },
        )
        return [flag]

    def _series(self, site: str, hists: list[np.ndarray]) -> dict:
        mean = np.mean(np.stack(hists), axis=0)
        return {"name": site, "counts": mean.round(4).tolist(), "color": _COLORS.get(site)}

    def _cov(self, values: list[float]) -> float:
        arr = np.asarray(values, dtype=float)
        return float(arr.std() / max(arr.mean(), 1e-6))


register(IntensityConsistencyCheck())
