"""Deterministic brain-region extractor: free-text hypothesis -> region coordinate.

Claude stays a pure orchestrator — this is a keyword/synonym match, and a region is
returned ONLY if it resolves to a real ``emb_*`` column in the cohort's
``df.attrs['region_columns']``. A region the cohort has no ROI column for is dropped
to ``("", [])``, so the number is always a real column's real AUROC, never invented.

The matcher covers the full dynamic Desikan-Killiany cortical + subcortical region
set the ETL emits: a small hand-curated alias table (``_ALIASES``) supplies
word-boundary-tolerant patterns and clinical abbreviations for the multi-word
anatomical names, and any cohort region NOT in that table falls back to a pattern
built from its bare slug. Longest / most-specific match wins, so multi-word regions
("inferior temporal", "posterior cingulate") and the group alias ("medial temporal")
beat bare "temporal" / "parietal".
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd

#: Curated slug -> regex. Word-boundary tolerant (optional -/space between the
#: CamelCase word parts the slug flattened) plus common clinical abbreviations.
#: Any cohort region NOT here falls back to a pattern built from its bare slug.
_ALIASES: dict[str, str] = {
    # --- subcortical / medial-temporal structures ---
    "hippocampus": r"hippocamp",
    "entorhinal": r"entorhinal|\bERC\b|\bEC\b",
    "parahippocampal": r"para[-\s]?hippocampal",
    "amygdala": r"amygdal",
    "thalamus": r"thalam",
    "putamen": r"putamen",
    "pallidum": r"pallid",
    "caudate": r"\bcaudate\b",
    "accumbensarea": r"accumbens",
    "ventraldc": r"ventral[-\s]?dc|ventral di?encephalon",
    "cerebellumcortex": r"cerebell",
    "lateralventricle": r"lateral[-\s]?ventricle",
    "inferiorlateralventricle": r"inferior[-\s]?lateral[-\s]?ventricle|temporal horn",
    # --- temporal lobe parcels ---
    "middletemporal": r"middle[-\s]?temporal|mid[-\s]?temporal|\bMTG\b",
    "inferiortemporal": r"inferior[-\s]?temporal|\bITG\b",
    "superiortemporal": r"superior[-\s]?temporal|\bSTG\b",
    "transversetemporal": r"transverse[-\s]?temporal|heschl",
    "temporalpole": r"temporal[-\s]?pole",
    "bankssts": r"banks?[-\s]?sts|\bbankssts\b|superior temporal sulcus",
    "fusiform": r"fusiform",
    # --- parietal lobe parcels ---
    "inferiorparietal": r"inferior[-\s]?parietal|\bIPL\b",
    "superiorparietal": r"superior[-\s]?parietal|\bSPL\b",
    "supramarginal": r"supra[-\s]?marginal",
    "postcentral": r"post[-\s]?central",
    "precuneus": r"precuneus",
    # --- occipital lobe parcels ---
    "lateraloccipital": r"lateral[-\s]?occipital",
    "pericalcarine": r"pericalcarine|calcarine",
    "cuneus": r"\bcuneus\b",
    "lingual": r"lingual",
    # --- frontal lobe parcels ---
    "superiorfrontal": r"superior[-\s]?frontal",
    "rostralmiddlefrontal": r"rostral[-\s]?middle[-\s]?frontal",
    "caudalmiddlefrontal": r"caudal[-\s]?middle[-\s]?frontal",
    "precentral": r"pre[-\s]?central",
    "paracentral": r"paracentral",
    "frontalpole": r"frontal[-\s]?pole",
    "lateralorbitofrontal": r"lateral[-\s]?orbito[-\s]?frontal",
    "medialorbitofrontal": r"medial[-\s]?orbito[-\s]?frontal",
    "parsopercularis": r"pars[-\s]?opercularis",
    "parsorbitalis": r"pars[-\s]?orbitalis",
    "parstriangularis": r"pars[-\s]?triangularis",
    # --- cingulate parcels ---
    "posteriorcingulate": r"posterior[-\s]?cingulate|\bPCC\b",
    "caudalanteriorcingulate": r"caudal[-\s]?anterior[-\s]?cingulate",
    "rostralanteriorcingulate": r"rostral[-\s]?anterior[-\s]?cingulate|\brACC\b",
    "isthmuscingulate": r"isthmus[-\s]?cingulate|isthmus of (the )?cingulate",
    # --- other cortical parcels ---
    "insula": r"\binsula\b|insular",
}

#: Anatomical GROUP -> regex. Checked AFTER every specific region so a multi-word
#: region beats a bare lobe; ``medial_temporal`` precedes bare temporal/parietal.
_GROUP_ALIASES: list[tuple[str, str]] = [
    ("medial_temporal", r"medial[-\s]?temporal(\s+lobe)?|\bMTL\b"),
    ("temporal", r"temporal"),
    ("parietal", r"parietal"),
]


def _bare_pattern(slug: str) -> str:
    """Fallback pattern for a cohort region not in ``_ALIASES``: the literal slug
    with optional separators tolerated between characters is overkill, so match the
    exact slug token (bounded)."""
    return r"\b" + re.escape(slug) + r"\b"


def _ordered_region_synonyms() -> list[tuple[str, re.Pattern]]:
    """Specific-region (slug, compiled-pattern) pairs, longest slug first so a more
    specific multi-word region is tested before a shorter overlapping one."""
    slugs = sorted(_ALIASES, key=len, reverse=True)
    return [(s, re.compile(_ALIASES[s], re.IGNORECASE)) for s in slugs]


_REGION_SYNONYMS = _ordered_region_synonyms()
_GROUP_SYNONYMS = [(g, re.compile(p, re.IGNORECASE)) for g, p in _GROUP_ALIASES]


def extract_region(text: str, df: Optional[pd.DataFrame]) -> tuple[str, list[str]]:
    """Return ``(region_slug, emb_columns)`` if the text names a region the cohort
    can restrict to, else ``("", [])``. Requires ``df.attrs['region_columns']``."""
    if df is None:
        return "", []
    region_map = df.attrs.get("region_columns") or {}
    if not region_map:
        return "", []
    low = text or ""

    # 1) Specific regions, longest slug first (curated aliases).
    for slug, pat in _REGION_SYNONYMS:
        if slug in region_map and pat.search(low):
            return slug, list(region_map[slug])

    # 2) Any cohort region without a curated alias -> bare-slug fallback, longest
    #    slug first so specific names beat shorter overlapping ones.
    covered = set(_ALIASES)
    fallbacks = sorted(
        (s for s in region_map if s not in covered and s not in {g for g, _ in _GROUP_ALIASES}),
        key=len, reverse=True,
    )
    for slug in fallbacks:
        if re.search(_bare_pattern(slug), low, re.IGNORECASE):
            return slug, list(region_map[slug])

    # 3) Anatomical groups last (medial temporal before bare temporal/parietal).
    for group, pat in _GROUP_SYNONYMS:
        if group in region_map and pat.search(low):
            return group, list(region_map[group])

    return "", []
