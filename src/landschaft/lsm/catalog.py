# SPDX-License-Identifier: GPL-3.0-or-later
"""Catalogue of the landscape metrics (same names as landscapemetrics)."""

from __future__ import annotations

import functools
from collections.abc import Iterable

import pandas as pd

__all__ = ["list_metrics", "canonical_name", "canonical_family", "LEVELS", "TYPES", "STATS"]

LEVELS = ("patch", "class", "landscape")
TYPES = (
    "area and edge metric",
    "shape metric",
    "core area metric",
    "aggregation metric",
    "diversity metric",
    "complexity metric",
)

# (metric, name, type, level), as landscapemetrics::lsm_abbreviations_names.
_CATALOG = [
    ("area", "patch area", "area and edge metric", "patch"),
    ("cai", "core area index", "core area metric", "patch"),
    ("circle", "related circumscribing circle", "shape metric", "patch"),
    ("contig", "contiguity index", "shape metric", "patch"),
    ("core", "core area", "core area metric", "patch"),
    ("enn", "euclidean nearest neighbor distance", "aggregation metric", "patch"),
    ("frac", "fractal dimension index", "shape metric", "patch"),
    ("gyrate", "radius of gyration", "area and edge metric", "patch"),
    ("ncore", "number of core areas", "core area metric", "patch"),
    ("para", "perimeter-area ratio", "shape metric", "patch"),
    ("perim", "patch perimeter", "area and edge metric", "patch"),
    ("shape", "shape index", "shape metric", "patch"),
    ("ai", "aggregation index", "aggregation metric", "class"),
    ("area_cv", "patch area", "area and edge metric", "class"),
    ("area_mn", "patch area", "area and edge metric", "class"),
    ("area_sd", "patch area", "area and edge metric", "class"),
    ("ca", "total (class) area", "area and edge metric", "class"),
    ("cai_cv", "core area index", "core area metric", "class"),
    ("cai_mn", "core area index", "core area metric", "class"),
    ("cai_sd", "core area index", "core area metric", "class"),
    ("circle_cv", "related circumscribing circle", "shape metric", "class"),
    ("circle_mn", "related circumscribing circle", "shape metric", "class"),
    ("circle_sd", "related circumscribing circle", "shape metric", "class"),
    ("clumpy", "clumpiness index", "aggregation metric", "class"),
    ("cohesion", "patch cohesion index", "aggregation metric", "class"),
    ("contig_cv", "contiguity index", "shape metric", "class"),
    ("contig_mn", "contiguity index", "shape metric", "class"),
    ("contig_sd", "contiguity index", "shape metric", "class"),
    ("core_cv", "core area", "core area metric", "class"),
    ("core_mn", "core area", "core area metric", "class"),
    ("core_sd", "core area", "core area metric", "class"),
    ("cpland", "core area percentage of landscape", "core area metric", "class"),
    ("dcad", "disjunct core area density", "core area metric", "class"),
    ("dcore_cv", "disjunct core area", "core area metric", "class"),
    ("dcore_mn", "disjunct core area", "core area metric", "class"),
    ("dcore_sd", "disjunct core area", "core area metric", "class"),
    ("division", "division index", "aggregation metric", "class"),
    ("ed", "edge density", "area and edge metric", "class"),
    ("enn_cv", "euclidean nearest neighbor distance", "aggregation metric", "class"),
    ("enn_mn", "euclidean nearest neighbor distance", "aggregation metric", "class"),
    ("enn_sd", "euclidean nearest neighbor distance", "aggregation metric", "class"),
    ("frac_cv", "fractal dimension index", "shape metric", "class"),
    ("frac_mn", "fractal dimension index", "shape metric", "class"),
    ("frac_sd", "fractal dimension index", "shape metric", "class"),
    ("gyrate_cv", "radius of gyration", "area and edge metric", "class"),
    ("gyrate_mn", "radius of gyration", "area and edge metric", "class"),
    ("gyrate_sd", "radius of gyration", "area and edge metric", "class"),
    ("iji", "interspersion and juxtaposition index", "aggregation metric", "class"),
    ("lpi", "largest patch index", "area and edge metric", "class"),
    ("lsi", "landscape shape index", "aggregation metric", "class"),
    ("mesh", "effective mesh size", "aggregation metric", "class"),
    ("ndca", "number of disjunct core areas", "core area metric", "class"),
    ("nlsi", "normalized landscape shape index", "aggregation metric", "class"),
    ("np", "number of patches", "aggregation metric", "class"),
    ("pafrac", "perimeter-area fractal dimension", "shape metric", "class"),
    ("para_cv", "perimeter-area ratio", "shape metric", "class"),
    ("para_mn", "perimeter-area ratio", "shape metric", "class"),
    ("para_sd", "perimeter-area ratio", "shape metric", "class"),
    ("pd", "patch density", "aggregation metric", "class"),
    ("pladj", "percentage of like adjacencies", "aggregation metric", "class"),
    ("pland", "percentage of landscape", "area and edge metric", "class"),
    ("shape_cv", "shape index", "shape metric", "class"),
    ("shape_mn", "shape index", "shape metric", "class"),
    ("shape_sd", "shape index", "shape metric", "class"),
    ("split", "splitting index", "aggregation metric", "class"),
    ("tca", "total core area", "core area metric", "class"),
    ("te", "total edge", "area and edge metric", "class"),
    ("ai", "aggregation index", "aggregation metric", "landscape"),
    ("area_cv", "patch area", "area and edge metric", "landscape"),
    ("area_mn", "patch area", "area and edge metric", "landscape"),
    ("area_sd", "patch area", "area and edge metric", "landscape"),
    ("cai_cv", "core area index", "core area metric", "landscape"),
    ("cai_mn", "core area index", "core area metric", "landscape"),
    ("cai_sd", "core area index", "core area metric", "landscape"),
    ("circle_cv", "related circumscribing circle", "shape metric", "landscape"),
    ("circle_mn", "related circumscribing circle", "shape metric", "landscape"),
    ("circle_sd", "related circumscribing circle", "shape metric", "landscape"),
    ("cohesion", "patch cohesion index", "aggregation metric", "landscape"),
    ("condent", "conditional entropy", "complexity metric", "landscape"),
    ("contag", "contagion index", "aggregation metric", "landscape"),
    ("contig_cv", "contiguity index", "shape metric", "landscape"),
    ("contig_mn", "contiguity index", "shape metric", "landscape"),
    ("contig_sd", "contiguity index", "shape metric", "landscape"),
    ("core_cv", "core area", "core area metric", "landscape"),
    ("core_mn", "core area", "core area metric", "landscape"),
    ("core_sd", "core area", "core area metric", "landscape"),
    ("dcad", "disjunct core area density", "core area metric", "landscape"),
    ("dcore_cv", "disjunct core area", "core area metric", "landscape"),
    ("dcore_mn", "disjunct core area", "core area metric", "landscape"),
    ("dcore_sd", "disjunct core area", "core area metric", "landscape"),
    ("division", "division index", "aggregation metric", "landscape"),
    ("ed", "edge density", "area and edge metric", "landscape"),
    ("enn_cv", "euclidean nearest neighbor distance", "aggregation metric", "landscape"),
    ("enn_mn", "euclidean nearest neighbor distance", "aggregation metric", "landscape"),
    ("enn_sd", "euclidean nearest neighbor distance", "aggregation metric", "landscape"),
    ("ent", "shannon entropy", "complexity metric", "landscape"),
    ("frac_cv", "fractal dimension index", "shape metric", "landscape"),
    ("frac_mn", "fractal dimension index", "shape metric", "landscape"),
    ("frac_sd", "fractal dimension index", "shape metric", "landscape"),
    ("gyrate_cv", "radius of gyration", "area and edge metric", "landscape"),
    ("gyrate_mn", "radius of gyration", "area and edge metric", "landscape"),
    ("gyrate_sd", "radius of gyration", "area and edge metric", "landscape"),
    ("iji", "interspersion and juxtaposition index", "aggregation metric", "landscape"),
    ("joinent", "joint entropy", "complexity metric", "landscape"),
    ("lpi", "largest patch index", "area and edge metric", "landscape"),
    ("lsi", "landscape shape index", "aggregation metric", "landscape"),
    ("mesh", "effective mesh size", "aggregation metric", "landscape"),
    ("msidi", "modified simpson's diversity index", "diversity metric", "landscape"),
    ("msiei", "modified simpson's evenness index", "diversity metric", "landscape"),
    ("mutinf", "mutual information", "complexity metric", "landscape"),
    ("ndca", "number of disjunct core areas", "core area metric", "landscape"),
    ("np", "number of patches", "aggregation metric", "landscape"),
    ("pafrac", "perimeter-area fractal dimension", "shape metric", "landscape"),
    ("para_cv", "perimeter-area ratio", "shape metric", "landscape"),
    ("para_mn", "perimeter-area ratio", "shape metric", "landscape"),
    ("para_sd", "perimeter-area ratio", "shape metric", "landscape"),
    ("pd", "patch density", "aggregation metric", "landscape"),
    ("pladj", "percentage of like adjacencies", "aggregation metric", "landscape"),
    ("pr", "patch richness", "diversity metric", "landscape"),
    ("prd", "patch richness density", "diversity metric", "landscape"),
    ("relmutinf", "relative mutual information", "complexity metric", "landscape"),
    ("rpr", "relative patch richness", "diversity metric", "landscape"),
    ("shape_cv", "shape index", "shape metric", "landscape"),
    ("shape_mn", "shape index", "shape metric", "landscape"),
    ("shape_sd", "shape index", "shape metric", "landscape"),
    ("shdi", "shannon's diversity index", "diversity metric", "landscape"),
    ("shei", "shannon's evenness index", "diversity metric", "landscape"),
    ("sidi", "simpson's diversity index", "diversity metric", "landscape"),
    ("siei", "simpson's evenness index", "diversity metric", "landscape"),
    ("split", "splitting index", "aggregation metric", "landscape"),
    ("ta", "total area", "area and edge metric", "landscape"),
    ("tca", "total core area", "core area metric", "landscape"),
    ("te", "total edge", "area and edge metric", "landscape"),
]

# Summary statistics of the patch metrics at the class and landscape level:
# landscapemetrics has mn, sd and cv; FRAGSTATS (and pylandstats) also the
# area-weighted mean, the median and the range.
STATS = ("mn", "sd", "cv", "am", "md", "ra")

# Patch metrics with summary statistics: (name, type). The perim statistics
# only exist in pylandstats; dcore is landscapemetrics' number of core areas
# per patch.
_DISTRIBUTIONS = {
    "area": ("patch area", "area and edge metric"),
    "cai": ("core area index", "core area metric"),
    "circle": ("related circumscribing circle", "shape metric"),
    "contig": ("contiguity index", "shape metric"),
    "core": ("core area", "core area metric"),
    "dcore": ("disjunct core area", "core area metric"),
    "enn": ("euclidean nearest neighbor distance", "aggregation metric"),
    "frac": ("fractal dimension index", "shape metric"),
    "gyrate": ("radius of gyration", "area and edge metric"),
    "para": ("perimeter-area ratio", "shape metric"),
    "perim": ("patch perimeter", "area and edge metric"),
    "shape": ("shape index", "shape metric"),
    "prox": ("proximity index", "aggregation metric"),
    "simi": ("similarity index", "aggregation metric"),
    "econ": ("edge contrast index", "contrast metric"),
}

# FRAGSTATS metrics that landscapemetrics does not have (besides the
# statistics above): (metric, name, type, level).
_FRAGSTATS = [
    ("prox", "proximity index", "aggregation metric", "patch"),
    ("simi", "similarity index", "aggregation metric", "patch"),
    ("econ", "edge contrast index", "contrast metric", "patch"),
    ("connect", "connectance index", "aggregation metric", "class"),
    ("cwed", "contrast-weighted edge density", "contrast metric", "class"),
    ("teci", "total edge contrast index", "contrast metric", "class"),
    ("connect", "connectance index", "aggregation metric", "landscape"),
    ("cwed", "contrast-weighted edge density", "contrast metric", "landscape"),
    ("teci", "total edge contrast index", "contrast metric", "landscape"),
]


def _extras() -> list[tuple[str, str, str, str]]:
    """Metrics beyond landscapemetrics (FRAGSTATS / pylandstats)."""
    known = {(m, lvl) for m, _, _, lvl in _CATALOG}
    rows = list(_FRAGSTATS)
    for level in ("class", "landscape"):
        for family, (name, kind) in _DISTRIBUTIONS.items():
            for stat in STATS:
                if (f"{family}_{stat}", level) not in known:
                    rows.append((f"{family}_{stat}", name, kind, level))
    return rows


_PREFIX = {"patch": "lsm_p_", "class": "lsm_c_", "landscape": "lsm_l_"}
_LEVEL_ORDER = {lvl: i for i, lvl in enumerate(LEVELS)}


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        ["level", "metric"], key=lambda s: s.map(_LEVEL_ORDER) if s.name == "level" else s
    ).reset_index(drop=True)


@functools.lru_cache(maxsize=None)
def _full_table() -> pd.DataFrame:
    rows = [(*r, False) for r in _CATALOG] + [(*r, True) for r in _extras()]
    df = pd.DataFrame(rows, columns=["metric", "name", "type", "level", "extra"])
    df["function_name"] = [_PREFIX[lvl] + m for m, lvl in zip(df["metric"], df["level"])]
    df["family"] = df["metric"].str.split("_").str[0]
    return _sorted(df)


def _table(extras: bool = False) -> pd.DataFrame:
    df = _full_table()
    return df if extras else df[~df["extra"]].reset_index(drop=True)


def _as_list(value) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(v) for v in value]
    raise TypeError(f"expected a string or a list of strings, got {value!r}")


def _subset(values: pd.Series, spec: list[str] | None, what: str) -> pd.Series:
    """Rows kept by one filter: all names, or positive or negative (``-x``) names."""
    if spec is None:
        return pd.Series(True, index=values.index)
    negative = [s.startswith("-") for s in spec]
    if any(negative) and not all(negative):
        raise ValueError(f"cannot mix positive and negative {what} names")
    if all(negative):
        return ~values.isin([s[1:] for s in spec])
    return values.isin(spec)


def select_metrics(level=None, metric=None, name=None, type=None, what=None, extras: bool = False) -> pd.DataFrame:
    """Catalogue rows chosen like landscapemetrics' ``list_lsm`` (plus the
    extras when ``extras``; extras named in ``what`` are always kept)."""
    table = _table(extras)
    what = _as_list(what)
    if what is not None:
        if any(w.startswith("-") for w in what):
            raise ValueError("negative names are not allowed in 'what'; use level/metric/name/type instead")
        levels = [w for w in what if w in LEVELS]
        rest = [w for w in what if w not in LEVELS]
        full = _full_table()
        unknown = set(rest) - set(full["function_name"])
        if unknown:
            raise ValueError(f"unknown metric function(s): {sorted(unknown)}; see list_metrics()")
        named = full[full["function_name"].isin(rest) & full["extra"]]
        if not named.empty:
            table = _sorted(pd.concat([table, named]).drop_duplicates("function_name"))
        keep = table["function_name"].isin(rest) | table["level"].isin(levels)
    else:
        level = _as_list(level)
        if level is not None:
            bad = {lv.lstrip("-") for lv in level} - set(LEVELS)
            if bad:
                raise ValueError(f"level must be one of {LEVELS}, got {sorted(bad)}")
        metric = _as_list(metric)
        if metric is not None:
            metric = [("-" if m.startswith("-") else "") + canonical_family(m.lstrip("-")) for m in metric]
        keep = (
            _subset(table["level"], level, "level")
            & _subset(table["family"], metric, "metric")
            & _subset(table["name"], _as_list(name), "name")
            & _subset(table["type"], _as_list(type), "type")
        )
    result = table[keep]
    if result.empty:
        raise ValueError("the selected metrics do not exist; see list_metrics()")
    return result.reset_index(drop=True)


def list_metrics(level=None, metric=None, name=None, type=None, what=None, *, extras: bool = False) -> pd.DataFrame:
    """Available landscape metrics.

    Filters work as in landscapemetrics' ``list_lsm``: ``level`` ("patch",
    "class", "landscape"), ``metric`` (family, e.g. ``"area"`` selects
    ``area``, ``area_mn``, ``area_sd`` and ``area_cv``), ``name`` and
    ``type`` (e.g. ``"aggregation metric"``) are combined, and names starting
    with ``-`` exclude. ``what`` lists levels and/or function names such as
    ``"lsm_c_ca"`` and overrides the other filters.

    ``extras=True`` adds the metrics landscapemetrics does not have
    (FRAGSTATS / pylandstats): the area-weighted mean ``_am``, the median
    ``_md`` and the range ``_ra`` of every patch metric, the statistics of
    ``perim``, and FRAGSTATS' proximity (``prox``), similarity (``simi``),
    connectance (``connect``) and edge-contrast (``econ``, ``cwed``,
    ``teci``) metrics. Extras named in ``what`` are always selected.

    Returns
    -------
    pandas.DataFrame
        Columns ``metric``, ``name``, ``type``, ``level``, ``extra`` (not in
        landscapemetrics) and ``function_name``.
    """
    return select_metrics(level, metric, name, type, what, extras).drop(columns="family")


# -- aliases ----------------------------------------------------------------------

# Full names (pylandstats' methods, FRAGSTATS' metric names) of the
# landscapemetrics abbreviations, per level.
_PATCH_NAMES = {
    "perimeter": "perim",
    "perimeter_area_ratio": "para",
    "shape_index": "shape",
    "fractal_dimension": "frac",
    "fractal_dimension_index": "frac",
    "core_area": "core",
    "number_of_core_areas": "ncore",
    "core_area_index": "cai",
    "euclidean_nearest_neighbor": "enn",
    "euclidean_nearest_neighbor_distance": "enn",
    "radius_of_gyration": "gyrate",
    "contiguity_index": "contig",
    "continguity_index": "contig",  # pylandstats' spelling
    "related_circumscribing_circle": "circle",
    "disjunct_core_area": "dcore",
    "proximity_index": "prox",
    "proximity": "prox",
    "similarity_index": "simi",
    "edge_contrast_index": "econ",
}
_CLASS_NAMES = {
    "total_area": "ca",
    "class_area": "ca",
    "proportion_of_landscape": "pland",
    "percentage_of_landscape": "pland",
    "number_of_patches": "np",
    "patch_density": "pd",
    "largest_patch_index": "lpi",
    "total_edge": "te",
    "edge_density": "ed",
    "total_core_area": "tca",
    "core_area_proportion_of_landscape": "cpland",
    "core_area_percentage_of_landscape": "cpland",
    "number_of_disjunct_core_areas": "ndca",
    "disjunct_core_area_density": "dcad",
    "landscape_shape_index": "lsi",
    "normalized_landscape_shape_index": "nlsi",
    "effective_mesh_size": "mesh",
    "splitting_index": "split",
    "landscape_division_index": "division",
    "division_index": "division",
    "aggregation_index": "ai",
    "clumpiness_index": "clumpy",
    "patch_cohesion_index": "cohesion",
    "interspersion_juxtaposition_index": "iji",
    "interspersion_and_juxtaposition_index": "iji",
    "percentage_of_like_adjacencies": "pladj",
    "perimeter_area_fractal_dimension": "pafrac",
    "connectance_index": "connect",
    "connectance": "connect",
    "contrast_weighted_edge_density": "cwed",
    "total_edge_contrast_index": "teci",
}
_LANDSCAPE_NAMES = {
    **{k: v for k, v in _CLASS_NAMES.items() if v not in ("ca", "pland", "cpland", "nlsi", "clumpy")},
    "total_area": "ta",
    "entropy": "ent",
    "shannon_diversity_index": "shdi",
    "shannon_evenness_index": "shei",
    "simpson_diversity_index": "sidi",
    "simpson_evenness_index": "siei",
    "modified_simpson_diversity_index": "msidi",
    "modified_simpson_evenness_index": "msiei",
    "patch_richness": "pr",
    "patch_richness_density": "prd",
    "relative_patch_richness": "rpr",
    "joint_entropy": "joinent",
    "conditional_entropy": "condent",
    "mutual_information": "mutinf",
    "relative_mutual_information": "relmutinf",
    "contagion": "contag",
}
_LEVEL_NAMES = {"patch": _PATCH_NAMES, "class": _CLASS_NAMES, "landscape": _LANDSCAPE_NAMES}


def canonical_family(family: str) -> str:
    """landscapemetrics family of a family name or alias (``"shape_index"``
    -> ``"shape"``, ``"AREA"`` -> ``"area"``)."""
    low = family.lower()
    return _PATCH_NAMES.get(low, low)


def canonical_name(name: str, level: str) -> str:
    """landscapemetrics abbreviation of a metric name at ``level``.

    Accepts FRAGSTATS abbreviations in any case (``"SHDI"``, ``"AREA_MN"``)
    and the full names of pylandstats / FRAGSTATS (``"shannon_diversity_index"``,
    ``"shape_index_am"``, ``"total_area"``: ``ca`` at the class level, ``ta``
    at the landscape level). Other names are returned lower-cased.
    """
    low = name.lower()
    table = _LEVEL_NAMES[level]
    if low in table:
        return table[low]
    if level != "patch":
        base, _, stat = low.rpartition("_")
        if base and stat in STATS:
            return f"{_PATCH_NAMES.get(base, base)}_{stat}"
    return low
