# SPDX-License-Identifier: GPL-3.0-or-later
"""The tools of landschaft a task may call by name (``ls.parallel.map``, ``monte_carlo``,
``tiled``): analyses of data objects, never readers, writers or downloads. Tools that
work in a moving neighbourhood also say how many cells around a tile they need
(its halo), so that a map cut into tiles gives the same values as the whole map.
"""

from __future__ import annotations

import importlib
import math

_TOP = ("accounting", "clustering_fac", "contagion_pff", "contortion", "distance_to_resistance",
        "euclidean_distance", "fad", "fad_multiscale", "fos", "gray_spatial_convolution", "gtn_components",
        "gtn_importance", "influence_zones", "landscape_mosaic", "landscape_mosaic_heatmap", "landscape_mosaic_stats",
        "mspa", "parcellation", "proximity", "reclassify", "restoration_status", "spa", "spatial_entropy",
        "spatial_entropy_map", "landscape_metrics", "window_metrics", "intactness", "zonal_stats")
_MODULES = {
    "surface": ("aacf", "bearing_area", "best_fit_plane", "curvature", "dissection", "find_peaks", "find_valleys",
                "fit_plane", "flat_area", "geomorphons", "gravity", "hilltop", "height_ba", "hli", "hsp", "metrics",
                "remove_plane", "s10z", "sa", "sbi", "sci", "scl", "sdc", "sdq", "sdq6", "sdr", "sds", "sfd", "sk",
                "sku", "smean", "sph", "spherical_sd", "spk", "sq", "srr", "srw", "ssc", "ssk", "std", "stxr",
                "surface_area", "sv", "svi", "svk", "terrain", "tpi", "trasp", "tri", "vrm", "window_metrics"),
    "hydro": ("breach", "connectivity_index", "distance_to_channel", "fill", "flow_accumulation", "flow_direction",
              "hand", "spi", "sti", "streams", "twi"),
    "services": ("carbon", "habitat_quality", "sdr", "valuation_constant", "water_yield"),
    "nlm": ("binarize", "classify", "curds", "distance_gradient", "edge_gradient", "fbm", "gaussian_field", "merge",
            "mosaic_field", "mosaic_gibbs", "mosaic_tess", "mpd", "neigh", "percolation", "planar_gradient", "random",
            "random_cluster", "random_rectangular_cluster", "rescale"),
    "circuit": ("omniscape", "circuitscape_advanced", "circuitscape_pairwise", "circuitscape_one_to_all",
                "source_from_resistance"),
}

TOOLS: dict = {name: ("landschaft", name) for name in _TOP}
for _mod, _names in _MODULES.items():
    for _n in _names:
        TOOLS[f"{_mod}.{_n}"] = (f"landschaft.{_mod}", _n)


def resolve(name: str):
    """The function of a registered tool."""
    spec = TOOLS.get(str(name))
    if spec is None:
        raise ValueError(f"{name!r} is not a tool that can run on helpers; see ls.parallel.TOOLS")
    return getattr(importlib.import_module(spec[0]), spec[1])


def name_of(tool) -> str:
    """A registered name for a tool given by name or as the function."""
    if isinstance(tool, str):
        resolve(tool)
        return tool
    for name, (module, attr) in TOOLS.items():
        if getattr(importlib.import_module(module), attr, None) is tool:
            return name
    raise ValueError(f"{getattr(tool, '__name__', tool)!r} is not a tool that can run on helpers "
                     "(see ls.parallel.TOOLS)")


def check_strings(value, path: str = "argument") -> None:
    """Strings sent to a helper name options, never files on it."""
    if isinstance(value, str):
        if any(c in value for c in "/\\:") or value.startswith("~") or value in (".", ".."):
            raise ValueError(f"{path}: {value!r} looks like a file path; tools on helpers take data, not files")
    elif isinstance(value, (list, tuple)):
        for i, x in enumerate(value):
            check_strings(x, f"{path}[{i}]")
    elif isinstance(value, dict):
        for k, x in value.items():
            check_strings(x, f"{path}[{k!r}]")


# --------------------------------------------------------------------------- halos

def _cells(v, res: float, units: str = "cells") -> int:
    return int(math.ceil(float(v) / res)) if units == "map" else int(math.ceil(float(v)))


HALO = {
    "fad": lambda p, res: int(p.get("window_size", 7)) // 2 + 1,
    "window_metrics": lambda p, res: int(p.get("window", p.get("window_size", 5))) // 2 + 1,
    "surface.geomorphons": lambda p, res: _cells(p.get("search", 3), res, p.get("units", "cells")) + 1,
    # (tpi, vrm, srr, hsp and spherical_sd sum their windows by running sums along the rows: a tile
    # changes where they start, so their tiles agree to about 1e-12 only; give them a halo yourself)
    "surface.tri": lambda p, res: int(p.get("s", 3)) // 2 + 1,
    "surface.dissection": lambda p, res: int(p.get("s", 5)) // 2 + 1,
    "surface.terrain": lambda p, res: 2,
    "surface.curvature": lambda p, res: 2,
    "surface.trasp": lambda p, res: 2,
    "services.carbon": lambda p, res: 0,
}
